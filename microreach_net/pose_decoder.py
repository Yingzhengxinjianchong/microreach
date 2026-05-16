"""
microreach_net/pose_decoder.py

Pose-Conditioned Cross-Attention Decoder（创新 2）。

输入:
    local_feat:  (B, M, feat_dim)         来自 backbone 的局部几何特征
    queries:     (B, M, K, 4)             K=24, 每个 (ψx, ψy, ψz, g_idx)
输出:
    out:         (B, M, K, feat_dim)      每个 (point, query) 的条件特征

机制:
    1. Query 编码: (ψ, g) → token (B, M, K, pose_token_dim)
       - ψ ∈ S²: 球坐标 (θ, φ) + 正弦位置编码 → 64 维
       - g ∈ {0,1,2}: nn.Embedding(3, 64)
       - concat → 128 维 token
    2. Cross-attention: query = pose token, key/value = repeat 的 local_feat
    3. 残差 + FFN，n_layers 层

这是 M1 的核心模块；M0 在 train.py 里旁路掉这部分（直接拿 local_feat 做单值预测）。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Query 编码 (ψ, g) -> token
# ──────────────────────────────────────────────

class QueryEncoder(nn.Module):
    """
    把 (ψ_x, ψ_y, ψ_z, g_idx) 编码成 pose_token_dim 维向量。
    """

    def __init__(self, pose_token_dim: int = 128, n_g: int = 3, n_freq: int = 4):
        super().__init__()
        assert pose_token_dim % 2 == 0
        half = pose_token_dim // 2

        self.n_freq = n_freq
        # ψ -> half 维（球坐标 + 正弦位置编码）
        # 输入特征: (θ, φ) + sin/cos 频率展开 = 2 + 2*2*n_freq
        psi_in_dim = 2 + 2 * 2 * n_freq
        self.psi_mlp = nn.Sequential(
            nn.Linear(psi_in_dim, half),
            nn.GELU(),
            nn.Linear(half, half),
        )

        # g -> half 维（embedding）
        self.g_embed = nn.Embedding(n_g, half)

        # 频率（log-linear 2^0 ... 2^(n_freq-1)）
        freqs = 2.0 ** torch.arange(n_freq, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        queries: (B, M, K, 4)   最后一维 (ψ_x, ψ_y, ψ_z, g_idx)
        return:  (B, M, K, pose_token_dim)
        """
        psi = queries[..., :3]                                    # (B, M, K, 3)
        g_idx = queries[..., 3].long().clamp(0, self.g_embed.num_embeddings - 1)

        # ψ -> 球坐标 (θ=arcsin(z), φ=atan2(y, x))
        # ψ 已归一化（fibonacci_sphere 保证）
        z = psi[..., 2].clamp(-1.0, 1.0)
        theta = torch.asin(z)
        phi = torch.atan2(psi[..., 1], psi[..., 0])

        # 正弦位置编码：[θ, φ, sin(freqs*θ), cos(freqs*θ), sin(freqs*φ), cos(freqs*φ)]
        # → 2 + 2*2*n_freq 维
        theta_freqs = theta.unsqueeze(-1) * self.freqs           # (..., n_freq)
        phi_freqs = phi.unsqueeze(-1) * self.freqs

        psi_features = torch.cat([
            theta.unsqueeze(-1), phi.unsqueeze(-1),
            torch.sin(theta_freqs), torch.cos(theta_freqs),
            torch.sin(phi_freqs), torch.cos(phi_freqs),
        ], dim=-1)                                                # (B, M, K, 2 + 4*n_freq)

        psi_token = self.psi_mlp(psi_features)                    # (B, M, K, half)
        g_token = self.g_embed(g_idx)                             # (B, M, K, half)
        return torch.cat([psi_token, g_token], dim=-1)            # (B, M, K, pose_token_dim)


# ──────────────────────────────────────────────
# Cross-Attention Block
# ──────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    单层 cross-attention + FFN，带残差和 LayerNorm（Pre-LN）。
    """

    def __init__(self, dim: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B*M, K, dim), kv: (B*M, 1, dim)
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))
        return q


# ──────────────────────────────────────────────
# Pose Decoder 主体
# ──────────────────────────────────────────────

class PoseConditionedDecoder(nn.Module):
    """
    输入:
        local_feat: (B, M, feat_dim)
        queries:    (B, M, K, 4)
    输出:
        cond_feat:  (B, M, K, feat_dim)
    """

    def __init__(
        self,
        feat_dim: int = 128,
        pose_token_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_g: int = 3,
    ):
        super().__init__()
        assert pose_token_dim == feat_dim, (
            f"为简化，要求 pose_token_dim==feat_dim，得到 {pose_token_dim} vs {feat_dim}"
        )

        self.query_encoder = QueryEncoder(pose_token_dim=pose_token_dim, n_g=n_g)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim=feat_dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        local_feat: torch.Tensor,      # (B, M, feat_dim)
        queries: torch.Tensor,         # (B, M, K, 4)
    ) -> torch.Tensor:
        B, M, K, _ = queries.shape
        D = local_feat.shape[-1]

        # 1. Query 编码 -> (B, M, K, D)
        q_tokens = self.query_encoder(queries)

        # 2. Cross-attention：把 local_feat 当成单 token 的 KV
        # reshape 让每 (B, M) 独立做 attention
        q = q_tokens.reshape(B * M, K, D)                          # (B*M, K, D)
        kv = local_feat.reshape(B * M, 1, D)                       # (B*M, 1, D)

        for block in self.blocks:
            q = block(q, kv)

        return q.reshape(B, M, K, D)                               # (B, M, K, D)


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PoseConditionedDecoder smoke test ===")
    decoder = PoseConditionedDecoder(feat_dim=128, pose_token_dim=128,
                                     n_heads=4, n_layers=2)
    decoder.eval()

    B, M, K = 2, 16, 24
    local_feat = torch.randn(B, M, 128)
    queries = torch.randn(B, M, K, 4)
    # 把 ψ 归一化
    psi = queries[..., :3]
    queries = torch.cat([
        psi / psi.norm(dim=-1, keepdim=True).clamp(min=1e-6),
        torch.randint(0, 3, (B, M, K, 1)).float(),
    ], dim=-1)

    with torch.no_grad():
        out = decoder(local_feat, queries)

    print(f"  input  local_feat: {tuple(local_feat.shape)}")
    print(f"  input  queries:    {tuple(queries.shape)}")
    print(f"  output cond_feat:  {tuple(out.shape)}")
    print(f"  num params: {sum(p.numel() for p in decoder.parameters()):,}")
    print("[OK] PoseConditionedDecoder smoke test passed")
