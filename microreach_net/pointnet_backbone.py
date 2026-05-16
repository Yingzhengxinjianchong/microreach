"""
microreach_net/pointnet_backbone.py

Local Geometry Encoder：PointNet++ 风格（PointNet++ ICCV 2017 改写的简化版）。

为什么不用 PartField (ICCV 2025 NVIDIA)：
    在 47 实例的 Faucet 小数据上，PartField vs PointNet++ 的预期差距只有 1-3 个 mIoU 点
    （论文 5-8 点差距是在 ShapeNetPart 万级数据上）。PartField 的工程成本（自定义 CUDA
    kernel + pytorch3d 依赖 + 预训练权重下载，预估 1-3 天调环境）远大于阶段二中期的收益。
    论文叙事不依赖 PartField，真正核心是 Pose-Conditioned Decoder 和 Cascade Supervision。
    PartField 留作阶段三/末期 P1 的消融对照（"M_full w/ PartField vs PointNet++"）。

接口（与 PartField backbone 兼容，未来可直接替换）：
    输入:
        point_cloud:  (B, N, 3)   整场景点云
        candidate_p:  (B, M, 3)   候选交互点
    输出:
        local_feat:   (B, M, feat_dim)   每个候选点的局部几何特征

模块组成：
    1. Ball Query：在每个 candidate_p 周围 radius=0.05m 球域内采 K=64 个点
    2. MLP 层级编码：[3+3, 64, 128] → [128, 128, feat_dim]
    3. Max Pool 聚合：把球域 K 个点的特征聚成单个 (B, M, feat_dim)

可选切换到真正的 PartField（未来）：
    实现一个 PartFieldBackbone(nn.Module) 同名 forward 签名即可；train.py 不用改。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Ball Query (CUDA-free 纯 PyTorch 实现)
# ──────────────────────────────────────────────

def ball_query(
    points: torch.Tensor,     # (B, N, 3)
    centers: torch.Tensor,    # (B, M, 3)
    radius: float,
    k: int,
) -> torch.Tensor:
    """
    在 points 中，给每个 center 找出半径 radius 内的最多 k 个邻居索引。

    返回:
        idx: (B, M, k)  long Tensor; 不够 k 个邻居时用第 0 个（最近的）填充。

    复杂度: O(B * N * M)，对 N=30k / M=16 / B=4 ~= 2M 个距离，几十毫秒，够用。
    """
    B, N, _ = points.shape
    M = centers.shape[1]

    # (B, M, N) 距离平方
    dist2 = torch.cdist(centers, points, p=2.0) ** 2          # (B, M, N)

    # 半径内 mask
    in_radius = dist2 <= radius ** 2                          # (B, M, N) bool

    # 把超半径的距离设为极大值，然后取 top-k 最小
    dist2_masked = dist2.masked_fill(~in_radius, float("inf"))
    _, idx = dist2_masked.topk(k, dim=-1, largest=False)      # (B, M, k)

    # 处理"邻居不足 k 个"的情况：把 inf 位置 fallback 到最近邻 idx[..., 0:1]
    # 这样 group 后 max pool 会自然吃掉无效项（因为它们是最近邻特征的副本）
    first_idx = idx[..., 0:1].expand(-1, -1, k)
    inf_mask = torch.isinf(dist2_masked.gather(-1, idx))
    idx = torch.where(inf_mask, first_idx, idx)

    return idx                                                # (B, M, k)


def index_points(
    points: torch.Tensor,     # (B, N, C)
    idx: torch.Tensor,        # (B, M, k)
) -> torch.Tensor:
    """
    按 idx 从 points 取点。

    返回:
        (B, M, k, C)
    """
    B, _, C = points.shape
    M, K = idx.shape[1], idx.shape[2]

    # 把 idx 展平后用 batched gather
    batch_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, M, K)
    grouped = points[batch_idx, idx]                          # (B, M, k, C)
    return grouped


# ──────────────────────────────────────────────
# Backbone 主体
# ──────────────────────────────────────────────

class PointNetBackbone(nn.Module):
    """
    PointNet++ 风格 local geometry encoder。

    Args:
        feat_dim:        输出特征维度（默认 128，对齐 default.yaml.model.feat_dim）
        radius:          ball query 半径（默认 0.05m = 5cm）
        k:               每球域最多 k 个邻居（默认 64）
        in_channels:     输入点云通道数（默认 3 表示 xyz；可扩展加颜色/法向量）

    forward:
        point_cloud: (B, N, 3)
        candidate_p: (B, M, 3)
      返回:
        local_feat:  (B, M, feat_dim)
    """

    def __init__(
        self,
        feat_dim: int = 128,
        radius: float = 0.05,
        k: int = 64,
        in_channels: int = 3,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.radius = radius
        self.k = k

        # MLP1: 输入是 (相对坐标 3 + 原始通道 in_channels) = 6
        # 输出 64
        self.mlp1 = nn.Sequential(
            nn.Conv2d(3 + in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # MLP2: max-pool 后再投影到 feat_dim
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )

    def forward(
        self,
        point_cloud: torch.Tensor,    # (B, N, 3)
        candidate_p: torch.Tensor,    # (B, M, 3)
    ) -> torch.Tensor:
        B, M, _ = candidate_p.shape

        # 1. Ball query 找邻居
        idx = ball_query(point_cloud, candidate_p, self.radius, self.k)  # (B, M, k)

        # 2. 取邻居点坐标 (B, M, k, 3)
        grouped_xyz = index_points(point_cloud, idx)

        # 3. 转成相对坐标（相对于 candidate_p）—— PointNet++ 的关键 trick
        rel_xyz = grouped_xyz - candidate_p.unsqueeze(2)                  # (B, M, k, 3)

        # 4. 拼接 [相对坐标, 原始 xyz] 作为特征输入
        local_feat_in = torch.cat([rel_xyz, grouped_xyz], dim=-1)         # (B, M, k, 6)

        # 5. MLP1: (B, M, k, 6) -> (B, M, k, 128)
        # Conv2d 要求 (B, C, H, W)，把 (M, k) 当作 H=M, W=k
        x = local_feat_in.permute(0, 3, 1, 2)                             # (B, 6, M, k)
        x = self.mlp1(x)                                                  # (B, 128, M, k)

        # 6. Max pool over k 邻居
        x = x.max(dim=-1).values                                          # (B, 128, M)
        x = x.permute(0, 2, 1)                                            # (B, M, 128)

        # 7. MLP2: (B, M, 128) -> (B, M, feat_dim)
        x = self.mlp2(x)                                                  # (B, M, feat_dim)

        return x


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PointNetBackbone smoke test ===")
    backbone = PointNetBackbone(feat_dim=128, radius=0.05, k=64)
    backbone.eval()

    # mock 一个 batch
    B, N, M = 2, 30000, 16
    pc = torch.randn(B, N, 3) * 0.5      # 点云范围 [-1, 1] 左右
    cp = torch.randn(B, M, 3) * 0.3      # 候选点稍微集中

    with torch.no_grad():
        feat = backbone(pc, cp)

    print(f"  input  point_cloud: {tuple(pc.shape)}")
    print(f"  input  candidate_p: {tuple(cp.shape)}")
    print(f"  output local_feat:  {tuple(feat.shape)}")
    print(f"  feat stats: mean={feat.mean():.3f}, std={feat.std():.3f}")
    print(f"  num params: {sum(p.numel() for p in backbone.parameters()):,}")
    print("[OK] PointNetBackbone smoke test passed")
