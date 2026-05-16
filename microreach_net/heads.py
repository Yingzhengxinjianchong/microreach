"""
microreach_net/heads.py

三层级联预测头：Geom / Contact / Exec。
阶段二中期只用 GeomHead；ContactHead / ExecHead 留接口供阶段三启用。

接口约定：
    所有 head 的 forward 都输出 LOGITS（未过 sigmoid 的原始分数）。
    BCEWithLogitsLoss 在 loss 里做 sigmoid，数值更稳定。
    评测/可视化时调用 head.predict() 拿 [0, 1] 概率。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPHead(nn.Module):
    """
    单个预测头：feat_dim -> hidden -> 1
    """

    def __init__(self, feat_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 任意 shape (..., feat_dim)
        return: (..., ) 去掉最后一维
        """
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class GeomHead(_MLPHead):
    """R_geom logits (阶段二中期必用)"""
    pass


class ContactHead(_MLPHead):
    """R_contact logits (推至阶段三)"""
    pass


class ExecHead(_MLPHead):
    """R_exec logits (推至阶段三)"""
    pass


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Heads smoke test ===")
    geom = GeomHead(feat_dim=128, hidden_dim=64)
    geom.eval()

    # per_query 模式输入 (B, M, K, 128)
    cond_feat = torch.randn(2, 16, 24, 128)
    with torch.no_grad():
        logits = geom(cond_feat)
        prob = geom.predict(cond_feat)
    print(f"  per_query: in={tuple(cond_feat.shape)} -> logits={tuple(logits.shape)}, prob range [{prob.min():.3f}, {prob.max():.3f}]")

    # per_point_mean 模式输入 (B, M, 128) (M0)
    local_feat = torch.randn(2, 16, 128)
    with torch.no_grad():
        logits2 = geom(local_feat)
    print(f"  per_point: in={tuple(local_feat.shape)} -> logits={tuple(logits2.shape)}")

    print(f"  num params (geom): {sum(p.numel() for p in geom.parameters()):,}")
    print("[OK] Heads smoke test passed")
