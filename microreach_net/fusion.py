"""
microreach_net/fusion.py

Cross-Scale Geometry Fusion（创新 1）。

阶段二中期：use_global_branch=False → 直接返回 local_feat（identity passthrough）
阶段三/末期：use_global_branch=True → 加入 Sparse Conv U-Net 全局分支，
            concat 后 MLP 融合（公式见 default.yaml 注释）

把这个抽出来单独一个模块的原因：以后切换全局分支只改这一个文件。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossScaleFusion(nn.Module):
    """
    阶段二中期是 identity；阶段三可扩展为 local + global 融合。

    Args:
        feat_dim:           输入/输出特征维度
        use_global_branch:  False → identity；True → 加 global 分支（未实现）
    """

    def __init__(
        self,
        feat_dim: int = 128,
        use_global_branch: bool = False,
    ):
        super().__init__()
        self.use_global_branch = use_global_branch
        self.feat_dim = feat_dim

        if use_global_branch:
            raise NotImplementedError(
                "Global Sparse Conv U-Net branch 推至阶段三。"
                "想现在启用：实现 SparseConvBranch 并在这里 fuse 后返回。"
            )

    def forward(
        self,
        local_feat: torch.Tensor,      # (B, M, feat_dim)
        point_cloud: torch.Tensor,     # (B, N, 3) 占位，未来 global branch 用
        candidate_p: torch.Tensor,     # (B, M, 3)  占位
    ) -> torch.Tensor:
        return local_feat


if __name__ == "__main__":
    print("=== Fusion smoke test ===")
    fuse = CrossScaleFusion(feat_dim=128, use_global_branch=False)
    local = torch.randn(2, 16, 128)
    pc = torch.randn(2, 30000, 3)
    cp = torch.randn(2, 16, 3)
    out = fuse(local, pc, cp)
    print(f"  identity passthrough: in {tuple(local.shape)} -> out {tuple(out.shape)}")
    print("[OK] Fusion smoke test passed")
