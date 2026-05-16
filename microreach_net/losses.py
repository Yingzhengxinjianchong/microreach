"""
microreach_net/losses.py

Loss 函数：阶段二中期只用 L_geom (BCEWithLogits with mask)。

cascade 一致性 loss (R_exec <= R_contact <= R_geom) 和 L_contact/L_exec 留接口给阶段三。

为什么 R_geom 是 [0, 1] 连续值也用 BCE 而不是 MSE：
    BCE 对 soft label 是合法的 (二项交叉熵的"成功概率"形式)，且在 [0, 1] 区间梯度
    比 MSE 更平滑、不易饱和。这是 affordance/reachability 任务的标准做法。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,        # (..., ) 任意 shape
    targets: torch.Tensor,       # 同 logits shape, 值 ∈ [0, 1]
    mask: torch.Tensor,          # broadcast 到 logits shape, 1=valid, 0=ignore
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    带 mask 的 BCEWithLogitsLoss，只在 mask=1 的位置算 loss，并按有效元素数归一化。

    支持 soft label (targets ∈ [0, 1])。
    """
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    raw = raw * mask
    return raw.sum() / (mask.sum() + eps)


def masked_focal_loss_with_logits(
    logits: torch.Tensor,        # (..., )
    targets: torch.Tensor,       # 同 logits shape, soft label ∈ [0, 1]
    mask: torch.Tensor,          # broadcast 到 logits shape
    alpha: float = 0.75,         # 正样本权重（数据 27% 正 → 偏向正样本）
    gamma: float = 2.0,          # focal 调节强度（论文默认）
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Soft-label adapted Focal Loss (Lin et al. ICCV 2017).

    解决 R_geom 数据 mean=0.155 + BCE → sigmoid 被压低 + 二值化后 mIoU 极低 的问题。

    公式（soft label 版）：
        p = sigmoid(logits)
        loss = -[α·(1-p)^γ·y·log(p) + (1-α)·p^γ·(1-y)·log(1-p)]

    关键点：
        - α=0.75: 正样本（高 R_geom）loss 加权，补偿样本不平衡
        - γ=2.0: 难样本（pred 离 gt 远）loss 加权，pull pred 靠近 gt
        - soft label 适配：保留 y · log(p) + (1-y) · log(1-p) 的连续性
    """
    p = torch.sigmoid(logits)
    p = p.clamp(min=eps, max=1.0 - eps)

    # soft-label cross entropy（不离散化）
    ce_pos = -targets * torch.log(p)               # y · log(p)
    ce_neg = -(1.0 - targets) * torch.log(1.0 - p)  # (1-y) · log(1-p)

    # focal factor: 难样本（pred 偏离）权重 ↑
    focal_pos = (1.0 - p) ** gamma
    focal_neg = p ** gamma

    # alpha 平衡：正样本 α，负样本 (1-α)
    loss = alpha * focal_pos * ce_pos + (1.0 - alpha) * focal_neg * ce_neg

    loss = loss * mask
    return loss.sum() / (mask.sum() + eps)


def microreach_loss_geom_only(
    pred_geom: torch.Tensor,         # (B, M, K) for M1 or (B, M) for M0  — logits
    target_geom: torch.Tensor,       # 同 pred_geom shape — ∈ [0, 1]
    mask: torch.Tensor,              # (B, M)  1=valid candidate point, 0=pad
    loss_type: str = "bce",          # "bce" | "focal"
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """
    阶段二中期主 loss。

    Args:
        loss_type: "bce" (默认，与之前训练一致) 或 "focal" (修复 mIoU 偏低)

    返回 dict 方便 train.py 打 W&B：
        'loss':        总 loss（即 L_geom）
        'l_geom':      同上
    """
    if pred_geom.dim() == 3:
        # M1: (B, M, K)
        mask_expand = mask.unsqueeze(-1).expand_as(pred_geom)         # (B, M, K)
    else:
        # M0: (B, M)
        mask_expand = mask                                            # (B, M)

    if loss_type == "bce":
        l_geom = masked_bce_with_logits(pred_geom, target_geom, mask_expand)
    elif loss_type == "focal":
        l_geom = masked_focal_loss_with_logits(
            pred_geom, target_geom, mask_expand,
            alpha=focal_alpha, gamma=focal_gamma,
        )
    else:
        raise ValueError(f"loss_type 必须是 'bce' 或 'focal'，得到 {loss_type!r}")

    return {
        "loss": l_geom,
        "l_geom": l_geom.detach(),
    }


def microreach_loss_full(
    pred_geom: torch.Tensor,
    pred_contact: torch.Tensor,
    pred_exec: torch.Tensor,
    target_geom: torch.Tensor,
    target_contact: torch.Tensor,
    target_exec: torch.Tensor,
    mask: torch.Tensor,
    mu: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    完整三层级联 loss（阶段三启用）。

    L = L_geom + L_contact + L_exec
        + mu * ( relu(pred_contact - pred_geom).mean()         (cascade1)
               + relu(pred_exec - pred_contact).mean() )       (cascade2)

    注意：cascade 项用 sigmoid 后的概率，不是 logits。
    """
    mask_e = mask.unsqueeze(-1).expand_as(pred_geom)

    l_geom    = masked_bce_with_logits(pred_geom,    target_geom,    mask_e)
    l_contact = masked_bce_with_logits(pred_contact, target_contact, mask_e)
    l_exec    = masked_bce_with_logits(pred_exec,    target_exec,    mask_e)

    p_geom    = torch.sigmoid(pred_geom)
    p_contact = torch.sigmoid(pred_contact)
    p_exec    = torch.sigmoid(pred_exec)

    cascade1 = F.relu(p_contact - p_geom)    # 应满足 p_contact <= p_geom
    cascade2 = F.relu(p_exec - p_contact)    # 应满足 p_exec <= p_contact

    # 只在有效位置算
    cascade1 = (cascade1 * mask_e).sum() / (mask_e.sum() + 1e-8)
    cascade2 = (cascade2 * mask_e).sum() / (mask_e.sum() + 1e-8)
    l_cascade = mu * (cascade1 + cascade2)

    total = l_geom + l_contact + l_exec + l_cascade
    return {
        "loss":      total,
        "l_geom":    l_geom.detach(),
        "l_contact": l_contact.detach(),
        "l_exec":    l_exec.detach(),
        "l_cascade": l_cascade.detach(),
    }


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Losses smoke test ===")

    B, M, K = 2, 16, 24

    # M1 测试
    pred_m1 = torch.randn(B, M, K)
    target_m1 = torch.rand(B, M, K)
    mask = torch.zeros(B, M)
    mask[:, :10] = 1.0   # 前 10 个 candidate point 有效
    out = microreach_loss_geom_only(pred_m1, target_m1, mask)
    print(f"  M1 mode: l_geom = {out['loss'].item():.4f}")

    # 验证 mask 起作用：把无效位置改成乱数后 loss 应该不变
    pred_m1_modified = pred_m1.clone()
    pred_m1_modified[:, 10:] = torch.randn_like(pred_m1_modified[:, 10:]) * 100
    out2 = microreach_loss_geom_only(pred_m1_modified, target_m1, mask)
    assert abs(out['loss'].item() - out2['loss'].item()) < 1e-5, \
        f"mask 不起作用！{out['loss'].item()} vs {out2['loss'].item()}"
    print(f"  M1 mask check: loss 不变 (改 invalid 位置 loss 仍 {out2['loss'].item():.4f})")

    # M0 测试
    pred_m0 = torch.randn(B, M)
    target_m0 = torch.rand(B, M)
    out_m0 = microreach_loss_geom_only(pred_m0, target_m0, mask)
    print(f"  M0 mode: l_geom = {out_m0['loss'].item():.4f}")

    print("[OK] Losses smoke test passed")
