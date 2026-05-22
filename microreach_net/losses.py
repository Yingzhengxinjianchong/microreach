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


def masked_dice_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice Loss for soft labels in [0, 1].

    Dice = 2 · sum(p · y) / (sum(p) + sum(y))
    Loss = 1 - Dice

    天然处理类别不平衡（少数类不会被多数类淹没）。Affordance / 医学图像分割
    的标准做法。TASA (AAAI 2026) 公式 11 用了这个。
    """
    p = torch.sigmoid(logits)
    p = p * mask
    y = targets * mask

    intersection = 2.0 * (p * y).sum()
    denom = p.sum() + y.sum() + eps

    return 1.0 - intersection / denom


def masked_iou_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft IoU Loss (Jaccard Loss) for soft labels.

    IoU = sum(p · y) / (sum(p) + sum(y) - sum(p · y))
    Loss = 1 - IoU

    直接优化 IoU 指标（让训练目标和 Micro-mIoU 评测目标对齐）。
    TASA (AAAI 2026) 公式 11 第 4 项。
    """
    p = torch.sigmoid(logits)
    p = p * mask
    y = targets * mask

    intersection = (p * y).sum()
    union = p.sum() + y.sum() - intersection + eps

    return 1.0 - intersection / union


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
    loss_type: str = "bce",          # "bce" | "focal" | "composite"
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    composite_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    阶段二中期主 loss。

    Args:
        loss_type:
            "bce"       — 单 BCE（baseline，与之前训练一致）
            "focal"     — 单 Focal（修复 mIoU 偏低，但 Recall 略降）
            "composite" — TASA (AAAI 2026) 公式 11 风格的复合 loss
                          L = w_bce·BCE + w_dice·Dice + w_focal·Focal + w_iou·IoU
                          默认权重 (0.3, 0.3, 0.2, 0.2) 来自 TASA 论文
        composite_weights: 自定义权重 dict，键 'bce'/'dice'/'focal'/'iou'
                           不传则用 TASA 默认值

    返回 dict（每个分量都返回，便于 W&B 拆分观察）：
        'loss':       总 loss
        'l_geom':     同 loss（兼容旧字段）
        'l_bce':      BCE 分量（仅 composite 模式）
        'l_dice':     Dice 分量
        'l_focal':    Focal 分量
        'l_iou':      IoU 分量
    """
    if pred_geom.dim() == 3:
        mask_expand = mask.unsqueeze(-1).expand_as(pred_geom)
    else:
        mask_expand = mask

    if loss_type == "bce":
        l_geom = masked_bce_with_logits(pred_geom, target_geom, mask_expand)
        return {"loss": l_geom, "l_geom": l_geom.detach()}

    elif loss_type == "focal":
        l_geom = masked_focal_loss_with_logits(
            pred_geom, target_geom, mask_expand,
            alpha=focal_alpha, gamma=focal_gamma,
        )
        return {"loss": l_geom, "l_geom": l_geom.detach()}

    elif loss_type == "composite":
        # TASA (AAAI 2026) Eq. 11: w_bce=0.3, w_dice=0.3, w_focal=0.2, w_iou=0.2
        weights = composite_weights or {"bce": 0.3, "dice": 0.3, "focal": 0.2, "iou": 0.2}

        l_bce   = masked_bce_with_logits(pred_geom, target_geom, mask_expand)
        l_dice  = masked_dice_loss_with_logits(pred_geom, target_geom, mask_expand)
        l_focal = masked_focal_loss_with_logits(
            pred_geom, target_geom, mask_expand,
            alpha=focal_alpha, gamma=focal_gamma,
        )
        l_iou   = masked_iou_loss_with_logits(pred_geom, target_geom, mask_expand)

        l_total = (weights["bce"]   * l_bce
                 + weights["dice"]  * l_dice
                 + weights["focal"] * l_focal
                 + weights["iou"]   * l_iou)

        return {
            "loss":    l_total,
            "l_geom":  l_total.detach(),
            "l_bce":   l_bce.detach(),
            "l_dice":  l_dice.detach(),
            "l_focal": l_focal.detach(),
            "l_iou":   l_iou.detach(),
        }

    else:
        raise ValueError(
            f"loss_type 必须是 'bce' | 'focal' | 'composite'，得到 {loss_type!r}"
        )


def microreach_loss_multihead(
    pred: Dict[str, torch.Tensor],          # {"geom": (B,M,K), "contact"?: ..., "exec"?: ...}
    target: Dict[str, torch.Tensor],         # 同结构（每个层独立标签）
    mask: torch.Tensor,                      # (B, M) 候选点 padding mask
    valid: Optional[Dict[str, torch.Tensor]] = None,  # {"contact"?: (B,M,K), "exec"?: ...} NaN-safe mask
    head_weights: Optional[Dict[str, float]] = None,
    cascade_consistency: bool = False,
    cascade_mu: float = 0.0,                 # 调用方传"已 warmup 后的 mu"
) -> Dict[str, torch.Tensor]:
    """
    阶段三多 head Loss（NaN-safe + cascade 一致性 + 可调权重）。

    NaN-safe：若某 head（如 contact）的 valid mask 全 0（hyh 还没补真值），
    该 head loss 直接给 0 但不影响其他 head 训练。
    cascade_mu：调用方负责 warmup 调度（前 50 epoch 给 0，之后线性 ramp）。

    Args:
        pred:    {"geom": ...} 必须；"contact"/"exec" 可选
        target:  与 pred 同 key
        mask:    (B, M) 1=valid candidate
        valid:   {head_name: (B, M, K)} 1=valid label position（NaN 处为 0）。
                 如不传则全部 valid。
        head_weights: 默认 0.5 (geom) + 0.3 (contact) + 0.3 (exec) - 多于 1 没关系
        cascade_consistency: 启用 cascade1 = relu(p_contact - p_geom)
                                + cascade2 = relu(p_exec - p_contact)
        cascade_mu: 当前 epoch 的 mu 值
    """
    if head_weights is None:
        head_weights = {"geom": 0.5, "contact": 0.3, "exec": 0.3}
    if valid is None:
        valid = {}

    # candidate-level mask broadcast 到 (B, M, K)
    mask_e = mask.unsqueeze(-1)                                     # (B, M, 1)

    out: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=mask.device, dtype=torch.float32)
    p_per_head: Dict[str, torch.Tensor] = {}

    for head_name in ["geom", "contact", "exec"]:
        if head_name not in pred:
            continue
        logits = pred[head_name]
        tgt    = target[head_name]
        w      = head_weights.get(head_name, 0.0)

        # 该 head 的逐位置 valid（默认全 valid）
        v = valid.get(head_name)
        if v is None:
            head_valid = mask_e.expand_as(logits)
        else:
            head_valid = v * mask_e                                 # (B, M, K)，相乘等价 AND

        # 完全没有 valid 标签（hyh 没补）→ loss = 0，不污染梯度
        if float(head_valid.sum().item()) < 0.5:
            l = torch.zeros((), device=mask.device, dtype=torch.float32)
        else:
            l = masked_bce_with_logits(logits, tgt, head_valid)

        out[f"l_{head_name}"] = l.detach()
        total = total + w * l
        p_per_head[head_name] = torch.sigmoid(logits)

    # Cascade 一致性正则
    if cascade_consistency and cascade_mu > 0.0:
        l_cascade = torch.zeros((), device=mask.device, dtype=torch.float32)
        # cascade1: p_contact <= p_geom
        if "geom" in p_per_head and "contact" in p_per_head:
            diff = F.relu(p_per_head["contact"] - p_per_head["geom"])
            l_cascade = l_cascade + (diff * mask_e).sum() / (mask_e.sum() * diff.size(-1) + 1e-8)
        # cascade2: p_exec <= p_contact
        if "contact" in p_per_head and "exec" in p_per_head:
            diff = F.relu(p_per_head["exec"] - p_per_head["contact"])
            l_cascade = l_cascade + (diff * mask_e).sum() / (mask_e.sum() * diff.size(-1) + 1e-8)
        total = total + cascade_mu * l_cascade
        out["l_cascade"] = l_cascade.detach()
    else:
        out["l_cascade"] = torch.zeros((), device=mask.device)

    out["loss"] = total
    return out


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

    # ─── 阶段三：multi-head loss 测试 ───
    print("\n=== Multi-head loss smoke test ===")

    # M2 (geom + contact)，contact 有 valid mask
    pred_m2 = {
        "geom":    torch.randn(B, M, K),
        "contact": torch.randn(B, M, K),
    }
    target_m2 = {
        "geom":    torch.rand(B, M, K),
        "contact": torch.rand(B, M, K),
    }
    valid_m2 = {
        "contact": torch.ones(B, M, K),
    }
    out_m2 = microreach_loss_multihead(
        pred_m2, target_m2, mask, valid=valid_m2,
        head_weights={"geom": 0.5, "contact": 0.3},
        cascade_consistency=True, cascade_mu=0.1,
    )
    print(f"  M2: total={out_m2['loss'].item():.4f}  "
          f"l_geom={out_m2['l_geom'].item():.4f}  "
          f"l_contact={out_m2['l_contact'].item():.4f}  "
          f"l_cascade={out_m2['l_cascade'].item():.4f}")
    assert out_m2["loss"].item() > 0

    # M_full (geom + contact + exec)
    pred_full = {**pred_m2, "exec": torch.randn(B, M, K)}
    target_full = {**target_m2, "exec": torch.rand(B, M, K)}
    valid_full = {**valid_m2, "exec": torch.ones(B, M, K)}
    out_full = microreach_loss_multihead(
        pred_full, target_full, mask, valid=valid_full,
        head_weights={"geom": 0.4, "contact": 0.3, "exec": 0.3},
        cascade_consistency=True, cascade_mu=0.1,
    )
    print(f"  M_full: total={out_full['loss'].item():.4f}  "
          f"l_geom={out_full['l_geom'].item():.4f}  "
          f"l_contact={out_full['l_contact'].item():.4f}  "
          f"l_exec={out_full['l_exec'].item():.4f}  "
          f"l_cascade={out_full['l_cascade'].item():.4f}")
    assert out_full["loss"].item() > 0

    # NaN-safe：contact head valid 全 0（hyh 没补标签）→ l_contact = 0
    valid_no_contact = {"contact": torch.zeros(B, M, K)}
    out_nan = microreach_loss_multihead(
        pred_m2, target_m2, mask, valid=valid_no_contact,
        head_weights={"geom": 0.5, "contact": 0.3},
    )
    assert out_nan["l_contact"].item() < 1e-6, "valid 全 0 时 l_contact 必须 = 0"
    print(f"  NaN-safe (contact valid=全0): l_contact={out_nan['l_contact'].item():.6f} (期望 0) ✓")

    # mu warmup = 0 时 cascade 项不参与
    out_no_cascade = microreach_loss_multihead(
        pred_m2, target_m2, mask, valid=valid_m2,
        cascade_consistency=True, cascade_mu=0.0,
    )
    assert out_no_cascade["l_cascade"].item() < 1e-6
    print(f"  mu=0 (warmup): l_cascade={out_no_cascade['l_cascade'].item():.6f} (期望 0) ✓")

    print("[OK] Multi-head loss smoke test passed")
    print("[OK] Losses smoke test passed")
