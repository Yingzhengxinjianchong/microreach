"""
eval/metrics.py

MicroReach 全部评测指标实现（阶段二中期必做 + 阶段三预留接口）。

阶段二中期必做：
    - micro_miou         : Micro-mIoU
    - pose_aware_recall_at_k : Pose-Aware Recall@1

阶段三激活（接口预留，输入合法则返回正确值，否则返回 None）：
    - cascade_consistency_rate : Cascade Consistency Rate（依赖三层标签）
    - sim_exec_success         : Sim Execution Success Rate（依赖 Isaac Sim）

工具函数（两阶段共用）：
    - build_micro_mask  : 从 tier 列表生成 micro_mask
    - apply_threshold   : 连续预测值二值化
    - evaluate_instance : 对单个实例跑全部当期指标
    - evaluate_dataset  : 批量评测，输出汇总表

单元测试：python metrics.py（覆盖极端情况 + mock 数据）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

DEFAULT_THRESHOLD: float = 0.5     # 连续值 → 二值的默认阈值
CASCADE_EPS:       float = 0.05    # 级联一致性容差


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def apply_threshold(
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """
    将连续预测值 / GT 分数二值化。

    Args:
        scores:    任意 shape 的 float array，值域 [0,1]
        threshold: 默认 0.5

    Returns:
        bool array，同 shape
    """
    return scores > threshold


def build_micro_mask(
    tiers: Sequence[str],
    n_queries_per_point: int,
) -> np.ndarray:
    """
    从每个候选点的 tier 标注，生成与 pred/gt 形状匹配的 micro_mask。

    Args:
        tiers:               长度 M 的 tier 列表，每元素 "micro"|"meso"|"macro"
        n_queries_per_point: 每个候选点的 query 数 K（= n_psi × n_g，如 24）

    Returns:
        micro_mask: bool array of shape (M, K)
                    True 表示该候选点属于 micro 档
    """
    mask_per_point = np.array([t == "micro" for t in tiers], dtype=bool)  # (M,)
    return np.broadcast_to(mask_per_point[:, None], (len(tiers), n_queries_per_point)).copy()


# ──────────────────────────────────────────────
# 核心指标
# ──────────────────────────────────────────────

def micro_miou(
    pred:        np.ndarray,
    gt:          np.ndarray,
    micro_mask:  np.ndarray,
    threshold:   float = DEFAULT_THRESHOLD,
    eps:         float = 1e-8,
) -> float:
    """
    Micro-mIoU：仅在 micro_mask=True 的位置计算预测与 GT 的 IoU。

    严格按方案 §3 公式：
        pred_b = pred > threshold
        gt_b   = gt   > threshold
        intersect = (pred_b & gt_b & micro_mask).sum()
        union     = ((pred_b | gt_b) & micro_mask).sum()
        miou      = intersect / (union + eps)

    Args:
        pred:       shape (M, K)，模型预测的连续值 ∈ [0,1]
        gt:         shape (M, K)，标签值 ∈ [0,1]
        micro_mask: shape (M, K)，bool，True=micro part
        threshold:  二值化阈值
        eps:        防零除

    Returns:
        float ∈ [0, 1]

    Raises:
        ValueError: 形状不匹配
    """
    _check_shape(pred, gt, micro_mask, "micro_miou")

    pred_b = apply_threshold(pred, threshold)
    gt_b   = apply_threshold(gt,   threshold)

    intersect = (pred_b & gt_b & micro_mask).sum()
    union     = ((pred_b | gt_b) & micro_mask).sum()

    return float(intersect / (union + eps))


def macro_miou(
    pred:        np.ndarray,
    gt:          np.ndarray,
    macro_mask:  np.ndarray,
    threshold:   float = DEFAULT_THRESHOLD,
    eps:         float = 1e-8,
) -> float:
    """
    Macro-mIoU：同 micro_miou，仅用 macro_mask。
    用于验证大件不退化（对照指标）。
    """
    _check_shape(pred, gt, macro_mask, "macro_miou")
    pred_b = apply_threshold(pred, threshold)
    gt_b   = apply_threshold(gt,   threshold)
    intersect = (pred_b & gt_b & macro_mask).sum()
    union     = ((pred_b | gt_b) & macro_mask).sum()
    return float(intersect / (union + eps))


def meso_miou(
    pred:       np.ndarray,
    gt:         np.ndarray,
    meso_mask:  np.ndarray,
    threshold:  float = DEFAULT_THRESHOLD,
    eps:        float = 1e-8,
) -> float:
    """
    Meso-mIoU：同 micro_miou，仅用 meso_mask（对照指标）。
    """
    _check_shape(pred, gt, meso_mask, "meso_miou")
    pred_b = apply_threshold(pred, threshold)
    gt_b   = apply_threshold(gt,   threshold)
    intersect = (pred_b & gt_b & meso_mask).sum()
    union     = ((pred_b | gt_b) & meso_mask).sum()
    return float(intersect / (union + eps))


def soft_iou(
    pred:        np.ndarray,
    gt:          np.ndarray,
    tier_mask:   Optional[np.ndarray] = None,
    eps:         float = 1e-8,
) -> float:
    """
    Per-query Soft IoU (Jaccard)：连续值 IoU，不二值化。

    阶段三新增。修复中期 BCE 模型 sigmoid 被压低导致二值化 Micro-mIoU≈0 的问题。
    soft IoU 直接在 [0,1] 连续值上算交并，相对排序学对了就有合理分数。

    公式：
        soft_IoU = Σ min(p, y) / Σ max(p, y)

    Args:
        pred:      shape (M, K)，连续预测 ∈ [0,1]
        gt:        shape (M, K)，标签 ∈ [0,1]
        tier_mask: shape (M, K) bool；None 表示所有位置都算
        eps:       防零除

    Returns:
        float ∈ [0, 1]
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred.shape={pred.shape} ≠ gt.shape={gt.shape}")

    if tier_mask is None:
        tier_mask = np.ones_like(pred, dtype=bool)
    elif tier_mask.shape != pred.shape:
        raise ValueError(
            f"tier_mask.shape={tier_mask.shape} ≠ pred.shape={pred.shape}"
        )

    if not tier_mask.any():
        return float("nan")

    p = np.where(tier_mask, pred, 0.0)
    g = np.where(tier_mask, gt,   0.0)

    inter = np.minimum(p, g).sum()
    union = np.maximum(p, g).sum()

    return float(inter / (union + eps))


def pose_aware_recall_at_k(
    pred:      np.ndarray,
    gt:        np.ndarray,
    k:         int   = 1,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """
    Pose-Aware Recall@k：对每个候选点 p，取预测值最高的 k 个 query，
    检查其中是否包含 GT 可达的 query（gt > threshold），
    然后对所有候选点求平均。

    验证条件场是否真正学到方向选择性：
        若模型没有学到 ψ 的差异，top-k 会集中于某几个方向 → Recall 低。

    Args:
        pred:      shape (M, K)，连续预测值 ∈ [0,1]
        gt:        shape (M, K)，GT 标签值 ∈ [0,1]
        k:         阶段二中期 k=1；阶段三 k=5
        threshold: GT 二值化阈值

    Returns:
        float ∈ [0, 1]

    Raises:
        ValueError: 形状不匹配或 k 超界
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred.shape={pred.shape} ≠ gt.shape={gt.shape}")
    M, K = pred.shape
    if k < 1 or k > K:
        raise ValueError(f"k={k} 超出范围 [1, {K}]")

    # 对每个候选点，取 pred 最大的 k 个位置
    top_k_idx = np.argsort(pred, axis=-1)[:, -k:]    # (M, k)

    # 查这 k 个位置的 GT 值
    gt_at_top_k = np.take_along_axis(gt, top_k_idx, axis=-1)  # (M, k)

    # 只要 k 个中任意一个 GT > threshold，该候选点 Recall = 1
    hit = (gt_at_top_k > threshold).any(axis=-1)     # (M,)

    return float(hit.mean())


def cascade_consistency_rate(
    pred_geom:    np.ndarray,
    pred_contact: np.ndarray,
    pred_exec:    np.ndarray,
    eps:          float = CASCADE_EPS,
) -> Optional[float]:
    """
    Cascade Consistency Rate：验证三层物理约束
        R_exec ≤ R_contact ≤ R_geom

    阶段二中期接口预留（只有一层 R_geom 时调用会返回 None）。
    阶段三三层标签齐全后自动激活。

    Args:
        pred_geom:    shape (M, K)
        pred_contact: shape (M, K)，若为 None 则返回 None
        pred_exec:    shape (M, K)，若为 None 则返回 None
        eps:          物理先验容差，允许小量违反

    Returns:
        float ∈ [0, 1]，或 None（标签不足时）
    """
    if pred_contact is None or pred_exec is None:
        return None  # 接口预留，阶段三激活

    if not (pred_geom.shape == pred_contact.shape == pred_exec.shape):
        raise ValueError("三个 pred 的 shape 必须一致")

    # R_contact ≤ R_geom + eps
    c1 = (pred_contact <= pred_geom + eps).astype(float).mean()
    # R_exec ≤ R_contact + eps
    c2 = (pred_exec <= pred_contact + eps).astype(float).mean()
    return float((c1 + c2) / 2.0)


def sim_exec_success(
    isaac_results: Optional[dict],
) -> Optional[float]:
    """
    Sim Execution Success Rate（阶段三，依赖 Isaac Sim + MoveIt 闭环）。

    接口预留：调用方把 Isaac Sim 执行结果以字典形式传入：
        isaac_results = {
            "n_success": int,
            "n_total":   int,
        }

    Args:
        isaac_results: dict 或 None（阶段二中期传 None）

    Returns:
        float ∈ [0, 1]，或 None（阶段二中期）
    """
    if isaac_results is None:
        return None   # 阶段二中期接口预留
    n_total = isaac_results.get("n_total", 0)
    if n_total == 0:
        return None
    return float(isaac_results["n_success"] / n_total)


# ──────────────────────────────────────────────
# 单实例评测（组合调用）
# ──────────────────────────────────────────────

@dataclass
class InstanceMetrics:
    """单个实例的评测结果，方便汇总。"""
    instance_id:              str
    micro_miou:               float
    meso_miou:                Optional[float]
    macro_miou:               Optional[float]
    pose_aware_recall_at_1:   float
    pose_aware_recall_at_5:   Optional[float]   # 阶段三
    cascade_consistency_rate: Optional[float]   # 阶段三
    sim_exec_success:         Optional[float]   # 阶段三
    # 阶段三新增：soft IoU（不二值化，直接连续 IoU），修复 BCE sigmoid 压低问题
    micro_soft_iou:           Optional[float] = None
    meso_soft_iou:            Optional[float] = None
    macro_soft_iou:           Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_instance(
    instance_id:   str,
    pred_geom:     np.ndarray,         # (M, K)
    gt_geom:       np.ndarray,         # (M, K)
    tiers:         List[str],          # 长度 M，每元素 "micro"|"meso"|"macro"
    pred_contact:  Optional[np.ndarray] = None,
    gt_contact:    Optional[np.ndarray] = None,
    pred_exec:     Optional[np.ndarray] = None,
    gt_exec:       Optional[np.ndarray] = None,
    isaac_results: Optional[dict]       = None,
    threshold:     float = DEFAULT_THRESHOLD,
) -> InstanceMetrics:
    """
    对单个实例跑全部当期指标。

    阶段二中期：只需传 pred_geom, gt_geom, tiers。
    阶段三：追加传 pred_contact/gt_contact/pred_exec/gt_exec/isaac_results。

    Args:
        pred_geom:    模型预测 R_geom，shape (M, K)
        gt_geom:      R_geom 标签，shape (M, K)
        tiers:        长度 M，每个候选点的 tier 分档
        pred_contact: 选填，阶段三使用
        gt_contact:   选填，阶段三使用
        pred_exec:    选填，阶段三使用
        gt_exec:      选填，阶段三使用
        isaac_results:选填，阶段三使用
        threshold:    二值化阈值

    Returns:
        InstanceMetrics
    """
    M, K = pred_geom.shape
    if len(tiers) != M:
        raise ValueError(f"tiers 长度 {len(tiers)} ≠ M={M}")

    # 三种 mask
    micro_mask = build_micro_mask(tiers, K)
    meso_mask  = np.array([t == "meso"  for t in tiers], dtype=bool)[:, None]
    meso_mask  = np.broadcast_to(meso_mask, (M, K)).copy()
    macro_mask = np.array([t == "macro" for t in tiers], dtype=bool)[:, None]
    macro_mask = np.broadcast_to(macro_mask, (M, K)).copy()

    # 必做指标（阶段二中期）
    m_micro = micro_miou(pred_geom, gt_geom, micro_mask, threshold)
    recall1 = pose_aware_recall_at_k(pred_geom, gt_geom, k=1, threshold=threshold)

    # 对照指标（若对应 mask 非空则计算，否则 None）
    m_meso  = meso_miou(pred_geom, gt_geom, meso_mask, threshold)  \
              if meso_mask.any() else None
    m_macro = macro_miou(pred_geom, gt_geom, macro_mask, threshold) \
              if macro_mask.any() else None

    # 阶段三指标
    recall5 = pose_aware_recall_at_k(pred_geom, gt_geom, k=min(5, K), threshold=threshold) \
              if K >= 5 else None

    ccr = cascade_consistency_rate(pred_geom, pred_contact, pred_exec)

    ses = sim_exec_success(isaac_results)

    # 阶段三新增：soft IoU（修复 BCE 压低导致 Micro-mIoU≈0 的反直觉现象）
    s_micro = soft_iou(pred_geom, gt_geom, micro_mask) if micro_mask.any() else None
    s_meso  = soft_iou(pred_geom, gt_geom, meso_mask)  if meso_mask.any()  else None
    s_macro = soft_iou(pred_geom, gt_geom, macro_mask) if macro_mask.any() else None

    return InstanceMetrics(
        instance_id              = instance_id,
        micro_miou               = m_micro,
        meso_miou                = m_meso,
        macro_miou               = m_macro,
        pose_aware_recall_at_1   = recall1,
        pose_aware_recall_at_5   = recall5,
        cascade_consistency_rate = ccr,
        sim_exec_success         = ses,
        micro_soft_iou           = s_micro,
        meso_soft_iou            = s_meso,
        macro_soft_iou           = s_macro,
    )


# ──────────────────────────────────────────────
# 数据集汇总
# ──────────────────────────────────────────────

@dataclass
class DatasetMetrics:
    """数据集级别的平均指标，用于输出对比表。"""
    micro_miou:               float
    meso_miou:                Optional[float]
    macro_miou:               Optional[float]
    pose_aware_recall_at_1:   float
    pose_aware_recall_at_5:   Optional[float]
    cascade_consistency_rate: Optional[float]
    sim_exec_success:         Optional[float]
    n_instances:              int
    # 阶段三新增
    micro_soft_iou:           Optional[float] = None
    meso_soft_iou:            Optional[float] = None
    macro_soft_iou:           Optional[float] = None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

    def print_table_row(self, method_name: str) -> None:
        """打印对比表中的一行（与 eval_main.py 格式对齐）。"""
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "  N/A "
        print(f"  {method_name:20s} | {fmt(self.micro_miou)} | "
              f"{fmt(self.meso_miou)} | {fmt(self.macro_miou)} | "
              f"{fmt(self.micro_soft_iou)} | "
              f"{fmt(self.meso_soft_iou)} | "
              f"{fmt(self.pose_aware_recall_at_1)} | "
              f"{fmt(self.pose_aware_recall_at_5)} | "
              f"{fmt(self.cascade_consistency_rate)} | "
              f"{fmt(self.sim_exec_success)}")


def evaluate_dataset(
    instance_metrics_list: List[InstanceMetrics],
) -> DatasetMetrics:
    """
    汇总所有实例的指标，取算术平均（忽略 None）。

    Args:
        instance_metrics_list: list of InstanceMetrics

    Returns:
        DatasetMetrics
    """
    def _mean(attr: str) -> Optional[float]:
        vals = [getattr(m, attr) for m in instance_metrics_list
                if getattr(m, attr) is not None]
        return float(np.mean(vals)) if vals else None

    return DatasetMetrics(
        micro_miou               = _mean("micro_miou"),
        meso_miou                = _mean("meso_miou"),
        macro_miou               = _mean("macro_miou"),
        pose_aware_recall_at_1   = _mean("pose_aware_recall_at_1"),
        pose_aware_recall_at_5   = _mean("pose_aware_recall_at_5"),
        cascade_consistency_rate = _mean("cascade_consistency_rate"),
        sim_exec_success         = _mean("sim_exec_success"),
        n_instances              = len(instance_metrics_list),
        micro_soft_iou           = _mean("micro_soft_iou"),
        meso_soft_iou            = _mean("meso_soft_iou"),
        macro_soft_iou           = _mean("macro_soft_iou"),
    )


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _check_shape(
    pred: np.ndarray,
    gt:   np.ndarray,
    mask: np.ndarray,
    func: str,
) -> None:
    if not (pred.shape == gt.shape == mask.shape):
        raise ValueError(
            f"{func}: pred={pred.shape}, gt={gt.shape}, mask={mask.shape} "
            "形状必须一致"
        )


# ──────────────────────────────────────────────
# 单元测试
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 65)
    print("metrics.py 单元测试")
    print("=" * 65)
    rng = np.random.default_rng(seed=0)

    # ─── 测试 1：micro_miou 极端情况 ───
    print("\n[Test 1] micro_miou 极端情况")

    M, K = 5, 24
    micro_mask = np.ones((M, K), dtype=bool)

    # 完美预测
    gt_ones = np.ones((M, K), dtype=np.float32)
    pred_ones = np.ones((M, K), dtype=np.float32)
    val = micro_miou(pred_ones, gt_ones, micro_mask)
    assert abs(val - 1.0) < 1e-6, f"完美预测应为 1.0, 得 {val}"
    print(f"  完美预测: {val:.4f} (期望 1.0) ✓")

    # 完全错误
    gt_ones2 = np.ones((M, K), dtype=np.float32)
    pred_zeros = np.zeros((M, K), dtype=np.float32)
    val = micro_miou(pred_zeros, gt_ones2, micro_mask)
    assert abs(val) < 1e-6, f"完全错误应为 0.0, 得 {val}"
    print(f"  完全错误: {val:.4f} (期望 0.0) ✓")

    # GT 全零（无正例）：union=0，应接近 0（eps 兜底）
    gt_zeros = np.zeros((M, K), dtype=np.float32)
    val = micro_miou(pred_zeros, gt_zeros, micro_mask)
    assert val < 1e-3, f"GT+pred 全零时期望趋于 0, 得 {val}"
    print(f"  GT+pred 全零: {val:.8f} (期望≈0) ✓")

    # micro_mask 全 False：无 micro 点，union=0，IoU 趋于 0
    no_micro = np.zeros((M, K), dtype=bool)
    val = micro_miou(pred_ones, gt_ones, no_micro)
    assert val < 1e-6
    print(f"  micro_mask 全 False: {val:.8f} (期望≈0) ✓")

    # 随机预测（期望 IoU 约 0.5 左右）
    pred_r = rng.random((M, K)).astype(np.float32)
    gt_r   = rng.random((M, K)).astype(np.float32)
    val = micro_miou(pred_r, gt_r, micro_mask)
    assert 0.0 <= val <= 1.0
    print(f"  随机预测: {val:.4f} (期望 ∈ [0,1]) ✓")

    # ─── 测试 2：pose_aware_recall_at_k ───
    print("\n[Test 2] pose_aware_recall_at_k")

    # 完美预测 k=1
    val = pose_aware_recall_at_k(pred_ones, gt_ones, k=1)
    assert abs(val - 1.0) < 1e-6
    print(f"  k=1 完美: {val:.4f} ✓")

    # top-1 完全不命中
    pred_zeros2 = np.zeros((M, K), dtype=np.float32)
    # 让 GT 在第 0 列为正，pred 在第 23 列最高（其他为 0，第 0 列 GT=1，pred=0）
    gt_col0 = np.zeros((M, K), dtype=np.float32)
    gt_col0[:, 0] = 1.0
    pred_col23 = np.zeros((M, K), dtype=np.float32)
    pred_col23[:, 23] = 1.0
    val = pose_aware_recall_at_k(pred_col23, gt_col0, k=1)
    assert abs(val) < 1e-6, f"top-1 不命中应为 0.0, 得 {val}"
    print(f"  k=1 不命中: {val:.4f} (期望 0.0) ✓")

    # k=5 包含正确答案
    val5 = pose_aware_recall_at_k(pred_col23, gt_col0, k=24)
    assert abs(val5 - 1.0) < 1e-6, f"k=K 全选应为 1.0, 得 {val5}"
    print(f"  k=K(全选): {val5:.4f} (期望 1.0) ✓")

    # k 越界
    try:
        pose_aware_recall_at_k(pred_ones, gt_ones, k=K + 1)
        assert False
    except ValueError as e:
        print(f"  k={K+1} 越界 → ValueError ✓")

    # ─── 测试 3：build_micro_mask 形状与内容 ───
    print("\n[Test 3] build_micro_mask")
    tiers = ["micro", "meso", "macro", "micro", "micro"]
    mask = build_micro_mask(tiers, 24)
    assert mask.shape == (5, 24)
    assert mask[0].all()    # micro
    assert not mask[1].any()  # meso
    assert not mask[2].any()  # macro
    assert mask[3].all()    # micro
    assert mask[4].all()    # micro
    print(f"  shape={mask.shape}, micro 行 all True, 非 micro 行 all False ✓")

    # ─── 测试 4：cascade_consistency_rate ───
    print("\n[Test 4] cascade_consistency_rate")

    # None 输入 → None
    val = cascade_consistency_rate(pred_ones, None, None)
    assert val is None
    print(f"  None 输入 → None ✓")

    # 完美满足约束（geom=1, contact=0.5, exec=0）
    pg = np.ones((M, K), dtype=np.float32)
    pc = np.full((M, K), 0.5, dtype=np.float32)
    pe = np.zeros((M, K), dtype=np.float32)
    val = cascade_consistency_rate(pg, pc, pe)
    assert abs(val - 1.0) < 1e-6, f"完美约束应为 1.0, 得 {val}"
    print(f"  完美约束 geom≥contact≥exec: {val:.4f} (期望 1.0) ✓")

    # 完全违反（geom=0, contact=1）
    pg2 = np.zeros((M, K), dtype=np.float32)
    pc2 = np.ones((M, K), dtype=np.float32)
    pe2 = np.zeros((M, K), dtype=np.float32)
    val = cascade_consistency_rate(pg2, pc2, pe2)
    # contact(1) > geom(0) + eps(0.05)，c1 应 < 1
    assert val < 0.9, f"完全违反时期望 < 0.9, 得 {val}"
    print(f"  完全违反 contact>geom: {val:.4f} (期望 < 0.9) ✓")

    # 边界：恰好满足（contact = geom + eps - 0.001，在容差内）
    pg3 = np.full((M, K), 0.5, dtype=np.float32)
    pc3 = pg3 + CASCADE_EPS - 0.001          # 恰好在容差内
    pe3 = np.zeros((M, K), dtype=np.float32)
    val = cascade_consistency_rate(pg3, pc3, pe3)
    assert abs(val - 1.0) < 1e-5, f"容差内应为 1.0, 得 {val}"
    print(f"  容差边界 (eps-0.001): {val:.4f} (期望 1.0) ✓")

    # ─── 测试 5：sim_exec_success 接口预留 ───
    print("\n[Test 5] sim_exec_success 接口预留")
    assert sim_exec_success(None) is None
    print("  None → None ✓")
    val = sim_exec_success({"n_success": 7, "n_total": 10})
    assert abs(val - 0.7) < 1e-6
    print(f"  7/10 → {val:.4f} (期望 0.7) ✓")
    assert sim_exec_success({"n_success": 0, "n_total": 0}) is None
    print("  0/0 → None ✓")

    # ─── 测试 6：evaluate_instance 组合调用 ───
    print("\n[Test 6] evaluate_instance 组合调用（阶段二中期模式）")
    tiers6 = ["micro"] * 3 + ["meso"] * 1 + ["macro"] * 1
    pred6 = rng.random((5, 24)).astype(np.float32)
    gt6   = rng.random((5, 24)).astype(np.float32)
    im = evaluate_instance("inst_test", pred6, gt6, tiers6)
    assert 0.0 <= im.micro_miou <= 1.0
    assert 0.0 <= im.pose_aware_recall_at_1 <= 1.0
    assert im.cascade_consistency_rate is None  # 阶段二中期应为 None
    assert im.sim_exec_success is None
    print(f"  micro_miou={im.micro_miou:.4f}, recall@1={im.pose_aware_recall_at_1:.4f}")
    print(f"  cascade=None ✓, sim_exec=None ✓")

    # ─── 测试 7：evaluate_dataset 汇总 ───
    print("\n[Test 7] evaluate_dataset 汇总（多实例）")
    all_ms = []
    for i in range(10):
        p = rng.random((5, 24)).astype(np.float32)
        g = rng.random((5, 24)).astype(np.float32)
        t = ["micro"] * 5
        all_ms.append(evaluate_instance(f"inst_{i:03d}", p, g, t))
    dm = evaluate_dataset(all_ms)
    assert dm.n_instances == 10
    assert 0.0 <= dm.micro_miou <= 1.0
    assert 0.0 <= dm.pose_aware_recall_at_1 <= 1.0
    assert dm.meso_miou is None   # 全 micro，无 meso 点
    print(f"  10 实例汇总: micro_miou={dm.micro_miou:.4f}, "
          f"recall@1={dm.pose_aware_recall_at_1:.4f}, "
          f"meso_miou={dm.meso_miou} ✓")

    # ─── 测试 8：形状不匹配应抛异常 ───
    print("\n[Test 8] 形状不匹配异常")
    try:
        micro_miou(
            np.zeros((5, 24)),
            np.zeros((5, 24)),
            np.zeros((5, 10), dtype=bool),   # 错误形状
        )
        assert False
    except ValueError as e:
        print(f"  形状不匹配 → ValueError ✓")

    try:
        pose_aware_recall_at_k(np.zeros((5, 24)), np.zeros((6, 24)))
        assert False
    except ValueError as e:
        print(f"  pred/gt 行数不同 → ValueError ✓")

    # ─── 测试 9：阶段三模式（三层标签都传入）───
    print("\n[Test 9] 阶段三模式（三层标签）")
    pred_geom3    = rng.random((5, 24)).astype(np.float32)
    pred_contact3 = pred_geom3 * 0.8                    # 保证 contact ≤ geom
    pred_exec3    = pred_contact3 * 0.7                 # 保证 exec ≤ contact
    gt_geom3      = rng.random((5, 24)).astype(np.float32)
    gt_contact3   = gt_geom3 * 0.8
    gt_exec3      = gt_contact3 * 0.7

    im3 = evaluate_instance(
        "stage3_test", pred_geom3, gt_geom3,
        tiers=["micro"] * 5,
        pred_contact=pred_contact3, gt_contact=gt_contact3,
        pred_exec=pred_exec3,       gt_exec=gt_exec3,
        isaac_results={"n_success": 8, "n_total": 10},
    )
    assert im3.cascade_consistency_rate is not None
    assert abs(im3.cascade_consistency_rate - 1.0) < 1e-5  # 保证满足约束
    assert abs(im3.sim_exec_success - 0.8) < 1e-6
    print(f"  cascade_rate={im3.cascade_consistency_rate:.4f} (期望 1.0) ✓")
    print(f"  sim_exec={im3.sim_exec_success:.4f} (期望 0.8) ✓")

    # ─── 测试 10：soft_iou（阶段三新增）───
    print("\n[Test 10] soft_iou (阶段三新增)")

    pred_full = np.ones((M, K), dtype=np.float32)
    gt_full   = np.ones((M, K), dtype=np.float32)
    val = soft_iou(pred_full, gt_full)
    assert abs(val - 1.0) < 1e-6, f"完美预测 soft_iou 应为 1.0, 得 {val}"
    print(f"  完美预测 soft_iou: {val:.4f} (期望 1.0) ✓")

    val = soft_iou(np.zeros_like(gt_full), gt_full)
    assert val < 1e-6, f"pred 全零应为 0, 得 {val}"
    print(f"  pred 全零: {val:.8f} (期望≈0) ✓")

    # 全零 pred + 全零 gt：union=0 → 0
    val = soft_iou(np.zeros((M, K)), np.zeros((M, K)))
    assert val < 1e-6
    print(f"  pred+gt 全零: {val:.8f} (期望≈0) ✓")

    # 关键性质：pred=0.4, gt=0.8 → soft_iou = 0.4/0.8 = 0.5
    pred_half = np.full((M, K), 0.4, dtype=np.float32)
    gt_eight  = np.full((M, K), 0.8, dtype=np.float32)
    val = soft_iou(pred_half, gt_eight)
    assert abs(val - 0.5) < 1e-4, f"pred=0.4/gt=0.8 应为 0.5, 得 {val}"
    print(f"  pred=0.4, gt=0.8 → {val:.4f} (期望 0.5) ✓")

    # 关键性质：BCE 压低问题的合理修复
    #   假设模型学到了 GT 的相对排序，但整体 sigmoid 被压低
    #   旧 hard IoU @ 0.5 阈值：pred 全 < 0.5 → IoU=0
    #   新 soft IoU：还能反映相对比例
    rng2 = np.random.default_rng(seed=42)
    gt_imbalanced = (rng2.random((M, K)) < 0.27).astype(np.float32)  # 27% 正样本
    pred_compressed = gt_imbalanced * 0.4 + rng2.random((M, K)) * 0.05  # 学对了排序但压低
    hard = micro_miou(pred_compressed, gt_imbalanced, np.ones((M, K), dtype=bool))
    soft = soft_iou(pred_compressed, gt_imbalanced)
    print(f"  压低数据 hard mIoU={hard:.4f} vs soft IoU={soft:.4f}")
    assert soft > hard, f"soft IoU 应高于压低后的 hard mIoU"
    print(f"  soft > hard ✓ (验证修复 BCE 压低问题)")

    # mask 选择性
    mask_partial = np.zeros((M, K), dtype=bool)
    mask_partial[:, :12] = True
    val = soft_iou(pred_full, gt_full, mask_partial)
    assert abs(val - 1.0) < 1e-6
    print(f"  mask 半选 + 完美预测: {val:.4f} (期望 1.0) ✓")

    # 全 False mask → nan
    val = soft_iou(pred_full, gt_full, np.zeros((M, K), dtype=bool))
    assert np.isnan(val)
    print(f"  mask 全 False → nan ✓")

    print("\n" + "=" * 65)
    print("全部 10 项测试通过 ✓")
    print("=" * 65)

    # 打印示例对比表 header
    print("\n示例对比表格式（eval_main.py 输出，含 soft IoU 阶段三新增列）：")
    print(f"  {'Method':20s} | micro  | meso  | macro | sIoUmi | sIoUme | R@1   | R@5   | Casc  | ExecS")
    print("  " + "─" * 100)
    dm.print_table_row("M1 (示例)")


if __name__ == "__main__":
    _run_tests()