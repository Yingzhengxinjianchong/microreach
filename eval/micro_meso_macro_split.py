"""
eval/micro_meso_macro_split.py

对每个 part 按其点云的 OBB（方向包围盒）最大边长自动分档：
    micro : OBB 最大边 ≤ 5 cm   （主战场，MicroReach 核心评测）
    meso  : 5 cm < OBB 最大边 ≤ 15 cm
    macro : OBB 最大边 > 15 cm  （验证大件不退化）

设计决策：
    - 单位：meters（PartNet-Mobility URDF 的标准单位）
      阈值写在 THRESHOLDS 常量里，外部可覆盖。
    - 输入可以是 (N,3) 点云或 trimesh.Trimesh 对象。
    - 阶段二中期 50 个 StorageFurniture 柜门把手基本全为 micro，
      但代码必须完整，阶段三可直接复用 200 实例评测集。
    - eval_set_200.json 格式见 load_eval_set / save_eval_set。

依赖：numpy, trimesh（已在 requirements.txt）
      open3d 作为可选后备（trimesh OBB 若异常时使用）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

import numpy as np

try:
    import trimesh
    _TRIMESH_OK = True
except ImportError:
    _TRIMESH_OK = False

try:
    import open3d as o3d
    _O3D_OK = True
except ImportError:
    _O3D_OK = False

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

# 单位：meters（与 PartNet-Mobility URDF 一致）
THRESHOLDS: Dict[str, float] = {
    "micro_upper": 0.05,   # ≤ 5 cm  → micro
    "meso_upper":  0.15,   # ≤ 15 cm → meso  （> 15cm → macro）
}

Tier = Literal["micro", "meso", "macro"]


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class PartRecord:
    """单个 part 的元数据 + 分档结果。"""
    instance_id: str
    part_id:     str
    obb_max_edge: float          # 单位 meters
    tier:         Tier
    n_points:     int            # 该 part 点数
    point_cloud_ratio: float     # 在全场景点云中的占比（0~1）

    def to_dict(self) -> dict:
        return {
            "instance_id":      self.instance_id,
            "part_id":          self.part_id,
            "obb_max_edge_m":   round(self.obb_max_edge, 6),
            "tier":             self.tier,
            "n_points":         self.n_points,
            "point_cloud_ratio":round(self.point_cloud_ratio, 6),
        }

    @staticmethod
    def from_dict(d: dict) -> "PartRecord":
        return PartRecord(
            instance_id       = d["instance_id"],
            part_id           = d["part_id"],
            obb_max_edge      = d["obb_max_edge_m"],
            tier              = d["tier"],
            n_points          = d["n_points"],
            point_cloud_ratio = d["point_cloud_ratio"],
        )


@dataclass
class InstanceRecord:
    """单个实例的评测集条目。"""
    instance_id:  str
    category:     str                        # e.g. "StorageFurniture"
    part_records: List[PartRecord] = field(default_factory=list)
    # 阶段三扩展字段，占位
    split:        str = "unassigned"         # train / val / test

    def to_dict(self) -> dict:
        return {
            "instance_id":  self.instance_id,
            "category":     self.category,
            "split":        self.split,
            "parts":        [p.to_dict() for p in self.part_records],
        }

    @staticmethod
    def from_dict(d: dict) -> "InstanceRecord":
        rec = InstanceRecord(
            instance_id = d["instance_id"],
            category    = d["category"],
            split       = d.get("split", "unassigned"),
        )
        rec.part_records = [PartRecord.from_dict(p) for p in d.get("parts", [])]
        return rec


# ──────────────────────────────────────────────
# OBB 计算
# ──────────────────────────────────────────────

def compute_obb_max_edge(
    points: np.ndarray,
    backend: str = "auto",
) -> float:
    """
    计算点云的 OBB（Oriented Bounding Box）最大边长。

    Args:
        points:  shape (N, 3)，单位 meters
        backend: "trimesh" | "open3d" | "auto"（优先 trimesh）

    Returns:
        obb_max_edge: float，单位 meters

    Raises:
        RuntimeError: 若两个 backend 都不可用
        ValueError:   若 points 太少（< 4 个）
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points 必须为 (N,3)，收到 {points.shape}")
    if len(points) < 4:
        raise ValueError(
            f"点数过少（{len(points)}），OBB 至少需要 4 个点。"
            "请检查 part_id 对应点是否正确过滤。"
        )

    use_trimesh = backend in ("trimesh", "auto") and _TRIMESH_OK
    use_o3d     = backend in ("open3d", "auto") and _O3D_OK

    if use_trimesh:
        try:
            return _obb_trimesh(points)
        except Exception as e:
            if not use_o3d:
                raise RuntimeError(f"trimesh OBB 失败: {e}") from e
            # fallthrough 到 open3d

    if use_o3d:
        try:
            return _obb_open3d(points)
        except Exception as e:
            raise RuntimeError(f"open3d OBB 失败: {e}") from e

    # 最后的降级方案：轴对齐包围盒（AABB），精度较低但总能跑通
    return _obb_aabb_fallback(points)


def _obb_trimesh(points: np.ndarray) -> float:
    """用 trimesh 的 PointCloud OBB 计算最大边长。"""
    pc = trimesh.PointCloud(points)
    obb = pc.bounding_box_oriented
    # obb.extents: (3,)，三条边长
    return float(obb.extents.max())


def _obb_open3d(points: np.ndarray) -> float:
    """用 Open3D 的 get_oriented_bounding_box 计算最大边长。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    return float(max(obb.extent))


def _obb_aabb_fallback(points: np.ndarray) -> float:
    """
    AABB 降级方案（不依赖任何第三方库，但不考虑旋转）。
    正式评测中应尽量用 OBB；此方案仅用于环境不完整时的兜底。
    """
    extents = points.max(axis=0) - points.min(axis=0)
    return float(extents.max())


# ──────────────────────────────────────────────
# 分档核心
# ──────────────────────────────────────────────

def classify_tier(
    obb_max_edge: float,
    thresholds: Dict[str, float] = THRESHOLDS,
) -> Tier:
    """
    按 OBB 最大边长对 part 分档。

    Args:
        obb_max_edge: 单位 meters
        thresholds:   可覆盖 THRESHOLDS

    Returns:
        "micro" | "meso" | "macro"
    """
    if obb_max_edge <= thresholds["micro_upper"]:
        return "micro"
    elif obb_max_edge <= thresholds["meso_upper"]:
        return "meso"
    else:
        return "macro"


def classify_part(
    instance_id:      str,
    part_id:          str,
    part_points:      np.ndarray,
    scene_n_points:   int,
    thresholds:       Dict[str, float] = THRESHOLDS,
    backend:          str = "auto",
) -> PartRecord:
    """
    对单个 part 完成 OBB 计算 + 分档 + 统计。

    Args:
        instance_id:    实例 ID 字符串
        part_id:        part ID 字符串
        part_points:    该 part 的点云，shape (N_part, 3)，单位 meters
        scene_n_points: 全场景点云总点数（用于计算 point_cloud_ratio）
        thresholds:     阈值字典，默认 THRESHOLDS
        backend:        OBB backend

    Returns:
        PartRecord
    """
    obb_max_edge = compute_obb_max_edge(part_points, backend=backend)
    tier = classify_tier(obb_max_edge, thresholds)
    n_points = len(part_points)
    ratio = n_points / max(scene_n_points, 1)

    return PartRecord(
        instance_id       = instance_id,
        part_id           = part_id,
        obb_max_edge      = obb_max_edge,
        tier              = tier,
        n_points          = n_points,
        point_cloud_ratio = ratio,
    )


def classify_instance(
    instance_id:         str,
    category:            str,
    parts_point_clouds:  Dict[str, np.ndarray],
    thresholds:          Dict[str, float] = THRESHOLDS,
    backend:             str = "auto",
) -> InstanceRecord:
    """
    对一个实例的所有 part 批量分档。

    Args:
        instance_id:        实例 ID 字符串
        category:           类别名称，e.g. "StorageFurniture"
        parts_point_clouds: {part_id: points (N_part, 3)}
        thresholds:         阈值字典
        backend:            OBB backend

    Returns:
        InstanceRecord（含所有 part 的 PartRecord）
    """
    scene_n_points = sum(len(pts) for pts in parts_point_clouds.values())

    record = InstanceRecord(instance_id=instance_id, category=category)
    for part_id, part_pts in parts_point_clouds.items():
        pr = classify_part(
            instance_id    = instance_id,
            part_id        = part_id,
            part_points    = part_pts,
            scene_n_points = scene_n_points,
            thresholds     = thresholds,
            backend        = backend,
        )
        record.part_records.append(pr)

    return record


# ──────────────────────────────────────────────
# eval_set_200.json 读写
# ──────────────────────────────────────────────

def load_eval_set(json_path: str) -> List[InstanceRecord]:
    """
    读取 data/eval_set_200.json，返回 InstanceRecord 列表。

    如果文件不存在，返回空列表（方便首次写入）。
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [InstanceRecord.from_dict(d) for d in raw]


def save_eval_set(json_path: str, records: List[InstanceRecord]) -> None:
    """
    将 InstanceRecord 列表写入 data/eval_set_200.json。
    覆盖写入，调用前请确保 records 完整。
    """
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2, ensure_ascii=False)
    print(f"[save_eval_set] 已保存 {len(records)} 条记录 → {json_path}")


def update_eval_set(
    json_path:     str,
    new_records:   List[InstanceRecord],
) -> List[InstanceRecord]:
    """
    增量合并：读取已有 JSON，按 instance_id 去重后写回。
    用于批量生成时逐批追加，而不是每次全量覆盖。
    """
    existing = {r.instance_id: r for r in load_eval_set(json_path)}
    for r in new_records:
        existing[r.instance_id] = r        # 覆盖旧条目
    merged = list(existing.values())
    save_eval_set(json_path, merged)
    return merged


# ──────────────────────────────────────────────
# 统计摘要
# ──────────────────────────────────────────────

def summarize_tier_distribution(records: List[InstanceRecord]) -> Dict[str, int]:
    """
    统计评测集中 micro / meso / macro part 的数量。

    Returns:
        {"micro": int, "meso": int, "macro": int, "total": int}
    """
    counts: Dict[str, int] = {"micro": 0, "meso": 0, "macro": 0}
    for inst in records:
        for pr in inst.part_records:
            counts[pr.tier] += 1
    counts["total"] = sum(counts.values())
    return counts


def print_summary(records: List[InstanceRecord]) -> None:
    """打印评测集分档统计摘要。"""
    dist = summarize_tier_distribution(records)
    total = max(dist["total"], 1)
    print(f"\n{'─'*40}")
    print(f"  评测集分档摘要  ({len(records)} 实例)")
    print(f"{'─'*40}")
    for tier in ("micro", "meso", "macro"):
        n = dist[tier]
        pct = 100.0 * n / total
        bar = "█" * int(pct / 5)
        print(f"  {tier:6s}: {n:4d} ({pct:5.1f}%)  {bar}")
    print(f"  {'total':6s}: {dist['total']:4d}")
    print(f"{'─'*40}\n")


# ──────────────────────────────────────────────
# 单元测试
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 55)
    print("micro_meso_macro_split.py 单元测试")
    print("=" * 55)

    rng = np.random.default_rng(seed=42)

    # ── 测试 1：classify_tier 边界值 ──
    print("\n[Test 1] classify_tier 边界值")
    cases = [
        (0.00,  "micro"),   # 0 cm
        (0.05,  "micro"),   # 恰好 5 cm（micro 上界，含）
        (0.051, "meso"),    # 5.1 cm
        (0.15,  "meso"),    # 恰好 15 cm（meso 上界，含）
        (0.151, "macro"),   # 15.1 cm
        (1.00,  "macro"),   # 1 m
    ]
    for edge_m, expected in cases:
        result = classify_tier(edge_m)
        assert result == expected, \
            f"edge={edge_m*100:.1f}cm: 期望 {expected}, 得到 {result}"
        print(f"  edge={edge_m*100:5.1f} cm → {result} ✓")

    # ── 测试 2：OBB 计算（已知轴对齐长方体，边长已知）──
    print("\n[Test 2] compute_obb_max_edge（轴对齐长方体）")
    # 2cm × 3cm × 4cm 的立方体（OBB = AABB = 最大边 4 cm）
    pts_box = np.array([
        [0.00, 0.00, 0.00],
        [0.02, 0.00, 0.00],
        [0.02, 0.03, 0.00],
        [0.00, 0.03, 0.00],
        [0.00, 0.00, 0.04],
        [0.02, 0.00, 0.04],
        [0.02, 0.03, 0.04],
        [0.00, 0.03, 0.04],
    ], dtype=np.float32)
    # 加一些噪声使点云更真实（不全是角点）
    pts_box_dense = pts_box.copy()
    for _ in range(50):
        t = rng.random((8, 3)).astype(np.float32)
        sample = (pts_box * t).sum(axis=0, keepdims=True) / t.sum()
        pts_box_dense = np.vstack([pts_box_dense, sample])

    edge = compute_obb_max_edge(pts_box_dense, backend="auto")
    print(f"  2×3×4 cm 盒子: 计算边长={edge*100:.2f} cm（期望≈4 cm）", end="")
    # trimesh OBB 对轴对齐体应当精确；允许 10% 误差
    assert abs(edge - 0.04) < 0.005, f"误差过大: {edge:.4f} m"
    print(" ✓")

    # ── 测试 3：classify_part 输出类型与字段完整性 ──
    print("\n[Test 3] classify_part 字段完整性")
    part_pts  = rng.random((200, 3)).astype(np.float32) * 0.03  # 3 cm 级别
    pr = classify_part(
        instance_id    = "inst_001",
        part_id        = "handle_0",
        part_points    = part_pts,
        scene_n_points = 10000,
    )
    assert isinstance(pr, PartRecord)
    assert pr.instance_id == "inst_001"
    assert pr.part_id     == "handle_0"
    assert pr.tier in ("micro", "meso", "macro")
    assert 0.0 <= pr.point_cloud_ratio <= 1.0
    assert pr.n_points == 200
    print(f"  OBB max edge={pr.obb_max_edge*100:.2f} cm, tier={pr.tier}, "
          f"ratio={pr.point_cloud_ratio:.4f} ✓")

    # ── 测试 4：classify_instance 批量处理 ──
    print("\n[Test 4] classify_instance 批量处理")
    parts = {
        "handle": rng.random((100, 3)).astype(np.float32) * 0.04,   # ~2 cm → micro
        "door":   rng.random((500, 3)).astype(np.float32) * 0.30,   # ~15 cm → meso/macro
        "cabinet":rng.random((300, 3)).astype(np.float32) * 0.60,   # ~30 cm → macro
    }
    inst_rec = classify_instance("inst_002", "StorageFurniture", parts)
    assert inst_rec.instance_id == "inst_002"
    assert len(inst_rec.part_records) == 3
    tiers = {pr.part_id: pr.tier for pr in inst_rec.part_records}
    print(f"  part tiers: {tiers}")
    assert tiers["handle"] == "micro",  f"handle 应为 micro，得 {tiers['handle']}"
    assert tiers["cabinet"] == "macro", f"cabinet 应为 macro，得 {tiers['cabinet']}"
    print("  handle=micro, cabinet=macro ✓")

    # ── 测试 5：JSON 序列化往返一致性 ──
    print("\n[Test 5] JSON 序列化往返一致性")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data", "eval_set_200.json")
        save_eval_set(path, [inst_rec])
        loaded = load_eval_set(path)
    assert len(loaded) == 1
    l_inst = loaded[0]
    assert l_inst.instance_id == "inst_002"
    assert len(l_inst.part_records) == 3
    assert abs(l_inst.part_records[0].obb_max_edge
               - inst_rec.part_records[0].obb_max_edge) < 1e-5
    print("  往返一致 ✓")

    # ── 测试 6：update_eval_set 增量合并（去重） ──
    print("\n[Test 6] update_eval_set 增量合并（去重）")
    inst_003 = classify_instance(
        "inst_003", "StorageFurniture",
        {"knob": rng.random((50, 3)).astype(np.float32) * 0.02}
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data", "eval_set_200.json")
        save_eval_set(path, [inst_rec])
        merged = update_eval_set(path, [inst_003, inst_rec])  # inst_rec 重复
    ids = [r.instance_id for r in merged]
    assert ids.count("inst_002") == 1, "去重失败"
    assert "inst_003" in ids
    assert len(merged) == 2
    print(f"  合并后 {len(merged)} 条（去重正确）✓")

    # ── 测试 7：summarize_tier_distribution ──
    print("\n[Test 7] summarize_tier_distribution 统计")
    dist = summarize_tier_distribution([inst_rec])
    assert dist["micro"] + dist["meso"] + dist["macro"] == dist["total"]
    assert dist["total"] == 3
    print(f"  分布: {dist} ✓")

    # ── 测试 8：点数不足抛异常 ──
    print("\n[Test 8] 点数不足异常")
    try:
        compute_obb_max_edge(np.zeros((2, 3)))
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"  2 点输入 → ValueError: {e} ✓")

    print("\n" + "=" * 55)
    print("全部 8 项测试通过 ✓")
    print("=" * 55)
    print_summary([inst_rec])


if __name__ == "__main__":
    _run_tests()