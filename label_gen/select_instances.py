"""
label_gen/select_instances.py

步骤 2.1：从 PartNet-Mobility 数据集中筛选 50 个 StorageFurniture 实例，
生成初始 data/eval_set_200.json。

PartNet-Mobility 目录结构（每个实例一个子目录）：
    <data_dir>/
        <instance_id>/          ← 纯数字，如 "45174"
            mobility.urdf
            meta.json           ← {"model_cat": "StorageFurniture", ...}
            semantics.txt       ← "link_0 hinge handle\nlink_1 ..."
            textured_objs/
                <link_name>/
                    textured.obj
                    ...

筛选条件：
    1. meta.json 中 model_cat == "StorageFurniture"
    2. semantics.txt 中至少有一个 handle/knob/lid 类型的 part
    3. 对应 part 的 mesh 文件存在且顶点数 ≥ 4（OBB 计算需要）

输出：
    - data/eval_set_200.json（调用 micro_meso_macro_split.py 的工具函数写入）

用法：
    python label_gen/select_instances.py \\
        --data-dir /root/autodl-fs/partnet_mobility \\
        --output   data/eval_set_200.json \\
        --n-instances 50 \\
        --seed 42

在服务器上跑：约 2-5 分钟（主要耗时是读 mesh + 算 OBB）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 把项目根目录加入 sys.path，使 eval/ 下的模块可以 import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from eval.micro_meso_macro_split import (
    classify_part,
    InstanceRecord,
    PartRecord,
    save_eval_set,
    THRESHOLDS,
)

try:
    import trimesh
    _TRIMESH_OK = True
except ImportError:
    _TRIMESH_OK = False
    print("[WARNING] trimesh 未安装，OBB 将退化为 AABB。请 pip install trimesh")

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

TARGET_CATEGORY = "StorageFurniture"

# 视为"交互 part"的语义关键词（小写匹配）
HANDLE_KEYWORDS = {"handle", "knob", "lid", "latch", "pull", "grip", "button"}

# 阶段三可扩展的类别映射（阶段二只用 StorageFurniture）
STAGE3_CATEGORIES = {
    "StorageFurniture": 50,
    "Microwave":        40,
    "Refrigerator":     40,
    "Dishwasher":       35,
    "Drawer":           35,
}


# ──────────────────────────────────────────────
# 读取实例元数据
# ──────────────────────────────────────────────

def read_meta(instance_dir: Path) -> Optional[dict]:
    """
    读取 meta.json，返回 dict；文件缺失或格式异常返回 None。
    """
    meta_path = instance_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def read_semantics(instance_dir: Path) -> List[Tuple[str, str, str]]:
    """
    读取 semantics.txt，返回 list of (link_name, joint_type, semantic_label)。

    格式兼容两种常见变体：
        link_0 hinge handle          ← 三列：link / joint_type / label
        link_0 handle                ← 两列：link / label
    """
    sem_path = instance_dir / "semantics.txt"
    if not sem_path.exists():
        return []

    result = []
    try:
        with open(sem_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    result.append((parts[0], parts[1], parts[2]))
                elif len(parts) == 2:
                    result.append((parts[0], "unknown", parts[1]))
    except OSError:
        pass
    return result


def find_handle_links(instance_dir: Path) -> List[Tuple[str, str]]:
    """
    返回该实例中被语义标注为"交互 part"的 link 列表。

    Returns:
        list of (link_name, semantic_label)，可能为空
    """
    semantics = read_semantics(instance_dir)
    handles = []
    for link_name, _joint_type, label in semantics:
        if any(kw in label.lower() for kw in HANDLE_KEYWORDS):
            handles.append((link_name, label))
    return handles


# ──────────────────────────────────────────────
# 读取 part mesh
# ──────────────────────────────────────────────

def find_mesh_files(instance_dir: Path, link_name: str) -> List[Path]:
    """
    在 textured_objs/<link_name>/ 或 part_objs/<link_name>/ 下找 .obj 文件。
    PartNet-Mobility 的两种常见目录名都兼容。
    """
    candidates = []
    for subdir in ("textured_objs", "part_objs"):
        link_dir = instance_dir / subdir / link_name
        if link_dir.is_dir():
            candidates.extend(link_dir.glob("*.obj"))
    return candidates


def load_part_vertices(instance_dir: Path, link_name: str) -> Optional[np.ndarray]:
    """
    加载一个 part 的所有 .obj 文件的顶点，合并为 (N, 3) 数组。
    返回 None 表示该 link 没有可用 mesh。
    """
    mesh_files = find_mesh_files(instance_dir, link_name)
    if not mesh_files:
        return None

    all_vertices = []
    for mf in mesh_files:
        try:
            if _TRIMESH_OK:
                mesh = trimesh.load(str(mf), force="mesh", process=False)
                if hasattr(mesh, "vertices") and len(mesh.vertices) >= 4:
                    all_vertices.append(np.array(mesh.vertices, dtype=np.float32))
            else:
                # 极简 fallback：直接解析 OBJ 文件读 "v x y z" 行
                verts = _parse_obj_vertices(mf)
                if len(verts) >= 4:
                    all_vertices.append(verts)
        except Exception:
            continue

    if not all_vertices:
        return None

    return np.concatenate(all_vertices, axis=0)


def _parse_obj_vertices(obj_path: Path) -> np.ndarray:
    """纯 Python 解析 OBJ 文件，仅提取顶点坐标（v x y z 行）。"""
    vertices = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        pass
    return np.array(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32)


# ──────────────────────────────────────────────
# 实例过滤
# ──────────────────────────────────────────────

def is_valid_instance(
    instance_dir: Path,
    target_category: str = TARGET_CATEGORY,
) -> Tuple[bool, str]:
    """
    检查单个实例是否满足筛选条件。

    Returns:
        (is_valid, reason_if_invalid)
    """
    # 条件1：目录存在
    if not instance_dir.is_dir():
        return False, "目录不存在"

    # 条件2：meta.json 存在且类别匹配
    meta = read_meta(instance_dir)
    if meta is None:
        return False, "meta.json 缺失或格式错误"
    if meta.get("model_cat", "") != target_category:
        return False, f"类别={meta.get('model_cat', '?')} ≠ {target_category}"

    # 条件3：至少一个 handle/knob 语义标注
    handles = find_handle_links(instance_dir)
    if not handles:
        return False, "无 handle/knob/lid 标注"

    # 条件4：至少一个 handle link 有可用 mesh
    for link_name, _label in handles:
        verts = load_part_vertices(instance_dir, link_name)
        if verts is not None and len(verts) >= 4:
            return True, ""

    return False, "handle link 无可用 mesh（顶点数 < 4）"


def scan_valid_instances(
    data_dir: str,
    target_category: str = TARGET_CATEGORY,
    verbose: bool = True,
) -> List[str]:
    """
    扫描 data_dir，返回满足筛选条件的实例 ID 列表。

    Args:
        data_dir:        PartNet-Mobility 数据集根目录
        target_category: 目标类别
        verbose:         是否打印进度

    Returns:
        list of instance_id strings（纯数字目录名）
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    all_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    valid_ids = []
    skipped = 0

    if verbose:
        print(f"[scan] 扫描 {len(all_dirs)} 个目录，过滤 {target_category}...")

    for i, d in enumerate(all_dirs):
        ok, reason = is_valid_instance(d, target_category)
        if ok:
            valid_ids.append(d.name)
        else:
            skipped += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(all_dirs)}] 已找到 {len(valid_ids)} 个有效实例")

    if verbose:
        print(f"[scan] 完成：{len(valid_ids)} 个有效实例，{skipped} 个跳过")

    return valid_ids


# ──────────────────────────────────────────────
# 随机抽样
# ──────────────────────────────────────────────

def select_instances(
    valid_ids: List[str],
    n: int = 50,
    seed: int = 42,
) -> List[str]:
    """
    从有效实例中随机抽取 n 个，固定 seed 保证可复现。

    Args:
        valid_ids: 所有有效实例 ID
        n:         抽取数量（阶段二=50，阶段三=200）
        seed:      随机种子

    Returns:
        selected: 抽取的 n 个实例 ID（已排序）

    Raises:
        ValueError: 若有效实例数 < n
    """
    if len(valid_ids) < n:
        raise ValueError(
            f"有效实例数 {len(valid_ids)} < 需要的 {n}，"
            f"请检查数据集完整性或放宽筛选条件。"
        )
    rng = random.Random(seed)
    selected = rng.sample(valid_ids, n)
    return sorted(selected)   # 排序方便 debug


# ──────────────────────────────────────────────
# 构建 eval_set JSON
# ──────────────────────────────────────────────

def build_instance_record(
    instance_id: str,
    data_dir: str,
    category: str = TARGET_CATEGORY,
) -> InstanceRecord:
    """
    为单个实例构建 InstanceRecord（包含所有 part 的 OBB + 分档信息）。

    Args:
        instance_id: 实例 ID 字符串
        data_dir:    PartNet-Mobility 根目录
        category:    类别名称

    Returns:
        InstanceRecord（part_records 由 handle link 的 mesh 驱动）
    """
    instance_dir = Path(data_dir) / instance_id
    semantics = read_semantics(instance_dir)

    # 收集所有 part 的点云（用于 OBB 计算）
    parts_vertices: Dict[str, np.ndarray] = {}
    for link_name, _joint_type, label in semantics:
        verts = load_part_vertices(instance_dir, link_name)
        if verts is not None and len(verts) >= 4:
            # 用 "link_名:语义" 作为 part_id，方便 sapien_loader 匹配
            part_key = f"{link_name}:{label}"
            parts_vertices[part_key] = verts

    if not parts_vertices:
        # fallback：把整个实例当一个 part（不应发生，因为已经过滤过）
        parts_vertices["default:unknown"] = np.zeros((4, 3), dtype=np.float32)

    # 统计全场景点数（所有 part 顶点之和，近似）
    scene_n_points = sum(len(v) for v in parts_vertices.values())

    # 调用 micro_meso_macro_split 的 classify_part
    record = InstanceRecord(instance_id=instance_id, category=category)
    for part_key, verts in parts_vertices.items():
        pr = classify_part(
            instance_id    = instance_id,
            part_id        = part_key,
            part_points    = verts,
            scene_n_points = scene_n_points,
        )
        record.part_records.append(pr)

    return record


def build_eval_set(
    selected_ids: List[str],
    data_dir: str,
    output_path: str,
    category: str = TARGET_CATEGORY,
    verbose: bool = True,
) -> List[InstanceRecord]:
    """
    批量构建 InstanceRecord 并写入 JSON。

    Args:
        selected_ids: 已选定的实例 ID 列表
        data_dir:     PartNet-Mobility 根目录
        output_path:  输出 JSON 路径（e.g. "data/eval_set_200.json"）
        category:     类别名称
        verbose:      打印进度

    Returns:
        list of InstanceRecord
    """
    records = []
    failed = []

    for i, iid in enumerate(selected_ids):
        try:
            rec = build_instance_record(iid, data_dir, category)
            records.append(rec)
            if verbose:
                n_micro = sum(1 for pr in rec.part_records if pr.tier == "micro")
                print(f"  [{i+1:2d}/{len(selected_ids)}] {iid}: "
                      f"{len(rec.part_records)} parts, {n_micro} micro")
        except Exception as e:
            failed.append((iid, str(e)))
            if verbose:
                print(f"  [{i+1:2d}/{len(selected_ids)}] {iid}: 失败 - {e}")

    if failed:
        print(f"\n[WARNING] {len(failed)} 个实例处理失败：")
        for iid, err in failed:
            print(f"  {iid}: {err}")

    save_eval_set(output_path, records)

    if verbose:
        _print_summary(records)

    return records


def _print_summary(records: List[InstanceRecord]) -> None:
    """打印筛选结果摘要。"""
    from eval.micro_meso_macro_split import summarize_tier_distribution, print_summary
    print(f"\n共 {len(records)} 个实例已写入 eval_set JSON")
    print_summary(records)


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从 PartNet-Mobility 筛选实例，生成 eval_set_200.json"
    )
    p.add_argument(
        "--data-dir", required=True,
        help="PartNet-Mobility 数据集根目录，e.g. /root/autodl-fs/partnet_mobility"
    )
    p.add_argument(
        "--output", default="data/eval_set_200.json",
        help="输出 JSON 路径（默认 data/eval_set_200.json）"
    )
    p.add_argument(
        "--n-instances", type=int, default=50,
        help="抽取实例数（阶段二=50，阶段三=200）"
    )
    p.add_argument(
        "--category", default=TARGET_CATEGORY,
        help=f"目标类别（默认 {TARGET_CATEGORY}）"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（保证可复现，默认 42）"
    )
    p.add_argument(
        "--verbose", action="store_true", default=True,
        help="打印详细进度"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[select_instances] 数据集目录: {args.data_dir}")
    print(f"[select_instances] 目标类别:   {args.category}")
    print(f"[select_instances] 抽取数量:   {args.n_instances}")
    print(f"[select_instances] 随机种子:   {args.seed}")
    print(f"[select_instances] 输出路径:   {args.output}")
    print()

    # 步骤1：扫描有效实例
    valid_ids = scan_valid_instances(args.data_dir, args.category, args.verbose)

    # 步骤2：随机抽取
    selected = select_instances(valid_ids, args.n_instances, args.seed)
    print(f"\n[select] 从 {len(valid_ids)} 个有效实例中抽取 {len(selected)} 个")

    # 步骤3：构建 InstanceRecord 并写入 JSON
    print(f"\n[build] 计算 OBB + 分档...")
    build_eval_set(selected, args.data_dir, args.output, args.category, args.verbose)

    print(f"\n[done] eval_set JSON 已写入: {args.output}")


# ──────────────────────────────────────────────
# 单元测试（不依赖真实数据，用 mock 目录结构）
# ──────────────────────────────────────────────

def _run_tests() -> None:
    import tempfile

    print("=" * 55)
    print("select_instances.py 单元测试")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "partnet_mobility"
        data_dir.mkdir()

        # 构建 mock 数据集
        # inst_001: StorageFurniture + handle → 应被选中
        _make_mock_instance(data_dir / "001", "StorageFurniture", [("link_0", "hinge", "handle")])
        # inst_002: StorageFurniture + knob → 应被选中
        _make_mock_instance(data_dir / "002", "StorageFurniture", [("link_0", "slider", "knob")])
        # inst_003: Microwave → 不应被选中（类别不符）
        _make_mock_instance(data_dir / "003", "Microwave", [("link_0", "hinge", "handle")])
        # inst_004: StorageFurniture 但无 handle → 不应被选中
        _make_mock_instance(data_dir / "004", "StorageFurniture", [("link_0", "hinge", "door")])
        # inst_005: StorageFurniture + lid → 应被选中
        _make_mock_instance(data_dir / "005", "StorageFurniture", [("link_0", "hinge", "lid")])

        # ── 测试 1：scan_valid_instances ──
        print("\n[Test 1] scan_valid_instances")
        valid = scan_valid_instances(str(data_dir), "StorageFurniture", verbose=False)
        assert set(valid) == {"001", "002", "005"}, f"期望 {{001,002,005}}，得 {set(valid)}"
        print(f"  有效实例: {sorted(valid)} ✓")

        # ── 测试 2：select_instances 抽样 ──
        print("\n[Test 2] select_instances 固定 seed")
        selected = select_instances(valid, n=2, seed=42)
        assert len(selected) == 2
        selected2 = select_instances(valid, n=2, seed=42)
        assert selected == selected2, "相同 seed 应产生相同结果"
        print(f"  抽取: {selected}，同 seed 结果一致 ✓")

        # ── 测试 3：数量不足应抛异常 ──
        print("\n[Test 3] 数量不足异常")
        try:
            select_instances(valid, n=10, seed=42)
            assert False, "应抛出 ValueError"
        except ValueError as e:
            print(f"  n=10 > valid=3 → ValueError ✓")

        # ── 测试 4：build_instance_record ──
        print("\n[Test 4] build_instance_record")
        rec = build_instance_record("001", str(data_dir), "StorageFurniture")
        assert rec.instance_id == "001"
        assert rec.category == "StorageFurniture"
        assert len(rec.part_records) >= 1
        pr = rec.part_records[0]
        assert pr.tier in ("micro", "meso", "macro")
        print(f"  实例001: {len(rec.part_records)} part(s), "
              f"首part OBB={pr.obb_max_edge*100:.2f}cm, tier={pr.tier} ✓")

        # ── 测试 5：build_eval_set 写入 JSON ──
        print("\n[Test 5] build_eval_set 写入 JSON")
        output = str(Path(tmpdir) / "data" / "eval_set_200.json")
        records = build_eval_set(["001", "002"], str(data_dir), output, verbose=False)
        assert os.path.exists(output)
        assert len(records) == 2
        with open(output) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["instance_id"] in ("001", "002")
        print(f"  写入 {len(records)} 条记录，JSON 格式正确 ✓")

        # ── 测试 6：read_semantics 两列格式兼容 ──
        print("\n[Test 6] read_semantics 两列格式兼容")
        sem_dir = data_dir / "006"
        sem_dir.mkdir()
        (sem_dir / "meta.json").write_text(json.dumps({"model_cat": "StorageFurniture"}))
        # 两列格式（无 joint_type）
        (sem_dir / "semantics.txt").write_text("link_0 handle\nlink_1 door\n")
        _make_mesh_for_link(sem_dir, "link_0")
        (sem_dir / "mobility.urdf").write_text("")
        sems = read_semantics(sem_dir)
        assert len(sems) == 2
        assert sems[0] == ("link_0", "unknown", "handle")
        print(f"  两列格式解析: {sems} ✓")

    print("\n" + "=" * 55)
    print("全部 6 项测试通过 ✓")
    print("=" * 55)


def _make_mock_instance(
    instance_dir: Path,
    category: str,
    links: List[Tuple[str, str, str]],
) -> None:
    """创建 mock 实例目录，用于单元测试。"""
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "meta.json").write_text(json.dumps({"model_cat": category}))
    sem_lines = "\n".join(f"{ln} {jt} {lbl}" for ln, jt, lbl in links)
    (instance_dir / "semantics.txt").write_text(sem_lines + "\n")
    (instance_dir / "mobility.urdf").write_text("")
    # 为每个 link 创建一个最小 .obj（8 个顶点，1cm 立方体）
    for link_name, _, _ in links:
        _make_mesh_for_link(instance_dir, link_name)


def _make_mesh_for_link(instance_dir: Path, link_name: str) -> None:
    """创建 1cm 立方体 .obj 文件。"""
    obj_dir = instance_dir / "textured_objs" / link_name
    obj_dir.mkdir(parents=True, exist_ok=True)
    obj_content = "\n".join([
        "v 0.00 0.00 0.00", "v 0.01 0.00 0.00",
        "v 0.01 0.01 0.00", "v 0.00 0.01 0.00",
        "v 0.00 0.00 0.01", "v 0.01 0.00 0.01",
        "v 0.01 0.01 0.01", "v 0.00 0.01 0.01",
        "f 1 2 3 4", "f 5 6 7 8",
    ])
    (obj_dir / "textured.obj").write_text(obj_content)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv or len(sys.argv) == 1:
        _run_tests()
    else:
        main()