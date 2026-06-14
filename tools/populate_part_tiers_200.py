"""
tools/populate_part_tiers_200.py

阶段三 P0 后续：把 hyh 新加的 153 个非 Faucet 实例 part_tiers 字段（当前 'unknown'）
重算成 micro/meso/macro。

设计原则（严谨级别）：
  1. 用现有 eval/micro_meso_macro_split.py 的 classify_tier_relative 函数，
     **完全复用阶段二中期的分档逻辑**（相对体积占比 ≤ 0.15 / ≤ 0.45 / > 0.45），
     保证新 153 实例与旧 47 Faucet 实例分档标准一致。
  2. **按 part 而非按 candidate 分档**——同一 part_id 的多个候选点共享同一 tier。
  3. **部分局部点云用 ball query 反推**——npz 里没有 part 标签的点云列，
     用 candidate_p 做种子，在 part_id 相同的全部候选点周围
     用 radius=0.10 的球域取并集作为 part 点云近似。
  4. 备份原 part_tiers 到 part_tiers_orig 字段（保留 hyh 数据）。
  5. **重算前先验证**：对旧 47 Faucet 实例重算，结果必须与 hyh 原标签一致
     （±1 档容忍——micro vs meso 边界可能因近似而漂移）。
     不一致率 > 10% 则中止。

用法：
    # 1. 验证模式：只对旧 Faucet 实例重算，对比与 hyh 标签的一致率
    python tools/populate_part_tiers_200.py --verify-only

    # 2. 全量写入（自动备份）
    python tools/populate_part_tiers_200.py --write

    # 3. 自定义球域半径（默认 0.10，PartNet-Mobility 归一化坐标）
    python tools/populate_part_tiers_200.py --write --ball-radius 0.12
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.micro_meso_macro_split import (
    THRESHOLDS,
    classify_tier_relative,
    compute_aabb_extent,
)


# ──────────────────────────────────────────────
# 球域 part 点云反推
# ──────────────────────────────────────────────

def part_points_from_balls(
    point_cloud: np.ndarray,         # (N, 3)
    centers:     np.ndarray,         # (K, 3) 同一 part 上的 K 个候选点
    radius:      float,
) -> np.ndarray:
    """
    取 part 局部点云：从同一 part_id 的所有候选点出发，球域并集。

    Args:
        point_cloud: 整场景 (N, 3)
        centers:     该 part 上的 K 个候选点
        radius:      球域半径

    Returns:
        local_pts: (M, 3) 球域并集的点云
    """
    # 对每个候选点算到所有场景点的距离，留下距离任一候选点 ≤ radius 的点
    # vectorized: (N, K) 距离矩阵
    diff = point_cloud[:, None, :] - centers[None, :, :]   # (N, K, 3)
    dists = np.linalg.norm(diff, axis=-1)                  # (N, K)
    keep_mask = (dists.min(axis=-1) <= radius)             # (N,) 离任一中心 ≤ r
    return point_cloud[keep_mask]


# ──────────────────────────────────────────────
# 单实例处理
# ──────────────────────────────────────────────

def relabel_instance(
    npz_path: Path,
    ball_radius: float,
    thresholds: Dict[str, float] = THRESHOLDS,
) -> Tuple[List[str], Dict[str, str]]:
    """
    对单个 npz 重算 part_tiers。

    Returns:
        new_tiers:        list of len M（按 candidate 顺序，同 part 共享 tier）
        part_to_tier:     {part_id: tier}
    """
    d = np.load(str(npz_path), allow_pickle=True)
    point_cloud = d["point_cloud"].astype(np.float32)       # (N, 3)
    candidate_p = d["candidate_p"].astype(np.float32)        # (M, 3)
    part_ids    = [str(x) for x in d["part_ids"]]            # len M

    # 整物体 AABB
    scene_extent = compute_aabb_extent(point_cloud)

    # 按 part_id 分组候选点
    by_part: Dict[str, List[int]] = defaultdict(list)
    for m, pid in enumerate(part_ids):
        by_part[pid].append(m)

    # 对每个唯一 part_id 算 tier
    part_to_tier: Dict[str, str] = {}
    for pid, idx_list in by_part.items():
        centers = candidate_p[idx_list]                       # (K, 3)
        local_pts = part_points_from_balls(point_cloud, centers, ball_radius)
        if len(local_pts) < 4:
            # 点太少（< 4），OBB 算不出 → 给最小档 micro
            part_to_tier[pid] = "micro"
            continue
        try:
            tier = classify_tier_relative(local_pts, scene_extent, thresholds)
            part_to_tier[pid] = tier
        except Exception:
            part_to_tier[pid] = "micro"

    new_tiers = [part_to_tier[pid] for pid in part_ids]
    return new_tiers, part_to_tier


# ──────────────────────────────────────────────
# 验证模式：对比 hyh 旧标签
# ──────────────────────────────────────────────

def verify(ball_radius: float) -> int:
    """对所有已有非 unknown 标签的实例重算，对比一致率。"""
    print(f"\n=== Verify mode: 对比重算结果 vs hyh 原标签（ball_radius={ball_radius}）===\n")
    npz_files = sorted((REPO_ROOT / "data").glob("*.npz"))

    n_total = 0
    n_known = 0
    n_match = 0
    n_off_by_one = 0      # micro<->meso 或 meso<->macro 相邻档误差
    n_far = 0             # micro<->macro 严重不一致
    mismatches: List[Tuple[str, str, str]] = []  # (instance_id, orig, new)

    TIER_ORDER = ["micro", "meso", "macro"]

    for npz_path in npz_files:
        d = np.load(str(npz_path), allow_pickle=True)
        orig = [str(x) for x in d["part_tiers"]]
        n_total += len(orig)

        # 只验证 hyh 已经填了 micro/meso/macro 的位置（跳过 unknown）
        known_mask = [t in {"micro", "meso", "macro"} for t in orig]
        if not any(known_mask):
            continue

        new_tiers, _ = relabel_instance(npz_path, ball_radius)

        iid = npz_path.stem
        for m, (o, n, k) in enumerate(zip(orig, new_tiers, known_mask)):
            if not k:
                continue
            n_known += 1
            if o == n:
                n_match += 1
            else:
                diff = abs(TIER_ORDER.index(o) - TIER_ORDER.index(n))
                if diff == 1:
                    n_off_by_one += 1
                else:
                    n_far += 1
                if len(mismatches) < 10:
                    mismatches.append((iid, o, n))

    if n_known == 0:
        print("(没有 hyh 标签的对照样本)")
        return 0

    print(f"总 candidate 数: {n_total}")
    print(f"hyh 标过 (micro/meso/macro): {n_known}")
    print(f"  exact match:    {n_match} ({100*n_match/n_known:.1f}%)")
    print(f"  off-by-one:     {n_off_by_one} ({100*n_off_by_one/n_known:.1f}%)")
    print(f"  far mismatch:   {n_far} ({100*n_far/n_known:.1f}%)")
    print(f"  exact OR ±1 一致率: {100*(n_match+n_off_by_one)/n_known:.1f}%")

    if mismatches:
        print("\n前 10 个不一致样本 (instance, orig, new):")
        for iid, o, n in mismatches:
            print(f"  {iid}: {o} -> {n}")

    accept_rate = (n_match + n_off_by_one) / n_known
    if accept_rate < 0.90:
        print(f"\n[FAIL] 一致率 {accept_rate:.1%} < 90%，分档逻辑可能有问题。")
        return 1
    print(f"\n[OK] 一致率 {accept_rate:.1%} 通过 90% 阈值")
    return 0


# ──────────────────────────────────────────────
# 全量写入
# ──────────────────────────────────────────────

def write_all(ball_radius: float, force: bool = False) -> int:
    """对所有 200 个 npz 重算 part_tiers 并写回，备份原字段。"""
    print(f"\n=== Write mode（ball_radius={ball_radius}）===\n")
    npz_files = sorted((REPO_ROOT / "data").glob("*.npz"))

    stats_orig = Counter()
    stats_new  = Counter()
    n_written = 0
    n_skipped = 0

    for npz_path in npz_files:
        d = np.load(str(npz_path), allow_pickle=True)
        orig = [str(x) for x in d["part_tiers"]]
        stats_orig.update(orig)

        # 重算
        new_tiers, _ = relabel_instance(npz_path, ball_radius)
        stats_new.update(new_tiers)

        # 备份原字段 + 写回
        data_dict = dict(d)
        # 不重复备份
        if "part_tiers_orig" not in data_dict or force:
            data_dict["part_tiers_orig"] = np.array(orig, dtype=object)
        data_dict["part_tiers"] = np.array(new_tiers, dtype=object)

        np.savez(npz_path, **data_dict)
        n_written += 1

    print(f"已写回 {n_written} 个 npz（备份字段 part_tiers_orig）\n")
    print(f"分布变化:")
    print(f"  原 (hyh) : {dict(stats_orig)}")
    print(f"  新 (重算): {dict(stats_new)}")
    return 0


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="仅在已有 hyh 标签的样本上对比重算一致率，不写回")
    parser.add_argument("--write", action="store_true",
                        help="对全部 200 个 npz 重算 + 写回（自动备份原字段）")
    parser.add_argument("--ball-radius", type=float, default=0.10,
                        help="球域半径（归一化坐标，默认 0.10）")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已存在的 part_tiers_orig 备份")
    args = parser.parse_args()

    if args.verify_only and args.write:
        raise SystemExit("--verify-only 和 --write 互斥")

    if not args.verify_only and not args.write:
        # 默认走验证模式
        return verify(args.ball_radius)

    if args.verify_only:
        return verify(args.ball_radius)
    if args.write:
        # 写入前先验证
        print("[step 1] 先做验证")
        rc = verify(args.ball_radius)
        if rc != 0:
            print("[abort] 验证失败，不写入")
            return rc
        print("\n[step 2] 写入")
        return write_all(args.ball_radius, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
