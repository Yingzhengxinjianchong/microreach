"""
microreach_net/dataset.py

PyTorch Dataset：读取 label_gen/batch_generate.py 生成的 .npz 标签文件。

数据接口（与 label_gen/batch_generate.py 第 311-330 行 save_dict 一致）：
    .npz 字段:
        point_cloud:  (N, 3) float32     # N=30000，整场景点云
        candidate_p:  (M, 3) float32     # M=10（实际可能 5-15），候选交互点
        queries:      (M, 24, 4) float32 # (ψx, ψy, ψz, g_idx)
        R_geom:       (M, 24) float32 ∈ [0, 1]
        R_contact:    (M, 24) float32    # 阶段二全 NaN（占位）
        R_exec:       (M, 24) float32    # 阶段二全 NaN（占位）
        part_ids:     (M,) str
        part_tiers:   (M,) str           # 当前是 'unknown'
        instance_id:  scalar str
        n_consistency_violations: scalar int

阶段二中期只读 R_geom；R_contact / R_exec 留给阶段三。

输出 batch（DataLoader 默认 collate）：
    point_cloud:  (B, N, 3) float32  Tensor
    candidate_p:  (B, M, 3)
    queries:      (B, M, 24, 4)
    target:       (B, M, 24) for per_query mode  或  (B, M) for per_point_mean mode
    instance_id:  list[str] of length B

注意：不同实例的 M（候选点数）可能不同（实际观察：5 / 10 / 15 等都有）。
本 Dataset 在 __getitem__ 里 pad 到 max_M（默认 16，足够），并返回 mask
区分有效点 / padding 点。Loss / metric 计算时按 mask 加权。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MicroReachDataset(Dataset):
    """
    Dataset for MicroReach 阶段二中期训练。

    Args:
        npz_dir:        .npz 标签所在目录（相对 repo 根或绝对路径）
        instance_ids:   要加载的实例 id 列表（如 ['1011', '1034', ...]）
                        由 split_dataset() 提前划分好
        target_mode:    'per_query' (M1: 直接监督 R_geom (M, 24))
                       | 'per_point_mean' (M0: 监督 R_geom.mean(-1) (M,))
        fields:         .npz 字段名映射（来自 default.yaml.data.fields）
        num_points:     场景点云下采样到的点数（None 表示不下采样）
    """

    def __init__(
        self,
        npz_dir: str,
        instance_ids: List[str],
        target_mode: str = "per_query",
        fields: Optional[Dict[str, str]] = None,
        num_points: Optional[int] = None,
        max_M: int = 16,
    ):
        self.npz_dir = Path(npz_dir)
        if not self.npz_dir.exists():
            raise FileNotFoundError(
                f"npz_dir 不存在: {self.npz_dir.absolute()}"
            )

        self.instance_ids = list(instance_ids)
        if len(self.instance_ids) == 0:
            raise ValueError("instance_ids 为空，没东西训")

        if target_mode not in {"per_query", "per_point_mean"}:
            raise ValueError(
                f"target_mode 必须是 per_query 或 per_point_mean，收到 {target_mode}"
            )
        self.target_mode = target_mode

        self.fields = fields or {
            "point_cloud": "point_cloud",
            "candidate_p": "candidate_p",
            "queries":     "queries",
            "r_geom":      "R_geom",
            "part_tiers":  "part_tiers",
        }
        self.num_points = num_points
        self.max_M = max_M

        self._check_files_exist()

    def _check_files_exist(self) -> None:
        """实例化时一次性检查所有 .npz 都在，避免训到一半才报缺文件。"""
        missing = []
        for iid in self.instance_ids:
            p = self.npz_dir / f"{iid}.npz"
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                f"缺失 {len(missing)} 个 .npz 文件，前 5 个：\n"
                + "\n".join(missing[:5])
            )

    def __len__(self) -> int:
        return len(self.instance_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        iid = self.instance_ids[idx]
        npz_path = self.npz_dir / f"{iid}.npz"

        # allow_pickle=True 因为有 str dtype 字段（part_ids / part_tiers / instance_id）
        data = np.load(str(npz_path), allow_pickle=True)

        F = self.fields
        point_cloud = data[F["point_cloud"]].astype(np.float32)  # (N, 3)
        candidate_p = data[F["candidate_p"]].astype(np.float32)  # (M, 3)
        queries     = data[F["queries"]].astype(np.float32)       # (M, 24, 4)
        r_geom      = data[F["r_geom"]].astype(np.float32)        # (M, 24)

        # 点云强制 resample 到 self.num_points（不同实例 N 不一致，必须对齐才能 batch）
        # 默认 30000，与 default.yaml 一致
        if self.num_points is None:
            target_N = 30000
        else:
            target_N = self.num_points
        N = point_cloud.shape[0]
        if N >= target_N:
            idx_sub = np.random.choice(N, target_N, replace=False)
        else:
            # 极少数实例点不足时，with replacement 补到 target_N
            idx_sub = np.random.choice(N, target_N, replace=True)
        point_cloud = point_cloud[idx_sub]

        # 构造目标
        if self.target_mode == "per_query":
            target = r_geom                              # (M, 24)
        else:  # per_point_mean
            target = r_geom.mean(axis=-1)                # (M,)

        # Pad candidate_p / queries / target 到 max_M，并生成 mask
        M = candidate_p.shape[0]
        if M > self.max_M:
            raise ValueError(
                f"实例 {iid} 的 M={M} 超过 max_M={self.max_M}，请增大 max_M"
            )

        mask = np.zeros(self.max_M, dtype=np.float32)
        mask[:M] = 1.0

        candidate_p_pad = np.zeros((self.max_M, 3), dtype=np.float32)
        candidate_p_pad[:M] = candidate_p

        queries_pad = np.zeros((self.max_M, 24, 4), dtype=np.float32)
        queries_pad[:M] = queries

        if self.target_mode == "per_query":
            target_pad = np.zeros((self.max_M, 24), dtype=np.float32)
            target_pad[:M] = target
        else:
            target_pad = np.zeros((self.max_M,), dtype=np.float32)
            target_pad[:M] = target

        return {
            "point_cloud": torch.from_numpy(point_cloud),    # (N, 3)
            "candidate_p": torch.from_numpy(candidate_p_pad), # (max_M, 3)
            "queries":     torch.from_numpy(queries_pad),     # (max_M, 24, 4)
            "target":      torch.from_numpy(target_pad),      # (max_M, 24) or (max_M,)
            "mask":        torch.from_numpy(mask),            # (max_M,) 1=valid, 0=pad
            "instance_id": iid,
        }


# ──────────────────────────────────────────────
# 数据划分工具
# ──────────────────────────────────────────────

def split_dataset(
    npz_dir: str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    扫描 npz_dir 下所有 .npz，按 ratios 划分 train/val/test。

    返回三个 instance_id 列表。
    """
    npz_dir = Path(npz_dir)
    all_files = sorted(npz_dir.glob("*.npz"))
    all_ids = [p.stem for p in all_files]

    if len(all_ids) == 0:
        raise FileNotFoundError(f"{npz_dir} 下找不到任何 .npz 文件")

    rng = np.random.RandomState(seed)
    rng.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:n_train + n_val]
    test_ids  = all_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


# ──────────────────────────────────────────────
# Smoke test：python -m microreach_net.dataset
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    npz_dir = repo_root / "data"

    print(f"扫描 {npz_dir}...")
    train_ids, val_ids, test_ids = split_dataset(str(npz_dir))
    print(f"  train: {len(train_ids)} 个 -> {train_ids[:3]}...")
    print(f"  val:   {len(val_ids)} 个 -> {val_ids}")
    print(f"  test:  {len(test_ids)} 个 -> {test_ids}")

    print("\n=== per_query 模式（M1 用）===")
    ds = MicroReachDataset(
        npz_dir=str(npz_dir),
        instance_ids=train_ids,
        target_mode="per_query",
    )
    print(f"  len: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, range=[{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k}: {v}")

    print("\n=== per_point_mean 模式（M0 用）===")
    ds_m0 = MicroReachDataset(
        npz_dir=str(npz_dir),
        instance_ids=train_ids,
        target_mode="per_point_mean",
    )
    sample = ds_m0[0]
    print(f"  target shape: {tuple(sample['target'].shape)} (应为 (M,))")

    print("\n=== DataLoader 测试 ===")
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=None,  # 默认 collate（要求所有样本同 shape）
    )
    try:
        batch = next(iter(loader))
        print(f"  point_cloud batch: {tuple(batch['point_cloud'].shape)}")
        print(f"  candidate_p batch: {tuple(batch['candidate_p'].shape)}")
        print(f"  queries batch:     {tuple(batch['queries'].shape)}")
        print(f"  target batch:      {tuple(batch['target'].shape)}")
        print(f"  instance_ids:      {batch['instance_id']}")
        print(f"  mask batch:        {tuple(batch['mask'].shape)}, valid points: {batch['mask'].sum(-1).tolist()}")
        print("\n[OK] Dataset smoke test passed")
    except RuntimeError as e:
        print(f"\n[FAIL] Default collate failed:\n  {e}")
