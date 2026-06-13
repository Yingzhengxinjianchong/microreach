"""
label_gen/patch_r_contact.py

Patch existing .npz files whose R_contact field is still all-NaN.

This script is intended for backfilling labels on already-generated .npz files.
It does not run SAPIEN and does not recompute R_geom. For newly generated
instances, label_gen/batch_generate.py should write R_contact directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repo root: <repo>/label_gen/patch_r_contact.py -> <repo>
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from label_gen.r_contact import batch_compute_r_contact

DATA = _ROOT / "data"
CONSISTENCY_TOL = 0.05


def s(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def patch_r_contact(data_dir: Path = DATA) -> dict:
    """
    Patch all .npz files under data_dir whose R_contact is all-NaN.

    Returns
    -------
    summary : dict
        Patch statistics.
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    total_violations = 0
    total_queries = 0
    patched = 0
    skipped = 0

    npz_paths = sorted(data_dir.glob("*.npz"))
    if not npz_paths:
        print(f"[WARNING] no .npz files found under {data_dir}")

    for path in npz_paths:
        iid = path.stem

        with np.load(path, allow_pickle=True) as d:
            arr = {k: d[k] for k in d.files}

        if "R_contact" not in arr:
            print(f"[跳过] {iid}（缺少 R_contact 字段）")
            skipped += 1
            continue

        # 已经 patch 过则跳过（R_contact 不全是 NaN）
        rc_existing = arr["R_contact"].astype(np.float32)
        if not np.isnan(rc_existing).all():
            print(f"[跳过] {iid}（R_contact 已有真值）")
            skipped += 1
            continue

        candidate_p = arr["candidate_p"].astype(np.float32)   # (M, 3)
        queries = arr["queries"].astype(np.float32)           # (M, 24, 4)
        R_geom = arr["R_geom"].astype(np.float32)             # (M, 24)
        M = candidate_p.shape[0]

        # 与旧版 patch 脚本保持一致：默认 normals / axes 全部 [0, 0, 1]
        part_normals = np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            (M, 1),
        )
        part_axes = np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            (M, 1),
        )

        R_contact = batch_compute_r_contact(
            candidate_p=candidate_p,
            queries=queries,
            part_normals=part_normals,
            part_axes=part_axes,
            seed=42,
        )  # (M, 24) float32

        # 一致性校验：R_contact <= R_geom + tol
        valid_mask = ~np.isnan(R_geom)
        n_violations = int(
            np.sum(R_contact[valid_mask] > R_geom[valid_mask] + CONSISTENCY_TOL)
        )
        n_queries = int(valid_mask.sum())
        viol_rate = n_violations / n_queries if n_queries > 0 else 0.0

        total_violations += n_violations
        total_queries += n_queries

        # 写回 .npz（全量字段保留，仅覆盖 R_contact 和 n_consistency_violations）
        arr["R_contact"] = R_contact
        arr["n_consistency_violations"] = np.int32(n_violations)
        np.savez_compressed(str(path), **arr)

        print(
            f"[完成] {iid}  M={M}  "
            f"R_contact mean={R_contact.mean():.3f}  "
            f"violations={n_violations}/{n_queries} ({viol_rate:.1%})"
        )
        patched += 1

    print()
    print("=" * 55)
    print(f"patch 完成：{patched} 个写入，{skipped} 个跳过")
    if total_queries > 0:
        print(
            f"一致性违反总数：{total_violations} / {total_queries} "
            f"({total_violations / total_queries:.1%})"
        )
    print("=" * 55)

    summary = {
        "num_patched": patched,
        "num_skipped": skipped,
        "total_queries": total_queries,
        "total_violations": total_violations,
        "violation_rate": (
            round(total_violations / total_queries, 4)
            if total_queries > 0
            else None
        ),
        "consistency_tol": CONSISTENCY_TOL,
        "data_dir": str(data_dir),
        "notes": (
            "R_contact computed by batch_compute_r_contact with default "
            "normals/axes=[0,0,1]. This script is for backfilling existing "
            ".npz files whose R_contact is all-NaN."
        ),
    }

    out = data_dir / "r_contact_patch_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved: {out}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch R_contact labels for existing MicroReach .npz files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA),
        help="Directory containing .npz files. Default: <repo>/data",
    )
    args = parser.parse_args()

    patch_r_contact(Path(args.data_dir))


if __name__ == "__main__":
    main()
