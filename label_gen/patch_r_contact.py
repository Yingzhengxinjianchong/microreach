"""
label_gen/patch_r_contact.py

Backfill or recompute R_contact labels for existing MicroReach .npz files.

Design:
    - label_gen/r_contact.py provides the raw contact scoring function.
    - This script reads existing .npz files, calls r_contact.py, and writes labels back.
    - R_contact_raw stores the raw contact score.
    - R_contact stores the training label.

For cascade training, use:
    R_contact = min(R_contact_raw, R_geom)

This enforces:
    R_contact <= R_geom

and keeps the original raw score for inspection / ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Repo root: <repo>/label_gen/patch_r_contact.py -> <repo>
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from label_gen.r_contact import batch_compute_r_contact

DATA = _ROOT / "data"
CONSISTENCY_TOL = 0.05


def _load_npz_as_dict(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _needs_patch(arr: Dict[str, Any], force: bool) -> bool:
    """
    Decide whether this file should be patched.

    Default behavior:
        patch only if R_contact is missing or all-NaN.

    force=True:
        recompute even if R_contact already has values.
    """
    if force:
        return True

    if "R_contact" not in arr:
        return True

    rc = arr["R_contact"].astype(np.float32)
    return np.isnan(rc).all()


def _compute_default_normals_axes(M: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep old 47-instance behavior: default normals / axes are all [0, 0, 1].

    This intentionally avoids using batch_generate.py part_info normals/axes,
    so old and newly generated instances can share the same R_contact label policy.
    """
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    part_normals = np.tile(normal, (M, 1))
    part_axes = np.tile(normal, (M, 1))
    return part_normals, part_axes


def _enforce_cascade(
    R_contact_raw: np.ndarray,
    R_geom: np.ndarray,
) -> np.ndarray:
    """
    Project raw contact scores onto the cascade-feasible set:

        R_contact <= R_geom

    NaN R_geom positions remain NaN in R_contact.
    """
    valid = ~np.isnan(R_geom)
    R_contact = np.full_like(R_contact_raw, np.nan, dtype=np.float32)
    R_contact[valid] = np.minimum(R_contact_raw[valid], R_geom[valid])
    return np.clip(R_contact, 0.0, 1.0)


def _count_violations(
    R_geom: np.ndarray,
    R_contact: np.ndarray,
    tol: float,
) -> tuple[int, int, float]:
    valid = (~np.isnan(R_geom)) & (~np.isnan(R_contact))
    n_queries = int(valid.sum())
    if n_queries == 0:
        return 0, 0, 0.0

    n_violations = int(np.sum(R_contact[valid] > R_geom[valid] + tol))
    rate = n_violations / n_queries
    return n_violations, n_queries, rate


def patch_r_contact(
    data_dir: Path = DATA,
    force: bool = False,
    enforce_cascade: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Patch .npz files under data_dir.

    Parameters
    ----------
    data_dir:
        Directory containing .npz files.
    force:
        If True, recompute R_contact even if it already has values.
    enforce_cascade:
        If True, write R_contact = min(R_contact_raw, R_geom).
        If False, write R_contact = R_contact_raw.
    dry_run:
        If True, print what would happen but do not write files.

    Returns
    -------
    summary:
        Patch statistics.
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    npz_paths = sorted(data_dir.glob("*.npz"))
    if not npz_paths:
        print(f"[WARNING] no .npz files found under {data_dir}")

    patched = 0
    skipped = 0
    total_queries = 0
    total_violations = 0
    total_raw_violations = 0

    for path in npz_paths:
        iid = path.stem
        arr = _load_npz_as_dict(path)

        required = ["candidate_p", "queries", "R_geom"]
        missing_fields = [k for k in required if k not in arr]
        if missing_fields:
            print(f"[跳过] {iid}（缺少字段: {missing_fields}）")
            skipped += 1
            continue

        if not _needs_patch(arr, force=force):
            print(f"[跳过] {iid}（R_contact 已有真值；如需重算请加 --force）")
            skipped += 1
            continue

        candidate_p = arr["candidate_p"].astype(np.float32)   # (M, 3)
        queries = arr["queries"].astype(np.float32)           # (M, 24, 4)
        R_geom = arr["R_geom"].astype(np.float32)             # (M, 24)
        M = candidate_p.shape[0]

        part_normals, part_axes = _compute_default_normals_axes(M)

        R_contact_raw = batch_compute_r_contact(
            candidate_p=candidate_p,
            queries=queries,
            part_normals=part_normals,
            part_axes=part_axes,
            seed=42,
        ).astype(np.float32)

        if enforce_cascade:
            R_contact = _enforce_cascade(R_contact_raw, R_geom)
        else:
            R_contact = R_contact_raw.copy()

        raw_v, raw_q, raw_rate = _count_violations(
            R_geom=R_geom,
            R_contact=R_contact_raw,
            tol=CONSISTENCY_TOL,
        )
        n_v, n_q, rate = _count_violations(
            R_geom=R_geom,
            R_contact=R_contact,
            tol=CONSISTENCY_TOL,
        )

        total_raw_violations += raw_v
        total_violations += n_v
        total_queries += n_q

        if not dry_run:
            arr["R_contact_raw"] = R_contact_raw
            arr["R_contact"] = R_contact
            arr["n_consistency_violations"] = np.int32(n_v)
            np.savez_compressed(str(path), **arr)

        print(
            f"[{'预览' if dry_run else '完成'}] {iid}  M={M}  "
            f"raw_mean={np.nanmean(R_contact_raw):.3f}  "
            f"final_mean={np.nanmean(R_contact):.3f}  "
            f"raw_viol={raw_v}/{raw_q} ({raw_rate:.1%})  "
            f"final_viol={n_v}/{n_q} ({rate:.1%})"
        )

        patched += 1

    print()
    print("=" * 70)
    print(f"patch 完成：{patched} 个{'预览' if dry_run else '写入'}，{skipped} 个跳过")
    if total_queries > 0:
        print(
            f"raw 一致性违反总数：{total_raw_violations} / {total_queries} "
            f"({total_raw_violations / total_queries:.1%})"
        )
        print(
            f"final 一致性违反总数：{total_violations} / {total_queries} "
            f"({total_violations / total_queries:.1%})"
        )
    print(f"enforce_cascade: {enforce_cascade}")
    print(f"dry_run: {dry_run}")
    print("=" * 70)

    summary = {
        "num_patched": patched,
        "num_skipped": skipped,
        "total_queries": total_queries,
        "total_raw_violations": total_raw_violations,
        "total_final_violations": total_violations,
        "raw_violation_rate": (
            round(total_raw_violations / total_queries, 4)
            if total_queries > 0
            else None
        ),
        "final_violation_rate": (
            round(total_violations / total_queries, 4)
            if total_queries > 0
            else None
        ),
        "consistency_tol": CONSISTENCY_TOL,
        "enforce_cascade": enforce_cascade,
        "dry_run": dry_run,
        "data_dir": str(data_dir),
        "notes": (
            "R_contact_raw is computed by batch_compute_r_contact with default "
            "normals/axes=[0,0,1]. If enforce_cascade=True, R_contact is written "
            "as min(R_contact_raw, R_geom), so the training label satisfies "
            "R_contact <= R_geom. R_contact_raw is kept for inspection."
        ),
    }

    if not dry_run:
        out = data_dir / "r_contact_patch_summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved: {out}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch or recompute R_contact labels for MicroReach .npz files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA),
        help="Directory containing .npz files. Default: <repo>/data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if R_contact already has values.",
    )
    parser.add_argument(
        "--enforce-cascade",
        action="store_true",
        help="Write R_contact = min(R_contact_raw, R_geom).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview patch statistics without writing .npz files.",
    )
    args = parser.parse_args()

    patch_r_contact(
        data_dir=Path(args.data_dir),
        force=args.force,
        enforce_cascade=args.enforce_cascade,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
