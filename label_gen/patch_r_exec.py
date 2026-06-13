#!/usr/bin/env python3
"""
Patch R_exec labels into existing MicroReach .npz files.

This script does not regenerate R_geom or R_contact. It reads existing .npz
files, computes analytic R_exec from PartNet-Mobility joint metadata, and
writes:
  - R_exec_raw
  - R_exec
  - exec_mapping_status
  - exec_mapped_child_links
  - exec_mapped_joint_types

Recommended P2 command:
  python label_gen/patch_r_exec.py --force --enforce-cascade
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Allow both:
#   python label_gen/patch_r_exec.py
# and:
#   PYTHONPATH=. python -m label_gen.patch_r_exec
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from label_gen.r_exec import compute_r_exec


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="data/eval_set_200.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--partnet-root", default="/root/autodl-tmp/datasets/partnet_mobility")
    ap.add_argument("--summary-out", default="data/r_exec_patch_summary.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--enforce-cascade", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--instances",
        default=None,
        help="Comma-separated instance ids for smoke test, e.g. 1011,40417",
    )
    return ap.parse_args()


def load_eval(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"{path} must contain a list")
    return obj


def save_npz_preserve(path: Path, updates: Dict[str, Any]):
    old = np.load(path, allow_pickle=True)
    payload = {k: old[k] for k in old.files}
    payload.update(updates)
    np.savez(path, **payload)


def main():
    setup_logging()
    args = parse_args()

    eval_path = Path(args.eval_set)
    data_dir = Path(args.data_dir)
    partnet_root = Path(args.partnet_root)

    eval_set = load_eval(eval_path)

    instance_filter = None
    if args.instances:
        instance_filter = {x.strip() for x in args.instances.split(",") if x.strip()}

    records = []
    for rec in eval_set:
        iid = str(rec["instance_id"])
        if instance_filter is not None and iid not in instance_filter:
            continue
        records.append(rec)

    if args.limit is not None:
        records = records[: args.limit]

    logging.info("records to process: %d", len(records))

    global_summary = {
        "processed": 0,
        "written": 0,
        "skipped_existing": 0,
        "missing_npz": [],
        "failed": [],
        "status_counts": {},
        "by_category": {},
        "raw_mean_sum": 0.0,
        "exec_mean_sum": 0.0,
    }

    for idx, rec in enumerate(records, 1):
        iid = str(rec["instance_id"])
        cat = rec.get("category", "unknown")

        npz_path = data_dir / f"{iid}.npz"
        inst_dir = partnet_root / iid

        logging.info("[%d/%d] %s %s", idx, len(records), iid, cat)

        if not npz_path.exists():
            logging.error("  missing npz: %s", npz_path)
            global_summary["missing_npz"].append(iid)
            continue

        try:
            d = np.load(npz_path, allow_pickle=True)

            if (
                "R_exec" in d.files
                and not np.isnan(d["R_exec"]).all()
                and not args.force
            ):
                logging.info("  R_exec exists, skip. Use --force to overwrite.")
                global_summary["skipped_existing"] += 1
                continue

            required = ["candidate_p", "queries", "part_ids", "R_contact"]
            for k in required:
                if k not in d.files:
                    raise KeyError(f"missing key {k} in {npz_path}")

            out = compute_r_exec(
                candidate_p=d["candidate_p"],
                queries=d["queries"],
                part_ids=d["part_ids"],
                partnet_instance_dir=inst_dir,
                r_contact=d["R_contact"],
                enforce_cascade=args.enforce_cascade,
            )

            summary = out["summary"]
            logging.info(
                "  raw_mean=%.4f exec_mean=%.4f status=%s",
                summary["raw_mean"],
                summary["exec_mean"],
                summary["status_counts"],
            )

            global_summary["processed"] += 1
            global_summary["raw_mean_sum"] += summary["raw_mean"]
            global_summary["exec_mean_sum"] += summary["exec_mean"]

            by_cat = global_summary["by_category"].setdefault(
                cat,
                {
                    "processed": 0,
                    "written": 0,
                    "status_counts": {},
                    "raw_mean_sum": 0.0,
                    "exec_mean_sum": 0.0,
                },
            )
            by_cat["processed"] += 1
            by_cat["raw_mean_sum"] += summary["raw_mean"]
            by_cat["exec_mean_sum"] += summary["exec_mean"]

            for status, n in summary["status_counts"].items():
                global_summary["status_counts"][status] = (
                    global_summary["status_counts"].get(status, 0) + int(n)
                )
                by_cat["status_counts"][status] = (
                    by_cat["status_counts"].get(status, 0) + int(n)
                )

            if not args.dry_run:
                save_npz_preserve(
                    npz_path,
                    {
                        "R_exec_raw": out["R_exec_raw"],
                        "R_exec": out["R_exec"],
                        "exec_mapping_status": out["exec_mapping_status"],
                        "exec_mapped_child_links": out["exec_mapped_child_links"],
                        "exec_mapped_joint_types": out["exec_mapped_joint_types"],
                    },
                )
                global_summary["written"] += 1
                by_cat["written"] += 1

        except Exception as e:
            logging.exception("  failed %s: %r", iid, e)
            global_summary["failed"].append({"instance_id": iid, "error": repr(e)})

    if global_summary["processed"] > 0:
        global_summary["raw_mean"] = (
            global_summary["raw_mean_sum"] / global_summary["processed"]
        )
        global_summary["exec_mean"] = (
            global_summary["exec_mean_sum"] / global_summary["processed"]
        )

    for cat, x in global_summary["by_category"].items():
        if x["processed"] > 0:
            x["raw_mean"] = x["raw_mean_sum"] / x["processed"]
            x["exec_mean"] = x["exec_mean_sum"] / x["processed"]

    if not args.dry_run:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, indent=2, ensure_ascii=False)
        logging.info("summary written: %s", out_path)

    logging.info("=" * 72)
    logging.info("processed: %d", global_summary["processed"])
    logging.info("written: %d", global_summary["written"])
    logging.info("skipped_existing: %d", global_summary["skipped_existing"])
    logging.info("missing_npz: %d", len(global_summary["missing_npz"]))
    logging.info("failed: %d", len(global_summary["failed"]))
    logging.info("status_counts: %s", global_summary["status_counts"])
    if global_summary["processed"] > 0:
        logging.info("raw_mean: %.4f", global_summary["raw_mean"])
        logging.info("exec_mean: %.4f", global_summary["exec_mean"])
    logging.info("=" * 72)

    if global_summary["failed"] or global_summary["missing_npz"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
