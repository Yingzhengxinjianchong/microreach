"""
eval/eval_w2a_overlap.py

阶段三 P2: Where2Act 47 实例 npz vs 我们的 200 实例 test set 只有 7 个共同实例。
在这 7 个交集实例上，公平对比我们 4 模型 + W2A，跑 paired t-test。

设计:
  - 输入: 我们的 4 个变体 ckpt + W2A npz
  - 自动算交集（M_full 200 实例 test split ∩ W2A npz instance_ids）
  - 对每个 ckpt 跑预测，但**只保留交集实例**
  - W2A 直接从 npz 取交集
  - 调 evaluate_instance + evaluate_dataset，输出对比表 + json

用法:
    python -m eval.eval_w2a_overlap \\
        --compare m1:configs/m1.yaml:ckpts/m1_seed42/best.pt \\
                  m2:configs/m2.yaml:ckpts/m2_seed42/best.pt \\
                  m_full:configs/m_full.yaml:ckpts/m_full_seed42/best.pt \\
                  m_full_nocascade:configs/m_full_nocascade.yaml:ckpts/m_full_nocascade_seed42/best.pt \\
        --w2a-npz eval/results/where2act_predictions.npz \\
        --json-out eval/results_stage3_w2a_overlap_seed42.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.eval_main import load_ckpt, predict_on_test, print_table_header
from eval.metrics import (
    DatasetMetrics,
    InstanceMetrics,
    evaluate_dataset,
    evaluate_instance,
)
from microreach_net.dataset import MicroReachDataset, split_dataset
from microreach_net.train import load_config


def compute_overlap_ids(w2a_npz_path: str, ref_config: str) -> List[str]:
    """算 200 实例 test split ∩ W2A npz instance_ids 的交集。"""
    cfg = load_config(ref_config)
    npz_dir = _REPO_ROOT / cfg["data"]["npz_dir"]
    _, _, test_ids = split_dataset(
        str(npz_dir),
        ratios=tuple(cfg["split"]["ratios"]),
        seed=cfg["split"]["seed"],
    )
    d = np.load(w2a_npz_path, allow_pickle=True)
    w2a_ids = set(str(x) for x in d["instance_ids"])
    return [iid for iid in test_ids if iid in w2a_ids]


def evaluate_variant_on_ids(
    name: str,
    config_path: str,
    ckpt_path: str,
    keep_ids: List[str],
    device: torch.device,
) -> Tuple[DatasetMetrics, List[InstanceMetrics]]:
    """加载 ckpt 跑预测，但只保留 keep_ids 列表里的实例。"""
    cfg = load_config(config_path)
    print(f"\n=== Evaluating {name} ===")
    print(f"  ckpt: {ckpt_path}")

    npz_dir = _REPO_ROOT / cfg["data"]["npz_dir"]
    # 直接用 keep_ids 作为"评测集"
    target_mode = cfg.get("target_mode", "per_query")
    data_cfg = cfg["data"]
    test_ds = MicroReachDataset(
        str(npz_dir), keep_ids, target_mode,
        data_cfg["fields"], num_points=data_cfg["num_points"],
        load_contact=bool(data_cfg.get("load_contact", False)),
        load_exec=bool(data_cfg.get("load_exec", False)),
    )
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=0)

    model = load_ckpt(ckpt_path, cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}, instances: {len(keep_ids)}")

    preds = predict_on_test(model, loader, device)
    inst_metrics: List[InstanceMetrics] = []
    for entry in preds:
        im = evaluate_instance(
            instance_id=entry["instance_id"],
            pred_geom=entry["pred_geom"],
            gt_geom=entry["gt_geom"],
            tiers=entry["tiers"],
            pred_contact=entry.get("pred_contact"),
            gt_contact=entry.get("gt_contact"),
            pred_exec=entry.get("pred_exec"),
            gt_exec=entry.get("gt_exec"),
            threshold=cfg["eval"]["bce_threshold"],
        )
        inst_metrics.append(im)

    return evaluate_dataset(inst_metrics), inst_metrics


def evaluate_w2a_on_ids(
    npz_path: str,
    keep_ids: List[str],
) -> Tuple[DatasetMetrics, List[InstanceMetrics]]:
    """W2A npz 评测，只用 keep_ids。"""
    print(f"\n=== Evaluating where2act ===")
    print(f"  npz: {npz_path}, instances: {len(keep_ids)}")

    data = np.load(npz_path, allow_pickle=True)
    all_ids = [str(x) for x in data["instance_ids"]]
    preds_all   = data["padded_predictions"]
    targets_all = data["padded_targets"]
    masks_all   = data["padded_valid_mask"]
    tiers_all   = data["padded_part_tiers"]

    id_to_row = {iid: i for i, iid in enumerate(all_ids)}

    inst_metrics: List[InstanceMetrics] = []
    for iid in keep_ids:
        row = id_to_row[iid]
        mask_b = masks_all[row]
        M_valid = int(mask_b.sum())
        pred = preds_all[row, :M_valid].astype(np.float32)
        # 用 npz 里的 W2A targets（按 candidate_p 推理时记录的 R_geom 标签）
        gt   = targets_all[row, :M_valid].astype(np.float32)
        tiers = [str(t) for t in tiers_all[row, :M_valid]]

        im = evaluate_instance(
            instance_id=iid, pred_geom=pred, gt_geom=gt,
            tiers=tiers, threshold=0.5,
        )
        inst_metrics.append(im)

    return evaluate_dataset(inst_metrics), inst_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs="+", required=True,
                        help='name:config:ckpt 三段 :')
    parser.add_argument("--w2a-npz", type=str,
                        default="eval/results/where2act_predictions.npz")
    parser.add_argument("--ref-config", type=str, default="configs/m_full.yaml",
                        help="读取 200 实例 test split 的参考 config")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    keep_ids = compute_overlap_ids(args.w2a_npz, args.ref_config)
    print(f"\n[overlap] 200 实例 test split ∩ W2A npz = {len(keep_ids)} 实例")
    print(f"   {keep_ids}")

    if len(keep_ids) == 0:
        raise SystemExit("交集为空，无法对比")

    results: Dict[str, Any] = {"overlap_ids": keep_ids}
    print_table_header()

    for spec in args.compare:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"--compare 项应为 name:config:ckpt: {spec}")
        name, cfg_p, ckpt_p = parts
        ds_metric, inst_list = evaluate_variant_on_ids(name, cfg_p, ckpt_p, keep_ids, device)
        ds_metric.print_table_row(name)
        results[name] = {
            "dataset": ds_metric.to_dict(),
            "instances": [m.to_dict() for m in inst_list],
        }

    # W2A
    ds_metric, inst_list = evaluate_w2a_on_ids(
        str(_REPO_ROOT / args.w2a_npz), keep_ids,
    )
    ds_metric.print_table_row("where2act")
    results["where2act"] = {
        "dataset": ds_metric.to_dict(),
        "instances": [m.to_dict() for m in inst_list],
        "source": "external_baseline_npz",
        "npz_path": args.w2a_npz,
    }

    print("  " + "-" * 140)

    if args.json_out:
        out = _REPO_ROOT / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
