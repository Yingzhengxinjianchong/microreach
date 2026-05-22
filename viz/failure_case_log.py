"""
viz/failure_case_log.py

阶段三 A3：失败案例库（CCF-C 投稿加分项）。

输入：eval_main.py 输出的 results_stage3_seed*.json（已含 instance-level pred / gt）

设计依据：
  - IROS 2023 standard practice（"affordance failure analysis"）：把失败 case 按预测
    向量与 GT 向量的几何关系分类，比单纯报告 mIoU/Recall 更具诊断价值。

输出：
  - eval/failure_log_<variant>.csv —— 一行一个 (instance, candidate, top1_pred_idx,
    top1_gt_idx, failure_class, pred_top1_score, gt_top1_score, sIoU)
  - 后续 failure_taxonomy.py 据此画饼图与按 tier 分组的失败模式分布

用法：
    python -m viz.failure_case_log \\
        --inputs eval/results_stage3_seed42.json \\
                 eval/results_stage3_seed43.json \\
        --variant m1_focal \\
        --out eval/failure_log_m1_focal.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# Fibonacci 球面 8 ψ × 3 g = 24 query
N_PSI = 8
N_G   = 3
G_NAMES = ["pinch", "power", "poke"]


def classify_failure(
    pred_geom: np.ndarray,    # (24,) 单个候选点的 24 query 预测
    gt_geom:   np.ndarray,    # (24,) 同
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    把单个候选点的预测 vs GT 分类为：

      "hit"               : top-1 命中（pred argmax 处 gt > threshold）
      "direction_flip"    : top-1 抓取构型对，但 ψ 方向错（"哪个方向"错了）
      "grasp_mismatch"    : top-1 ψ 方向对，但抓取构型错（pinch/power/poke 选错）
      "low_confidence"    : top-1 既不命中 ψ 也不命中 g，但整体 sigmoid 输出偏低
                            （pred max < 0.4 即认为是 "我不知道" 型失败）
      "wrong_quadrant"    : top-1 ψ 与 g 都错（完全押错宝）
      "no_gt"             : GT 全 0（数据本身没正样本，跳过统计）

    返回 dict 含 failure_class + top1 信息。
    """
    if (gt_geom > threshold).sum() == 0:
        return {
            "failure_class":    "no_gt",
            "top1_pred_idx":    int(pred_geom.argmax()),
            "top1_gt_idx":      -1,
            "pred_top1_score":  float(pred_geom.max()),
            "gt_top1_score":    0.0,
        }

    top1_pred = int(pred_geom.argmax())
    top1_gt   = int(gt_geom.argmax())
    pred_score = float(pred_geom[top1_pred])
    gt_score   = float(gt_geom[top1_gt])

    # 命中：pred argmax 的位置 GT 也 > threshold
    if gt_geom[top1_pred] > threshold:
        cls = "hit"
    else:
        # query 索引 → (g_idx, psi_idx)：query[g*8 + psi]
        pred_g, pred_psi = divmod(top1_pred, N_PSI)
        gt_g,   gt_psi   = divmod(top1_gt,   N_PSI)

        if pred_score < 0.4:
            cls = "low_confidence"
        elif pred_g == gt_g and pred_psi != gt_psi:
            cls = "direction_flip"
        elif pred_g != gt_g and pred_psi == gt_psi:
            cls = "grasp_mismatch"
        else:
            cls = "wrong_quadrant"

    return {
        "failure_class":    cls,
        "top1_pred_idx":    top1_pred,
        "top1_gt_idx":      top1_gt,
        "pred_top1_score":  pred_score,
        "gt_top1_score":    gt_score,
    }


def soft_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """单候选点的 soft IoU。"""
    p, g = pred, gt
    inter = np.minimum(p, g).sum()
    union = np.maximum(p, g).sum()
    return float(inter / (union + 1e-8))


def aggregate_from_jsons(
    json_paths: List[str],
    variant: str,
) -> List[Dict[str, Any]]:
    """
    从多个 results json 读 variant 的 (instance, candidate) 级失败信息。

    Returns: list of dict per candidate point.
    """
    rows: List[Dict[str, Any]] = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if variant not in data:
            print(f"  [warn] {variant!r} 不在 {path}，跳过")
            continue

        seed_idx = Path(path).stem.split("seed")[-1]   # 'seed42' → '42'
        instances = data[variant].get("instances", [])
        if not instances:
            continue

        # 重新读 npz 拿原始 R_geom / part_tiers + 之前的预测要从 json metric 反推
        # —— 这里走更简单路径：直接从 npz 读 GT 24-dim；
        # pred 在阶段二 eval_main 的 json 里只存了 instance-level 聚合 metric，
        # 没存逐 candidate 24-dim 预测。所以 failure_case_log 需要重新加载 ckpt 预测。
        # 简化方案：本脚本只做"输入是逐 candidate pred/gt npz"模式的工具函数；
        # 命令行用法暂时禁用 json 路径，改为通过 ckpt+config 重新推理。
        raise NotImplementedError(
            "failure_case_log 当前仅支持通过 ckpt 重新推理；results json 没存逐 candidate 预测。"
            "请用 --ckpt --config 模式。"
        )
    return rows


def run_from_ckpt(
    config_path: str,
    ckpt_path: str,
    out_csv: str,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """通过加载 ckpt + 跑 test 集，按 candidate 粒度记录失败案例。"""
    import sys
    import torch
    from torch.utils.data import DataLoader
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    from eval.eval_main import load_ckpt, predict_on_test
    from microreach_net.dataset import MicroReachDataset, split_dataset
    from microreach_net.train import load_config

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    npz_dir = REPO_ROOT / cfg["data"]["npz_dir"]
    _train, _val, test_ids = split_dataset(
        str(npz_dir),
        ratios=tuple(cfg["split"]["ratios"]),
        seed=cfg["split"]["seed"],
    )

    target_mode = cfg.get("target_mode", "per_query")
    test_ds = MicroReachDataset(
        str(npz_dir), test_ids, target_mode,
        cfg["data"]["fields"], num_points=cfg["data"]["num_points"],
        load_contact=cfg["data"].get("load_contact", False),
        load_exec=cfg["data"].get("load_exec", False),
    )
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=0)

    model = load_ckpt(ckpt_path, cfg, device)
    preds = predict_on_test(model, loader, device)

    rows: List[Dict[str, Any]] = []
    for entry in preds:
        iid = entry["instance_id"]
        pred = entry["pred_geom"]                   # (M, 24)
        gt   = entry["gt_geom"]                      # (M, 24)
        tiers = entry["tiers"]

        M_valid = pred.shape[0]
        for m in range(M_valid):
            cls_info = classify_failure(pred[m], gt[m], threshold)
            cls_info["instance_id"] = iid
            cls_info["candidate_idx"] = m
            cls_info["tier"] = tiers[m] if m < len(tiers) else "unknown"
            cls_info["soft_iou"] = soft_iou(pred[m], gt[m])
            rows.append(cls_info)

    # 写 CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "instance_id", "candidate_idx", "tier",
        "failure_class",
        "top1_pred_idx", "top1_gt_idx",
        "pred_top1_score", "gt_top1_score", "soft_iou",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[saved] {out_csv}  ({len(rows)} candidates)")

    # 简单分类统计
    from collections import Counter
    cnt = Counter(r["failure_class"] for r in rows)
    n_eff = sum(v for k, v in cnt.items() if k != "no_gt")
    print(f"\nFailure mode distribution (excl. no_gt, total={n_eff}):")
    for cls, n in sorted(cnt.items(), key=lambda x: -x[1]):
        if cls == "no_gt":
            print(f"  {cls:18s}: {n} (跳过，GT 全 0)")
        else:
            pct = 100.0 * n / max(n_eff, 1)
            print(f"  {cls:18s}: {n:3d}  ({pct:.1f}%)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt",   type=str, required=True)
    parser.add_argument("--out",    type=str, required=True,
                        help="输出 CSV 路径，如 eval/failure_log_m1_focal_seed42.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_from_ckpt(args.config, args.ckpt, args.out, args.threshold)


if __name__ == "__main__":
    main()
