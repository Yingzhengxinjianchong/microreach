"""
eval/eval_main.py

阶段二中期评测主入口：加载 ckpt -> 在 test 集跑预测 -> 调用 metrics.py 输出对比表。

用法：
    # 评测单个变体
    python -m eval.eval_main --ckpt ckpts/m1_seed42/best.pt --config configs/m1.yaml

    # 对比多个变体（生成完整表格）
    python -m eval.eval_main --compare \\
        m0:configs/m0.yaml:ckpts/m0_seed42/best.pt \\
        m1:configs/m1.yaml:ckpts/m1_seed42/best.pt

    # 加入外部基线（队友预存的 Where2Act npz）
    python -m eval.eval_main \\
        --compare m0:configs/m0.yaml:ckpts/m0_seed42/best.pt m1:configs/m1.yaml:ckpts/m1_seed42/best.pt \\
        --baseline where2act:baselines/where2act_predictions.npz \\
        --json-out eval/results_stage2_full.json

设计要点：
    - 队友黄弋涵已写完 eval/metrics.py，本文件只负责"加载 ckpt + 喂数据 + 调用 metrics"
    - 接口完全沿用 evaluate_instance(pred_geom, gt_geom, tiers, ...)
    - M0 预测的是 (M,) 标量，要广播到 (M, 24) 才能跟 GT 比较（M0 不学方向，所有 query 共享同一个预测值）
    - 外部基线（如 Where2Act）由队友推理后保存 npz；本脚本按 test split 切片后调同一个 evaluate_instance

外部基线 npz 字段约定（队友 baselines/where2act_predictions.npz 已采用）：
    instance_ids:      (N,) object   每个实例 id 字符串
    padded_predictions: (N, max_M, K) float32   sigmoid 后的预测概率
    padded_targets:     (N, max_M, K) float32   .npz 里的 R_geom（GT）
    padded_valid_mask:  (N, max_M)   bool       有效候选点位置
    padded_part_tiers:  (N, max_M)   object     'micro'/'meso'/'macro'/'unknown'
    baseline_name:      scalar str
    description:        scalar str
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# 把项目根加入 sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import (
    DatasetMetrics,
    InstanceMetrics,
    evaluate_dataset,
    evaluate_instance,
)
from microreach_net.dataset import MicroReachDataset, split_dataset
from microreach_net.train import MicroReachNet, load_config


# ──────────────────────────────────────────────
# 加载 ckpt + 跑预测
# ──────────────────────────────────────────────

def load_ckpt(ckpt_path: str, cfg: Dict[str, Any], device: torch.device) -> MicroReachNet:
    """加载 best.pt 到模型。"""
    model = MicroReachNet(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_on_test(
    model: MicroReachNet,
    test_loader: DataLoader,
    device: torch.device,
) -> List[Tuple[str, np.ndarray, np.ndarray, List[str]]]:
    """
    对 test 集所有实例跑预测，返回每实例的 (id, pred (M,24), gt (M,24), tiers list)。

    M0 模型输出 (B, M) 被广播成 (B, M, K) 以便跟 GT 形状一致（M0 不学方向，所有 query 共享）。
    """
    n_queries = 24   # 阶段二中期 8ψ × 3g
    out: List[Tuple[str, np.ndarray, np.ndarray, List[str]]] = []

    for batch in test_loader:
        pc = batch["point_cloud"].to(device)
        cp = batch["candidate_p"].to(device)
        q  = batch["queries"].to(device)
        mask = batch["mask"].to(device)
        instance_ids = batch["instance_id"]

        if model.use_pose_decoder:
            logits = model(pc, cp, q)                    # (B, M_max, K)
            pred = torch.sigmoid(logits)                 # (B, M_max, K)
        else:
            logits = model(pc, cp)                       # (B, M_max)
            pred_scalar = torch.sigmoid(logits)          # (B, M_max)
            pred = pred_scalar.unsqueeze(-1).expand(-1, -1, n_queries)  # broadcast (B, M_max, K)

        target = batch["target"].to(device)              # 可能 (B, M_max, K) 或 (B, M_max)
        if target.dim() == 2:
            # M0 用 per_point_mean 训的，target 是 (B, M_max)；评测时我们要 (M, K)
            # 不过 evaluate 时只看预测/GT 在 (M, K) 上的关系——
            # 这里加载真实 R_geom 标签 (M, K) 来评测才公平。
            # 重新从 .npz 读 R_geom（per_query 形式）：
            target_full_list = []
            for iid in instance_ids:
                npz = np.load(_REPO_ROOT / "data" / f"{iid}.npz", allow_pickle=True)
                target_full_list.append(npz["R_geom"])           # (M, K)，未 padding
            # 不用 padding，直接逐实例处理
            for b, iid in enumerate(instance_ids):
                M_valid = int(mask[b].sum().item())
                pred_b = pred[b, :M_valid].cpu().numpy()         # (M, K)
                gt_b   = target_full_list[b]                     # (M, K)，可能 M_valid 一致
                gt_b = gt_b[:M_valid]                            # 防 padding 引入
                tiers = _load_tiers(iid, M_valid)
                out.append((iid, pred_b, gt_b, tiers))
        else:
            # M1：target 已经是 (B, M_max, K)
            for b, iid in enumerate(instance_ids):
                M_valid = int(mask[b].sum().item())
                pred_b = pred[b, :M_valid].cpu().numpy()
                gt_b   = target[b, :M_valid].cpu().numpy()
                tiers = _load_tiers(iid, M_valid)
                out.append((iid, pred_b, gt_b, tiers))

    return out


def _load_tiers(instance_id: str, M_valid: int) -> List[str]:
    """从原始 .npz 读 part_tiers 字段（队友当前用的是 'unknown'，等他更新就生效）。"""
    npz_path = _REPO_ROOT / "data" / f"{instance_id}.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    if "part_tiers" in data.keys():
        tiers = [str(t) for t in data["part_tiers"]][:M_valid]
    else:
        tiers = ["unknown"] * M_valid
    return tiers


# ──────────────────────────────────────────────
# 评测单个变体
# ──────────────────────────────────────────────

def evaluate_variant(
    name: str,
    config_path: str,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[DatasetMetrics, List[InstanceMetrics]]:
    """加载 ckpt + 跑 test + 调 metrics.py 汇总。"""
    cfg = load_config(config_path)
    print(f"\n=== Evaluating {name} ===")
    print(f"  config: {config_path}")
    print(f"  ckpt:   {ckpt_path}")

    # 数据
    npz_dir = _REPO_ROOT / cfg["data"]["npz_dir"]
    _train_ids, _val_ids, test_ids = split_dataset(
        str(npz_dir),
        ratios=tuple(cfg["split"]["ratios"]),
        seed=cfg["split"]["seed"],
    )
    print(f"  test instances: {len(test_ids)} -> {test_ids}")

    target_mode = cfg.get("target_mode", "per_query")
    test_ds = MicroReachDataset(
        str(npz_dir), test_ids, target_mode,
        cfg["data"]["fields"], num_points=cfg["data"]["num_points"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    # 模型
    model = load_ckpt(ckpt_path, cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # 跑预测
    preds = predict_on_test(model, test_loader, device)

    # 调队友的 evaluate_instance
    inst_metrics: List[InstanceMetrics] = []
    for iid, pred, gt, tiers in preds:
        im = evaluate_instance(
            instance_id=iid,
            pred_geom=pred,
            gt_geom=gt,
            tiers=tiers,
            threshold=cfg["eval"]["bce_threshold"],
        )
        inst_metrics.append(im)

    ds_metric = evaluate_dataset(inst_metrics)
    return ds_metric, inst_metrics


# ──────────────────────────────────────────────
# 外部基线（队友预存 npz）评测
# ──────────────────────────────────────────────

def evaluate_baseline_from_npz(
    name: str,
    npz_path: str,
    test_ids: List[str],
    threshold: float = 0.5,
) -> Tuple[DatasetMetrics, List[InstanceMetrics]]:
    """
    读队友预存的 baseline npz，按 test_ids 切片后调 evaluate_instance。

    Args:
        name:     基线名（输出表格里用，如 "where2act"）
        npz_path: 预存预测 npz 路径
        test_ids: 与自训模型相同的 test 集 instance id 列表（保证公平比较）
        threshold: 二值化阈值（与自训模型相同）
    """
    print(f"\n=== Evaluating {name} (external baseline) ===")
    print(f"  npz: {npz_path}")

    data = np.load(str(_REPO_ROOT / npz_path), allow_pickle=True)
    all_ids = [str(x) for x in data["instance_ids"]]
    preds_all   = data["padded_predictions"]   # (N, max_M, K)
    targets_all = data["padded_targets"]       # (N, max_M, K)
    masks_all   = data["padded_valid_mask"]    # (N, max_M)
    tiers_all   = data["padded_part_tiers"]    # (N, max_M)

    # 解析基线元信息
    bname = str(data["baseline_name"]) if "baseline_name" in data.files else name
    desc = str(data["description"]) if "description" in data.files else ""
    print(f"  baseline: {bname}")
    if desc:
        print(f"  note: {desc[:120]}{'...' if len(desc) > 120 else ''}")

    # 按 test_ids 切片
    id_to_row = {iid: i for i, iid in enumerate(all_ids)}
    missing = [iid for iid in test_ids if iid not in id_to_row]
    if missing:
        raise ValueError(
            f"基线 npz 缺少 test 实例: {missing}（共 {len(missing)} 个）"
        )

    inst_metrics: List[InstanceMetrics] = []
    for iid in test_ids:
        row = id_to_row[iid]
        mask_b = masks_all[row]                                  # (max_M,) bool
        M_valid = int(mask_b.sum())

        pred = preds_all[row, :M_valid].astype(np.float32)        # (M, K)
        gt   = targets_all[row, :M_valid].astype(np.float32)
        tiers = [str(t) for t in tiers_all[row, :M_valid]]

        im = evaluate_instance(
            instance_id=iid,
            pred_geom=pred,
            gt_geom=gt,
            tiers=tiers,
            threshold=threshold,
        )
        inst_metrics.append(im)

    ds_metric = evaluate_dataset(inst_metrics)
    return ds_metric, inst_metrics


def get_test_ids_from_config(config_path: str) -> List[str]:
    """从任一自训变体的 yaml 配置读 test split。所有变体共用同一个 split。"""
    cfg = load_config(config_path)
    npz_dir = _REPO_ROOT / cfg["data"]["npz_dir"]
    _train, _val, test_ids = split_dataset(
        str(npz_dir),
        ratios=tuple(cfg["split"]["ratios"]),
        seed=cfg["split"]["seed"],
    )
    return test_ids


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def print_table_header() -> None:
    print()
    print("  " + "-" * 140)
    print(f"  {'method':20s} | {'micro':6s} | {'meso':6s} | {'macro':6s} | "
          f"{'sIoUmi':6s} | {'sIoUme':6s} | "
          f"{'rec@1':6s} | {'rec@5':6s} | {'cascade':6s} | {'execSucc':8s}")
    print("  " + "-" * 140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="单变体评测：ckpt 路径")
    parser.add_argument("--config", type=str, default=None,
                        help="单变体评测：配置路径")
    parser.add_argument("--compare", nargs="+", default=None,
                        help='多变体对比：name:config:ckpt 三段以 : 分隔，空格分多个')
    parser.add_argument("--baseline", nargs="+", default=None,
                        help='外部基线（队友预存 npz）：name:npz_path 两段以 : 分隔，空格分多个')
    parser.add_argument("--baseline-config-ref", type=str, default="configs/m1.yaml",
                        help="从哪个 yaml 读 test split（所有变体共用 split，默认 m1.yaml）")
    parser.add_argument("--json-out", type=str, default=None,
                        help="结果写入 json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: Dict[str, Any] = {}

    has_any = bool(args.compare or args.baseline or (args.ckpt and args.config))
    if not has_any:
        parser.error("必须提供 --ckpt + --config 或 --compare 或 --baseline")

    print_table_header()

    # 1. 自训模型（实时 ckpt 推理）
    if args.compare:
        for spec in args.compare:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"--compare 项应为 name:config:ckpt，得到 {spec}")
            name, cfg_p, ckpt_p = parts
            ds_metric, inst_list = evaluate_variant(name, cfg_p, ckpt_p, device)
            ds_metric.print_table_row(name)
            results[name] = {
                "dataset": ds_metric.to_dict(),
                "instances": [m.to_dict() for m in inst_list],
            }
    elif args.ckpt and args.config:
        ds_metric, inst_list = evaluate_variant("variant", args.config, args.ckpt, device)
        ds_metric.print_table_row("variant")
        results["variant"] = {
            "dataset": ds_metric.to_dict(),
            "instances": [m.to_dict() for m in inst_list],
        }

    # 2. 外部基线（队友预存 npz）
    if args.baseline:
        # 取 test split：用 --baseline-config-ref 指向的 yaml；
        # 若已经有 --compare，复用第一项的 config 保证 split 一致
        if args.compare:
            cfg_ref = args.compare[0].split(":")[1]
        else:
            cfg_ref = args.baseline_config_ref
        test_ids = get_test_ids_from_config(cfg_ref)
        print(f"\n(baselines use test split from {cfg_ref}: {len(test_ids)} instances)")

        for spec in args.baseline:
            parts = spec.split(":", 1)   # 只切第一个 :，npz 路径里没 :
            if len(parts) != 2:
                raise ValueError(f"--baseline 项应为 name:npz_path，得到 {spec}")
            name, npz_p = parts
            ds_metric, inst_list = evaluate_baseline_from_npz(
                name, npz_p, test_ids,
                threshold=0.5,
            )
            ds_metric.print_table_row(name)
            results[name] = {
                "dataset": ds_metric.to_dict(),
                "instances": [m.to_dict() for m in inst_list],
                "source": "external_baseline_npz",
                "npz_path": npz_p,
            }

    print("  " + "-" * 116)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
