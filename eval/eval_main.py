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

    # 输出 json 报告
    python -m eval.eval_main --compare ... --json-out eval/results.json

设计要点：
    - 队友黄弋涵已写完 eval/metrics.py，本文件只负责"加载 ckpt + 喂数据 + 调用 metrics"
    - 接口完全沿用 evaluate_instance(pred_geom, gt_geom, tiers, ...)
    - M0 预测的是 (M,) 标量，要广播到 (M, 24) 才能跟 GT 比较（M0 不学方向，所有 query 共享同一个预测值）
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
# 主入口
# ──────────────────────────────────────────────

def print_table_header() -> None:
    print()
    print("  " + "-" * 116)
    print(f"  {'method':20s} | {'micro':6s} | {'meso':6s} | {'macro':6s} | "
          f"{'rec@1':6s} | {'rec@5':6s} | {'cascade':6s} | {'execSucc':8s}")
    print("  " + "-" * 116)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="单变体评测：ckpt 路径")
    parser.add_argument("--config", type=str, default=None,
                        help="单变体评测：配置路径")
    parser.add_argument("--compare", nargs="+", default=None,
                        help='多变体对比：name:config:ckpt 三段以 : 分隔，空格分多个')
    parser.add_argument("--json-out", type=str, default=None,
                        help="结果写入 json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: Dict[str, Any] = {}

    if args.compare:
        print_table_header()
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
        print("  " + "-" * 116)
    elif args.ckpt and args.config:
        ds_metric, inst_list = evaluate_variant("variant", args.config, args.ckpt, device)
        print_table_header()
        ds_metric.print_table_row("variant")
        print("  " + "-" * 116)
        results["variant"] = {
            "dataset": ds_metric.to_dict(),
            "instances": [m.to_dict() for m in inst_list],
        }
    else:
        parser.error("必须提供 --ckpt + --config 或 --compare")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
