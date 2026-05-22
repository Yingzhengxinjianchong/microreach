"""
eval/significance.py

阶段三 A2：多 seed 显著性检验。

输入：多个 eval_main.py 跑出的 results_*.json，每个对应一个 seed
输出：
  1. mean ± std 对比表（每变体在每个指标上的统计量）
  2. paired t-test：M1 系列 vs M0、M1 系列 vs Where2Act，每个指标一个 p-value

用法：
    # 阶段三：3 seed × 5 模型聚合
    python -m eval.significance \\
        --inputs eval/results_stage3_seed42.json \\
                 eval/results_stage3_seed43.json \\
                 eval/results_stage3_seed44.json \\
        --baseline m0 \\
        --json-out eval/significance_stage3.json

设计要点：
  - paired t-test：因为每个 seed 在同一 test split 上跑，配对自然
  - 配对维度：3 seed × |test_set|=6 instance = 18 个配对样本/指标
  - 显著性阈值：p < 0.05 标 *，p < 0.01 标 **，p < 0.001 标 ***
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


METRICS = [
    "micro_miou",
    "meso_miou",
    "macro_miou",
    "micro_soft_iou",
    "meso_soft_iou",
    "macro_soft_iou",
    "pose_aware_recall_at_1",
    "pose_aware_recall_at_5",
]


def load_results(json_paths: List[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    加载多个 results json。

    Returns:
        {variant_name: {instance_id: [seed1_metrics, seed2_metrics, ...]}}
    """
    per_variant: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for path in json_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"缺失结果文件：{path}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        for variant, entry in data.items():
            instances = entry.get("instances", [])
            if not instances:
                continue
            per_variant.setdefault(variant, {})
            for im in instances:
                iid = im["instance_id"]
                per_variant[variant].setdefault(iid, []).append(im)

    return per_variant


def aggregate_mean_std(
    per_variant: Dict[str, Dict[str, List[Dict[str, Any]]]],
    metric: str,
) -> Dict[str, Tuple[float, float, int]]:
    """
    对每个变体，把所有 (seed, instance) 配对值取均值和标准差。

    Returns:
        {variant: (mean, std, n_samples)}
    """
    out: Dict[str, Tuple[float, float, int]] = {}
    for variant, by_inst in per_variant.items():
        vals: List[float] = []
        for iid, list_metrics in by_inst.items():
            for m in list_metrics:
                v = m.get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
        if not vals:
            out[variant] = (float("nan"), float("nan"), 0)
            continue
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[variant] = (float(np.mean(vals)), std_val, len(vals))
    return out


def paired_arrays(
    per_variant: Dict[str, Dict[str, List[Dict[str, Any]]]],
    variant_a: str,
    variant_b: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 (instance_id, seed_index) 配对，取出 a 和 b 在 metric 上的值。

    Returns:
        (a_vals, b_vals) 等长 numpy array
    """
    a_vals: List[float] = []
    b_vals: List[float] = []

    if variant_a not in per_variant or variant_b not in per_variant:
        return np.array([]), np.array([])

    by_inst_a = per_variant[variant_a]
    by_inst_b = per_variant[variant_b]

    common_iids = sorted(set(by_inst_a.keys()) & set(by_inst_b.keys()))

    for iid in common_iids:
        a_list = by_inst_a[iid]
        b_list = by_inst_b[iid]
        n = min(len(a_list), len(b_list))
        for i in range(n):
            va = a_list[i].get(metric)
            vb = b_list[i].get(metric)
            if va is None or vb is None:
                continue
            if isinstance(va, float) and np.isnan(va):
                continue
            if isinstance(vb, float) and np.isnan(vb):
                continue
            a_vals.append(float(va))
            b_vals.append(float(vb))

    return np.array(a_vals), np.array(b_vals)


def paired_t_test(
    per_variant: Dict[str, Dict[str, List[Dict[str, Any]]]],
    variant_a: str,
    variant_b: str,
    metric: str,
) -> Dict[str, Any]:
    """Paired t-test: H0 = a 与 b 在该指标上均值相等"""
    a, b = paired_arrays(per_variant, variant_a, variant_b, metric)
    if len(a) < 2 or len(b) < 2:
        return {
            "n_pairs": int(len(a)),
            "mean_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "stars": "",
        }

    diff = a - b
    t_stat, p_value = stats.ttest_rel(a, b)

    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = ""

    return {
        "n_pairs": int(len(a)),
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "stars": stars,
    }


def format_mean_std_table(
    per_variant: Dict[str, Dict[str, List[Dict[str, Any]]]],
    variant_order: Optional[List[str]] = None,
) -> str:
    """Mean ± std 对比表"""
    if variant_order is None:
        variant_order = sorted(per_variant.keys())

    lines = []
    lines.append("=" * 140)
    lines.append("Mean ± Std (across seeds × instances)")
    lines.append("=" * 140)

    header = f"  {'variant':16s} | " + " | ".join(f"{m[:14]:>14s}" for m in METRICS)
    lines.append(header)
    lines.append("  " + "-" * 138)

    metric_stats = {metric: aggregate_mean_std(per_variant, metric) for metric in METRICS}

    for v in variant_order:
        if v not in per_variant:
            continue
        row = f"  {v:16s} | "
        cells = []
        for metric in METRICS:
            mean, std, n = metric_stats[metric][v]
            if np.isnan(mean):
                cells.append(f"{'N/A':>14s}")
            else:
                cells.append(f"{mean:.3f}±{std:.3f}".rjust(14))
        row += " | ".join(cells)
        lines.append(row)

    lines.append("=" * 140)
    return "\n".join(lines)


def format_pvalue_table(
    per_variant: Dict[str, Dict[str, List[Dict[str, Any]]]],
    baseline: str,
    comparators: Optional[List[str]] = None,
) -> str:
    """每个变体 vs baseline 的 paired t-test"""
    if comparators is None:
        comparators = [v for v in per_variant.keys() if v != baseline]

    lines = []
    lines.append("=" * 160)
    lines.append(f"Paired t-test: comparator vs baseline ({baseline!r})")
    lines.append("=" * 160)

    header = f"  {'comparator':16s} | " + " | ".join(f"{m[:14]:>18s}" for m in METRICS)
    lines.append(header)
    lines.append("  " + "-" * 158)

    for comp in comparators:
        row = f"  {comp:16s} | "
        cells = []
        for metric in METRICS:
            t = paired_t_test(per_variant, comp, baseline, metric)
            mean_diff = t["mean_diff"]
            p = t["p_value"]
            stars = t["stars"]
            if np.isnan(mean_diff):
                cells.append(f"{'N/A':>18s}")
            else:
                cells.append(f"{mean_diff:+.3f} p={p:.3f}{stars}".rjust(18))
        row += " | ".join(cells)
        lines.append(row)

    lines.append("")
    lines.append("  Significance: * p<0.05, ** p<0.01, *** p<0.001")
    lines.append("  mean_diff > 0  → comparator 高于 baseline；mean_diff < 0  → 低于")
    lines.append("=" * 160)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="多个 results json（不同 seed）")
    parser.add_argument("--baseline", type=str, default="m0",
                        help="paired t-test 的 baseline variant 名")
    parser.add_argument("--variant-order", nargs="*", default=None,
                        help="表格中变体的展示顺序；默认按字典序")
    parser.add_argument("--json-out", type=str, default=None,
                        help="把统计结果写到 json")
    args = parser.parse_args()

    print("=== Significance test ===")
    print(f"  inputs:   {len(args.inputs)} json")
    for p in args.inputs:
        print(f"            {p}")
    print(f"  baseline: {args.baseline}")

    per_variant = load_results(args.inputs)
    print(f"  variants: {sorted(per_variant.keys())}")

    n_inst = {v: len(by_inst) for v, by_inst in per_variant.items()}
    n_seed_max = {v: max(len(lst) for lst in by_inst.values()) if by_inst else 0
                  for v, by_inst in per_variant.items()}
    print(f"  per variant instances: {n_inst}")
    print(f"  per variant max seeds: {n_seed_max}")
    print()

    table1 = format_mean_std_table(per_variant, args.variant_order)
    print(table1)
    print()

    if args.baseline not in per_variant:
        print(f"[warn] baseline {args.baseline!r} 不在结果里，跳过 t-test")
    else:
        comparators = args.variant_order or sorted(per_variant.keys())
        comparators = [v for v in comparators if v != args.baseline]
        table2 = format_pvalue_table(per_variant, args.baseline, comparators)
        print(table2)

    if args.json_out:
        out: Dict[str, Any] = {
            "inputs": args.inputs,
            "baseline": args.baseline,
            "n_variants": len(per_variant),
            "mean_std": {},
            "pairwise_vs_baseline": {},
        }
        for metric in METRICS:
            ms = aggregate_mean_std(per_variant, metric)
            out["mean_std"][metric] = {
                v: {"mean": m, "std": s, "n": n} for v, (m, s, n) in ms.items()
            }
        for v in per_variant.keys():
            if v == args.baseline:
                continue
            out["pairwise_vs_baseline"][v] = {
                metric: paired_t_test(per_variant, v, args.baseline, metric)
                for metric in METRICS
            }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()
