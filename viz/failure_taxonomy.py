"""
viz/failure_taxonomy.py

阶段三 A3：把 failure_case_log.py 输出的 CSV 聚合成失败分布饼图与按 tier 分组的堆叠图。

设计依据：
  - CVPR / ICRA 投稿中常见"failure analysis" figure（pie + stacked bar）
    比 quantitative table 更直观说明"模型在什么场景下失败"。

输出：
  - eval/failure_taxonomy_<variant>.png  —— 双 panel：
      Panel A 失败类别饼图（hit / direction_flip / grasp_mismatch / low_confidence / wrong_quadrant）
      Panel B 按 tier 分组的堆叠柱状图（micro / meso / macro 各失败模式占比）

用法：
    python -m viz.failure_taxonomy \\
        --inputs eval/failure_log_m1_focal_seed42.csv \\
                 eval/failure_log_m1_focal_seed43.csv \\
                 eval/failure_log_m1_focal_seed44.csv \\
        --title "M1 Focal failure taxonomy (3 seeds)" \\
        --out eval/failure_taxonomy_m1_focal.png
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSES = [
    "hit",
    "direction_flip",
    "grasp_mismatch",
    "low_confidence",
    "wrong_quadrant",
]
CLASS_LABELS = {
    "hit":             "Hit (top-1 ok)",
    "direction_flip":  "Direction flip\n(g ok, psi wrong)",
    "grasp_mismatch":  "Grasp mismatch\n(psi ok, g wrong)",
    "low_confidence":  "Low confidence\n(pred max < 0.4)",
    "wrong_quadrant":  "Wrong quadrant\n(psi + g both wrong)",
}
COLORS_BW = ["#444444", "#777777", "#999999", "#BBBBBB", "#DDDDDD"]


def load_csvs(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("failure_class") == "no_gt":
                    continue
                rows.append(row)
    return rows


def plot(rows: List[Dict[str, str]], title: str, out_path: str) -> None:
    if not rows:
        print("[empty] no valid samples")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13)

    # Panel A: pie
    cnt = Counter(r["failure_class"] for r in rows)
    total = sum(cnt.values())
    sizes = [cnt.get(c, 0) for c in CLASSES]
    labels = [f"{CLASS_LABELS[c]}\n{cnt.get(c,0)} ({100*cnt.get(c,0)/total:.1f}%)" for c in CLASSES]
    non_zero = [(s, l, col) for s, l, col in zip(sizes, labels, COLORS_BW) if s > 0]
    if non_zero:
        s_n, l_n, c_n = zip(*non_zero)
        axes[0].pie(s_n, labels=l_n, colors=c_n, autopct="", startangle=90,
                     wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    axes[0].set_title(f"Failure distribution (n={total})", fontsize=11)

    # Panel B: stacked bar by tier
    tiers = ["micro", "meso", "macro"]
    grid: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        grid[r["tier"]][r["failure_class"]] += 1

    bottoms = np.zeros(len(tiers), dtype=float)
    for ci, cls in enumerate(CLASSES):
        heights = []
        for t in tiers:
            total_t = sum(grid[t].values())
            if total_t == 0:
                heights.append(0)
            else:
                heights.append(100.0 * grid[t].get(cls, 0) / total_t)
        axes[1].bar(tiers, heights, bottom=bottoms, color=COLORS_BW[ci],
                    label=cls, edgecolor="white", linewidth=0.8)
        for i, h in enumerate(heights):
            if h > 4:
                axes[1].text(i, bottoms[i] + h/2, f"{h:.0f}%",
                              ha="center", va="center", fontsize=8,
                              color="white" if ci < 2 else "black")
        bottoms += heights

    axes[1].set_ylabel("Proportion (%)")
    axes[1].set_title("Failure mode by part tier", fontsize=11)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=9, frameon=False)
    for i, t in enumerate(tiers):
        total_t = sum(grid[t].values())
        axes[1].text(i, 102, f"n={total_t}", ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="failure_case_log.py 输出的多个 CSV")
    parser.add_argument("--title", type=str, default="Failure taxonomy")
    parser.add_argument("--out",   type=str, required=True)
    args = parser.parse_args()

    rows = load_csvs(args.inputs)
    print(f"loaded {len(rows)} non-trivial candidates (excluded 'no_gt')")
    plot(rows, args.title, args.out)


if __name__ == "__main__":
    main()
