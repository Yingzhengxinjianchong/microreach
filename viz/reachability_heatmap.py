"""
viz/reachability_heatmap.py

R_geom 在 8ψ × 3g 上的极坐标玫瑰图（文档 §3.4 + §8）。

用法：
    # 单实例的 GT 与 M1 预测对比
    python -m viz.reachability_heatmap \\
        --instance 153 \\
        --ckpt ckpts/m1_seed42/best.pt \\
        --config configs/m1.yaml \\
        --out viz/figs

图布局（每个候选点一个 panel）：
    - 角度方向：8 个 ψ 方向（用 fibonacci 球面投影到方位角 φ）
    - 径向 3 圈：3 个 g 类型（内 pinch / 中 power / 外 poke）
    - 颜色：R_geom 分数（越红越高）
    - 标题：part_id + part_tier
    - 左 GT / 右 PRED 双 panel

阶段二中期：实例 153（mean R_geom=0.48）和 1741（mean=0.40）方向选择性最强，画这两个。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # 服务器无显示环境
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from microreach_net.train import MicroReachNet, load_config


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_instance(instance_id: str) -> dict:
    """从 data/<id>.npz 读所有字段。"""
    path = _REPO_ROOT / "data" / f"{instance_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"实例 {instance_id} 不在 data/")
    return dict(np.load(str(path), allow_pickle=True))


@torch.no_grad()
def predict_one(
    ckpt_path: str,
    config_path: str,
    instance: dict,
    device: torch.device,
) -> np.ndarray:
    """对单个实例跑预测，返回 pred R_geom (M, K)."""
    cfg = load_config(config_path)
    model = MicroReachNet(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    pc = torch.from_numpy(instance["point_cloud"]).float().unsqueeze(0).to(device)   # (1, N, 3)
    cp = torch.from_numpy(instance["candidate_p"]).float().unsqueeze(0).to(device)   # (1, M, 3)
    q  = torch.from_numpy(instance["queries"]).float().unsqueeze(0).to(device)       # (1, M, K, 4)

    # 强制点云对齐到 30000（与训练一致）
    target_N = cfg["data"]["num_points"]
    if pc.shape[1] != target_N:
        idx = torch.randperm(pc.shape[1])[:target_N] if pc.shape[1] >= target_N \
              else torch.randint(0, pc.shape[1], (target_N,))
        pc = pc[:, idx]

    if model.use_pose_decoder:
        logits = model(pc, cp, q)           # 单头: (1, M, K) | 多头: dict
        # 阶段三：多头模型（M2 / M_full）forward 返回 dict，需要取 geom 头
        if isinstance(logits, dict):
            logits = logits["geom"]
        pred = torch.sigmoid(logits)
    else:
        logits = model(pc, cp)              # (1, M)
        K = q.shape[2]
        pred = torch.sigmoid(logits).unsqueeze(-1).expand(-1, -1, K)
    return pred[0].cpu().numpy()            # (M, K)


# ──────────────────────────────────────────────
# 极坐标绘图
# ──────────────────────────────────────────────

def plot_polar_for_point(
    ax: plt.Axes,
    queries: np.ndarray,        # (K, 4)，K=24=8*3
    scores: np.ndarray,         # (K,) ∈ [0, 1]
    title: str = "",
    n_psi: int = 8,
    n_g: int = 3,
    cmap_name: str = "Reds",
    vmax: Optional[float] = None,    # 颜色映射的最大值；None 表示用 scores.max()
) -> None:
    """
    在 ax 上画一个极坐标玫瑰图：
        - 角度：从 queries[:, :3] 的方向 ψ 投影到方位角 φ = atan2(ψ_y, ψ_x)
        - 径向：g_idx ∈ {0,1,2} 对应内 / 中 / 外圈
        - 颜色：scores 经过 [0, vmax] 归一化后映射

    重要：vmax 默认 = scores.max()（相对归一化），让 PRED 也能看出方向选择性。
    R_geom 数据 mean 只有 0.155，绝对映射 [0, 1] 会让 PRED 颜色非常浅，
    看不出"哪个方向相对最高"。
    """
    psi = queries[:, :3]
    g_idx = queries[:, 3].astype(int)
    phi = np.arctan2(psi[:, 1], psi[:, 0])

    cmap = plt.get_cmap(cmap_name)
    if vmax is None:
        vmax = max(float(scores.max()), 1e-6)

    # 标题里加上 vmax 提示，让看图人知道这是"相对"颜色
    title_with_vmax = f"{title}  (vmax={vmax:.3f})"

    for g in range(n_g):
        mask = g_idx == g
        phi_g = phi[mask]
        score_g = scores[mask]

        r_inner = 0.2 + 0.25 * g
        r_outer = 0.2 + 0.25 * (g + 1)

        sector_width = 2 * np.pi / n_psi

        order = np.argsort(phi_g)
        phi_sorted = phi_g[order]
        score_sorted = score_g[order]

        for p, s in zip(phi_sorted, score_sorted):
            # 归一化到 [0.15, 1.0]，避免低分扇区完全空白看不见
            normalized = 0.15 + 0.85 * (s / vmax)
            normalized = float(np.clip(normalized, 0.0, 1.0))
            ax.bar(
                p, r_outer - r_inner, width=sector_width,
                bottom=r_inner, color=cmap(normalized),
                edgecolor="white", linewidth=0.5,
            )

    g_labels = ["pinch", "power", "poke"]
    for g, label in enumerate(g_labels):
        ax.text(0, 0.2 + 0.25 * (g + 0.5), label,
                ha="center", va="center", fontsize=7, color="black")

    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, n_psi, endpoint=False))
    ax.set_xticklabels([f"ψ{i}" for i in range(n_psi)], fontsize=7)
    ax.set_title(title_with_vmax, fontsize=9, pad=10)


def plot_instance(
    instance_id: str,
    instance: dict,
    pred: np.ndarray,        # (M, K)
    out_dir: Path,
    out_name: Optional[str] = None,    # 自定义文件名（不含扩展名）；None 则用 polar_<id>
    variant_label: Optional[str] = None,  # 加到 suptitle 上区分模型（如 "M1 BCE", "M1+composite"）
) -> None:
    """画一个实例：每个候选点一对 (GT, PRED) panel。"""
    gt = instance["R_geom"]                          # (M, K)
    queries = instance["queries"]                    # (M, K, 4)
    tiers = [str(t) for t in instance.get("part_tiers", ["unknown"] * gt.shape[0])]
    part_ids = [str(p) for p in instance.get("part_ids", [f"p{i}" for i in range(gt.shape[0])])]

    M, K = gt.shape
    fig, axes = plt.subplots(
        M, 2, figsize=(8, 3.2 * M),
        subplot_kw={"projection": "polar"},
    )
    if M == 1:
        axes = axes[None, :]

    for i in range(M):
        plot_polar_for_point(
            axes[i, 0], queries[i], gt[i],
            title=f"GT  | part={part_ids[i]} tier={tiers[i]}",
        )
        pred_title = f"PRED | part={part_ids[i]} tier={tiers[i]}"
        if variant_label:
            pred_title = f"PRED ({variant_label}) | part={part_ids[i]} tier={tiers[i]}"
        plot_polar_for_point(
            axes[i, 1], queries[i], pred[i],
            title=pred_title,
        )

    suptitle = f"Instance {instance_id} | M={M} candidate points × {K} queries"
    if variant_label:
        suptitle += f"  [{variant_label}]"
    fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (out_name or f"polar_{instance_id}") + ".png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True,
                        help="实例 ID，如 153 / 1380")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="viz/figs")
    parser.add_argument("--out_name", type=str, default=None,
                        help="自定义输出文件名（不含 .png 后缀）。None 用 polar_<instance>")
    parser.add_argument("--variant_label", type=str, default=None,
                        help="模型标签，会写在 PRED panel 标题和 suptitle 上（如 'M1 BCE' / 'M1+composite'）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance = load_instance(args.instance)
    pred = predict_one(args.ckpt, args.config, instance, device)
    plot_instance(args.instance, instance, pred, _REPO_ROOT / args.out,
                  out_name=args.out_name, variant_label=args.variant_label)


if __name__ == "__main__":
    main()
