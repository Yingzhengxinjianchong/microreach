import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_GLOB = "data/*.npz"
OUT_DIR = Path("data/polar_figs_auto")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SELECT = 5
N_PSI = 8
N_G = 3


def safe_norm_score(x, target, scale):
    """
    分数越接近 1 越好；距离 target 越远分数越低
    """
    return max(0.0, 1.0 - abs(x - target) / scale)


def compute_candidate_visual_score(r24):
    """
    对单个 candidate 的 24 维 R_geom 打一个“适合画图”的分数
    r24 shape: (24,)
    """
    r = r24.reshape(N_G, N_PSI)  # (3, 8), g-major

    mean_val = float(np.mean(r))
    nonzero = float(np.mean(r > 0))

    # 方向差异：8 个方向上有起伏更好看
    psi_profile = r.mean(axis=0)   # (8,)
    psi_std = float(np.std(psi_profile))

    # ring 差异：3 个 g 层之间有差异更好看
    g_profile = r.mean(axis=1)     # (3,)
    g_std = float(np.std(g_profile))

    # 全局对比度
    overall_std = float(np.std(r))

    # 目标偏好：
    # mean 不要太低也不要太高
    # nonzero 不要太低也不要太高
    # 标准差越明显越好
    mean_score = safe_norm_score(mean_val, target=0.18, scale=0.18)
    nz_score   = safe_norm_score(nonzero, target=0.35, scale=0.35)

    # 这些标准差不是越大越无限好，因此做一个饱和型得分
    psi_score = min(1.0, psi_std / 0.12)
    g_score   = min(1.0, g_std / 0.10)
    var_score = min(1.0, overall_std / 0.18)

    # 综合分数
    score = (
        0.28 * mean_score +
        0.24 * nz_score +
        0.20 * psi_score +
        0.12 * g_score +
        0.16 * var_score
    )

    stats = {
        "mean": mean_val,
        "nonzero": nonzero,
        "psi_std": psi_std,
        "g_std": g_std,
        "overall_std": overall_std,
        "score": score,
    }
    return score, stats


def choose_best_candidate(R_geom):
    """
    从一个实例内所有 candidate 里挑最适合画图的那个
    R_geom shape: (M, 24)
    """
    best_idx = None
    best_score = -1.0
    best_stats = None

    for i in range(R_geom.shape[0]):
        score, stats = compute_candidate_visual_score(R_geom[i])
        if score > best_score:
            best_score = score
            best_idx = i
            best_stats = stats

    return best_idx, best_score, best_stats


def score_instance(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    R_geom = d["R_geom"]  # (M, 24)

    best_idx, best_score, best_stats = choose_best_candidate(R_geom)

    # 实例整体统计，用于辅助排序
    inst_mean = float(np.mean(R_geom))
    inst_nonzero = float(np.mean(R_geom > 0))

    # 实例总分主要看“最佳 candidate 是否好看”
    # 但也轻微考虑整个实例不要太极端
    inst_mean_score = safe_norm_score(inst_mean, target=0.18, scale=0.18)
    inst_nz_score   = safe_norm_score(inst_nonzero, target=0.35, scale=0.35)

    total_score = 0.75 * best_score + 0.15 * inst_mean_score + 0.10 * inst_nz_score

    info = {
        "path": npz_path,
        "instance_id": Path(npz_path).stem,
        "best_candidate_idx": best_idx,
        "best_candidate_score": best_score,
        "best_candidate_stats": best_stats,
        "instance_mean": inst_mean,
        "instance_nonzero": inst_nonzero,
        "total_score": total_score,
    }
    return info


def plot_instance(info, rank):
    d = np.load(info["path"], allow_pickle=True)

    instance_id = info["instance_id"]
    candidate_idx = info["best_candidate_idx"]
    candidate_p = d["candidate_p"][candidate_idx]
    R_geom = d["R_geom"][candidate_idx].reshape(N_G, N_PSI)

    theta_edges = np.linspace(0, 2*np.pi, N_PSI + 1)
    radial_edges = np.arange(N_G + 1)  # 0,1,2,3

    fig = plt.figure(figsize=(6.2, 5.3))
    ax = fig.add_subplot(111, projection="polar")

    # pcolormesh 使用边界数组时，C 的 shape 应为
    # (len(radial_edges)-1, len(theta_edges)-1) = (3, 8)
    mesh = ax.pcolormesh(theta_edges, radial_edges, R_geom, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.12)
    cbar.set_label("R_geom")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(theta_edges[:-1]), labels=[f"ψ{i}" for i in range(N_PSI)])
    ax.set_rgrids([0.5, 1.5, 2.5], labels=["g0", "g1", "g2"], angle=22.5)
    ax.set_rlim(0, N_G)

    st = info["best_candidate_stats"]
    ax.set_title(
        f"Rank {rank} | Instance {instance_id} | candidate #{candidate_idx}\n"
        f"mean={st['mean']:.3f}, nonzero={st['nonzero']:.1%}, "
        f"ψ-std={st['psi_std']:.3f}, g-std={st['g_std']:.3f}\n"
        f"p=[{candidate_p[0]:.3f}, {candidate_p[1]:.3f}, {candidate_p[2]:.3f}]"
    )

    out_path = OUT_DIR / f"{rank:02d}_{instance_id}_cand{candidate_idx}_polar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise RuntimeError("No NPZ files found under data/*.npz")

    infos = [score_instance(f) for f in files]
    infos.sort(key=lambda x: x["total_score"], reverse=True)

    selected = infos[:N_SELECT]

    summary_lines = []
    summary_lines.append("Auto-selected instances for polar visualization")
    summary_lines.append("=" * 60)

    print("[info] selected instances:")
    for rank, info in enumerate(selected, start=1):
        line = (
            f"{rank:02d}. instance={info['instance_id']}  "
            f"inst_mean={info['instance_mean']:.4f}  "
            f"inst_nonzero={info['instance_nonzero']:.1%}  "
            f"best_cand={info['best_candidate_idx']}  "
            f"cand_score={info['best_candidate_score']:.4f}  "
            f"total_score={info['total_score']:.4f}"
        )
        print(line)
        summary_lines.append(line)

    summary_lines.append("")
    summary_lines.append("Saved figures:")
    for rank, info in enumerate(selected, start=1):
        out_path = plot_instance(info, rank)
        summary_lines.append(str(out_path))
        print(f"[saved] {out_path}")

    summary_path = OUT_DIR / "selection_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[saved] {summary_path}")
    print(f"[done] {len(selected)} figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
