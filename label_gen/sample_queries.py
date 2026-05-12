"""
label_gen/sample_queries.py

阶段二中期：Fibonacci 球面采样 8 个方向 ψ × 3 种抓取类型 g = 24 query / 候选点
阶段三扩展：32 个方向 × 3 = 96 query / 候选点（改 N_PSI=32 即可，其余代码不变）

输出 shape：
    queries: (num_candidate_p, N_PSI * N_G, 4)
    最后一维含义: (ψ_x, ψ_y, ψ_z, g_idx)  g_idx ∈ {0=pinch, 1=power, 2=poke}

参考：
    R(p, ψ, g) ∈ [0,1]  —— MicroReach 5D Pose-Conditioned Reachability Field
"""

import numpy as np
from typing import Tuple

# ──────────────────────────────────────────────
# 全局常量（改这里即可在阶段三切换到 96 query 模式）
# ──────────────────────────────────────────────
N_PSI: int = 8          # 阶段二中期；阶段三改为 32
N_G:   int = 3          # 固定：pinch / power / poke
G_NAMES = ["pinch", "power", "poke"]


# ──────────────────────────────────────────────
# 核心工具函数
# ──────────────────────────────────────────────

def fibonacci_sphere(n: int) -> np.ndarray:
    """
    Fibonacci 球面采样：在单位球面 S² 上生成 n 个近似均匀分布的方向向量。

    算法来源：
        Álvaro González (2010). "Measurement of Areas on a Sphere Using
        Fibonacci and Latitude-Longitude Lattices." Mathematical Geosciences.

    原理：把球面用黄金角 (2π/φ²) 螺旋展开，避免聚集极点的纬度均匀采样缺陷。

    Args:
        n: 采样点数

    Returns:
        directions: ndarray of shape (n, 3), 每行是单位方向向量
    """
    if n < 1:
        raise ValueError(f"n 必须 ≥ 1，收到 {n}")

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0  # φ ≈ 1.6180
    i = np.arange(n, dtype=float)

    # 仰角 θ ∈ [-π/2, π/2]（arcsin 保证面积均匀）
    theta = np.arcsin(1.0 - 2.0 * i / (n - 1)) if n > 1 else np.array([0.0])

    # 方位角 φ ∈ [0, 2π)，黄金角螺旋
    phi = 2.0 * np.pi * i / golden_ratio
    phi = phi % (2.0 * np.pi)  # 保证 ∈ [0, 2π)

    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    directions = np.stack([x, y, z], axis=-1)

    # 数值归一化（防浮点误差积累）
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / np.clip(norms, 1e-9, None)

    return directions.astype(np.float32)


def sample_queries_for_point(
    psi_directions: np.ndarray
) -> np.ndarray:
    """
    给单个候选点 p 生成完整的 (ψ, g) query 网格。

    Args:
        psi_directions: shape (N_PSI, 3)，单位方向向量

    Returns:
        queries: shape (N_PSI * N_G, 4)
                 每行 = [ψ_x, ψ_y, ψ_z, g_idx]
                 g_idx: 0=pinch, 1=power, 2=poke
    """
    n_psi = len(psi_directions)
    n_queries = n_psi * N_G
    queries = np.zeros((n_queries, 4), dtype=np.float32)

    idx = 0
    for g_idx in range(N_G):
        for psi in psi_directions:
            queries[idx, :3] = psi
            queries[idx, 3]  = float(g_idx)
            idx += 1

    return queries


def sample_queries_for_instance(
    candidate_p: np.ndarray,
    n_psi: int = N_PSI,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    给一个实例的所有候选点批量生成 query 网格。

    所有候选点**共享同一套 ψ 方向**（Fibonacci 球面采样只做一次）。
    这是刻意设计的：确保跨点方向对齐，消融实验中 ψ 轴一致。

    Args:
        candidate_p: shape (M, 3)，候选交互点的 3D 坐标，M 通常为 5
        n_psi:       ψ 采样数量，默认 N_PSI（阶段二=8，阶段三=32）

    Returns:
        psi_directions: shape (n_psi, 3)，本实例使用的 ψ 方向集合
        queries:        shape (M, n_psi * N_G, 4)，每行 [ψ_x, ψ_y, ψ_z, g_idx]
    """
    if candidate_p.ndim != 2 or candidate_p.shape[1] != 3:
        raise ValueError(
            f"candidate_p 应为 (M, 3)，收到 {candidate_p.shape}"
        )

    M = len(candidate_p)
    psi_directions = fibonacci_sphere(n_psi)          # (n_psi, 3)

    queries = np.stack(
        [sample_queries_for_point(psi_directions) for _ in range(M)],
        axis=0
    )  # (M, n_psi * N_G, 4)

    return psi_directions, queries


def decode_query(query_row: np.ndarray) -> dict:
    """
    将单行 query [ψ_x, ψ_y, ψ_z, g_idx] 解码为可读字典。

    Args:
        query_row: shape (4,)

    Returns:
        dict with keys: psi (np.ndarray shape (3,)), g_idx (int), g_name (str)
    """
    psi = query_row[:3]
    g_idx = int(round(query_row[3]))
    return {
        "psi":    psi,
        "g_idx":  g_idx,
        "g_name": G_NAMES[g_idx],
    }


def save_queries(
    save_path: str,
    instance_id: str,
    point_cloud: np.ndarray,
    candidate_p: np.ndarray,
    psi_directions: np.ndarray,
    queries: np.ndarray,
) -> None:
    """
    将单个实例的 query 结果以 .npz 格式保存。

    约定（与 batch_generate.py / r_geom.py 共用同一文件）：
      - point_cloud:   (N, 3)
      - candidate_p:   (M, 3)
      - psi_directions:(n_psi, 3)
      - queries:       (M, n_psi * N_G, 4)
      - R_geom:        (M, n_psi * N_G)   ← 由 r_geom.py 填充，此处缺省 NaN
      - R_contact:     (M, n_psi * N_G)   ← 由 r_contact.py 填充，此处缺省 NaN
      - R_exec:        (M, n_psi * N_G)   ← 由 r_exec.py 填充，此处缺省 NaN
    """
    M, K, _ = queries.shape
    np.savez(
        save_path,
        instance_id   = np.array([instance_id]),
        point_cloud   = point_cloud.astype(np.float32),
        candidate_p   = candidate_p.astype(np.float32),
        psi_directions= psi_directions.astype(np.float32),
        queries       = queries.astype(np.float32),
        R_geom        = np.full((M, K), np.nan, dtype=np.float32),
        R_contact     = np.full((M, K), np.nan, dtype=np.float32),
        R_exec        = np.full((M, K), np.nan, dtype=np.float32),
    )


def load_queries(load_path: str) -> dict:
    """
    读取由 save_queries 或 batch_generate 写入的 .npz 文件。

    Returns:
        dict，key 同 save_queries 中的约定字段
    """
    data = np.load(load_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ──────────────────────────────────────────────
# 可视化（可选，不依赖 SAPIEN，仅需 matplotlib）
# ──────────────────────────────────────────────

def visualize_psi_directions(
    psi_directions: np.ndarray,
    title: str = "Fibonacci Sphere Sampling",
    save_path: str = None,
) -> None:
    """
    3D 散点图可视化 ψ 方向的球面分布，用于肉眼验证均匀性。

    Args:
        psi_directions: shape (n_psi, 3)
        title:          图标题
        save_path:      若指定则保存为 PNG，否则 plt.show()
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = psi_directions[:, 0], psi_directions[:, 1], psi_directions[:, 2]
    ax.scatter(xs, ys, zs, c=np.arange(len(xs)), cmap="plasma", s=80, depthshade=True)

    # 画参考单位球（wireframe）
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color="lightgray", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"{title}\n(n={len(psi_directions)})")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# 单元测试（直接 python sample_queries.py 运行）
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 55)
    print("sample_queries.py 单元测试")
    print("=" * 55)

    # ── 测试 1：fibonacci_sphere 输出形状与归一化 ──
    print("\n[Test 1] fibonacci_sphere 形状与单位范数")
    for n in [1, 8, 32, 100]:
        dirs = fibonacci_sphere(n)
        assert dirs.shape == (n, 3), f"形状错误: {dirs.shape}"
        norms = np.linalg.norm(dirs, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5), \
            f"n={n}: 存在非单位向量, norms range [{norms.min():.6f}, {norms.max():.6f}]"
        print(f"  n={n:3d}: shape={dirs.shape}, "
              f"norm ∈ [{norms.min():.6f}, {norms.max():.6f}] ✓")

    # ── 测试 2：球面覆盖均匀性（最近邻距离标准差应较小）──
    print("\n[Test 2] 球面均匀性（最近邻角度标准差）")
    for n in [8, 32]:
        dirs = fibonacci_sphere(n)
        # 计算所有点对夹角
        cos_mat = np.clip(dirs @ dirs.T, -1.0, 1.0)
        np.fill_diagonal(cos_mat, -1.0)   # 排除自身
        # 每个点的最近邻夹角（越小越近）
        min_angles = np.degrees(np.arccos(cos_mat.max(axis=1)))
        std = min_angles.std()
        mean = min_angles.mean()
        print(f"  n={n:2d}: 最近邻角度 mean={mean:.2f}°, std={std:.2f}° "
              f"(std 越小均匀性越好) ✓")

    # ── 测试 3：sample_queries_for_instance 输出形状 ──
    print("\n[Test 3] sample_queries_for_instance 输出形状")
    M = 5
    candidate_p = np.random.randn(M, 3).astype(np.float32)
    for n_psi in [8, 32]:
        psi_dirs, queries = sample_queries_for_instance(candidate_p, n_psi=n_psi)
        expected_k = n_psi * N_G
        assert psi_dirs.shape == (n_psi, 3),      f"psi_dirs 形状错误: {psi_dirs.shape}"
        assert queries.shape == (M, expected_k, 4), f"queries 形状错误: {queries.shape}"
        print(f"  n_psi={n_psi:2d}: psi_dirs={psi_dirs.shape}, "
              f"queries={queries.shape} ✓")

    # ── 测试 4：g_idx 值域检查 ──
    print("\n[Test 4] g_idx 值域（必须全在 {0,1,2}）")
    psi_dirs, queries = sample_queries_for_instance(
        np.zeros((5, 3), dtype=np.float32), n_psi=8
    )
    g_vals = queries[:, :, 3].astype(int)
    unique_g = np.unique(g_vals)
    assert set(unique_g) == {0, 1, 2}, f"g_idx 异常: {unique_g}"
    # 三种 g 数量应相等
    for g_idx, g_name in enumerate(G_NAMES):
        count = (g_vals == g_idx).sum()
        expected = 5 * 8  # M × n_psi
        assert count == expected, f"{g_name} 数量应为 {expected}, 实为 {count}"
        print(f"  g={g_name}({g_idx}): count={count} ✓")

    # ── 测试 5：decode_query 解码正确 ──
    print("\n[Test 5] decode_query 解码")
    psi_dirs, queries = sample_queries_for_instance(
        np.eye(3, dtype=np.float32), n_psi=8
    )
    row = queries[0, 0]
    decoded = decode_query(row)
    assert decoded["g_idx"] in {0, 1, 2}
    assert decoded["g_name"] in G_NAMES
    assert np.allclose(np.linalg.norm(decoded["psi"]), 1.0, atol=1e-5)
    print(f"  首行 query: ψ={decoded['psi'].round(4)}, "
          f"g={decoded['g_name']}({decoded['g_idx']}) ✓")

    # ── 测试 6：save / load 往返一致性 ──
    print("\n[Test 6] save_queries / load_queries 往返一致性")
    import tempfile, os
    candidate_p = np.random.randn(5, 3).astype(np.float32)
    point_cloud  = np.random.randn(100, 3).astype(np.float32)
    psi_dirs, queries = sample_queries_for_instance(candidate_p, n_psi=8)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_instance")
        save_queries(path, "test_001", point_cloud, candidate_p, psi_dirs, queries)
        loaded = load_queries(path + ".npz")
    assert np.allclose(loaded["queries"],       queries,       atol=1e-6), "queries 不一致"
    assert np.allclose(loaded["candidate_p"],   candidate_p,   atol=1e-6), "candidate_p 不一致"
    assert np.allclose(loaded["psi_directions"],psi_dirs,      atol=1e-6), "psi_directions 不一致"
    assert np.all(np.isnan(loaded["R_geom"])),    "R_geom 初始化应全为 NaN"
    assert np.all(np.isnan(loaded["R_contact"])), "R_contact 初始化应全为 NaN"
    assert np.all(np.isnan(loaded["R_exec"])),    "R_exec 初始化应全为 NaN"
    print("  往返一致，R_geom/R_contact/R_exec 初始全为 NaN ✓")

    # ── 测试 7：n=1 边界情况 ──
    print("\n[Test 7] n=1 边界情况")
    dirs = fibonacci_sphere(1)
    assert dirs.shape == (1, 3)
    assert np.allclose(np.linalg.norm(dirs), 1.0, atol=1e-5)
    print(f"  n=1: dirs={dirs} ✓")

    # ── 测试 8：非法输入应抛出异常 ──
    print("\n[Test 8] 非法输入异常检测")
    try:
        fibonacci_sphere(0)
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"  fibonacci_sphere(0) → ValueError: {e} ✓")
    try:
        sample_queries_for_instance(np.zeros((5, 2)))  # 错误维度
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"  candidate_p shape (5,2) → ValueError: {e} ✓")

    print("\n" + "=" * 55)
    print("全部 8 项测试通过 ✓")
    print("=" * 55)

    # ── 可选：可视化 ──
    print("\n生成 8 方向球面分布可视化（保存为 psi_8.png）...")
    try:
        dirs8 = fibonacci_sphere(8)
        visualize_psi_directions(dirs8, title="Fibonacci Sphere (n=8)", save_path="psi_8.png")
        dirs32 = fibonacci_sphere(32)
        visualize_psi_directions(dirs32, title="Fibonacci Sphere (n=32)", save_path="psi_32.png")
        print("已保存 psi_8.png 和 psi_32.png")
    except Exception as e:
        print(f"可视化跳过（{e}）")


if __name__ == "__main__":
    _run_tests()