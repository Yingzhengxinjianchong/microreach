"""
label_gen/r_contact.py

职责：对每个 (p, ψ, g) query 计算力闭合评分 R_contact ∈ [0, 1]。
      P1 加分项，完全独立于 SAPIEN，本地即可运行和测试。

评分公式：
    R_contact = sigmoid(0.4 * fc_score + 0.3 * contact_area + 0.3 * axis_alignment)

三个子分量：
    fc_score        : 摩擦锥力闭合评分，基于 wrench 空间凸包体积近似
    contact_area    : 接触面积分数（归一化到 [0,1]），power > pinch > poke
    axis_alignment  : 抓取主轴与 part 法线/轴线的对齐程度

抓取类型 g_idx 映射（与 sample_queries.py 一致）：
    0 = pinch   (精捏)
    1 = power   (握持)
    2 = poke    (戳入)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MU_FRICTION = 0.6            # 默认摩擦系数
N_CONE_EDGES = 8             # 摩擦锥离散化边数
SIGMOID_SCALE = 5.0          # sigmoid 斜率，让评分曲线更陡
EPS = 1e-8

# 每种抓取类型的典型接触面积比例（相对于最大可能接触面积）
_AREA_BASE = {
    0: 0.15,   # pinch  — 两指尖，面积小
    1: 0.55,   # power  — 手掌+四指，面积大
    2: 0.05,   # poke   — 单指尖，面积最小
}

# 接触点数量（每种抓取类型用于采样的虚拟接触点数）
_N_CONTACTS = {
    0: 2,   # pinch  — 两个接触点
    1: 5,   # power  — 五个接触点
    2: 1,   # poke   — 单接触点
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _sigmoid(x: float, scale: float = SIGMOID_SCALE) -> float:
    """数值稳定的 sigmoid。"""
    x_scaled = scale * x
    if x_scaled >= 0:
        return 1.0 / (1.0 + np.exp(-x_scaled))
    else:
        e = np.exp(x_scaled)
        return e / (1.0 + e)


def _normalize(v: np.ndarray) -> np.ndarray:
    """归一化向量，零向量返回零向量。"""
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def _sample_contact_points(
    p: np.ndarray,
    psi: np.ndarray,
    g_idx: int,
    part_normal: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    根据抓取类型在 p 周围采样虚拟接触点。

    Parameters
    ----------
    p           : 候选交互点，shape (3,)
    psi         : 抓取方向单位向量，shape (3,)
    g_idx       : 抓取类型索引 (0=pinch, 1=power, 2=poke)
    part_normal : part 表面法线，shape (3,)，用于扰动接触点
    rng         : numpy 随机数生成器

    Returns
    -------
    contacts    : shape (N_contacts, 3)，虚拟接触点坐标
    """
    n_c = _N_CONTACTS[g_idx]
    contacts = np.empty((n_c, 3), dtype=np.float64)

    # 构造局部坐标系（psi 为主轴）
    psi_n = _normalize(psi)
    # 找一个与 psi_n 不平行的向量来构造切平面
    ref = np.array([1.0, 0.0, 0.0]) if abs(psi_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tang1 = _normalize(np.cross(psi_n, ref))
    tang2 = _normalize(np.cross(psi_n, tang1))

    if g_idx == 0:  # pinch：两指在 tang1 方向对称
        offset = 0.01  # 1 cm 手指间距
        contacts[0] = p + offset * tang1
        contacts[1] = p - offset * tang1

    elif g_idx == 1:  # power：5 个接触点，圆形分布
        radii = np.array([0.0, 0.015, 0.015, 0.015, 0.015])
        angles = np.array([0.0, 0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        for i in range(n_c):
            contacts[i] = (
                p
                + radii[i] * (np.cos(angles[i]) * tang1 + np.sin(angles[i]) * tang2)
            )

    else:  # poke：单接触点，就是 p 本身加微小法线偏移
        contacts[0] = p + 0.002 * _normalize(part_normal)

    return contacts


def _friction_cone_wrenches(
    contact: np.ndarray,
    normal: np.ndarray,
    mu: float = MU_FRICTION,
    n_edges: int = N_CONE_EDGES,
) -> np.ndarray:
    """
    为单个接触点生成摩擦锥的离散化 wrench 向量。

    每条锥边产生一个 6D wrench [f; r×f]，其中 r = contact - origin。

    Returns
    -------
    wrenches : shape (n_edges, 6)
    """
    normal_n = _normalize(normal)

    # 构造切平面基向量
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _normalize(np.cross(normal_n, ref))
    t2 = _normalize(np.cross(normal_n, t1))

    angles = np.linspace(0, 2 * np.pi, n_edges, endpoint=False)
    wrenches = np.empty((n_edges, 6), dtype=np.float64)

    for i, theta in enumerate(angles):
        # 锥边方向：法线 + mu*(cos*t1 + sin*t2)，归一化
        f = _normalize(normal_n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2))
        torque = np.cross(contact, f)
        wrenches[i] = np.concatenate([f, torque])

    return wrenches


def _fc_score_from_contacts(
    contacts: np.ndarray,
    normals: np.ndarray,
    mu: float = MU_FRICTION,
) -> float:
    """
    力闭合评分：用 GWS（Grasp Wrench Space）的 L1-ball 近似体积。

    精确凸包需要 scipy.spatial.ConvexHull，但在高维（6D）情况下可能
    不稳定。这里用 epsilon-metric 近似：
        fc_score = clip(min_sv / (max_sv + eps), 0, 1)
    其中 sv 是所有 wrench 向量组成矩阵的奇异值。
    值越接近 1 表示 wrench 空间越"各向同性"（力闭合越好）。

    Parameters
    ----------
    contacts : (N, 3)
    normals  : (N, 3)，每个接触点对应的表面法线
    """
    all_wrenches = []
    for c, n in zip(contacts, normals):
        all_wrenches.append(_friction_cone_wrenches(c, n, mu=mu))
    W = np.vstack(all_wrenches)  # shape (N * n_edges, 6)

    if W.shape[0] < 6:
        return 0.0

    try:
        sv = np.linalg.svd(W, compute_uv=False)
        score = float(np.clip(sv[-1] / (sv[0] + EPS), 0.0, 1.0))
    except np.linalg.LinAlgError:
        score = 0.0

    return score


# ---------------------------------------------------------------------------
# 核心公共接口
# ---------------------------------------------------------------------------

def compute_r_contact(
    p: np.ndarray,
    psi: np.ndarray,
    g_idx: int,
    part_normal: np.ndarray,
    part_axis: Optional[np.ndarray] = None,
    mu: float = MU_FRICTION,
    seed: Optional[int] = None,
) -> float:
    """
    计算单个 (p, ψ, g) query 的 R_contact ∈ [0, 1]。

    Parameters
    ----------
    p           : 候选交互点，shape (3,)
    psi         : 抓取方向单位向量，shape (3,)
    g_idx       : 抓取类型索引 (0=pinch, 1=power, 2=poke)
    part_normal : part 表面法线（朝外），shape (3,)
    part_axis   : part 的主轴方向（如铰链轴），None 时用 part_normal 代替
    mu          : 摩擦系数，默认 0.6
    seed        : 随机种子（用于接触点采样的微小扰动，保证可复现）

    Returns
    -------
    R_contact   : float ∈ [0, 1]
    """
    p = np.asarray(p, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    part_normal = np.asarray(part_normal, dtype=np.float64)

    if part_axis is None:
        part_axis = part_normal.copy()
    else:
        part_axis = np.asarray(part_axis, dtype=np.float64)

    rng = np.random.default_rng(seed)

    # ---- 1. 采样接触点 ----
    contacts = _sample_contact_points(p, psi, g_idx, part_normal, rng)
    n_c = contacts.shape[0]

    # 每个接触点的法线：近似为 part_normal（平面假设）
    normals = np.tile(_normalize(part_normal), (n_c, 1))

    # ---- 2. fc_score ----
    fc_score = _fc_score_from_contacts(contacts, normals, mu=mu)

    # ---- 3. contact_area ----
    # 基础面积 + poke 类型惩罚（单点接触天然面积小）
    base_area = _AREA_BASE[g_idx]
    # power 抓取在方向与法线对齐时面积更大
    psi_n = _normalize(psi)
    normal_n = _normalize(part_normal)
    align_bonus = abs(float(np.dot(psi_n, -normal_n)))  # 抓取方向朝向表面时最好
    contact_area = float(np.clip(base_area + 0.2 * align_bonus, 0.0, 1.0))

    # ---- 4. axis_alignment ----
    # 对 pinch/power：抓取方向与 part 主轴的对齐度
    # 对 poke：戳入方向与法线对齐度（垂直表面时最好）
    axis_n = _normalize(part_axis)
    if g_idx == 2:  # poke：垂直表面最优
        axis_alignment = float(abs(np.dot(psi_n, -normal_n)))
    else:           # pinch/power：与 part 主轴垂直时最优（横向抓取）
        perp = float(np.sqrt(max(0.0, 1.0 - np.dot(psi_n, axis_n) ** 2)))
        axis_alignment = perp

    # ---- 5. 加权求和 + sigmoid ----
    raw = 0.4 * fc_score + 0.3 * contact_area + 0.3 * axis_alignment
    r_contact = float(_sigmoid(raw - 0.5))   # 中心平移：raw=0.5 → R_contact≈0.5

    return float(np.clip(r_contact, 0.0, 1.0))


def batch_compute_r_contact(
    candidate_p: np.ndarray,
    queries: np.ndarray,
    part_normals: np.ndarray,
    part_axes: Optional[np.ndarray] = None,
    mu: float = MU_FRICTION,
    seed: int = 42,
) -> np.ndarray:
    """
    批量计算 R_contact，与 batch_generate.py 接口对齐。

    Parameters
    ----------
    candidate_p  : (M, 3)，M 个候选交互点
    queries      : (M, Q, 4)，每点 Q 个 query，最后一维 (ψ_x, ψ_y, ψ_z, g_idx)
    part_normals : (M, 3)，每点对应的 part 表面法线
    part_axes    : (M, 3) or None，每点对应的 part 主轴
    mu           : 摩擦系数
    seed         : 随机种子基数

    Returns
    -------
    R_contact    : (M, Q)，float32
    """
    M, Q, _ = queries.shape
    R_contact = np.zeros((M, Q), dtype=np.float32)

    for i in range(M):
        p_i = candidate_p[i]
        n_i = part_normals[i]
        ax_i = part_axes[i] if part_axes is not None else None

        for j in range(Q):
            psi = queries[i, j, :3]
            g_idx = int(queries[i, j, 3])
            R_contact[i, j] = compute_r_contact(
                p=p_i,
                psi=psi,
                g_idx=g_idx,
                part_normal=n_i,
                part_axis=ax_i,
                mu=mu,
                seed=seed + i * Q + j,
            )

    return R_contact


# ---------------------------------------------------------------------------
# 内置测试套件
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            print(f"  ✓  {name}")
            passed += 1
        else:
            print(f"  ✗  {name}  {detail}")
            failed += 1

    print("=" * 60)
    print("r_contact.py  内置测试套件")
    print("=" * 60)

    # ---- T1: 输出范围 ----
    print("\n[T1] 输出范围 ∈ [0, 1]")
    rng = np.random.default_rng(0)
    for g in range(3):
        p = np.array([0.0, 0.0, 0.0])
        psi = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 0.0, 1.0])
        r = compute_r_contact(p, psi, g, normal, seed=g)
        check(f"g_idx={g} → R_contact={r:.4f}", 0.0 <= r <= 1.0)

    # ---- T2: 随机方向输出范围 ----
    print("\n[T2] 随机方向 × 所有抓取类型 × 100 次，均在 [0,1]")
    all_in_range = True
    for _ in range(100):
        p = rng.standard_normal(3)
        psi = rng.standard_normal(3)
        normal = rng.standard_normal(3)
        g = int(rng.integers(0, 3))
        r = compute_r_contact(p, psi, g, normal, seed=int(rng.integers(0, 9999)))
        if not (0.0 <= r <= 1.0):
            all_in_range = False
            break
    check("所有 100 次随机调用均在 [0,1]", all_in_range)

    # ---- T3: poke vs pinch/power 的接触面积排序 ----
    print("\n[T3] 接触面积：power > pinch > poke（基础值）")
    check(
        "area_base: power > pinch",
        _AREA_BASE[1] > _AREA_BASE[0],
        f"{_AREA_BASE[1]} vs {_AREA_BASE[0]}",
    )
    check(
        "area_base: pinch > poke",
        _AREA_BASE[0] > _AREA_BASE[2],
        f"{_AREA_BASE[0]} vs {_AREA_BASE[2]}",
    )

    # ---- T4: fc_score 在接触点数为 1 时退化 ----
    print("\n[T4] 单接触点（poke）不会崩溃")
    p = np.array([0.0, 0.0, 0.0])
    psi = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 0.0, 1.0])
    r_poke = compute_r_contact(p, psi, g_idx=2, part_normal=normal, seed=0)
    check(f"poke R_contact={r_poke:.4f} ∈ [0,1]", 0.0 <= r_poke <= 1.0)

    # ---- T5: 批量接口形状 ----
    print("\n[T5] batch_compute_r_contact 输出形状")
    M, Q = 5, 24
    cand_p = rng.standard_normal((M, 3))
    queries = np.concatenate(
        [rng.standard_normal((M, Q, 3)), rng.integers(0, 3, (M, Q, 1)).astype(float)],
        axis=-1,
    )
    part_normals = _normalize(rng.standard_normal((M, 3)))
    if part_normals.ndim == 1:  # edge case
        part_normals = np.tile(part_normals, (M, 1))
    else:
        part_normals = np.array([_normalize(part_normals[i]) for i in range(M)])
    R_batch = batch_compute_r_contact(cand_p, queries, part_normals)
    check(f"输出 shape={R_batch.shape} == ({M},{Q})", R_batch.shape == (M, Q))
    check("批量输出均在 [0,1]", bool(np.all((R_batch >= 0.0) & (R_batch <= 1.0))))

    # ---- T6: 一致性——相同输入相同输出 ----
    print("\n[T6] 确定性：相同 seed 给相同结果")
    p0 = np.array([0.1, 0.2, 0.3])
    psi0 = np.array([0.0, 1.0, 0.0])
    n0 = np.array([0.0, 0.0, 1.0])
    r_a = compute_r_contact(p0, psi0, 0, n0, seed=99)
    r_b = compute_r_contact(p0, psi0, 0, n0, seed=99)
    check(f"两次相同调用 {r_a:.6f} == {r_b:.6f}", abs(r_a - r_b) < 1e-9)

    # ---- T7: sigmoid 工具函数 ----
    print("\n[T7] sigmoid 数值稳定性")
    check("sigmoid(0)=0.5", abs(_sigmoid(0.0) - 0.5) < 1e-6)
    check("sigmoid(100) 不溢出", 0.0 < _sigmoid(100.0) <= 1.0)
    check("sigmoid(-100) 不下溢", 0.0 <= _sigmoid(-100.0) < 1.0)

    # ---- T8: 接触点采样形状 ----
    print("\n[T8] 接触点采样形状")
    rng2 = np.random.default_rng(0)
    for g in range(3):
        pts = _sample_contact_points(
            p=np.zeros(3),
            psi=np.array([0.0, 0.0, 1.0]),
            g_idx=g,
            part_normal=np.array([0.0, 0.0, 1.0]),
            rng=rng2,
        )
        expected_n = _N_CONTACTS[g]
        check(
            f"g_idx={g} 接触点数={pts.shape[0]} == {expected_n}",
            pts.shape == (expected_n, 3),
        )

    # ---- 汇总 ----
    print()
    print("=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"全部 {total} 项测试通过 ✓")
    else:
        print(f"{passed}/{total} 项通过，{failed} 项失败 ✗")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)