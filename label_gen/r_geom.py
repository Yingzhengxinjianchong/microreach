"""
label_gen/r_geom.py

步骤 2.4：计算 R_geom 标签，吸收 MoMaGen ICLR 2026 硬+软约束分级。

公式：
    R_geom(p, ψ, g) ∈ [0, 1]

    Hard 部分（任一失败 → R_geom = 0.0）：
        1. IK 求解：给定目标末端位姿，是否存在无奇异 IK 解
        2. 碰撞检测：IK 解对应的关节角下，机器人是否与场景/自身碰撞

    Soft 部分（连续值，体现"好不好"）：
        joint_margin  : 关节角到极限的归一化余量 ∈ [0, 1]
        yoshikawa     : Yoshikawa 操作度指标（归一化）∈ [0, 1]
        R_geom_soft   = 0.5 × joint_margin + 0.5 × yoshikawa

机器人：Franka Panda（7-DOF）
IK：SAPIEN 2.x Jacobian 伪逆迭代（不依赖外部 IK 库）

服务器运行示例（单实例调试）：
    python label_gen/r_geom.py \\
        --npz-path    data/45174.npz \\
        --data-dir    /root/autodl-fs/partnet_mobility \\
        --robot-urdf  assets/franka/panda.urdf \\
        --visualize   # 输出极坐标玫瑰图

测试（不需要 GPU，用 mock 数据）：
    python label_gen/r_geom.py --test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from label_gen.sample_queries import load_queries, G_NAMES

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

# IK 迭代参数
IK_MAX_ITER:     int   = 100
IK_POS_TOL:      float = 1e-3      # 位置误差阈值（1mm）
IK_ORI_TOL:      float = 1e-2      # 姿态误差阈值（rad）
IK_DAMP:         float = 1e-3      # 阻尼最小二乘正则化系数
IK_LR:           float = 0.5       # Jacobian 伪逆步长

# 末端逼近距离（p 是接触点，末端目标 = p - approach_dist × ψ）
APPROACH_DIST:   float = 0.10      # 10cm 逼近距离

# 抓取类型 → 末端偏移（沿 ψ 方向的额外偏置，单位 m）
GRASP_OFFSET: dict = {
    0: 0.00,   # pinch：直接对准接触点
    1: 0.02,   # power：包裹抓，稍微靠近 2cm
    2: -0.02,  # poke：戳，末端稍微超过接触点
}

# 关节角余量归一化使用的"安全区"占总范围比例
JOINT_MARGIN_SAFE_RATIO: float = 0.1

# Yoshikawa 指标的归一化参考值（Panda 的典型最大操作度）
# 实际值由批量统计确定，这里给一个经验估计值
YOSHIKAWA_REF: float = 0.05


# ──────────────────────────────────────────────
# 目标末端位姿计算
# ──────────────────────────────────────────────

def compute_target_pose(
    p:   np.ndarray,
    psi: np.ndarray,
    g:   int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定交互点 p、逼近方向 ψ、抓取类型 g，
    计算末端执行器的目标位置和姿态。

    约定：
        - 末端 Z 轴（gripper forward）与 ψ 对齐（沿 ψ 方向逼近）
        - 末端位置 = p - (APPROACH_DIST + grasp_offset[g]) × ψ

    Args:
        p:   (3,) 交互点世界坐标
        psi: (3,) 单位逼近方向向量
        g:   抓取类型索引（0=pinch, 1=power, 2=poke）

    Returns:
        (target_pos, target_rot_mat)
        target_pos:     (3,) float32
        target_rot_mat: (3, 3) float32，列：right, up, forward（=ψ）
    """
    psi = psi / (np.linalg.norm(psi) + 1e-9)
    offset = APPROACH_DIST + GRASP_OFFSET.get(g, 0.0)
    target_pos = (p - offset * psi).astype(np.float32)

    # 旋转矩阵：让 gripper 的 Z 轴（forward）对准 ψ
    forward = psi.astype(np.float32)

    # 选取不与 forward 共线的 up 向量
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(forward, world_up)) > 0.9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    right = right / max(right_norm, 1e-9)
    up = np.cross(forward, right)

    # 旋转矩阵列：[right, up, forward]
    target_rot = np.stack([right, up, forward], axis=-1)  # (3, 3)

    return target_pos, target_rot


# ──────────────────────────────────────────────
# SAPIEN 机器人接口（Franka Panda）
# ──────────────────────────────────────────────

class RobotController:
    """
    封装 SAPIEN 2.x 中 Franka Panda 的 FK / 碰撞检测 / Jacobian 操作。

    接口设计为"无状态输入/输出"，方便在批量 query 中多次调用。
    """

    def __init__(self, scene, robot_urdf: str):
        """
        Args:
            scene:      SAPIEN scene（已初始化）
            robot_urdf: panda.urdf 绝对路径
        """
        import sapien.core as sapien
        self.scene = scene

        # 加载机器人
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(robot_urdf)
        if self.robot is None:
            raise RuntimeError(f"机器人 URDF 加载失败: {robot_urdf}")

        # 关节信息（只考虑 active joints，排除 gripper 的 2 个手指）
        self.active_joints = self.robot.get_active_joints()
        # Franka Panda：前 7 个是臂关节，后 2 个是手指（finger joint）
        self.arm_joints    = self.active_joints[:7]
        self.n_dof         = len(self.arm_joints)

        # 关节极限
        self.joint_limits = np.array([
            [j.get_limits()[0][0], j.get_limits()[0][1]]
            for j in self.arm_joints
        ], dtype=np.float32)   # (7, 2)

        # 末端 link（Franka 的 panda_hand 或 panda_link8）
        self.ee_link = self._find_ee_link()
        self.ee_link_idx = [l.get_index() for l in self.robot.get_links()].index(
            self.ee_link.get_index()
        )

        # 中立构型（Franka 的常用起始构型）
        self.neutral_qpos = np.array(
            [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8], dtype=np.float32
        )
        self._set_qpos(self.neutral_qpos)

    def _find_ee_link(self):
        """找末端 link（panda_hand 优先，否则取最后一个 link）。"""
        for link in self.robot.get_links():
            if "hand" in link.get_name() or "ee" in link.get_name():
                return link
        return self.robot.get_links()[-1]

    def _set_qpos(self, qpos: np.ndarray) -> None:
        """设置臂关节角（不超出极限）。"""
        qpos_clipped = np.clip(qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])
        full_qpos = self.robot.get_qpos().copy()
        full_qpos[:self.n_dof] = qpos_clipped
        self.robot.set_qpos(full_qpos)

    def forward_kinematics(self, qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        正运动学：给定关节角，返回末端位置和旋转矩阵。

        Args:
            qpos: (7,) 臂关节角

        Returns:
            (ee_pos, ee_rot_mat)
            ee_pos:     (3,)
            ee_rot_mat: (3, 3)
        """
        self._set_qpos(qpos)
        self.scene.step()   # 让 SAPIEN 更新 FK

        ee_pose = self.ee_link.get_pose()
        ee_pos  = np.array(ee_pose.p, dtype=np.float32)
        # Pose.q 是 [w, x, y, z]
        quat_wxyz = np.array(ee_pose.q, dtype=np.float32)
        ee_rot = _quat_wxyz_to_mat(quat_wxyz)

        return ee_pos, ee_rot

    def compute_jacobian(self, qpos: np.ndarray) -> np.ndarray:
        """
        数值差分 Jacobian（6 × n_dof），用于 IK 迭代。

        精确解析 Jacobian 需要 pinocchio；这里用数值差分代替，
        精度足够 IK 收敛（步长 1e-4 rad）。

        Args:
            qpos: (7,) 当前关节角

        Returns:
            J: (6, 7) Jacobian，前 3 行位置，后 3 行姿态（旋转向量）
        """
        eps = 1e-4
        J = np.zeros((6, self.n_dof), dtype=np.float32)

        pos0, rot0 = self.forward_kinematics(qpos)
        rvec0 = _rotmat_to_rvec(rot0)

        for i in range(self.n_dof):
            dq = np.zeros(self.n_dof, dtype=np.float32)
            dq[i] = eps
            pos1, rot1 = self.forward_kinematics(qpos + dq)
            rvec1 = _rotmat_to_rvec(rot1)
            J[:3, i] = (pos1 - pos0) / eps
            J[3:, i] = (rvec1 - rvec0) / eps

        # 恢复原始 qpos
        self._set_qpos(qpos)
        self.scene.step()

        return J

    def check_collision(self) -> bool:
        """
        检查当前 qpos 下是否存在碰撞（自碰撞 + 与场景物体碰撞）。

        Returns:
            True = 有碰撞（不可达）
        """
        self.scene.step()
        contacts = self.scene.get_contacts()
        for contact in contacts:
            # 过滤 gripper 手指之间的预期接触
            a0 = contact.actor0.get_name()
            a1 = contact.actor1.get_name()
            if "finger" in a0 and "finger" in a1:
                continue
            if len(contact.points) > 0:
                return True
        return False


# ──────────────────────────────────────────────
# IK 求解器（Jacobian 伪逆迭代）
# ──────────────────────────────────────────────

def solve_ik(
    controller:   RobotController,
    target_pos:   np.ndarray,
    target_rot:   np.ndarray,
    init_qpos:    Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Damped Least Squares IK（Levenberg-Marquardt 变体）。

    Args:
        controller:  RobotController 实例
        target_pos:  (3,) 目标末端位置
        target_rot:  (3, 3) 目标末端旋转矩阵
        init_qpos:   (7,) 初始关节角，None 则用中立构型

    Returns:
        qpos: (7,) IK 解，或 None（无解）
    """
    qpos = (init_qpos.copy() if init_qpos is not None
            else controller.neutral_qpos.copy())

    for _iter in range(IK_MAX_ITER):
        cur_pos, cur_rot = controller.forward_kinematics(qpos)

        # 位置误差
        pos_err = target_pos - cur_pos   # (3,)

        # 姿态误差（旋转向量形式）
        R_err = target_rot @ cur_rot.T   # 误差旋转矩阵
        ori_err = _rotmat_to_rvec(R_err) # (3,)

        err = np.concatenate([pos_err, ori_err])  # (6,)

        # 收敛判定
        if (np.linalg.norm(pos_err) < IK_POS_TOL and
                np.linalg.norm(ori_err) < IK_ORI_TOL):
            return qpos  # 收敛

        # Jacobian + Damped Least Squares
        J = controller.compute_jacobian(qpos)
        JJT = J @ J.T   # (6, 6)
        damp_mat = IK_DAMP * np.eye(6, dtype=np.float32)
        dq = IK_LR * J.T @ np.linalg.solve(JJT + damp_mat, err)   # (7,)

        # 更新并夹到关节极限
        qpos = np.clip(
            qpos + dq,
            controller.joint_limits[:, 0],
            controller.joint_limits[:, 1],
        )

    # 超过最大迭代次数，检查最终误差
    cur_pos, cur_rot = controller.forward_kinematics(qpos)
    pos_err = np.linalg.norm(target_pos - cur_pos)
    ori_err = np.linalg.norm(_rotmat_to_rvec(target_rot @ cur_rot.T))

    if pos_err < IK_POS_TOL * 5 and ori_err < IK_ORI_TOL * 5:
        return qpos   # 松弛收敛

    return None   # IK 失败


# ──────────────────────────────────────────────
# 软约束计算
# ──────────────────────────────────────────────

def compute_joint_margin(
    qpos:         np.ndarray,
    joint_limits: np.ndarray,
) -> float:
    """
    计算关节角到极限的归一化余量 ∈ [0, 1]。

    定义：
        对每个关节 i，计算到上/下极限的距离：
            d_i = min(q_i - lo_i, hi_i - q_i) / (hi_i - lo_i)
        取所有关节的最小值（最危险的那个关节决定整体余量）。

    Args:
        qpos:         (7,) 当前关节角
        joint_limits: (7, 2) [lo, hi]

    Returns:
        margin ∈ [0, 1]，越大越远离极限
    """
    lo, hi = joint_limits[:, 0], joint_limits[:, 1]
    range_  = hi - lo
    range_  = np.maximum(range_, 1e-6)  # 避免除零
    d_lo = (qpos - lo) / range_         # 到下极限的归一化距离
    d_hi = (hi - qpos) / range_         # 到上极限的归一化距离
    d_min = np.minimum(d_lo, d_hi)      # 到最近极限的归一化距离
    return float(np.clip(d_min.min(), 0.0, 1.0))


def compute_yoshikawa(
    controller: RobotController,
    qpos:       np.ndarray,
    ref:        float = YOSHIKAWA_REF,
) -> float:
    """
    Yoshikawa 操作度指标（归一化）∈ [0, 1]。

    原始定义：w(q) = sqrt(det(J · J^T))
        - w 越大，末端在各方向的可操作性越好
        - w = 0 时处于奇异构型

    归一化：w_norm = min(w / ref, 1.0)
        ref 来自 Franka Panda 的经验最大操作度（~0.05）

    Args:
        controller: RobotController
        qpos:       (7,) 关节角
        ref:        归一化参考值

    Returns:
        float ∈ [0, 1]
    """
    J = controller.compute_jacobian(qpos)    # (6, 7)
    J_pos = J[:3, :]                         # 只用位置部分（3×7）
    det_val = np.linalg.det(J_pos @ J_pos.T) # det(J_pos · J_pos^T)，标量
    w = float(np.sqrt(max(det_val, 0.0)))    # Yoshikawa 操作度
    return float(np.clip(w / max(ref, 1e-9), 0.0, 1.0))


# ──────────────────────────────────────────────
# 核心：compute_r_geom
# ──────────────────────────────────────────────

def compute_r_geom(
    controller: RobotController,
    p:          np.ndarray,
    psi:        np.ndarray,
    g:          int,
) -> float:
    """
    计算单个 (p, ψ, g) query 的 R_geom ∈ [0, 1]。

    严格按项目计划 §3.4 + MoMaGen ICLR 2026 硬+软约束分级：

        Hard：IK 存在 + 无碰撞 → 失败则返回 0.0
        Soft：joint_margin × 0.5 + yoshikawa × 0.5 → ∈ (0, 1]

    Args:
        controller: RobotController（已加载机器人和场景物体）
        p:          (3,) 交互点世界坐标
        psi:        (3,) 单位逼近方向向量
        g:          抓取类型索引

    Returns:
        R_geom ∈ [0.0, 1.0]
    """
    # ── Step 1：计算目标末端位姿 ──
    target_pos, target_rot = compute_target_pose(p, psi, g)

    # ── Step 2 (Hard)：IK 求解 ──
    qpos = solve_ik(controller, target_pos, target_rot)
    if qpos is None:
        return 0.0   # IK 无解

    # ── Step 3 (Hard)：碰撞检测 ──
    controller._set_qpos(qpos)
    if controller.check_collision():
        return 0.0   # 碰撞

    # ── Step 4 (Soft)：关节余量 ──
    margin = compute_joint_margin(qpos, controller.joint_limits)

    # ── Step 5 (Soft)：Yoshikawa 操作度 ──
    yoshi = compute_yoshikawa(controller, qpos)

    # ── Step 6：合成 R_geom ──
    r_geom = 0.5 * margin + 0.5 * yoshi
    return float(np.clip(r_geom, 0.0, 1.0))


# ──────────────────────────────────────────────
# 批量填充单个实例的 R_geom
# ──────────────────────────────────────────────

def fill_r_geom_for_instance(
    npz_path:    str,
    controller:  RobotController,
    verbose:     bool = True,
) -> str:
    """
    读取实例 .npz，对所有 query 计算 R_geom 并写回（in-place 更新）。

    Args:
        npz_path:   data/<instance_id>.npz 的路径
        controller: RobotController（已加载该实例的场景物体）
        verbose:    打印进度

    Returns:
        npz_path（写回后的路径）
    """
    data = load_queries(npz_path)
    candidate_p  = data["candidate_p"]    # (M, 3)
    queries      = data["queries"]         # (M, K, 4)
    R_geom       = data["R_geom"].copy()   # (M, K)，初始全 NaN

    M, K, _ = queries.shape
    total = M * K
    done  = 0

    for m in range(M):
        for k in range(K):
            psi   = queries[m, k, :3]
            g_idx = int(round(queries[m, k, 3]))
            R_geom[m, k] = compute_r_geom(controller, candidate_p[m], psi, g_idx)
            done += 1

        if verbose:
            progress = done / total * 100
            r_vals = R_geom[m, :][~np.isnan(R_geom[m, :])]
            mean_r = r_vals.mean() if len(r_vals) > 0 else float("nan")
            print(f"  p[{m:2d}]: K={K} queries, "
                  f"R_geom mean={mean_r:.3f}, 0-rate={(r_vals == 0).mean():.2f}")

    # 写回（覆盖原文件）
    np.savez(
        npz_path.replace(".npz", ""),
        **{k: data[k] for k in data if k != "R_geom"},
        R_geom=R_geom,
    )

    if verbose:
        valid = R_geom[~np.isnan(R_geom)]
        print(f"  [r_geom] {os.path.basename(npz_path)}: "
              f"total={total}, "
              f"positive={int((valid > 0).sum())} ({(valid > 0).mean()*100:.1f}%), "
              f"mean={valid.mean():.3f}")

    return npz_path


# ──────────────────────────────────────────────
# 可视化：极坐标玫瑰图
# ──────────────────────────────────────────────

def visualize_polar_r_geom(
    npz_path:   str,
    save_path:  Optional[str] = None,
    point_idx:  int = 0,
) -> None:
    """
    对单个候选点，画 R_geom 关于 ψ 方向的极坐标玫瑰图。
    用于肉眼验证方向选择性是否合理。

    Args:
        npz_path:  .npz 文件路径
        save_path: 若指定则保存图片，否则 plt.show()
        point_idx: 画第几个候选点（默认 0）
    """
    import matplotlib.pyplot as plt

    data         = load_queries(npz_path)
    queries      = data["queries"]      # (M, K, 4)
    R_geom       = data["R_geom"]       # (M, K)
    instance_id  = str(data["instance_id"][0])

    if point_idx >= len(queries):
        raise ValueError(f"point_idx={point_idx} 超出 M={len(queries)}")

    q_row = queries[point_idx]     # (K, 4)
    r_row = R_geom[point_idx]      # (K,)

    # 用 ψ 的方位角作为极坐标角度，R_geom 作为半径
    psi_vecs = q_row[:, :3]             # (K, 3)
    phi_angles = np.arctan2(psi_vecs[:, 1], psi_vecs[:, 0])  # 方位角
    g_indices  = q_row[:, 3].astype(int)

    colors = ["#2196F3", "#4CAF50", "#FF5722"]   # pinch, power, poke
    labels = G_NAMES

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))

    for g_idx, (color, label) in enumerate(zip(colors, labels)):
        mask  = g_indices == g_idx
        phi_g = phi_angles[mask]
        r_g   = r_row[mask]
        valid = ~np.isnan(r_g)
        ax.scatter(phi_g[valid], r_g[valid], c=color, label=label,
                   s=80, alpha=0.8, edgecolors="white", linewidths=0.5)

    ax.set_ylim(0, 1)
    ax.set_title(f"R_geom 极坐标图\n实例 {instance_id}，候选点 {point_idx}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# 数学工具函数
# ──────────────────────────────────────────────

def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """
    四元数 [w, x, y, z] → 3×3 旋转矩阵。
    不依赖 scipy（批量 query 中频繁调用，避免 import overhead）。
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def _rotmat_to_rvec(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵 → 旋转向量（Rodrigues 参数），用于姿态误差计算。
    ‖rvec‖ = 旋转角（rad），方向 = 旋转轴。
    """
    # 使用 scipy 的 Rotation 转换（精确且稳定）
    try:
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(R).as_rotvec().astype(np.float32)
    except Exception:
        # fallback：直接从迹计算（精度稍差，但不依赖 scipy）
        trace = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        angle = np.arccos(trace)
        if abs(angle) < 1e-7:
            return np.zeros(3, dtype=np.float32)
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2 * np.sin(angle))
        return (angle * axis).astype(np.float32)


# ──────────────────────────────────────────────
# 单元测试（不依赖 SAPIEN，用 mock 数据验证数学部分）
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 55)
    print("r_geom.py 单元测试（数学部分，不依赖 SAPIEN）")
    print("=" * 55)
    rng = np.random.default_rng(42)

    # ── 测试 1：compute_target_pose 位置正确 ──
    print("\n[Test 1] compute_target_pose")
    p   = np.array([0.5, 0.0, 0.5], dtype=np.float32)
    psi = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 沿 X 轴逼近

    for g, name in enumerate(G_NAMES):
        pos, rot = compute_target_pose(p, psi, g)
        offset = APPROACH_DIST + GRASP_OFFSET[g]
        expected_x = p[0] - offset   # 目标位置在 X 轴负方向退 offset
        assert abs(pos[0] - expected_x) < 1e-5, \
            f"g={name}: pos[0]={pos[0]:.4f} ≠ {expected_x:.4f}"
        # 旋转矩阵正交性验证
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5), \
            f"g={name}: 旋转矩阵不正交"
        # forward 方向（rot 第三列）应与 psi 对齐
        assert np.allclose(rot[:, 2], psi, atol=1e-5), \
            f"g={name}: forward ≠ psi"
        print(f"  g={name}: pos={pos.round(4)}, forward=rot[:,2]={rot[:,2].round(4)} ✓")

    # ── 测试 2：compute_joint_margin 边界 ──
    print("\n[Test 2] compute_joint_margin")
    limits = np.array([[-1.0, 1.0]] * 7, dtype=np.float32)

    # 中央构型 → 余量应为 0.5（到任一极限距离=0.5，范围=2，归一化=0.25...
    # 等等，公式是 (q-lo)/(hi-lo) → 0.5 到 1.0 中间，min = 0.5
    q_center = np.zeros(7, dtype=np.float32)
    m = compute_joint_margin(q_center, limits)
    assert abs(m - 0.5) < 1e-5, f"中央构型余量应为 0.5, 得 {m}"
    print(f"  中央构型 margin={m:.4f} (期望 0.5) ✓")

    # 紧贴上极限 → 余量应接近 0
    q_at_limit = np.ones(7, dtype=np.float32) * 0.99
    m2 = compute_joint_margin(q_at_limit, limits)
    assert m2 < 0.02, f"紧贴极限余量应<0.02, 得 {m2}"
    print(f"  紧贴上极限 margin={m2:.4f} (期望≈0) ✓")

    # ── 测试 3：_quat_wxyz_to_mat 正交性 ──
    print("\n[Test 3] _quat_wxyz_to_mat 正交性")
    for _ in range(10):
        q = rng.random(4).astype(np.float32)
        q = q / np.linalg.norm(q)
        R = _quat_wxyz_to_mat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), f"旋转矩阵不正交: {R}"
    print("  10 个随机四元数 → 旋转矩阵全部正交 ✓")

    # ── 测试 4：_rotmat_to_rvec 往返一致性 ──
    print("\n[Test 4] _rotmat_to_rvec 往返一致性")
    try:
        from scipy.spatial.transform import Rotation
        for _ in range(5):
            rvec_gt = rng.random(3).astype(np.float32) * 0.5  # 小角度
            R = Rotation.from_rotvec(rvec_gt).as_matrix().astype(np.float32)
            rvec_out = _rotmat_to_rvec(R)
            assert np.allclose(rvec_out, rvec_gt, atol=1e-4), \
                f"rvec 不一致: {rvec_out} ≠ {rvec_gt}"
        print("  5 个随机旋转向量往返一致 ✓")
    except ImportError:
        print("  scipy 未安装，跳过 ✓")

    # ── 测试 5：GRASP_OFFSET 合理性 ──
    print("\n[Test 5] GRASP_OFFSET 覆盖所有 g_idx")
    assert set(GRASP_OFFSET.keys()) == {0, 1, 2}, f"GRASP_OFFSET 键不完整: {GRASP_OFFSET}"
    print(f"  GRASP_OFFSET = {GRASP_OFFSET} ✓")

    print("\n" + "=" * 55)
    print("全部 5 项数学测试通过 ✓")
    print("=" * 55)
    print("\nSAPEN 相关测试（IK、碰撞检测）需在服务器上运行：")
    print("  python label_gen/r_geom.py \\")
    print("    --npz-path  data/<instance_id>.npz \\")
    print("    --data-dir  /root/autodl-fs/partnet_mobility \\")
    print("    --robot-urdf assets/franka/panda.urdf \\")
    print("    --visualize")
    print("\n验证要点：")
    print("  1. R_geom 分布不退化（不全 0 或全 1）")
    print("  2. positive rate 约 20-60%（StorageFurniture 把手一般可达）")
    print("  3. 极坐标图显示方向选择性（正面方向 R_geom 高，侧面或背面低）")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="计算 R_geom 标签")
    p.add_argument("--npz-path",   required=False, default=None,
                   help="单个实例 .npz 路径（调试用）")
    p.add_argument("--data-dir",   required=False, default=None,
                   help="PartNet-Mobility 根目录（用于加载场景 URDF）")
    p.add_argument("--robot-urdf", default="assets/franka/panda.urdf",
                   help="Franka Panda URDF 路径")
    p.add_argument("--visualize",  action="store_true",
                   help="生成极坐标玫瑰图")
    p.add_argument("--test",       action="store_true",
                   help="运行单元测试（不需要 SAPIEN）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.test:
        _run_tests()
        return

    if args.npz_path is None or args.data_dir is None:
        print("请传入 --npz-path 和 --data-dir，或使用 --test 运行单元测试")
        return

    # 初始化 SAPIEN + 机器人
    from label_gen.sapien_loader import init_sapien, load_urdf, reset_scene
    import sapien.core as sapien

    engine, renderer, scene = init_sapien(headless=True)

    # 加载场景（实例 URDF）
    instance_id  = Path(args.npz_path).stem
    instance_dir = Path(args.data_dir) / instance_id
    _, _ = load_urdf(scene, str(instance_dir / "mobility.urdf"))

    # 加载机器人
    controller = RobotController(scene, args.robot_urdf)

    # 计算并写回 R_geom
    fill_r_geom_for_instance(args.npz_path, controller, verbose=True)

    # 可选：极坐标可视化
    if args.visualize:
        save_path = args.npz_path.replace(".npz", "_polar.png")
        visualize_polar_r_geom(args.npz_path, save_path=save_path)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv or len(sys.argv) == 1:
        _run_tests()
    else:
        main()