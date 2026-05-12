"""
label_gen/sapien_loader.py

步骤 2.2：用 SAPIEN 模拟单视角 RGB-D 观测，生成候选交互点。

功能：
    1. 初始化 SAPIEN Engine（Headless，服务器无显示器）
    2. 加载实例 mobility.urdf
    3. 用 3 个随机相机位姿各渲染一张深度图 + 语义分割图
    4. 反投影为点云（~10k 点），附带 part 标签
    5. 对 handle/knob/lid part 上的点做 FPS，每 part 采 5 个候选点 p
    6. 调用 sample_queries.py 生成 query，初始化 .npz 文件

依赖：sapien==2.2.0, numpy, open3d（可视化用）
      Franka Panda URDF：放在 <project_root>/assets/franka/panda.urdf
      （下载：https://github.com/haosulab/ManiSkill/tree/main/mani_skill/assets/robots/panda）

服务器运行示例：
    python label_gen/sapien_loader.py \\
        --data-dir    /root/autodl-fs/partnet_mobility \\
        --eval-set    data/eval_set_200.json \\
        --output-dir  data/ \\
        --instance-id 45174      # 只处理一个实例（调试用）
        --visualize              # 可选：Open3D 可视化验证

    # 批量处理由 batch_generate.py 调用，无需手动传 --instance-id
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from label_gen.sample_queries import sample_queries_for_instance, save_queries, N_PSI

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

N_CAMERA_POSES:   int   = 3       # 每实例渲染几个相机视角
IMG_WIDTH:        int   = 640
IMG_HEIGHT:       int   = 480
CAM_FOVY:         float = np.radians(60)  # 垂直视角 60°
CAM_NEAR:         float = 0.01
CAM_FAR:          float = 5.0
MAX_POINTS:       int   = 10_000  # 每视角保留点数上限（随机下采样）
N_CANDIDATES_PER_PART: int = 5   # 每个 handle part FPS 采 5 个候选点

# 相机围绕物体的轨道半径（米）
CAM_ORBIT_RADIUS: float = 1.2

# handle 语义关键词（与 select_instances.py 保持一致）
HANDLE_KEYWORDS = {"handle", "knob", "lid", "latch", "pull", "grip", "button"}


# ──────────────────────────────────────────────
# SAPIEN 初始化
# ──────────────────────────────────────────────

def init_sapien(headless: bool = True):
    """
    初始化 SAPIEN Engine + Renderer + Scene。

    Args:
        headless: True = 服务器无界面模式（必须为 True）

    Returns:
        (engine, renderer, scene)

    注：SAPIEN 2.x 在 headless 下用 OffscreenRenderer 或 VulkanRenderer+headless。
        若服务器没有显示器，VulkanRenderer 需要 Xvfb 或 EGL。
        AutoDL RTX 3060 镜像自带 Xvfb，用 `export DISPLAY=:0` 即可。
    """
    import sapien.core as sapien

    engine = sapien.Engine()
    engine.set_log_level("warning")   # 减少日志噪音

    # SAPIEN 2.2.0 headless 渲染配置
    renderer = sapien.VulkanRenderer(offscreen_only=headless)
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 240.0)    # 物理步长（标签生成不需要精细物理）
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, -1, -1], [1, 1, 1])

    return engine, renderer, scene


def reset_scene(engine, renderer):
    """
    销毁旧 scene，创建新 scene（用于批量处理时切换实例）。
    SAPIEN 2.x 不支持在同一 scene 内直接卸载 actor，需整体重建。
    """
    import sapien.core as sapien

    scene = engine.create_scene()
    scene.set_timestep(1 / 240.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, -1, -1], [1, 1, 1])
    return scene


# ──────────────────────────────────────────────
# URDF 加载
# ──────────────────────────────────────────────

def load_urdf(
    scene,
    urdf_path: str,
    fix_root: bool = True,
) -> Tuple[object, Dict[str, int]]:
    """
    加载 PartNet-Mobility 实例的 mobility.urdf。

    Args:
        scene:     SAPIEN scene
        urdf_path: mobility.urdf 的绝对路径
        fix_root:  是否固定根 link（通常是）

    Returns:
        (articulation, link_name_to_id)
        link_name_to_id: {link_name: link_index} 用于分割图解码
    """
    import sapien.core as sapien

    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root
    loader.scale = 1.0

    articulation = loader.load(urdf_path)
    if articulation is None:
        raise RuntimeError(f"URDF 加载失败: {urdf_path}")

    # 构建 link name → segmentation id 映射
    # SAPIEN 2.x 中每个 link 的 per_object_segmentation id = link 索引 + 1
    link_name_to_id: Dict[str, int] = {}
    for i, link in enumerate(articulation.get_links()):
        link_name_to_id[link.get_name()] = i + 1   # +1 因为 0 是背景

    return articulation, link_name_to_id


# ──────────────────────────────────────────────
# 相机位姿生成
# ──────────────────────────────────────────────

def generate_camera_poses(
    n_poses: int = N_CAMERA_POSES,
    seed:    int = 0,
    radius:  float = CAM_ORBIT_RADIUS,
) -> List[np.ndarray]:
    """
    生成 n_poses 个相机到世界坐标系的变换矩阵（4×4）。

    策略：
        在单位球面上用 Fibonacci 采样（借用 sample_queries.py 的思路），
        然后把相机放在 radius 处，看向原点（场景中心）。

    Returns:
        list of (4, 4) float32 camera-to-world matrices
    """
    rng = np.random.default_rng(seed)

    poses = []
    # Fibonacci 球面采样保证视角均匀
    golden = (1 + np.sqrt(5)) / 2
    i_arr = np.arange(n_poses, dtype=float)
    theta = np.arcsin(1 - 2 * i_arr / max(n_poses - 1, 1))
    phi   = 2 * np.pi * i_arr / golden % (2 * np.pi)
    # 相机位置（上半球偏移，避免从地面以下看）
    theta = np.clip(theta, np.radians(10), np.radians(80))

    for i in range(n_poses):
        # 相机位置
        x = radius * np.cos(theta[i]) * np.cos(phi[i])
        y = radius * np.cos(theta[i]) * np.sin(phi[i])
        z = radius * np.sin(theta[i])
        cam_pos = np.array([x, y, z], dtype=np.float32)

        # look-at：相机看向原点，up = [0,0,1]
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-9)  # 指向原点
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:                 # 相机正好在 Z 轴上，避免退化
            world_up = np.array([0, 1, 0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
        right = right / right_norm
        up = np.cross(right, forward)

        # 构造 4×4 camera-to-world（列：right, up, -forward）
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward   # SAPIEN 相机 -Z 轴朝前
        c2w[:3, 3] = cam_pos

        poses.append(c2w)

    return poses


# ──────────────────────────────────────────────
# 渲染 + 反投影
# ──────────────────────────────────────────────

def render_rgbd_seg(
    scene,
    renderer,
    c2w: np.ndarray,
    img_w: int = IMG_WIDTH,
    img_h: int = IMG_HEIGHT,
    fovy:  float = CAM_FOVY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在给定相机位姿下渲染深度图和语义分割图。

    Args:
        scene:  SAPIEN scene（已加载 URDF）
        renderer: SAPIEN renderer
        c2w:    (4, 4) camera-to-world 矩阵

    Returns:
        (depth, seg, intrinsics)
        depth:      (H, W) float32，单位 meters，0 表示无效
        seg:        (H, W) uint32，每像素对应 link id（0=背景）
        intrinsics: (3, 3) float32 相机内参矩阵
    """
    import sapien.core as sapien

    # 相机安装在 dummy actor 上（SAPIEN 2.x 的标准做法）
    dummy = scene.create_actor_builder().build_static(name="cam_dummy")

    # camera-to-world → SAPIEN Pose
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    from scipy.spatial.transform import Rotation
    quat_xyzw = Rotation.from_matrix(R).as_quat()          # scipy [x,y,z,w]
    quat_wxyz  = np.roll(quat_xyzw, 1)                     # SAPIEN [w,x,y,z]
    pose = sapien.Pose(p=t.tolist(), q=quat_wxyz.tolist())

    camera = scene.add_mounted_camera(
        name        = "depth_cam",
        actor       = dummy,
        pose        = pose,
        width       = img_w,
        height      = img_h,
        fovy        = fovy,
        near        = CAM_NEAR,
        far         = CAM_FAR,
    )

    scene.step()
    scene.update_render()
    camera.take_picture()

    # 获取深度：SAPIEN 2.x "Position" texture → (H, W, 4)，前三维是相机空间 XYZ
    position = camera.get_float_texture("Position")    # (H, W, 4)
    depth    = -position[..., 2]                       # SAPIEN 相机 -Z 轴朝前
    depth    = np.clip(depth, CAM_NEAR, CAM_FAR).astype(np.float32)

    # 获取分割图：(H, W, 4) uint32，[0] 通道是 per-object seg id
    seg_raw = camera.get_uint32_texture("Segmentation")  # (H, W, 4)
    seg = seg_raw[..., 0].astype(np.uint32)              # (H, W)

    # 相机内参（从 fovy + 分辨率推算）
    fy = img_h / (2 * np.tan(fovy / 2))
    fx = fy   # 正方形像素
    cx, cy = img_w / 2.0, img_h / 2.0
    intrinsics = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float32)

    # 清理：移除相机和 dummy actor
    scene.remove_camera(camera)
    scene.remove_actor(dummy)

    return depth, seg, intrinsics


def backproject_depth(
    depth:      np.ndarray,
    seg:        np.ndarray,
    intrinsics: np.ndarray,
    c2w:        np.ndarray,
    max_points: int = MAX_POINTS,
    rng:        Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将深度图反投影为世界坐标系下的 3D 点云。

    Args:
        depth:      (H, W) float32，米
        seg:        (H, W) uint32，link id
        intrinsics: (3, 3) 相机内参
        c2w:        (4, 4) camera-to-world
        max_points: 随机下采样上限
        rng:        numpy random Generator（None 则新建）

    Returns:
        (points, labels)
        points: (N, 3) float32，世界坐标系
        labels: (N,)  uint32，对应 link seg id
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 有效像素（深度在合理范围内）
    valid = (depth > CAM_NEAR) & (depth < CAM_FAR - 0.01)
    u_valid = np.where(valid)[1]   # 列索引
    v_valid = np.where(valid)[0]   # 行索引
    d_valid = depth[valid]
    s_valid = seg[valid]

    if len(d_valid) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint32)

    # 随机下采样
    if len(d_valid) > max_points:
        idx = rng.choice(len(d_valid), size=max_points, replace=False)
        u_valid = u_valid[idx]
        v_valid = v_valid[idx]
        d_valid = d_valid[idx]
        s_valid = s_valid[idx]

    # 像素坐标 → 相机坐标
    x_cam = (u_valid - cx) * d_valid / fx
    y_cam = (v_valid - cy) * d_valid / fy
    z_cam = d_valid
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # 相机坐标 → 世界坐标
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t[None, :]   # (N, 3)

    return pts_world.astype(np.float32), s_valid


# ──────────────────────────────────────────────
# FPS 采样
# ──────────────────────────────────────────────

def fps_sample(
    points: np.ndarray,
    k:      int,
    seed:   int = 0,
) -> np.ndarray:
    """
    贪心最远点采样（Farthest Point Sampling）。

    Args:
        points: (N, 3) float32
        k:      采样点数
        seed:   首个点的随机起点种子

    Returns:
        indices: (k,) int，选中点在 points 中的索引
    """
    N = len(points)
    if N == 0:
        return np.array([], dtype=np.int64)
    k = min(k, N)

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, N))]
    distances = np.full(N, np.inf, dtype=np.float32)

    for _ in range(k - 1):
        last = points[selected[-1]]
        dist = np.sum((points - last) ** 2, axis=-1)
        distances = np.minimum(distances, dist)
        selected.append(int(np.argmax(distances)))

    return np.array(selected, dtype=np.int64)


# ──────────────────────────────────────────────
# semantics.txt → handle link id 映射
# ──────────────────────────────────────────────

def load_semantics(instance_dir: Path) -> Dict[str, str]:
    """
    返回 {link_name: semantic_label}，只保留 handle 类 part。
    """
    sem_path = instance_dir / "semantics.txt"
    result = {}
    if not sem_path.exists():
        return result
    with open(sem_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                link_name, _, label = parts
            elif len(parts) == 2:
                link_name, label = parts
            else:
                continue
            if any(kw in label.lower() for kw in HANDLE_KEYWORDS):
                result[link_name] = label
    return result


# ──────────────────────────────────────────────
# 主流程：处理单个实例
# ──────────────────────────────────────────────

def process_instance(
    instance_id:   str,
    data_dir:      str,
    output_dir:    str,
    engine,
    renderer,
    n_cam_poses:   int = N_CAMERA_POSES,
    n_candidates:  int = N_CANDIDATES_PER_PART,
    n_psi:         int = N_PSI,
    cam_seed:      int = 0,
    visualize:     bool = False,
) -> str:
    """
    处理单个实例：渲染 → 点云 → FPS → query → 保存 .npz。

    Args:
        instance_id:  实例 ID 字符串
        data_dir:     PartNet-Mobility 根目录
        output_dir:   .npz 输出目录（e.g. "data/"）
        engine, renderer: SAPIEN 引擎（外部共享，避免重复初始化）
        n_cam_poses:  渲染相机数
        n_candidates: 每 part FPS 采样点数
        n_psi:        ψ 方向数（阶段二=8）
        cam_seed:     相机位姿随机种子
        visualize:    是否打开 Open3D 可视化窗口

    Returns:
        output_path: 写入的 .npz 文件路径

    Raises:
        RuntimeError: URDF 加载失败或无 handle 候选点
    """
    instance_dir = Path(data_dir) / instance_id
    urdf_path    = str(instance_dir / "mobility.urdf")

    # ── 1. 重建 scene，加载 URDF ──
    scene = reset_scene(engine, renderer)
    articulation, link_name_to_id = load_urdf(scene, urdf_path)

    # ── 2. 读 handle link 映射 ──
    handle_links = load_semantics(instance_dir)   # {link_name: label}
    if not handle_links:
        raise RuntimeError(f"实例 {instance_id} 无 handle/knob/lid 标注")

    handle_link_ids = {
        link_name_to_id[ln]: lbl
        for ln, lbl in handle_links.items()
        if ln in link_name_to_id
    }
    if not handle_link_ids:
        raise RuntimeError(f"实例 {instance_id} 的 handle link 未在 URDF 中找到")

    # ── 3. 生成相机位姿，渲染并反投影 ──
    cam_poses = generate_camera_poses(n_cam_poses, seed=cam_seed)
    rng = np.random.default_rng(cam_seed)

    all_points: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for c2w in cam_poses:
        depth, seg, intrinsics = render_rgbd_seg(scene, renderer, c2w)
        pts, lbls = backproject_depth(depth, seg, intrinsics, c2w, rng=rng)
        if len(pts) > 0:
            all_points.append(pts)
            all_labels.append(lbls)

    if not all_points:
        raise RuntimeError(f"实例 {instance_id} 所有视角点云为空，请检查 URDF")

    all_points = np.concatenate(all_points, axis=0)  # (N_total, 3)
    all_labels = np.concatenate(all_labels, axis=0)  # (N_total,)

    # ── 4. 提取 handle part 点 + FPS 采样候选点 ──
    candidate_list: List[np.ndarray] = []

    for link_id, label in handle_link_ids.items():
        part_mask   = (all_labels == link_id)
        part_points = all_points[part_mask]

        if len(part_points) < n_candidates:
            # 点太少时全取
            if len(part_points) > 0:
                candidate_list.append(part_points)
            continue

        fps_idx  = fps_sample(part_points, k=n_candidates, seed=int(link_id))
        selected = part_points[fps_idx]     # (n_candidates, 3)
        candidate_list.append(selected)

    if not candidate_list:
        raise RuntimeError(
            f"实例 {instance_id} 的 handle part 在点云中无可见点，"
            "请检查相机位姿或 URDF 尺寸是否异常。"
        )

    candidate_p = np.concatenate(candidate_list, axis=0)  # (M, 3)

    # ── 5. 生成 (ψ, g) query 网格 ──
    psi_directions, queries = sample_queries_for_instance(candidate_p, n_psi=n_psi)

    # ── 6. 保存 .npz ──
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, instance_id)
    save_queries(
        save_path      = output_path,
        instance_id    = instance_id,
        point_cloud    = all_points,
        candidate_p    = candidate_p,
        psi_directions = psi_directions,
        queries        = queries,
    )

    # ── 7. 可选：Open3D 可视化 ──
    if visualize:
        _visualize_candidates(all_points, candidate_p, instance_id)

    print(f"[sapien_loader] {instance_id}: "
          f"{len(all_points)} pts, {len(candidate_p)} candidates, "
          f"queries shape={queries.shape} → {output_path}.npz")

    return output_path + ".npz"


# ──────────────────────────────────────────────
# 可视化（Open3D）
# ──────────────────────────────────────────────

def _visualize_candidates(
    point_cloud:  np.ndarray,
    candidate_p:  np.ndarray,
    instance_id:  str,
) -> None:
    """
    Open3D 可视化：灰色点云 + 红色候选交互点。
    用于肉眼验证候选点是否落在 part 表面。
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[visualize] open3d 未安装，跳过可视化")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # 候选点用红色大球
    spheres = []
    for p in candidate_p:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh.translate(p)
        mesh.paint_uniform_color([1.0, 0.0, 0.0])
        spheres.append(mesh)

    o3d.visualization.draw_geometries(
        [pcd] + spheres,
        window_name=f"MicroReach: {instance_id} 候选交互点",
    )


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAPIEN 点云渲染 + 候选点生成")
    p.add_argument("--data-dir",    required=True,
                   help="PartNet-Mobility 根目录")
    p.add_argument("--output-dir",  default="data/",
                   help="输出 .npz 目录（默认 data/）")
    p.add_argument("--eval-set",    default="data/eval_set_200.json",
                   help="eval_set JSON 路径（批量处理时用）")
    p.add_argument("--instance-id", default=None,
                   help="只处理单个实例 ID（调试用；不传则读 eval_set）")
    p.add_argument("--n-psi",       type=int, default=N_PSI,
                   help=f"ψ 方向数（默认 {N_PSI}）")
    p.add_argument("--visualize",   action="store_true",
                   help="处理完每个实例后打开 Open3D 可视化（仅调试用）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engine, renderer, _ = init_sapien(headless=True)

    if args.instance_id:
        instance_ids = [args.instance_id]
    else:
        with open(args.eval_set, "r") as f:
            records = json.load(f)
        instance_ids = [r["instance_id"] for r in records]

    print(f"[sapien_loader] 处理 {len(instance_ids)} 个实例...")

    success, failed = 0, []
    for iid in instance_ids:
        try:
            process_instance(
                instance_id = iid,
                data_dir    = args.data_dir,
                output_dir  = args.output_dir,
                engine      = engine,
                renderer    = renderer,
                n_psi       = args.n_psi,
                visualize   = args.visualize,
            )
            success += 1
        except Exception as e:
            print(f"[ERROR] {iid}: {e}")
            failed.append(iid)

    print(f"\n[done] 成功 {success}，失败 {len(failed)}")
    if failed:
        print(f"失败实例: {failed}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        print("sapien_loader.py 的测试需要在服务器上运行（依赖 SAPIEN + GPU）。")
        print("请在服务器上用以下命令验证：")
        print("  python label_gen/sapien_loader.py \\")
        print("    --data-dir /root/autodl-fs/partnet_mobility \\")
        print("    --instance-id <任意一个实例ID> \\")
        print("    --visualize")
        print("\n验证要点：")
        print("  1. 控制台输出 '? pts, ? candidates, queries shape=(M, 24, 4)'")
        print("  2. Open3D 窗口中灰色点云 + 红色球体落在把手表面（不是空中或地板）")
        print("  3. data/<instance_id>.npz 文件已生成，R_geom 全为 NaN")
    else:
        main()