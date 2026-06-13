"""
label_gen/batch_generate.py

职责：串联完整 pipeline，批量处理 eval_set_200.json 中的 50 个实例，
      输出每实例的 .npz 标签文件。

依赖（按顺序）：
    sapien_loader.py    → 加载 URDF，生成点云 + 候选点 p
    sample_queries.py   → 生成 8ψ × 3g = 24 个 query
    r_geom.py           → 计算 R_geom 标签
    r_contact.py        → 计算 R_contact 标签（可选，P1 加分）

运行方式（服务器上）：
    # 完整跑全部 50 个实例
    python label_gen/batch_generate.py

    # 从断点续跑（--start-from 接实例ID）
    python label_gen/batch_generate.py --start-from 47645

    # 跑时跳过 r_contact（默认不跳过，若 r_contact 未完成则自动跳过）
    python label_gen/batch_generate.py --skip-contact

    # 后台运行
    nohup python label_gen/batch_generate.py > logs/batch.log 2>&1 &

输出：
    data/<instance_id>.npz    每个实例一个
    data/batch_progress.json  断点记录（每10个实例更新）
    data/failure_cases.jsonl  一致性校验失败的案例
    logs/batch.log            运行日志（nohup 时使用）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 路径设置：把项目根目录加入 sys.path，保证各模块可导入
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 延迟导入（服务器上才会有 SAPIEN，本地写代码时不崩溃）
def _import_pipeline_modules():
    """尝试导入 pipeline 依赖模块，返回成功导入的模块字典。"""
    modules = {}
    try:
        from label_gen import sapien_loader
        modules["sapien_loader"] = sapien_loader
    except ImportError as e:
        logging.warning(f"sapien_loader 导入失败（需在服务器上运行）: {e}")

    try:
        from label_gen import sample_queries
        modules["sample_queries"] = sample_queries
    except ImportError as e:
        logging.warning(f"sample_queries 导入失败: {e}")

    try:
        from label_gen import r_geom
        modules["r_geom"] = r_geom
    except ImportError as e:
        logging.warning(f"r_geom 导入失败: {e}")

    try:
        from label_gen import r_contact
        modules["r_contact"] = r_contact
    except ImportError as e:
        logging.info(f"r_contact 未找到，将跳过接触力标签: {e}")

    return modules


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DATA_DIR = _REPO_ROOT / "data"
LOGS_DIR = _REPO_ROOT / "logs"
EVAL_SET_PATH = DATA_DIR / "eval_set_200.json"
PROGRESS_PATH = DATA_DIR / "batch_progress.json"
FAILURE_LOG_PATH = DATA_DIR / "failure_cases.jsonl"

CHECKPOINT_EVERY = 10        # 每处理 N 个实例保存一次断点
CONSISTENCY_TOL = 0.05       # R_contact ≤ R_geom 容差
GIT_AUTO_PUSH = False         # 全部完成后手动 git push
MAX_CANDIDATES = 16           # 与 microreach_net.dataset 默认 max_M 对齐




def cap_candidate_points(
    candidate_p: np.ndarray,
    part_info: Dict[str, Any],
    instance_id: str,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Limit candidate points to max_candidates.

    New categories can produce M=30 or more candidate points, while the
    training Dataset currently pads to max_M=16 by default. This deterministic
    cap keeps generated .npz files compatible with the existing training code.
    """
    M = int(candidate_p.shape[0])
    if M <= max_candidates:
        return candidate_p, part_info

    # Deterministic per-instance subsampling.
    seed = int(instance_id) if str(instance_id).isdigit() else sum(ord(c) for c in str(instance_id))
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(M, size=max_candidates, replace=False)).astype(int)

    logging.warning(
        f"  [cap] {instance_id}: candidate_p M={M} > {max_candidates}, "
        f"keep indices={keep.tolist()}"
    )

    candidate_p = candidate_p[keep]
    part_info = dict(part_info or {})

    # Slice list/array fields aligned with candidate_p.
    for key in ("part_ids", "part_tiers"):
        if key not in part_info:
            continue
        v = part_info[key]
        try:
            if len(v) == M:
                if isinstance(v, np.ndarray):
                    part_info[key] = v[keep]
                else:
                    part_info[key] = [v[i] for i in keep]
        except TypeError:
            pass

    # Remap candidate-indexed dict fields, used by r_contact when not skipped.
    for key in ("normals", "axes"):
        if key not in part_info:
            continue
        v = part_info[key]
        if isinstance(v, dict):
            new_v = {}
            default = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            for new_i, old_i in enumerate(keep):
                old_i_int = int(old_i)
                new_v[new_i] = v.get(old_i_int, v.get(str(old_i_int), default))
            part_info[key] = new_v
        else:
            try:
                if len(v) == M:
                    if isinstance(v, np.ndarray):
                        part_info[key] = v[keep]
                    else:
                        part_info[key] = [v[i] for i in keep]
            except TypeError:
                pass

    return candidate_p, part_info


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

def setup_logging(log_file: Optional[Path] = None) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# 断点续跑
# ---------------------------------------------------------------------------

def load_progress() -> Dict[str, Any]:
    """加载上次运行的进度记录。"""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_update": None}


def save_progress(progress: Dict[str, Any]) -> None:
    """持久化当前进度到 JSON 文件。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 失败案例记录
# ---------------------------------------------------------------------------

def log_failure_case(case: Dict[str, Any]) -> None:
    """把一致性校验失败的案例追加写入 JSONL 文件。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 一致性校验
# ---------------------------------------------------------------------------

def check_consistency(
    instance_id: str,
    R_geom: np.ndarray,
    R_contact: Optional[np.ndarray],
    tol: float = CONSISTENCY_TOL,
) -> int:
    """
    验证 R_contact ≤ R_geom + tol（接触力评分不应高于几何可达性）。

    Returns
    -------
    n_violations : 违反约束的 query 数量
    """
    if R_contact is None:
        return 0

    violations = R_contact > (R_geom + tol)
    n_v = int(np.sum(violations))

    if n_v > 0:
        # 找出违反的具体 index
        viol_idx = np.argwhere(violations)
        logging.warning(
            f"  [一致性] {instance_id}: {n_v} 个 query 违反 R_contact ≤ R_geom+{tol}"
        )
        # 记录前 5 个违反案例
        for idx in viol_idx[:5]:
            i, j = int(idx[0]), int(idx[1])
            log_failure_case(
                {
                    "instance_id": instance_id,
                    "candidate_idx": i,
                    "query_idx": j,
                    "R_geom": float(R_geom[i, j]),
                    "R_contact": float(R_contact[i, j]),
                    "delta": float(R_contact[i, j] - R_geom[i, j]),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    return n_v


# ---------------------------------------------------------------------------
# 单实例处理
# ---------------------------------------------------------------------------

def process_instance(
    instance_id: str,
    instance_meta: Dict[str, Any],
    modules: Dict[str, Any],
    skip_contact: bool = False,
    partnet_root: Optional[Path] = None,
) -> bool:
    """
    对单个实例跑完整 pipeline，保存 .npz。

    Returns
    -------
    success : bool
    """
    out_path = DATA_DIR / f"{instance_id}.npz"

    # 已完成则跳过
    if out_path.exists():
        logging.info(f"  [跳过] {instance_id}（.npz 已存在）")
        return True

    logging.info(f"  [开始] {instance_id}")
    t0 = time.time()

    try:
        # ---- 确定 URDF 路径 ----
        if partnet_root is None:
            partnet_root = Path("/root/autodl-fs/partnet_mobility")
        urdf_path = partnet_root / instance_id / "mobility.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF 未找到: {urdf_path}")

        # ---- Step 1: sapien_loader → 点云 + 候选交互点 ----
        loader = modules.get("sapien_loader")
        if loader is None:
            raise RuntimeError("sapien_loader 未导入，请在服务器上运行")

        point_cloud, candidate_p, part_info = loader.load_instance(
            urdf_path=str(urdf_path),
            instance_id=instance_id,
        )
        # point_cloud : (N_pts, 3)
        # candidate_p : (M, 3)
        # part_info   : dict，含 normals, axes, part_ids 等

        logging.info(
            f"  [loader] 点云={point_cloud.shape}，候选点={candidate_p.shape[0]}"
        )

        candidate_p, part_info = cap_candidate_points(
            candidate_p=candidate_p,
            part_info=part_info,
            instance_id=instance_id,
        )

        # ---- Step 2: sample_queries → queries ----
        sq = modules.get("sample_queries")
        if sq is None:
            raise RuntimeError("sample_queries 未导入")

        # sample_queries.py 当前接口为 sample_queries_for_instance(...)
        # 返回 (psi_directions, queries)，这里只需要 queries。
        if hasattr(sq, "generate_queries"):
            queries = sq.generate_queries(candidate_p)  # 兼容旧接口
        elif hasattr(sq, "sample_queries_for_instance"):
            _psi_directions, queries = sq.sample_queries_for_instance(candidate_p)
        else:
            raise RuntimeError(
                "sample_queries.py 中既没有 generate_queries，也没有 "
                "sample_queries_for_instance，请检查接口。"
            )

        logging.info(f"  [queries] shape={queries.shape}")

        # ---- Step 3: r_geom → R_geom ----
        rg = modules.get("r_geom")
        if rg is None:
            raise RuntimeError("r_geom 未导入")

        R_geom = rg.batch_compute_r_geom(
            candidate_p=candidate_p,
            queries=queries,
            urdf_path=str(urdf_path),
            instance_id=instance_id,
        )  # (M, 24), float32
        logging.info(
            f"  [r_geom] mean={R_geom.mean():.3f}, nonzero={np.mean(R_geom > 0):.1%}"
        )

        # ---- Step 4: r_contact → R_contact（可选）----
        R_contact: Optional[np.ndarray] = None
        rc = modules.get("r_contact")
        if rc is not None and not skip_contact:
            part_normals = np.array(
                [part_info["normals"].get(i, np.array([0.0, 0.0, 1.0])) for i in range(len(candidate_p))]
            )
            part_axes = np.array(
                [part_info.get("axes", {}).get(i, np.array([0.0, 0.0, 1.0])) for i in range(len(candidate_p))]
            )
            R_contact = rc.batch_compute_r_contact(
                candidate_p=candidate_p,
                queries=queries,
                part_normals=part_normals,
                part_axes=part_axes,
            )  # (M, 24), float32
            logging.info(
                f"  [r_contact] mean={R_contact.mean():.3f}"
            )
        else:
            logging.info("  [r_contact] 已跳过")

        # ---- Step 5: 一致性校验 ----
        n_violations = check_consistency(instance_id, R_geom, R_contact)

        # ---- Step 6: 保存 .npz ----
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_dict: Dict[str, Any] = {
            "instance_id": instance_id,
            "point_cloud": point_cloud.astype(np.float32),
            "candidate_p": candidate_p.astype(np.float32),
            "queries": queries.astype(np.float32),
            "R_geom": R_geom,
            # R_contact 存在则写入，否则用 NaN 占位（保持 .npz 字段一致）
            "R_contact": R_contact if R_contact is not None
                         else np.full_like(R_geom, np.nan),
            # R_exec 留给阶段三 Isaac Sim 填写
            "R_exec": np.full_like(R_geom, np.nan),
            "n_consistency_violations": np.int32(n_violations),
        }
        # 写入 part_info 的可序列化字段
        if "part_ids" in part_info:
            save_dict["part_ids"] = np.array(part_info["part_ids"])
        if "part_tiers" in part_info:
            save_dict["part_tiers"] = np.array(
                [str(t) for t in part_info["part_tiers"]]
            )

        np.savez_compressed(str(out_path), **save_dict)

        elapsed = time.time() - t0
        logging.info(f"  [完成] {instance_id}  耗时 {elapsed:.1f}s → {out_path.name}")
        return True

    except Exception:
        logging.error(f"  [失败] {instance_id}\n{traceback.format_exc()}")
        return False


# ---------------------------------------------------------------------------
# Git 自动提交
# ---------------------------------------------------------------------------

def git_push_data(batch_label: str) -> bool:
    """把 data/ 下的新 .npz 和 JSON 文件 add-commit-push。"""
    try:
        subprocess.run(
            ["git", "add", "data/*.npz", "data/eval_set_200.json",
             "data/batch_progress.json", "data/failure_cases.jsonl"],
            cwd=_REPO_ROOT,
            check=False,  # 部分路径不存在时 add 会报错，不 check
        )
        # 检查是否有变更
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=_REPO_ROOT,
        )
        if result.returncode == 0:
            logging.info("[git] 无新变更，跳过 commit")
            return True

        subprocess.run(
            ["git", "commit", "-m", f"data: R_geom labels {batch_label}"],
            cwd=_REPO_ROOT,
            check=True,
        )
        subprocess.run(["git", "push"], cwd=_REPO_ROOT, check=True)
        logging.info("[git] push 成功")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"[git] push 失败: {e}")
        return False


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(
    start_from: Optional[str] = None,
    skip_contact: bool = False,
    partnet_root: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Parameters
    ----------
    start_from   : 从该 instance_id 开始（断点续跑），None 表示从头
    skip_contact : 跳过 r_contact 计算
    partnet_root : PartNet-Mobility 数据集根目录，None 时用默认路径
    dry_run      : 仅打印计划，不实际运行（用于调试）
    """
    setup_logging(LOGS_DIR / "batch.log")
    logging.info("=" * 60)
    logging.info("batch_generate.py 启动")
    logging.info(f"  start_from   = {start_from}")
    logging.info(f"  skip_contact = {skip_contact}")
    logging.info(f"  partnet_root = {partnet_root}")
    logging.info(f"  dry_run      = {dry_run}")
    logging.info("=" * 60)

    # ---- 读取实例列表 ----
    if not EVAL_SET_PATH.exists():
        logging.error(f"eval_set_200.json 不存在: {EVAL_SET_PATH}")
        logging.error("请先运行 label_gen/select_instances.py 生成实例列表。")
        sys.exit(1)

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set_raw = json.load(f)

    # 兼容两种 eval set 格式：
    # 1) 旧格式: {"instances": [...]}
    # 2) 当前 save_eval_set 输出格式: [{instance_id, category, parts, ...}, ...]
    if isinstance(eval_set_raw, list):
        instances: List[Dict[str, Any]] = eval_set_raw
    elif isinstance(eval_set_raw, dict):
        instances = eval_set_raw.get("instances", [])
    else:
        logging.error(f"eval_set_200.json 格式错误: {type(eval_set_raw)}")
        sys.exit(1)

    if not instances:
        logging.error("eval_set_200.json 中实例列表为空！")
        sys.exit(1)

    logging.info(f"实例总数: {len(instances)}")

    # ---- 加载进度，确定起始点 ----
    progress = load_progress()
    completed_set = set(progress["completed"])
    failed_set = set(progress["failed"])

    # 找到起始实例的索引
    start_idx = 0
    if start_from is not None:
        for i, inst in enumerate(instances):
            if str(inst.get("instance_id", inst.get("id", ""))) == start_from:
                start_idx = i
                break
        else:
            logging.warning(
                f"--start-from {start_from} 未在列表中找到，从头开始"
            )

    remaining = instances[start_idx:]
    logging.info(
        f"待处理: {len(remaining)} 个（已完成 {len(completed_set)}，"
        f"曾失败 {len(failed_set)}）"
    )

    if dry_run:
        logging.info("[dry_run] 计划处理的实例:")
        for inst in remaining:
            iid = str(inst.get("instance_id", inst.get("id", "")))
            status = "completed" if iid in completed_set else "pending"
            logging.info(f"  {iid}  [{status}]")
        return

    # ---- 导入 pipeline 模块 ----
    modules = _import_pipeline_modules()
    if "sapien_loader" not in modules:
        logging.error(
            "sapien_loader 导入失败！请确认在 AutoDL 服务器上运行，且已安装 SAPIEN。"
        )
        sys.exit(1)

    partnet_root_path = Path(partnet_root) if partnet_root else None

    # ---- 主循环 ----
    batch_start_time = time.time()
    processed_this_run = 0

    for idx, inst in enumerate(remaining):
        instance_id = str(inst.get("instance_id", inst.get("id", "")))

        if instance_id in completed_set:
            logging.info(f"[{idx+1}/{len(remaining)}] {instance_id} 已完成，跳过")
            continue

        logging.info(f"[{idx+1}/{len(remaining)}] 处理 {instance_id}")

        success = process_instance(
            instance_id=instance_id,
            instance_meta=inst,
            modules=modules,
            skip_contact=skip_contact,
            partnet_root=partnet_root_path,
        )

        if success:
            completed_set.add(instance_id)
            progress["completed"].append(instance_id)
            # 从 failed 中移除（重试成功）
            if instance_id in failed_set:
                failed_set.discard(instance_id)
                progress["failed"] = [x for x in progress["failed"] if x != instance_id]
        else:
            failed_set.add(instance_id)
            if instance_id not in progress["failed"]:
                progress["failed"].append(instance_id)

        processed_this_run += 1

        # ---- 断点保存（每 CHECKPOINT_EVERY 个实例）----
        if processed_this_run % CHECKPOINT_EVERY == 0:
            save_progress(progress)
            if GIT_AUTO_PUSH:
                batch_label = (
                    f"instances-{progress['completed'][0]}-"
                    f"{progress['completed'][-1]}"
                    f"({len(progress['completed'])}个)"
                )
                git_push_data(batch_label)
            logging.info(
                f"[checkpoint] 已完成 {len(completed_set)} 个，"
                f"失败 {len(failed_set)} 个"
            )

    # ---- 最终汇总 ----
    save_progress(progress)
    total_time = time.time() - batch_start_time

    logging.info("=" * 60)
    logging.info("批量生成完成")
    logging.info(f"  总耗时       : {total_time/60:.1f} 分钟")
    logging.info(f"  成功实例数   : {len(completed_set)}")
    logging.info(f"  失败实例数   : {len(failed_set)}")
    if failed_set:
        logging.info(f"  失败列表     : {sorted(failed_set)}")
    logging.info(f"  失败案例记录 : {FAILURE_LOG_PATH}")
    logging.info("=" * 60)

    # ---- 最终 git push ----
    if GIT_AUTO_PUSH:
        git_push_data(f"final-{len(completed_set)}instances")

    if failed_set:
        logging.warning(
            f"有 {len(failed_set)} 个实例失败，可用 --start-from 指定实例续跑。"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量生成 MicroReach 标签（R_geom / R_contact）"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        metavar="INSTANCE_ID",
        help="从指定 instance_id 开始（断点续跑）",
    )
    parser.add_argument(
        "--skip-contact",
        action="store_true",
        default=False,
        help="跳过 r_contact 计算（R_contact 字段填 NaN）",
    )
    parser.add_argument(
        "--partnet-root",
        type=str,
        default=None,
        metavar="PATH",
        help="PartNet-Mobility 数据集根目录（默认 /root/autodl-fs/partnet_mobility）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="仅打印计划，不实际运行",
    )
    args = parser.parse_args()

    main(
        start_from=args.start_from,
        skip_contact=args.skip_contact,
        partnet_root=args.partnet_root,
        dry_run=args.dry_run,
    )