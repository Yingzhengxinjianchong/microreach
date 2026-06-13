"""
Analytic R_exec label generation for MicroReach.

This module computes an execution feasibility score R_exec(p, psi, g)
from articulated-joint metadata. It is intentionally file-light and
contains no batch write logic; use patch_r_exec.py for .npz updates.

P2 initial version:
- Parse mobility.urdf and semantics.txt.
- Map candidate part_id to a movable joint by:
  1) exact link match when valid;
  2) semantic name match plus geometric disambiguation otherwise.
- Score execution by alignment between query direction psi and the
  joint-induced local motion direction.
- Static / unmapped candidates get R_exec_raw = 0.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


MOVABLE_URDF_TYPES = {"revolute", "continuous", "prismatic"}
MOVABLE_SEM_TYPES = {"hinge", "slider", "revolute", "prismatic"}


def _as_float_vec3(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, str):
        vals = [float(v) for v in x.split()]
    else:
        vals = [float(v) for v in x]
    if len(vals) != 3:
        return None
    return np.asarray(vals, dtype=np.float32)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return v / n


def _parse_part_id(part_id: str) -> Tuple[str, str]:
    part_id = str(part_id)
    if ":" not in part_id:
        return part_id, ""
    link, name = part_id.split(":", 1)
    return link, name


def parse_semantics(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse semantics.txt lines like:
      link_0 hinge switch
      link_2 static faucet_base
    """
    path = Path(path)
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 3:
            continue

        link = toks[0]
        joint_sem = toks[1]
        name = " ".join(toks[2:])
        out[link] = {
            "link": link,
            "joint_sem": joint_sem,
            "name": name,
            "movable_sem": joint_sem in MOVABLE_SEM_TYPES,
        }
    return out


def parse_urdf_joints(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Return joints indexed by child link.
    """
    path = Path(path)
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    tree = ET.parse(path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "")

        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        origin = joint.find("origin")
        limit = joint.find("limit")

        child_link = child.attrib.get("link") if child is not None else None
        parent_link = parent.attrib.get("link") if parent is not None else None
        if not child_link:
            continue

        axis_v = None
        if axis is not None:
            axis_v = _as_float_vec3(axis.attrib.get("xyz"))
            if axis_v is not None:
                axis_v = _normalize(axis_v)

        origin_v = np.zeros(3, dtype=np.float32)
        if origin is not None and "xyz" in origin.attrib:
            parsed_origin = _as_float_vec3(origin.attrib.get("xyz"))
            if parsed_origin is not None:
                origin_v = parsed_origin

        lower = None
        upper = None
        if limit is not None:
            if "lower" in limit.attrib:
                lower = float(limit.attrib["lower"])
            if "upper" in limit.attrib:
                upper = float(limit.attrib["upper"])

        out[child_link] = {
            "joint_name": joint_name,
            "joint_type": joint_type,
            "parent": parent_link,
            "child": child_link,
            "axis": axis_v,
            "origin": origin_v,
            "lower": lower,
            "upper": upper,
            "movable_urdf": joint_type in MOVABLE_URDF_TYPES,
        }

    return out


def _point_line_distance(p: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    axis = _normalize(axis)
    if float(np.linalg.norm(axis)) < 1e-8:
        return float("inf")
    return float(np.linalg.norm(np.cross(p - origin, axis)))


def _is_valid_movable_joint(joint: Dict[str, Any]) -> bool:
    if not joint.get("movable_urdf", False):
        return False
    if joint.get("axis") is None:
        return False
    if float(np.linalg.norm(joint["axis"])) < 1e-8:
        return False
    return True


def map_candidate_to_joint(
    part_id: str,
    p: np.ndarray,
    semantics: Dict[str, Dict[str, Any]],
    joints: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Map one candidate part to a movable joint.

    Returns:
      (joint, status)
    status:
      exact
      semantic
      static_or_unmapped
    """
    link, name = _parse_part_id(part_id)

    # 1. Exact link match when the parsed link is reliable.
    sem_exact = semantics.get(link)
    joint_exact = joints.get(link)
    if (
        sem_exact is not None
        and joint_exact is not None
        and sem_exact.get("name") == name
        and _is_valid_movable_joint(joint_exact)
    ):
        return joint_exact, "exact"

    # 2. Semantic match. This handles the known link_N offset issue.
    candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for child_link, joint in joints.items():
        if not _is_valid_movable_joint(joint):
            continue

        sem = semantics.get(child_link, {})
        if sem.get("name") != name:
            continue

        dist = _point_line_distance(p, joint["origin"], joint["axis"])
        candidates.append((dist, child_link, joint))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2], "semantic"

    # 3. Static frame/body/base or otherwise unmapped candidate.
    return None, "static_or_unmapped"


def _score_prismatic(psi: np.ndarray, joint: Dict[str, Any]) -> float:
    axis = _normalize(joint["axis"])
    psi = _normalize(psi)
    if float(np.linalg.norm(psi)) < 1e-8:
        return 0.0

    # Bidirectional analytic score: push or pull can actuate a slider.
    return float(np.clip(abs(float(np.dot(psi, axis))), 0.0, 1.0))


def _score_revolute(p: np.ndarray, psi: np.ndarray, joint: Dict[str, Any]) -> float:
    axis = _normalize(joint["axis"])
    origin = np.asarray(joint["origin"], dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    psi = _normalize(psi)

    if float(np.linalg.norm(psi)) < 1e-8:
        return 0.0

    r = p - origin
    tangent = np.cross(axis, r)

    if float(np.linalg.norm(tangent)) < 1e-8:
        return 0.0

    tangent = _normalize(tangent)

    # Bidirectional score: both opening and closing directions are executable.
    return float(np.clip(abs(float(np.dot(psi, tangent))), 0.0, 1.0))


def score_query_exec(
    p: np.ndarray,
    psi: np.ndarray,
    joint: Optional[Dict[str, Any]],
) -> float:
    if joint is None:
        return 0.0

    joint_type = joint.get("joint_type", "")

    if joint_type == "prismatic":
        return _score_prismatic(psi, joint)

    if joint_type in {"revolute", "continuous"}:
        return _score_revolute(p, psi, joint)

    return 0.0


def compute_r_exec(
    candidate_p: np.ndarray,
    queries: np.ndarray,
    part_ids: np.ndarray | List[str],
    partnet_instance_dir: str | Path,
    r_contact: Optional[np.ndarray] = None,
    enforce_cascade: bool = False,
) -> Dict[str, Any]:
    """
    Compute R_exec for one instance.

    Args:
      candidate_p: (M, 3)
      queries: (M, K, 4), where queries[..., :3] is psi.
      part_ids: length M, strings like 'link_2:switch'
      partnet_instance_dir: PartNet-Mobility instance folder.
      r_contact: optional (M, K). Used only when enforce_cascade=True.
      enforce_cascade: if true, R_exec = min(R_exec_raw, R_contact)

    Returns:
      dict with R_exec_raw, R_exec, mapping_status, summary.
    """
    candidate_p = np.asarray(candidate_p, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    part_ids = [str(x) for x in list(part_ids)]

    if candidate_p.ndim != 2 or candidate_p.shape[1] != 3:
        raise ValueError(f"candidate_p must be (M, 3), got {candidate_p.shape}")
    if queries.ndim != 3 or queries.shape[0] != candidate_p.shape[0] or queries.shape[2] < 3:
        raise ValueError(f"queries must be (M, K, >=3), got {queries.shape}")
    if len(part_ids) != candidate_p.shape[0]:
        raise ValueError(f"part_ids length {len(part_ids)} != M {candidate_p.shape[0]}")

    inst_dir = Path(partnet_instance_dir)
    semantics = parse_semantics(inst_dir / "semantics.txt")
    joints = parse_urdf_joints(inst_dir / "mobility.urdf")

    M, K = queries.shape[:2]
    r_exec_raw = np.zeros((M, K), dtype=np.float32)
    mapping_status: List[str] = []
    mapped_child_links: List[str] = []
    mapped_joint_types: List[str] = []

    status_counts: Dict[str, int] = {}

    for i in range(M):
        joint, status = map_candidate_to_joint(
            part_id=part_ids[i],
            p=candidate_p[i],
            semantics=semantics,
            joints=joints,
        )
        mapping_status.append(status)
        status_counts[status] = status_counts.get(status, 0) + 1

        if joint is None:
            mapped_child_links.append("")
            mapped_joint_types.append("")
            continue

        mapped_child_links.append(str(joint.get("child", "")))
        mapped_joint_types.append(str(joint.get("joint_type", "")))

        for j in range(K):
            psi = queries[i, j, :3]
            r_exec_raw[i, j] = score_query_exec(candidate_p[i], psi, joint)

    r_exec_raw = np.nan_to_num(r_exec_raw, nan=0.0, posinf=0.0, neginf=0.0)
    r_exec_raw = np.clip(r_exec_raw, 0.0, 1.0).astype(np.float32)

    if enforce_cascade:
        if r_contact is None:
            raise ValueError("r_contact is required when enforce_cascade=True")
        r_contact = np.asarray(r_contact, dtype=np.float32)
        if r_contact.shape != r_exec_raw.shape:
            raise ValueError(f"r_contact shape {r_contact.shape} != {r_exec_raw.shape}")
        r_exec = np.minimum(r_exec_raw, np.nan_to_num(r_contact, nan=0.0))
    else:
        r_exec = r_exec_raw.copy()

    r_exec = np.clip(r_exec, 0.0, 1.0).astype(np.float32)

    summary = {
        "M": int(M),
        "K": int(K),
        "status_counts": status_counts,
        "raw_mean": float(np.mean(r_exec_raw)),
        "raw_nonzero_ratio": float(np.mean(r_exec_raw > 1e-6)),
        "exec_mean": float(np.mean(r_exec)),
        "exec_nonzero_ratio": float(np.mean(r_exec > 1e-6)),
    }

    return {
        "R_exec_raw": r_exec_raw,
        "R_exec": r_exec,
        "exec_mapping_status": np.asarray(mapping_status),
        "exec_mapped_child_links": np.asarray(mapped_child_links),
        "exec_mapped_joint_types": np.asarray(mapped_joint_types),
        "summary": summary,
    }
