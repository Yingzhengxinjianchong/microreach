import sys
import os
from pathlib import Path

import numpy as np
import torch


MICROREACH_ROOT = Path("/root/autodl-tmp/microreach")
WHERE2ACT_CODE = Path("/root/autodl-tmp/where2act/code")
POINTNET2_ROOT = Path("/root/autodl-tmp/Pointnet2_PyTorch")

sys.path.insert(0, str(WHERE2ACT_CODE))
sys.path.insert(0, str(POINTNET2_ROOT))

from models.model_3d_legacy import Network  # noqa: E402


DATA_DIR = MICROREACH_ROOT / "data"
OUT_PATH = MICROREACH_ROOT / "eval" / "results" / "where2act_predictions.npz"

EXP_NAME = "finalexp-model_all_final-pulling-None-train_all_v1"
MODEL_EPOCH = 81

N_POINTS = 10000
N_QUERY = 24
SEED = 42

def as_str_array(x):
    return np.array([str(v) for v in x], dtype=object)


def sample_scene_with_candidates(point_cloud, candidate_p, n_points=N_POINTS, seed=SEED):
    """
    构造 Where2Act 输入点云：
    - 前 M 个点强制放 candidate_p
    - 后面用 scene point cloud 补齐到 10000
    这样 inference_action_score 输出的前 M 个 score 就对应 candidate_p。
    """
    rng = np.random.default_rng(seed)

    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    candidate_p = np.asarray(candidate_p, dtype=np.float32)

    m = candidate_p.shape[0]
    if m >= n_points:
        return candidate_p[:n_points].astype(np.float32), n_points

    need = n_points - m
    n_scene = point_cloud.shape[0]

    if n_scene >= need:
        idx = rng.choice(n_scene, size=need, replace=False)
    else:
        idx = rng.choice(n_scene, size=need, replace=True)

    sampled = point_cloud[idx]
    pcs = np.concatenate([candidate_p, sampled], axis=0).astype(np.float32)

    return pcs, m


def load_where2act_model(device):
    conf_path = WHERE2ACT_CODE / "logs" / EXP_NAME / "conf.pth"
    ckpt_path = WHERE2ACT_CODE / "logs" / EXP_NAME / "ckpts" / f"{MODEL_EPOCH}-network.pth"

    print("[info] conf:", conf_path)
    print("[info] ckpt:", ckpt_path)

    conf = torch.load(conf_path, map_location="cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    net = Network(conf.feat_dim, conf.rv_dim, conf.rv_cnt).to(device)
    ret = net.load_state_dict(ckpt, strict=False)
    print("[info] load_state_dict:", ret)

    net.eval()
    return net

def main():
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No npz files found in {DATA_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[info] device:", device)
    print("[info] npz files:", len(npz_files))

    net = load_where2act_model(device)

    instance_ids = []
    predictions = []
    candidate_scores = []
    targets = []
    valid_masks = []
    candidate_points = []
    part_tiers = []
    part_ids = []

    max_m = 0

    for rank, f in enumerate(npz_files, start=1):
        d = np.load(f, allow_pickle=True)

        iid = str(d["instance_id"]) if "instance_id" in d.files else f.stem
        pc = d["point_cloud"].astype(np.float32)
        cand = d["candidate_p"].astype(np.float32)
        r_geom = d["R_geom"].astype(np.float32)

        pcs_np, m = sample_scene_with_candidates(pc, cand, N_POINTS, seed=SEED + rank)
        pcs = torch.from_numpy(pcs_np[None]).float().to(device)

        with torch.no_grad():
            scores_all = net.inference_action_score(pcs)

        scores_all = scores_all.detach().cpu().numpy()[0]
        scores = scores_all[:m].astype(np.float32)

        pred = np.repeat(scores[:, None], N_QUERY, axis=1).astype(np.float32)

        instance_ids.append(iid)
        predictions.append(pred)
        candidate_scores.append(scores)
        targets.append(r_geom)
        valid_masks.append(np.ones((m,), dtype=bool))
        candidate_points.append(cand)

        if "part_tiers" in d.files:
            part_tiers.append(as_str_array(d["part_tiers"]))
        else:
            part_tiers.append(np.array(["unknown"] * m, dtype=object))

        if "part_ids" in d.files:
            part_ids.append(as_str_array(d["part_ids"]))
        else:
            part_ids.append(np.array(["unknown"] * m, dtype=object))

        max_m = max(max_m, m)

        print(
            f"[{rank:02d}/{len(npz_files)}] {iid}: "
            f"M={m}, score_mean={scores.mean():.4f}, "
            f"min={scores.min():.4f}, max={scores.max():.4f}"
        )

    n = len(instance_ids)

    padded_predictions = np.full((n, max_m, N_QUERY), np.nan, dtype=np.float32)
    padded_targets = np.full((n, max_m, N_QUERY), np.nan, dtype=np.float32)
    padded_scores = np.full((n, max_m), np.nan, dtype=np.float32)
    padded_valid_mask = np.zeros((n, max_m), dtype=bool)
    padded_candidate_p = np.full((n, max_m, 3), np.nan, dtype=np.float32)
    padded_part_tiers = np.full((n, max_m), "unknown", dtype=object)
    padded_part_ids = np.full((n, max_m), "unknown", dtype=object)

    for i in range(n):
        m = candidate_scores[i].shape[0]
        padded_predictions[i, :m] = predictions[i]
        padded_targets[i, :m] = targets[i]
        padded_scores[i, :m] = candidate_scores[i]
        padded_valid_mask[i, :m] = True
        padded_candidate_p[i, :m] = candidate_points[i]
        padded_part_tiers[i, :m] = part_tiers[i]
        padded_part_ids[i, :m] = part_ids[i]

    np.savez_compressed(
        OUT_PATH,
        instance_ids=np.array(instance_ids, dtype=object),
        predictions=np.array(predictions, dtype=object),
        candidate_scores=np.array(candidate_scores, dtype=object),
        targets=np.array(targets, dtype=object),
        valid_masks=np.array(valid_masks, dtype=object),
        candidate_p=np.array(candidate_points, dtype=object),
        part_tiers=np.array(part_tiers, dtype=object),
        part_ids=np.array(part_ids, dtype=object),
        padded_predictions=padded_predictions,
        padded_targets=padded_targets,
        padded_candidate_scores=padded_scores,
        padded_valid_mask=padded_valid_mask,
        padded_candidate_p=padded_candidate_p,
        padded_part_tiers=padded_part_tiers,
        padded_part_ids=padded_part_ids,
        baseline_name=np.array("Where2Act", dtype=object),
        exp_name=np.array(EXP_NAME, dtype=object),
        model_epoch=np.array(MODEL_EPOCH),
        description=np.array(
            "Official Where2Act pretrained actionability model. "
            "P(action) is evaluated at each MicroReach candidate point and "
            "broadcast to 24 pose queries as R(p). Pose-Aware Recall is N/A.",
            dtype=object,
        ),
    )

    print("[done] saved:", OUT_PATH)
    print("[done] instances:", n)
    print("[done] max_M:", max_m)


if __name__ == "__main__":
    main()
