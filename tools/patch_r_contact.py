import sys
from pathlib import Path
from collections import Counter
import numpy as np

# 把项目根目录加入 sys.path，保证 label_gen 可导入
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from label_gen.r_contact import batch_compute_r_contact

DATA = Path("data")
CONSISTENCY_TOL = 0.05


def s(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


total_rc = Counter()      # 统计各实例 R_contact 均值分布用
total_violations = 0
total_queries = 0
patched = 0
skipped = 0

for path in sorted(DATA.glob("*.npz")):
    iid = path.stem
    d = np.load(path, allow_pickle=True)
    arr = {k: d[k] for k in d.files}

    # 已经 patch 过则跳过（R_contact 不全是 NaN）
    rc_existing = arr["R_contact"].astype(np.float32)
    if not np.isnan(rc_existing).all():
        print(f"[跳过] {iid}（R_contact 已有真值）")
        skipped += 1
        continue

    candidate_p = arr["candidate_p"].astype(np.float32)   # (M, 3)
    queries     = arr["queries"].astype(np.float32)        # (M, 24, 4)
    R_geom      = arr["R_geom"].astype(np.float32)         # (M, 24)
    M           = candidate_p.shape[0]

    # normals / axes 与 load_instance() 默认值一致，全部 [0,0,1]
    part_normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (M, 1))
    part_axes    = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (M, 1))

    R_contact = batch_compute_r_contact(
        candidate_p  = candidate_p,
        queries      = queries,
        part_normals = part_normals,
        part_axes    = part_axes,
        seed         = 42,
    )  # (M, 24) float32

    # 一致性校验：R_contact <= R_geom + tol
    valid_mask   = ~np.isnan(R_geom)
    n_violations = int(np.sum(R_contact[valid_mask] > R_geom[valid_mask] + CONSISTENCY_TOL))
    n_queries    = int(valid_mask.sum())
    viol_rate    = n_violations / n_queries if n_queries > 0 else 0.0

    total_violations += n_violations
    total_queries    += n_queries

    # 写回 .npz（全量字段保留，仅覆盖 R_contact 和 n_consistency_violations）
    arr["R_contact"]                 = R_contact
    arr["n_consistency_violations"]  = np.int32(n_violations)
    np.savez_compressed(str(path), **arr)

    print(
        f"[完成] {iid}  M={M}  "
        f"R_contact mean={R_contact.mean():.3f}  "
        f"violations={n_violations}/{n_queries} ({viol_rate:.1%})"
    )
    patched += 1

print()
print("=" * 55)
print(f"patch 完成：{patched} 个写入，{skipped} 个跳过")
print(f"一致性违反总数：{total_violations} / {total_queries} "
      f"({total_violations/total_queries:.1%})" if total_queries > 0 else "")
print("=" * 55)

# 保存汇总 JSON（与 part_tiers_population_summary.json 同目录）
import json
summary = {
    "num_patched":         patched,
    "num_skipped":         skipped,
    "total_queries":       total_queries,
    "total_violations":    total_violations,
    "violation_rate":      round(total_violations / total_queries, 4) if total_queries > 0 else None,
    "consistency_tol":     CONSISTENCY_TOL,
    "notes": (
        "R_contact computed by batch_compute_r_contact with default normals/axes=[0,0,1], "
        "matching load_instance() behavior in batch_generate.py"
    ),
}
out = DATA / "r_contact_patch_summary.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"saved: {out}")