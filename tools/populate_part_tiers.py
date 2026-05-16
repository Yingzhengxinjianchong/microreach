import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

DATA = Path("data")
EVAL = DATA / "eval_set_47_faucet_relative_success.json"

def s(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)

def sem_of(pid):
    return str(pid).split(":", 1)[1] if ":" in str(pid) else str(pid)

def shift_minus_2(pid):
    if ":" not in pid:
        return pid
    link, sem = pid.split(":", 1)
    if not link.startswith("link_"):
        return pid
    try:
        idx = int(link.replace("link_", ""))
    except ValueError:
        return pid
    return f"link_{idx - 2}:{sem}"

with open(EVAL, "r", encoding="utf-8") as f:
    records = json.load(f)

exact = {}
sem_inst = {}
sem_global = defaultdict(list)

for r in records:
    iid = str(r["instance_id"])
    exact[iid] = {}
    sem_inst[iid] = defaultdict(list)

    for p in r["parts"]:
        pid = str(p["part_id"])
        tier = str(p["tier"])
        sem = sem_of(pid)

        exact[iid][pid] = tier
        sem_inst[iid][sem].append(tier)
        sem_global[sem].append(tier)

def majority(xs):
    return Counter(xs).most_common(1)[0][0]

total = Counter()
methods = Counter()
unknown = Counter()

for path in sorted(DATA.glob("*.npz")):
    iid = path.stem
    d = np.load(path, allow_pickle=True)
    arr = {k: d[k] for k in d.files}

    tiers = []

    for pid_raw in arr["part_ids"]:
        pid = s(pid_raw)
        pid2 = shift_minus_2(pid)
        sem = sem_of(pid)

        if pid2 in exact.get(iid, {}):
            tier = exact[iid][pid2]
            method = "shift_minus_2"
        elif sem in sem_inst.get(iid, {}):
            tier = majority(sem_inst[iid][sem])
            method = "instance_semantic"
        elif sem in sem_global:
            tier = majority(sem_global[sem])
            method = "global_semantic"
        else:
            tier = "unknown"
            method = "unknown"
            unknown[(iid, pid, pid2)] += 1

        tiers.append(tier)
        methods[method] += 1

    tiers = np.array(tiers, dtype="<U7")
    arr["part_tiers"] = tiers
    np.savez_compressed(path, **arr)
    total.update(tiers.tolist())

print("candidate-level tier counts:", dict(total))
print("mapping methods:", dict(methods))
print("unknown count:", sum(unknown.values()))

if unknown:
    print("unknown examples:")
    for (iid, pid, pid2), c in list(unknown.items())[:20]:
        print(iid, pid, "->", pid2, c)
    raise SystemExit(2)

summary = {
    "num_npz": len(list(DATA.glob("*.npz"))),
    "candidate_level_tier_counts": dict(total),
    "mapping_methods": dict(methods),
    "notes": "part_tiers populated by shift_minus_2, instance semantic fallback, then global semantic fallback",
}

with open(DATA / "part_tiers_population_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("saved:", DATA / "part_tiers_population_summary.json")
