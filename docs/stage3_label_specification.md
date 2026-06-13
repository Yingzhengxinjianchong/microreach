# Stage 3 Label Schema

This document describes the Stage 3 label files for the 200-instance MicroReach eval set.

## Files

Eval-set index:

```text
data/eval_set_200.json
```

Per-instance label files:

```text
data/<instance_id>.npz
```

Dataset composition:

| Category | Count |
|---|---:|
| Faucet | 47 |
| StorageFurniture | 60 |
| Switch | 40 |
| CoffeeMachine | 40 |
| Dishwasher | 13 |
| Total | 200 |

## NPZ Fields

Each `.npz` file contains:

| Field | Shape | Description |
|---|---:|---|
| `instance_id` | scalar | PartNet-Mobility instance id |
| `point_cloud` | `(N, 3)` | Object point cloud |
| `candidate_p` | `(M, 3)` | Candidate interaction points |
| `queries` | `(M, K, 4)` | Pose-conditioned queries |
| `R_geom` | `(M, K)` | Geometry reachability label |
| `R_contact_raw` | `(M, K)` | Raw contact feasibility score |
| `R_contact` | `(M, K)` | Cascade-safe contact label |
| `R_exec_raw` | `(M, K)` | Raw analytic execution feasibility score |
| `R_exec` | `(M, K)` | Cascade-safe execution label |
| `part_ids` | `(M,)` | Candidate part identifiers |
| `part_tiers` | `(M,)` | Candidate part scale tiers |
| `exec_mapping_status` | `(M,)` | R_exec candidate-to-joint mapping status |
| `exec_mapped_child_links` | `(M,)` | URDF child links used by R_exec |
| `exec_mapped_joint_types` | `(M,)` | URDF joint types used by R_exec |

## Cascade Constraint

The final labels enforce:

```text
R_exec <= R_contact <= R_geom
```

`R_contact` is generated from:

```text
R_contact = min(R_contact_raw, R_geom)
```

`R_exec` is generated from:

```text
R_exec = min(R_exec_raw, R_contact)
```

Final validation:

```text
R_exec > R_contact violations: 0 / 43656
R_contact > R_geom violations: 0 / 43656
```

## R_exec Generation

`R_exec_raw` is an analytic execution-feasibility label based on PartNet-Mobility articulation metadata.

The implementation reads:

```text
mobility.urdf
semantics.txt
```

Candidate parts are mapped to movable joints by:

1. exact link match when reliable;
2. semantic-name match plus geometric disambiguation otherwise.

If multiple joints share the same semantic name, the nearest joint axis to the candidate point is selected.

Execution scoring:

| Joint type | Rule |
|---|---|
| `prismatic` | alignment between query direction and joint axis |
| `revolute` / `continuous` | alignment between query direction and local tangential motion |
| static / unmapped | `R_exec_raw = 0` |

For revolute / continuous joints, local tangential motion is computed as:

```text
cross(axis, candidate_p - joint_origin)
```

Final mapping summary:

| Mapping status | Count |
|---|---:|
| `semantic` | 1354 |
| `exact` | 295 |
| `static_or_unmapped` | 170 |

Final joint-type summary:

| Joint type | Count |
|---|---:|
| `revolute` | 815 |
| `prismatic` | 561 |
| `continuous` | 273 |
| static / unmapped | 170 |

Overall label means:

```text
R_exec_raw mean: 0.4704
R_exec mean: 0.0420
```

## Training Usage

Recommended usage:

| Model | Labels |
|---|---|
| M1 | `R_geom` |
| M2 | `R_geom`, `R_contact` |
| M_full | `R_geom`, `R_contact`, `R_exec` |

For M2:

```yaml
data:
  load_contact: true
  load_exec: false
```

For M_full:

```yaml
data:
  load_contact: true
  load_exec: true
```

## Note

`R_exec` is an analytic label, not a full dynamic simulation label.

Isaac Sim / MoveIt closed-loop execution is a downstream evaluation step and is separate from this label schema.
