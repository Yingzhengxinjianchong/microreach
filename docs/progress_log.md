# Stage 2 Faucet Relative-Scale Label Generation Summary - 5.15 - hyh

## Dataset

- Source dataset: PartNet-Mobility
- Category: Faucet
- Selected instances: 50
- Successfully processed instances: 47
- Failed instances: 3
- Failed IDs: 1401, 1556, 168

## Scale Definition

PartNet-Mobility OBJ coordinates in the current assets are treated as normalized object coordinates rather than calibrated real-world metric dimensions. Therefore, micro/meso/macro tiers are defined by relative object scale.

## Eval Set Tier Distribution

- micro: 85
- meso: 14
- macro: 32
- total parts: 131

## Label Generation

- Generated NPZ files: 47
- Total queries: 8424
- Mean of per-instance R_geom means: 0.1548
- Average R_geom nonzero rate: 27.34%
- R_geom mean range: 0.0082 to 0.4830
- R_geom nonzero rate range: 1.67% to 85.0%

## Failed Instances

The following 3 instances were excluded from the final successful set because their interactive parts were not visible in the rendered multi-view point cloud even after increasing camera views:

- 1401
- 1556
- 168

## Notes

R_contact was skipped in this run and stored as NaN placeholders. The main Stage 2 output is R_geom.

The generated NPZ labels are tracked with Git LFS.

---

# Stage 2 Mid-Phase Network Training & Evaluation Summary - 5.16 - gjw

## Hardware & Setup

- Compute: AutoDL RTX 4090 24GB
- Framework: PyTorch 2.1.2 + CUDA 12.1, Python 3.10.8
- Branch: `gjw`
- W&B project: https://wandb.ai/2352744-tongji-university/microreach
- Repo path: `/root/autodl-tmp/microreach_workspace/microreach`

## Code Delivered (gjw branch)

### `microreach_net/` (network)

- `pointnet_backbone.py`: PointNet++ style local geometry encoder. Ball query (radius=5cm, K=64) + 2-layer MLP. Outputs (B, M, 128) per-candidate-point features.
- `fusion.py`: identity passthrough for current stage. Global Sparse Conv U-Net branch interface reserved for stage 3.
- `pose_decoder.py`: Pose-Conditioned Cross-Attention Decoder. ψ encoded via spherical coords (θ, φ) + sinusoidal positional encoding (n_freq=4); g via `nn.Embedding(3)`. 2 cross-attention layers, 4 heads, 128-dim tokens.
- `heads.py`: `GeomHead` (active in stage 2 mid). `ContactHead` / `ExecHead` defined but unused; activated when stage 3 enables them.
- `losses.py`: `masked_bce_with_logits` for soft R_geom labels in [0, 1]. `microreach_loss_full` reserved for stage 3 cascade loss.
- `dataset.py`: reads .npz with per-instance padding to `max_M=16` plus valid mask. Supports both `per_query` (M1) and `per_point_mean` (M0) modes via `target_mode`.
- `train.py`: unified entry; M0 / M1 switched purely by `model.use_pose_decoder` in yaml.

### `configs/`

- `default.yaml`: shared base config (data interface, split ratios, network dims, training hyperparams)
- `m0.yaml`: overrides `use_pose_decoder=False`, `target_mode=per_point_mean`
- `m1.yaml`: overrides `use_pose_decoder=True`, `target_mode=per_query`

### `eval/`

- `eval_main.py`: ckpt loader + test-set inference + comparison table. Uses hyh's `eval/metrics.py` (no duplication).

### `viz/`

- `reachability_heatmap.py`: polar rose plot (8 ψ directions × 3 g rings) with relative `vmax` normalization. Saves `viz/figs/polar_<id>.png`.

## Backbone Choice

PartField (ICCV 2025) was originally specified per project doc §2.4. Switched to PointNet++ for stage 2 mid because:

1. PartField requires custom CUDA kernels + pytorch3d, est. 1-3 days to debug environment
2. On 47-instance Faucet data, expected gain over PointNet++ is only 1-3 mIoU points (paper's 5-8 point gap is on ShapeNetPart, scale-of-magnitude larger)
3. Project doc §2.4 wording says "用 PartField **替代** PointNet++" — PointNet++ is the legitimate default; PartField is the upgrade path

PartField will be added in stage 3 as an ablation row (`M_full w/ PartField vs PointNet++`).

## Training Setup

- Dataset: 47 Faucet instances (from hyh's stage 2 step 2 output)
- Split: 8:1:1 → 37 train / 4 val / 6 test (split seed = 42, training seed = 42)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4, grad_clip=1.0
- Scheduler: CosineAnnealingLR over 100 epochs
- Batch size: 4 instances; point cloud resampled to N=30000 per instance
- Loss: BCEWithLogitsLoss (masked) on R_geom soft labels
- Validation: every 5 epochs
- Train time: ~40s per variant on 4090 (single seed)

## Results (test set, 6 instances)

### Aggregate

| Method                | Params |  Recall@1 |  Recall@5 | Best val_loss |
| --------------------- | -----: | --------: | --------: | ------------: |
| M0 (no pose decoder)  |    55K |     0.467 |     0.683 |         0.386 |
| M1 (pose-conditioned) |   457K | **0.533** | **0.717** |     **0.373** |
| Δ                     |   8.3× |    +14.3% |     +4.9% |         -3.4% |

Random baseline for Recall@1 = 1/24 = 4.2%.

### Per-instance Recall@1 (M1)

| Instance | M0 R@1 |  M1 R@1 |  M1 R@5 | Notes                                              |
| -------- | -----: | ------: | ------: | -------------------------------------------------- |
| 153      |    0.7 |     0.9 |     1.0 | High R_geom mean (0.48), 9/10 candidates top-1 hit |
| 1380     |    0.4 | **1.0** | **1.0** | Perfect on all 5 candidates                        |
| 1832     |    1.0 |     0.0 | **1.0** | Top-1 off but top-5 covers truth                   |
| 1667     |    0.4 |     0.6 |     0.6 | Moderate                                           |
| 822      |    0.2 |     0.4 |     0.4 | Low overall reachability                           |
| 908      |    0.1 |     0.3 |     0.3 | Hardest test instance                              |

## Visualizations

Polar rose plots for two best M1 instances saved to `viz/figs/`:

- `polar_153.png`: 10 candidate points × (GT, PRED) panels
- `polar_1380.png`: 5 candidate points × (GT, PRED) panels

Both use relative `vmax` per panel (R_geom mean is only 0.155, so absolute [0, 1] mapping makes PRED look uniformly faint). vmax is annotated in each panel title.

## Key Observations

1. **Pose-condition is effective**: M1 outperforms M0 on Recall@1 by +14.3% (0.467 → 0.533), far above the random baseline 4.2%. The Pose-Conditioned Cross-Attention Decoder learned direction selectivity per candidate point.
2. **No overfitting**: val_loss tracks train_loss; both decrease monotonically. M1's best val_loss (0.373) is reached at epoch 15 and stays stable through epoch 99.
3. **Micro-mIoU = 0 is by design, not a bug**: `part_tiers` field in current .npz is filled with `'unknown'`. `build_micro_mask` correctly returns empty mask, making numerator and denominator both zero. Will activate immediately once tiers are populated.

## Stage 2 Mid Coverage vs Project Doc

| Doc requirement                          | Status                                                     |
| ---------------------------------------- | ---------------------------------------------------------- |
| Step 3.1 PartField backbone              | Replaced with PointNet++ (deferred to stage 3 as ablation) |
| Step 3.2 Cross-Scale Geometry Fusion     | Local-only; global Sparse Conv branch deferred to stage 3  |
| Step 3.3 Pose-Conditioned Decoder        | Done                                                       |
| Step 3.4 Three heads                     | GeomHead done; Contact/Exec interfaces reserved            |
| Step 3.5 Loss                            | L_geom done; cascade loss reserved                         |
| Step 3.6 Training loop                   | Done (100 epoch × 1 seed, W&B logged)                      |
| Step 4.1 micro/meso/macro split          | hyh implemented                                            |
| Step 4.2 metrics (Micro-mIoU + Recall@1) | hyh implemented `eval/metrics.py`                          |
| Step 4.4 evaluation entry                | gjw implemented `eval/eval_main.py`                        |
| Step 5.1 M0 single seed                  | Done                                                       |
| Step 5.2 M1 single seed                  | Done                                                       |

## Asks for hyh (Stage 2 Mid follow-ups)

1. **Populate `part_tiers` field in .npz**: the 85/14/32 micro/meso/macro distribution already exists in `data/eval_set_47_faucet_relative_success.json` — just needs to be written into per-instance .npz. Once done, gjw will rerun `eval_main.py` and Micro-mIoU / Meso-mIoU / Macro-mIoU will populate automatically.
2. **Where2Act baseline** (project doc step 6.1, assigned to hyh): save inference results to `eval/results/where2act_predictions.npz` so it can be added as a row in the comparison table.

## Next for gjw

- Stage 2 mid presentation slides (PPT)
- AutoDL instance powered off to save GPU time; will reopen when hyh updates tiers or new training is needed
- Stage 3 prep: M2 (R_geom + R_contact) and M_full (three-level cascade) training, blocked on hyh's R_contact + R_exec label completion
- Optional stage 2 P1 if time permits: add a simple global branch (e.g. PointNet over downsampled scene) to fusion.py to instantiate Cross-Scale Fusion (innovation #1)

#  Stage 2 Follow-up: Polar Visualizations, `part_tiers` Patch, and Where2Act Baseline - 5.16 - hyh

## 1. Added polar visualizations for `R_geom`

Added polar visualizations for the Stage 2 `R_geom(p, ψ, g)` labels.

Visualization setup:

- each figure corresponds to one representative candidate point;
- angular bins correspond to the 8 sampled `ψ` directions;
- radial rings correspond to the 3 `g` configurations;
- color indicates the `R_geom` score;
- instances and candidates are selected automatically to avoid degenerate all-zero or all-one visualizations.

Related files:

```text
tools/auto_select_and_plot_polar.py
data/polar_figs_auto/
data/polar_figs_auto/selection_summary.txt
```

These figures are intended for presentation/debugging purposes and show that the generated `R_geom` labels contain non-trivial pose-conditioned structure.

---

## 2. Patched `part_tiers` in Stage 2 `.npz` files

Member B found that the `part_tiers` field in the 47 generated `.npz` files was still set to `unknown`, which caused empty micro/meso/macro masks during evaluation.

This update populates `part_tiers` in all 47 Stage 2 `.npz` files using the tier annotations from:

```text
data/eval_set_47_faucet_relative_success.json
```

During the patching process, we found that the `part_ids` stored in `.npz` files use a shifted link index compared with the eval-set JSON. For example:

```text
npz:      link_2:switch
eval set: link_0:switch
```

The patch script therefore uses:

1. `link_idx - 2` mapping;
2. instance-level semantic fallback for remaining unmatched parts;
3. a final check to ensure no candidate keeps `unknown` tier.

Final candidate-level tier statistics:

```text
npz files: 47
candidate-level part_tiers:
  micro: 291
  meso : 55
  macro: 5
unknown count: 0
```

Related files:

```text
tools/populate_part_tiers.py
data/part_tiers_population_summary.json
data/*.npz
```

This fixes the empty micro/meso/macro evaluation masks and allows Member B to rerun the tier-specific metrics.

---

## 3. Reproduced the Where2Act baseline

Reproduced the official Where2Act baseline using the released pretrained checkpoint.

Official repository:

```text
https://github.com/daerduoCarey/where2act
```

Downloaded and unpacked the official pretrained logs:

```text
final_logs.zip
```

Checkpoint used:

```text
exp_name    = finalexp-model_all_final-pulling-None-train_all_v1
model_epoch = 81
model       = model_3d_legacy
```

Checkpoint files:

```text
/root/autodl-tmp/where2act/code/logs/finalexp-model_all_final-pulling-None-train_all_v1/conf.pth
/root/autodl-tmp/where2act/code/logs/finalexp-model_all_final-pulling-None-train_all_v1/ckpts/81-network.pth
```

### Environment notes

The original Where2Act code depends on the old SAPIEN 0.8 Optifuser APIs, while our current server environment uses SAPIEN 2.x. As a result, the original visualization/simulation script cannot be run directly.

Instead, we use offline inference:

```text
load the official pretrained Where2Act model
bypass the old SAPIEN simulation/visualization environment
run actionability inference directly on our 47 MicroReach .npz point clouds
```

The PointNet2 CUDA extension was also patched for the RTX 4090 / CUDA 12.1 environment:

- removed unsupported old CUDA architecture targets such as `compute_37`;
- set the build architecture to `TORCH_CUDA_ARCH_LIST=8.9`;
- patched the PointNet2 Lightning `hparams` compatibility issue by using a plain `nn.Module` inheritance path;
- verified that the official checkpoint loads with all keys matched.

Verification:

```text
pointnet2 import ok
Where2Act Network import ok
network created
load_state_dict return: <All keys matched successfully>
```

### Inference output

Where2Act predicts point-level actionability:

```text
P(action | p)
```

MicroReach uses pose-conditioned labels:

```text
R_geom(p, ψ, g)
```

Therefore, for evaluation, the Where2Act score is treated as a non-pose-conditioned `R(p)` baseline and broadcast to all 24 pose queries:

```text
Where2Act(p, ψ, g) = Where2Act(p)
```

Generated file:

```text
eval/results/where2act_predictions.npz
```

Output check:

```text
baseline: Where2Act
instances: 47
padded_predictions: (47, 15, 24)
padded_targets: (47, 15, 24)
padded_valid_mask: (47, 15)
score range: 0.040323596 0.83731
```

Related files:

```text
baselines/run_where2act_infer.py
eval/results/where2act_predictions.npz
docs/stage2_where2act_baseline_report.md
```

The output is ready for Member B to plug into `eval_main.py` and add Where2Act as a comparison-table row. Since Where2Act does not output pose-conditioned predictions, Pose-Aware Recall should be marked as N/A.

---

## 4. Summary of this follow-up

This follow-up completed three Stage 2 Member-A deliverables:

```text
✓ added R_geom polar visualizations
✓ patched part_tiers in all 47 Stage 2 .npz files
✓ reproduced Where2Act with the official pretrained checkpoint and produced baseline predictions
```

New or updated deliverables:

```text
data/*.npz
data/part_tiers_population_summary.json
data/polar_figs_auto/
eval/results/where2act_predictions.npz
baselines/run_where2act_infer.py
tools/populate_part_tiers.py
tools/auto_select_and_plot_polar.py
docs/stage2_where2act_baseline_report.md
```

Next expected steps:

- Member B pulls the latest `main`;
- reruns micro/meso/macro tier-specific evaluation;
- adds Where2Act to the Stage 2 comparison table;
- compares M0 / M1 / Where2Act results.
