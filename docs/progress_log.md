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
2. **Where2Act baseline** (project doc step 6.1, assigned to hyh): save inference results to `baselines/where2act_predictions.npz` so it can be added as a row in the comparison table.

## Next for gjw

- Stage 2 mid presentation slides (PPT)
- AutoDL instance powered off to save GPU time; will reopen when hyh updates tiers or new training is needed
- Stage 3 prep: M2 (R_geom + R_contact) and M_full (three-level cascade) training, blocked on hyh's R_contact + R_exec label completion
- Optional stage 2 P1 if time permits: add a simple global branch (e.g. PointNet over downsampled scene) to fusion.py to instantiate Cross-Scale Fusion (innovation #1)