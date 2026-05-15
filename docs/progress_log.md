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