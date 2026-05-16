# 阶段二 Where2Act 基线复现报告

## 1. 目标

本阶段的目标是复现 Where2Act 作为 MicroReach 的外部基线，并在当前阶段二评测集上生成可用于统一评测的预测结果。

根据阶段二计划，Where2Act 基线需要完成以下内容：

- clone 官方 Where2Act 仓库；
- 安装并适配推理环境；
- 加载作者发布的 pretrained checkpoint；
- 在当前 47 个有效 Faucet 实例上运行 inference；
- 将 Where2Act 输出的 actionability score `P(action)` 作为非姿态条件的 `R(p)`；
- 将 `R(p)` 广播到 24 个 pose queries，以便与 MicroReach 的 `(p, ψ, g)` 标签格式对齐；
- Pose-Aware Recall 标记为 N/A，因为 Where2Act 不输出姿态条件 `ψ/g`。

最终输出文件为：

```text
eval/results/where2act_predictions.npz
```

该文件供成员 B 的 `eval_main.py` 加入 comparison table。

---

## 2. 官方代码与 checkpoint 获取

本次使用的 Where2Act 官方仓库为：

```text
https://github.com/daerduoCarey/where2act
```

官方 README 中说明，预训练模型需要额外下载 `final_logs.zip`，并解压到 `code/logs/` 目录。我们通过作者提供的资源链接下载了 pretrained checkpoints。

解压后，使用的 checkpoint 为：

```text
/root/autodl-tmp/where2act/code/logs/finalexp-model_all_final-pulling-None-train_all_v1/conf.pth
/root/autodl-tmp/where2act/code/logs/finalexp-model_all_final-pulling-None-train_all_v1/ckpts/81-network.pth
```

对应配置为：

```text
exp_name    = finalexp-model_all_final-pulling-None-train_all_v1
model_epoch = 81
model       = model_3d_legacy
primitive   = pulling
```

官方 checkpoint 能够成功读取，并且模型权重加载结果为：

```text
All keys matched successfully
```

说明 pretrained checkpoint 与官方模型结构匹配。

---

## 3. 环境适配过程

Where2Act 原始代码较老，官方 README 中说明其测试环境为：

```text
Ubuntu 18.04
CUDA 10.1
Python 3.6
PyTorch 1.7.0
SAPIEN 0.8
```

而当前服务器环境为：

```text
RTX 4090
CUDA 12.1
Python 3.10
PyTorch 2.x
SAPIEN 2.x
```

因此直接运行官方 visualization / simulation 脚本会遇到兼容性问题。例如官方 `visu_action_heatmap_proposals.py` 会 import：

```python
from env import Env
```

而 `env.py` 使用了旧版 SAPIEN 的：

```python
OptifuserConfig
OptifuserRenderer
OptifuserController
```

当前 SAPIEN 2.x 中不再提供这些接口，因此官方可视化脚本无法直接运行。

为避免被旧版仿真接口阻塞，本阶段采用更适合当前评测集的方式：

```text
不运行 Where2Act 的仿真环境；
只加载官方 pretrained model；
直接对 MicroReach 已生成的 47 个 .npz 点云进行 offline inference。
```

这仍然满足阶段二基线需求：使用作者发布的 checkpoint，在我们的 47 实例评测集上输出 Where2Act 的 `P(action)`。

---

## 4. PointNet2 依赖适配

Where2Act 的 3D 模型依赖 `Pointnet2_PyTorch`，其中包括 CUDA extension：

```text
pointnet2_ops
```

当前服务器为 RTX 4090，对应 CUDA 架构为：

```text
sm_89
```

原始 PointNet2 编译脚本默认包含旧架构：

```text
compute_37
```

在 CUDA 12.1 下会报错：

```text
nvcc fatal: Unsupported gpu architecture 'compute_37'
```

因此我们将编译架构改为：

```text
TORCH_CUDA_ARCH_LIST=8.9
```

并成功编译安装：

```text
Successfully installed pointnet2-ops-3.0.0
```

此外，PointNet2 原代码继承 `pytorch_lightning.LightningModule`，在当前环境中会触发 `hparams` 只读属性冲突。由于 Where2Act 仅将该模块作为普通 `nn.Module` 使用，我们将其兼容性修改为继承 `nn.Module`，从而成功完成模型实例化和 checkpoint 加载。

最终验证结果：

```text
pointnet2 import ok
Where2Act Network import ok
network created
load_state_dict return: <All keys matched successfully>
```

---

## 5. 推理方法

Where2Act 的 `model_3d_legacy.Network` 提供了 actionability inference 接口：

```python
inference_action_score(pcs)
```

其输入输出为：

```text
input : pcs, shape = (B, N, 3)
output: actionability score, shape = (B, N)
```

在随机点云上测试成功：

```text
score shape: torch.Size([1, 10000])
score min: 0.1235
score max: 0.8804
score mean: 0.6504
```

正式推理时，对于每个 MicroReach `.npz` 实例：

1. 读取 `point_cloud` 和 `candidate_p`；
2. 构造 Where2Act 输入点云；
3. 将 `candidate_p` 放在点云前 `M` 个位置；
4. 从原始 scene point cloud 中采样补齐到 10000 点；
5. 调用 `inference_action_score(pcs)`；
6. 取输出前 `M` 个分数作为每个 candidate 的 `P(action)`；
7. 因为 Where2Act 不预测 `ψ/g`，将每个 candidate 的 `P(action)` 广播到 24 个 queries；
8. 保存为 `(M, 24)` 的预测矩阵。

换言之，Where2Act 在本评测中的定义为：

```text
Where2Act predicts R(p)
MicroReach predicts R(p, ψ, g)
```

因此：

```text
Where2Act prediction for query (p, ψ, g) = P(action | p)
```

Pose-Aware Recall 对 Where2Act 标记为 N/A。

---

## 6. 输出文件格式

最终输出文件：

```text
eval/results/where2act_predictions.npz
```

包含以下字段：

```text
instance_ids
predictions
candidate_scores
targets
valid_masks
candidate_p
part_tiers
part_ids

padded_predictions
padded_targets
padded_candidate_scores
padded_valid_mask
padded_candidate_p
padded_part_tiers
padded_part_ids

baseline_name
exp_name
model_epoch
description
```

其中：

- `predictions`：object array，每个元素为 `(M, 24)`；
- `candidate_scores`：object array，每个元素为 `(M,)`；
- `targets`：对应的 MicroReach `R_geom`，shape 为 `(M, 24)`；
- `part_tiers`：candidate 级 micro / meso / macro；
- `padded_predictions`：padding 后统一 tensor，shape 为 `(47, 15, 24)`；
- `padded_targets`：padding 后 GT，shape 为 `(47, 15, 24)`；
- `padded_valid_mask`：有效 candidate mask，shape 为 `(47, 15)`。

---

## 7. 结果检查

生成后的文件检查结果如下：

```text
keys:
['instance_ids', 'predictions', 'candidate_scores', 'targets', 'valid_masks',
 'candidate_p', 'part_tiers', 'part_ids', 'padded_predictions',
 'padded_targets', 'padded_candidate_scores', 'padded_valid_mask',
 'padded_candidate_p', 'padded_part_tiers', 'padded_part_ids',
 'baseline_name', 'exp_name', 'model_epoch', 'description']

baseline: Where2Act
instances: 47
padded_predictions: (47, 15, 24)
padded_targets: (47, 15, 24)
padded_valid_mask: (47, 15)
first iid: 1011
first pred shape: (10, 24)
first tiers: ['micro' 'micro' 'micro' 'micro' 'micro' 'micro' 'micro' 'micro' 'micro' 'micro']
score range: 0.040323596 0.83731
```

这说明：

- 47 个实例均已完成推理；
- 预测结果与当前标签格式对齐；
- `part_tiers` 已正常保留；
- Where2Act 输出分数非退化；
- 可以交由成员 B 加入 `eval_main.py` 的 comparison table。

---

## 8. 与 MicroReach 的对齐方式

Where2Act 输出的是点级 actionability：

```text
P(action | p)
```

而 MicroReach 标签是姿态条件化可达性：

```text
R_geom(p, ψ, g)
```

因此对齐方式为：

```text
Where2Act(p, ψ, g) = Where2Act(p)
```

即同一个 candidate point 下 24 个 pose queries 使用相同分数。

这种设计体现了 Where2Act 作为非姿态条件 baseline 的限制：

- 可以判断哪里可能可交互；
- 不能判断以什么方向 `ψ`、什么末端构型 `g` 去交互；
- 因此不计算 Pose-Aware Recall；
- 主要用于比较 Micro-mIoU / Meso-mIoU / Macro-mIoU 等位置级或 broadcast 后的指标。

---

## 9. 当前结论

本阶段已完成 Where2Act 基线复现的核心交付：

```text
✅ 官方 Where2Act 仓库 clone 成功
✅ 作者 pretrained checkpoint 下载并加载成功
✅ PointNet2 CUDA extension 在 RTX 4090 上适配成功
✅ 官方模型 `model_3d_legacy` 成功实例化
✅ checkpoint 权重完全匹配
✅ 在 47 个 MicroReach Faucet 实例上完成 offline inference
✅ 输出 `eval/results/where2act_predictions.npz`
```

该基线结果可以进入阶段二 comparison table。Where2Act 在本阶段作为非姿态条件 actionability baseline，用于和 MicroReach 的姿态条件模型进行对比。
