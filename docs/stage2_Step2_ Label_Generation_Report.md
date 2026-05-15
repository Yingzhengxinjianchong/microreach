# 阶段二 Step 2 标签生成 Pipeline 报告

## 1. 本阶段目标与实际交付

本报告记录阶段二Step 2：标签生成 pipeline 的实现、调试过程与最终产出。当前阶段的核心目标是为后续网络训练提供可用的姿态条件化几何可达性标签，即：

$$
R_{\text{geom}}(p, \psi, g) \in [0, 1]
$$

其中：

* ($p$)：候选交互点位置；
* ($\psi$)：末端逼近方向；
* ($g$)：末端构型或交互类型；
* ($R_{\text{geom}}$)：由 IK 可达性与连续软评分构成的几何可达性标签。

本阶段最终完成了以下内容：

* 从 PartNet-Mobility 中筛选并构建 Faucet 类别评测集；
* 采用相对物体尺度完成 micro / meso / macro 分档；
* 使用 SAPIEN 渲染点云并采样候选交互点；
* 为每个候选点生成 ($8\psi \times 3g = 24$) 个姿态条件 query；
* 计算连续值 ($R_{\text{geom}}$) 标签；
* 生成 47 个有效实例的 `.npz` 标签文件；
* 将代码、评测集、统计文件和标签数据通过 Git / Git LFS 推送到仓库。

当前阶段重点是完成可训练、可复现、非退化的 `R_geom` 标签数据。`R_contact` 和 `R_exec` 在当前 `.npz` 文件中保留为占位字段，用于保持数据结构一致。

---

## 2. 数据集选择与问题发现

### 2.1 初始选择：StorageFurniture

最初按照项目计划，我们优先尝试使用 PartNet-Mobility 中的 `StorageFurniture` 类别。这个选择的直觉是：柜子、抽屉、储物家具通常具有门、抽屉、把手等结构，似乎适合用来构建微小交互部件的可达性标签。

实际扫描数据后发现，`StorageFurniture` 虽然实例数量较多，但当前数据版本中的语义粒度较粗。许多实例只标注到：

* `furniture_body`
* `rotation_door`
* `translation_door`
* `drawer`

而缺少真正细粒度的：

* `handle`
* `knob`
* `button`
* `latch`
* `pull`
* `grip`

这导致我们最初虽然能筛出大量 `StorageFurniture` 实例，但这些实例中的交互部件更多是门板、抽屉整体或家具主体，而不是我们想重点研究的“微小交互部件”。

### 2.2 StorageFurniture 的分档问题

在 `StorageFurniture` 上运行早期分档脚本后，出现了明显异常：几乎所有 part 都被划为 `macro`，`micro` 数量为 0。进一步检查输出后发现：

* part 的 `obb_max_edge_m` 通常接近 0.5 到 1.0；
* 如果直接按照绝对物理尺度阈值 5 cm / 15 cm 分档，绝大多数 part 都会被判为大尺度部件；
* 语义中也确实没有足够的真实 micro-interaction 关键词。

这说明问题不是代码简单报错，而是数据本身与当前任务目标不匹配：`StorageFurniture` 在该数据版本中更适合粗粒度铰接部件分析，而不适合作为阶段二中期验证“微小交互部件可达性”的主要类别。

### 2.3 切换到 Faucet

之后我们检查了 PartNet-Mobility 中其他类别的语义标注，发现 `Faucet` 更适合当前阶段目标。`Faucet` 类别中常见的 part 语义包括：

* `switch`
* `stem`
* `spout`
* `faucet_base`

其中 `switch` 和 `stem` 更接近我们希望分析的微小或中小尺度交互部件。相较于 `StorageFurniture`，`Faucet` 的优点是：

1. 实例数量足够；
2. 交互语义更清晰；
3. 部件之间相对尺度差异明显；
4. 更容易在阶段二中验证姿态条件化可达性的有效性。

因此，最终阶段二 Step 2 使用 `Faucet` 类别构建评测集。这个调整体现了我们在工程实现中根据数据实际质量进行问题定位和方案修正的过程。

---

## 3. micro / meso / macro 分档方式修正

### 3.1 绝对尺度分档的问题

原始设计中，micro / meso / macro 按照物理尺寸划分：

* micro：部件 OBB 最大边不超过 5 cm；
* meso：5 cm 到 15 cm；
* macro：大于 15 cm。

但在实际检查 PartNet-Mobility 的 OBJ 文件和 URDF 后，我们发现当前数据中的坐标不能直接当作真实米制尺寸使用。例如在 Faucet 实例中，一些开关或水龙头部件的 OBB 最大边会达到 0.5 甚至 1.0 左右。如果直接解释为米，这显然不符合真实水龙头开关尺度。

进一步检查 URDF 中的 mesh 引用和 OBJ 顶点范围后，我们判断：当前数据更适合被视为归一化物体坐标，而不是已经严格标定到真实物理单位的尺寸。继续使用 5 cm / 15 cm 的绝对阈值会导致所有或绝大多数部件被划为 macro，从而无法服务于 micro-part 分层评测。

### 3.2 改为相对物体尺度

为解决这个问题，我们将阶段二的分档方式改为相对尺度。具体做法是：对每个实例，先计算整件物体的 AABB 体积，再计算每个 part 的 AABB 体积，使用二者的比例作为分档依据：

$$
\text{relative scale} = \frac{\text{part AABB volume}}{\text{object AABB volume}}
$$

当前阶段使用的阈值为：

* micro：relative volume ≤ 0.15；
* meso：0.15 < relative volume ≤ 0.45；
* macro：relative volume > 0.45。

这样做的好处是：

1. 不依赖数据集坐标是否为真实米制单位；
2. 能反映同一物体内部不同部件的相对大小；
3. 更适合当前阶段的 Faucet 类别；
4. 保持了 micro / meso / macro 分层评测的思想。

### 3.3 分档结果

在最终 Faucet 评测集中，分档结果为：

| tier  |  数量 |    比例 |
| ----- | --: | ----: |
| micro |  85 | 64.9% |
| meso  |  14 | 10.7% |
| macro |  32 | 24.4% |
| total | 131 |  100% |

这个结果比最初 StorageFurniture 的 100% macro 明显合理得多。尤其是 `switch` 和部分 `stem` 主要进入 micro / meso 档，符合我们对 Faucet 微小交互部件的直觉。

需要强调的是：本次修改没有改变 eval set 的 JSON schema，只改变了 `tier` 的判定方式。因此后续脚本仍然可以按原来的字段读取：

* `instance_id`
* `category`
* `parts`
* `part_id`
* `obb_max_edge_m`
* `tier`
* `n_points`
* `point_cloud_ratio`

---

## 4. 标签生成 Pipeline

最终标签生成 pipeline 包含以下步骤：

1. 读取 Faucet eval set；
2. 使用 SAPIEN 加载 PartNet-Mobility URDF；
3. 多视角渲染 RGB-D / segmentation；
4. 从 Position texture 反投影得到世界坐标点云；
5. 根据语义筛选交互 part；
6. 在交互 part 可见点上 FPS 采样候选点；
7. 对每个候选点生成 ($8\psi \times 3g = 24$) 个 query；
8. 使用 Franka Panda 机器人模型计算 ($R_{\text{geom}}$)；
9. 保存每个实例的 `.npz` 标签文件。

### 4.1 SAPIEN 渲染与点云生成

在实现过程中，我们遇到了两个主要工程问题。

第一个问题是 Vulkan 初始化失败。SAPIEN 的 renderer 依赖 Vulkan，而服务器环境中虽然 CUDA 可以看到 GPU，但 Vulkan ICD 配置最初没有稳定选中 NVIDIA 设备。我们通过指定 NVIDIA ICD：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XDG_RUNTIME_DIR=/tmp/runtime-root
```

解决了 SAPIEN renderer 初始化失败的问题。

第二个问题是相机坐标系。最初相机渲染出的 segmentation 全为 0，点云为空。检查 SAPIEN 的 camera pose 约定后，我们修正了相机位姿生成方式：

* SAPIEN camera local +X 为 forward；
* local +Y 为 left；
* local +Z 为 up；
* Position texture 的前三维再通过 `camera.get_model_matrix()` 转为世界坐标。

修正后，渲染得到的 segmentation 和点云均正常。最终每个实例点云约为 30000 点。

### 4.2 交互候选点采样

交互 part 通过语义关键词筛选。对于 Faucet，主要关注：

* `switch`
* `stem`
* `spout`

同时保留少量 `faucet_base` 作为宏观对照部件。每个可见交互 part 上使用 FPS 采样候选点。候选点数量随实例和可见 part 数量变化，常见为 5、10 或 15 个。

部分实例失败的主要原因是：交互 part 虽然在语义文件中存在，但在当前多视角渲染点云中没有可见点，导致无法采样候选点。

### 4.3 Query 采样

每个候选点生成 24 个 query：

$$
8\psi \times 3g = 24
$$

其中：

* ($\psi$)：8 个球面采样的末端逼近方向；
* ($g$)：3 类末端构型或交互模式；
* query 的最后一维为 ($(\psi_x, \psi_y, \psi_z, g_{idx})$)。

最终每个实例中的 `queries` 形状为：

```text
(M, 24, 4)
```

其中 (M) 是该实例中的候选点数量。

### 4.4 R_geom 计算

`R_geom` 的目标是衡量给定 $(p, \psi, g)$)下，机器人末端是否能够以相应姿态接近目标点，并给出连续可达性分数。

当前实现包含：

1. 根据 $(p, \psi, g)$ 计算目标末端 pose；
2. 先尝试完整 6D IK；
3. 如果完整 IK 失败，使用 position-only IK fallback；
4. 根据关节余量和 Yoshikawa 操作度计算连续软评分；
5. 当前阶段将 collision 作为 warning 统计，不作为 hard reject。

最初直接使用完整 6D IK 和严格碰撞过滤时，`R_geom` 全部退化为 0。通过 debug 发现主要原因是 IK 失败过多，部分 IK 成功样本又被 collision hard reject 全部归零。为避免阶段二标签无法训练，我们将当前版本调整为：

* 保留完整 IK；
* 增加 position-only fallback；
* collision 暂时只记录 warning，不将样本直接归零；
* 使用连续软评分保证 `R_geom` 分布非退化。

这种处理使当前阶段能够生成有效训练信号。严格碰撞判定可以在后续更完整的物理标定后进一步加强。

---

## 5. 最终结果统计

最终从 50 个 Faucet 候选实例中成功处理 47 个，失败 3 个。

### 5.1 实例与 query 统计

| 项目                   |   数值 |
| -------------------- | ---: |
| selected instances   |   50 |
| successful instances |   47 |
| failed instances     |    3 |
| total queries        | 8424 |

失败实例为：

```text
1401, 1556, 168
```

### 5.2 R_geom 分布

| 指标                     |     数值 |
| ---------------------- | -----: |
| mean of instance means | 0.1548 |
| average nonzero rate   | 27.34% |
| min R_geom mean        | 0.0082 |
| max R_geom mean        | 0.4830 |
| min nonzero rate       |  1.67% |
| max nonzero rate       |  85.0% |

从统计结果可以看出，`R_geom` 既不是全 0，也不是全 1；不同实例之间存在明显差异。这说明当前标签具有有效的监督信号，可以用于后续 M0 / M1 训练。

### 5.3 代表性实例

部分实例的 `R_geom` 分布如下：

| instance |   mean | nonzero rate |
| -------- | -----: | -----------: |
| 153      | 0.4830 |        85.0% |
| 1741     | 0.4048 |        70.8% |
| 1488     | 0.3122 |        54.2% |
| 1626     | 0.2347 |        41.7% |
| 1011     | 0.1143 |        20.4% |
| 1052     | 0.0082 |         1.7% |

这些样本覆盖了从较难到较易的不同可达性情况，有助于模型学习区分不同姿态与不同交互点的可达性模式。

---

## 6. 失败实例分析

最终失败的 3 个实例为：

* 1401
* 1556
* 168

失败原因均为：

```text
交互 part 在多视角渲染点云中无可见点
```

这意味着实例语义中存在交互 part，但在当前渲染视角下，对应 link 没有被采样到有效点，因此无法生成 candidate point。

我们已经尝试增加相机视角数量。增加视角后，成功实例数从 44 个提升到 47 个，但上述 3 个实例仍然失败。因此当前阶段将其从成功 eval set 中剔除，并在 `batch_progress.json` 和 summary 文件中保留失败记录。

这种处理方式避免了将空候选点或无效标签混入训练集，同时保留了失败案例，便于后续进一步分析。

---

## 7. 生成文件清单

本阶段主要产出包括：

### 7.1 数据与标签

```text
data/eval_set_200.json
data/eval_set_47_faucet_relative_success.json
data/stage2_faucet_47_summary.json
data/batch_progress.json
data/*.npz
```

其中 `data/*.npz` 共有 47 个文件，由 Git LFS 管理。

每个 `.npz` 文件包含：

* `point_cloud`
* `candidate_p`
* `queries`
* `R_geom`
* `R_contact` 占位字段
* `R_exec` 占位字段
* `part_ids`
* `part_tiers`

### 7.2 主要代码

```text
label_gen/select_instances.py
label_gen/sapien_loader.py
label_gen/sample_queries.py
label_gen/r_geom.py
label_gen/batch_generate.py
eval/micro_meso_macro_split.py
```

### 7.3 文档

```text
docs/stage2_faucet_47_summary.md
docs/stage2_step2_label_generation_report.md
```

---

## 8. 极坐标可视化占位

极坐标图用于展示同一个 candidate point 在不同 (\psi) 和 (g) 下的 `R_geom` 分布。

图中计划采用：

* 角度方向：8 个 $(\psi)$ 方向；
* 径向三层：3 个 (g) 类型；
* 颜色：`R_geom` 分数。

当前由于 GPU 实例暂时不可用，极坐标图留作占位。待实例恢复后，可直接使用已保存的 `.npz` 文件在 CPU 上生成，不需要重新跑标签生成。

占位：

```text
[TODO] Polar Visualization 1
[TODO] Polar Visualization 2
[TODO] Polar Visualization 3
[TODO] Polar Visualization 4
[TODO] Polar Visualization 5
```

后续生成后将保存到：

```text
data/polar_figs_auto/
```

---

## 9. 当前结论

本阶段已经完成Step 2中的核心交付：构建了 Faucet 类别的相对尺度评测集，并生成了 47 个有效实例的姿态条件化 `R_geom` 标签。

最终结果表明：

1. 标签生成 pipeline 可以完整跑通；
2. SAPIEN 渲染、候选点采样、query 采样和 `R_geom` 计算均可复现；
3. `R_geom` 分布非退化，具有训练信号；
4. 标签数据已通过 Git LFS 推送，可供成员 B 开始后续模型训练。

当前产出已经满足启动 M0 / M1 训练所需的数据条件。
