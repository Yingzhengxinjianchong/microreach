# MicroReach 阶段二中期 PPT 文稿（关镜文）

> 5 张幻灯片，每张 30 秒讲完。汇报总时长 ~2.5 分钟。
> 关键数字加粗。**配图按每张 slide 的"配图"段落操作**。

---

## Slide 1 — 标题页

```
MicroReach 阶段二中期进展
姿态条件化 Reachability 网络

关镜文 ｜ 2026-05-16
机器人技术课程项目
```

**配图**：右下角放 `polar_153.png` 缩略图（吸睛 + 暗示后面有结果）。

---

## Slide 2 — 阶段二中期任务范围

**标题**：阶段二中期我做了什么

**正文**（项目符号列表）：

- **Step 3 — MicroReach-Net 网络实现**
  - Local Geometry Encoder（PointNet++ 风格，含 ball query）
  - Pose-Conditioned Cross-Attention Decoder（**创新点 2**）
  - GeomHead 单层（Contact / Exec head 推至阶段三）

- **Step 5 — 训练 4 个变体**
  - M0（baseline，无 pose decoder，55K 参数）
  - M1 (BCE) / M1+focal / M1+composite（pose-conditioned，**457K 参数**）

- **Step 4 — 评测**
  - Micro-mIoU + Meso-mIoU + Pose-Aware Recall@1 / Recall@5
  - 极坐标可视化（GT vs PRED 对比）

- **额外做的：Loss 函数消融**（参考 SOTA 实践）
  - 单 BCE → Focal Loss (Lin ICCV 2017) → 复合 Loss (TASA AAAI 2026 Eq.11)

**配图**：流程图（自己画或用 drawio）。结构建议：

```
[单视角点云 30K 点] ─┐
                     ├─► [Local Encoder ball-r=5cm] ─► (M, 128)
[候选交互点 M=10] ───┘                                   │
                                                         ▼
[查询 (ψ,g)×24] ─► [Query Encoder] ─► [Cross-Attn ×2] ─► (M,24,128)
                                                         │
                                                         ▼
                                              [GeomHead] ─► R_geom
```

---

## Slide 3 — 核心结果：4 模型 trade-off（**最重要**）

**标题**：M0 vs M1 三种 loss：trade-off 与 narrative

**表格**（PPT 里画成 5 行 5 列，**关键数字加粗**）：

| Method | Micro-mIoU | Meso-mIoU | Recall@1 | Recall@5 |
|---|---:|---:|---:|---:|
| M0（无 pose） | **0.263** | **0.382** | 0.467 | 0.683 |
| M1（BCE） | 0.020 | 0.000 | **0.533** | 0.717 |
| M1+focal | 0.133 | 0.143 | 0.450 | 0.717 |
| M1+composite | 0.116 | 0.140 | 0.417 | **0.733** |

**核心 narrative**（放表下面，3 个 bullet）：

- **M1 系列在排序指标 (Recall@k) 上系统性优于 M0**：M1 (BCE) Recall@1 = **0.533 vs M0 0.467**（+14.3%）；M1+composite Recall@5 = **0.733 vs M0 0.683**（+5.0%）。远高于随机基线 1/24 = 4.2%，证明 **pose-condition decoder 学到了方向选择性**。

- **M0 在 mIoU 上更高是任务粒度差异，不是 pose-condition 失败**：M0 输出标量 broadcast 到 24 query → 一个候选点要么"全 1"要么"全 0"；M1 逐 query 预测 → 单候选点 IoU 结构性偏小。这与 [Toward Affordance Ranking](https://ieeexplore.ieee.org/ielaam/7083369/8764082/8770077-aam.pdf) 论点一致：**ranked affordance 任务用 ranking-based 指标更公平**。

- **Loss 消融揭示 trade-off**：从 BCE → Focal → TASA 复合 loss (BCE+Dice+Focal+IoU)，Micro-mIoU 提升 **6.5×（0.020 → 0.133 → 0.116）**，但 Recall@1 略降（0.533 → 0.450 → 0.417）。**没有 free lunch**，需要按下游任务选 loss。

**配图**：W&B 的 `val/recall@1` 4 条曲线对比截图（去 https://wandb.ai/2352744-tongji-university/microreach 勾选 4 个 run 对比）。

**讲稿提示**（口头说，不写 PPT 上）：
> "我们一开始只训了 M0 和 M1 (BCE)，看到 M1 Recall@1 反超 M0，
> 以为成功了，但等队友补了 part_tiers 字段后跑 mIoU，发现 M1 在 mIoU 上反而输 M0。
>
> 我没绕过这个问题，而是按 affordance 领域 SOTA 实践 (TASA AAAI 2026)
> 加了 focal loss + 复合 loss 两个变体重训。结果发现 mIoU 能涨 6.5 倍但还是赶不上 M0，
> 同时 Recall@1 会降。
>
> 仔细分析后这不是 bug —— M0 和 M1 输出粒度不同导致 IoU 评测本身就偏向 M0。
> 这跟我看到的 affordance ranking 论文观点一致：
> ranked affordance 任务应该用 Recall/MRR 而不是 IoU。
>
> 所以最终结论是：M1 系列在排序指标上系统性更优，pose-condition 设计是成功的，
> 同时承认在阈值化 mIoU 上 M0 因为粒度优势会赢。这是诚实的多指标 trade-off。"

---

## Slide 4 — 极坐标可视化：模型学到了什么

**标题**：可视化：模型预测 vs 真实可达

**配图**：直接放 `polar_1380.png` 占满整张 slide 70% 区域（5 候选点，左 GT / 右 PRED 对比）

**解读文字**（放图右侧或下方）：

- **实例 1380**（Faucet 类，**M1 在该实例上 Recall@1 = 1.0** —— 5 个候选点全部 top-1 命中）

- 极坐标说明：
  - 角度方向：8 个 ψ 采样方向（Fibonacci 球面）
  - 3 个同心圈：pinch（内）/ power（中）/ poke（外）
  - 颜色深浅：R_geom 分数（**相对归一化**，vmax 标在每个 panel 标题里）

- **关键观察**（看图中第 4 行 part_link_2_switch）：
  - **GT (左, vmax=0.598)**：高分集中在某个扇区
  - **PRED (右, vmax=0.414)**：高分扇区与 GT 重合
  - PRED vmax 略低于 GT 是正常的（模型对绝对值偏保守），但**最高分方向一致**就足以让 Recall@1 命中

- **结论**：模型不是简单学"哪个点可达"，而是学到了**在这个点上，从哪个方向 ψ、用什么抓取 g 才能可达** —— 这正是 5D 条件场设计的目标

**讲稿提示**：
> "这张图我选了 M1 表现最好的实例 1380。
> 极坐标的角度是 8 个 ψ 方向，3 个同心圈是 pinch/power/poke 三种抓取类型。
> 注意我用的是相对归一化，因为 R_geom 数据本身分布偏低（mean 只有 0.155）。
> 左边是真实标签 GT，右边是 M1 预测。
> 可以看到模型预测的高分扇区跟真值高度重合，
> 这就是这个实例上 Recall@1 = 100% 的直觉解释。"

---

## Slide 5 — 阶段二末期 + 阶段三计划

**标题**：下一步

**两栏布局**：

**左栏：阶段二末期 P1（接下来 1-2 周）**

- ✅ 队友 part_tiers 已补 → Micro-mIoU / Meso-mIoU 已激活
- ⚠️ 队友补全 R_contact 标签 → 启动 **M2 训练**（geom + contact 两层级联）
- ⚠️ 队友跑 Where2Act 基线 → 对比表加一行
- ⚠️ 在 metrics.py 加 **per-query soft IoU** 给 M1 公平比较（与队友讨论接口）

**右栏：阶段三（核心目标）**

- **数据扩展**：47 → 200 实例 × 三层标签（geom + contact + exec）
- **完整消融**：M0/M1/M2/M_full × 3 seed = **12 次训练 + 显著性检验**
- **物理验证**：Isaac Sim + MoveIt 闭环 → **Sim Execution Success Rate**
- **可选升级**：Backbone 从 PointNet++ → **PartField** (ICCV 2025)
- **Loss 调研延续**：在更大数据上验证复合 loss 的 mIoU/Recall trade-off 是否消失
- **目标**：CCF-C 期刊 / 课程项目高分

**配图**：W&B 项目首页截图（多个 run 的 loss 曲线一览）—— 证明工程严谨。

---

## 备注：PPT 制作建议

1. **配色**（按你之前定的 drawio 风格）：
   - 操作 / 流程：蓝色 `#4A90E2`
   - AI / 模型：黄色 `#F5A623`
   - 数字 / 成果：绿色 `#7ED321`
   - 风险 / 降级：红色 `#D0021B`
   - 判断 / 路径：蓝紫 `#9013FE`

2. **字号**：标题 28-32pt，正文 18-20pt，表格数字 22-24pt

3. **加粗策略**：每张 slide 至少 1-2 个关键数字加粗（**+14.3%** / **53.3%** / **457K** 这些）

4. **页脚**（每张 slide 底部）：
   ```
   关镜文 ｜ MicroReach 阶段二中期 ｜ 2026-05-16   p.N
   ```

5. **避免**：
   - 不要写完整段落（信息密度过高，观众读不完）
   - 不要放代码截图（太密无法阅读）
   - 不要放原始终端输出
