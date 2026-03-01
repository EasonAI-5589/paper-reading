# VLAW 批读笔记 · Experiments

---

## 5. Experiments

三个核心实验问题：
1. 能否学到高保真的 action-conditioned world model（对 contact-rich 任务，能同时建模成功和失败轨迹）？
2. World model 生成的合成数据能否提升 VLA policy 性能？
3. Policy 和 world model 能否通过迭代训练持续改进（multi-task 设置）？

---

### 5.1. Experimental Settings

**Platform & Tasks.** DROID 平台：Franka Panda + Robotiq gripper，两个第三人称相机 + 一个腕部相机。五类 contact-rich 任务：

- **Stacking（积木堆叠）**: "stack block A on block B"，4色积木随机放置
- **Open Book（开书）**: "open the book cover"，4种不同的书
- **Erase Marks（擦白板）**: "erase all marks using a tissue"，1-3条记号笔线
- **Scooping（勺取）**: "transfer some A to the bowl"，花生/糖果/杏仁
- **Drawing（画圆）**: "draw a complete circle on a whiteboard"

![](../images/ee5005add94ba4b0a500ca49033d84c0c4cd88908db1ca32baea4da9d3a925d8.jpg)

*Figure 4: DROID 平台上的 5 类实验任务，均涉及复杂接触或可变形物体。*

> 💡 **任务选择的用意**：这 5 个任务都有明确的 contact 要求，是传统仿真里极难建模的场景（deformable objects、摩擦力依赖），也是现有 world model 最难处理的地方。这个 benchmark 选择直接挑战了"world model 只能做 pick-and-place"的现状。

**Base Models & Setup.** π₀.₅ 作为 VLA，Ctrl-World 作为 world model，Qwen3-VL-4B-Instruct 作为 reward model。每类任务先用 **25 条 expert demo** fine-tune π₀.₅ 作为 base policy。

> 💡 **起点不是零**：base policy 已经在 25 条 demo 上 fine-tune 过了，不是直接用 pretrained π₀.₅。这意味着 base policy 对任务已有基本理解，VLAW 是在此基础上做进一步提升，提升空间相对有限但更有实际意义。

**Hyperparameters.** 每次迭代：50 real rollouts/task → world model 50K steps fine-tune → 生成 500 synthetic trajectories/task → policy 2K steps fine-tune（bs=256）。共 **2 次迭代**。

---

### 5.2. World Model Quality: Action Replay & Policy-in-the-loop

**Action Replay 评估.** 从真实轨迹随机选初始帧，将真实动作序列喂给 world model，auto-regressively 预测 5 秒视频。与两个 baseline 比较：pretrained Ctrl-World、仅用 expert demo fine-tune 的 Ctrl-World。

> 💡 **Replay vs. Rollout**：这里是 open-loop replay（固定动作），比 closed-loop rollout 更简单，误差只来自视频预测。Replay 更适合独立评估 world model 质量。

**两类 Metric：**
- **(1) 视频质量指标**：PSNR、SSIM（像素级）；LPIPS、FID、FVD（感知/分布级）
- **(2) 交互事件混淆矩阵**：对 50 个包含物理接触的片段，标注交互结果（成功/失败），对比 world model 预测与真实结果

> 💡 **混淆矩阵 metric 非常关键**：纯视频质量指标（PSNR 高）不代表 world model 理解了物理交互结果。一个模型可能背景预测很准（PSNR 高），但对"积木有没有叠起来"这类事件完全预测错误。这个额外 metric 是更面向任务的评估，是本文实验设计的亮点。

**Table 1 结果：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ | TP↑ | FN↓ | TN↑ | FP↓ |
|------|-------|-------|--------|------|------|-----|-----|-----|-----|
| Pretrained Ctrl-World | 16.32 | 0.634 | 0.347 | 41.03 | 225.13 | - | - | - | - |
| + Expert Rollout | 19.87 | 0.748 | 0.189 | 12.76 | 99.98 | 28 | 2 | 9 | **11** |
| + Expert + **Online Rollout** | **21.77** | **0.784** | **0.136** | **9.58** | **64.12** | 26 | 4 | **19** | **1** |

> 💡 **关键发现：online rollout 的价值在于消除过度乐观（FP: 11→1）**
> - 只用 expert rollout：FP=11，即 world model 把很多失败的物理交互预测为成功（过度乐观）
> - 加入 online rollout（含 failure）后：FP 降到 1，world model 学会了"谨慎"
> - 代价：TN 从 9 升到 19（好事），但 TP 从 28 降到 26、FN 从 2 升到 4（world model 变保守）
>
> 对 policy 训练来说，**保守型 > 乐观型**：FP 合成数据会把失败轨迹当成功来训练 policy（有害），FN 只是损失了一些有效训练数据（无害但浪费）。

![](../images/0955c0e8c1ef4c7b0c277fa69f2e1c44064559c34ea5ec4a64a38ec4748fd4cf.jpg)

*Figure 6: 相同初始帧 + 相同动作序列，三种 world model 对比。Expert-only 训练的 world model 过度乐观；Online rollout fine-tune 后准确捕捉物理动力学。*

**Policy-in-the-loop Rollout.** Post-trained world model 在 closed-loop 下能稳定跑长达 20 秒的长视野轨迹。

![](../images/9cf345b746d1a60bcd9612769422fb840ff896444b084da6acbe652a37e78daa.jpg)

*Figure 5: Policy 在 world model 内的 closed-loop rollout（20 秒，20 步）。上：铲花生入碗；下：用纸巾擦白板。Post-trained world model 准确捕捉接触动力学。*

> 💡 **20 秒 closed-loop 稳定**：diffusion world model 在 closed-loop 下容易累积误差，能稳定 20 秒说明模型物理预测相当可靠——这是整个"在想象中搜索成功轨迹"能奏效的前提。

---

### 5.3. Policy Improvement: Results & Ablations

**Baselines：**
- **(1) Filtered BC**：只用真实 rollout 中的成功轨迹做 SFT（50 rollouts/task，与 VLAW 相同）
- **(2) DSRL**（Wagenmaker et al. 2025）：在 π₀.₅ 的 noise space 做 online RL（只评估 10 次，"too time-consuming"）

> 💡 **DSRL 只评 10 次**：样本量太少，统计显著性低，对比价值有限。这是实验设计的一个瑕疵。

**Large-scale Rollout Visualization:**

![](../images/6e36a1a4d16192047bb3511e5ad694fbbf558f4d9bc6591bd7a0f2b142fb90ab.jpg)

*Figure 8: 从同一真实初始帧出发，world model 想象出 15 条不同轨迹。真实 rollout 里 robot 失败了，world model 能搜索到成功轨迹作为 policy 训练的监督信号。*

> 💡 **类 HER（Hindsight Experience Replay）的效果**：当 policy 在真实环境中失败，world model 提供"如果做对了应该是怎样"的模拟，给 policy 展示正确示范——帮助 policy 从失败经历中学习。

**Table 2 详细成功率（每任务 50 次评估）：**

| 方法 | Stacking | Wiping | Open Book | Scooping | Drawing | **Mean** |
|------|----------|--------|-----------|----------|---------|---------|
| Base model | 0.62 | 0.46 | 0.56 | 0.44 | 0.22 | 0.460 |
| DSRL | 0.70 | 0.40 | 0.50 | 0.60 | 0.30 | 0.500 |
| Filtered BC-1 | 0.80 | 0.62 | 0.72 | 0.64 | 0.46 | 0.648 |
| Filtered BC-2 | 0.88 | 0.76 | 0.82 | 0.74 | 0.56 | 0.752 |
| **Ours-1** | 0.80 | 0.72 | 0.80 | 0.72 | **0.68** | 0.744 |
| **Ours-2** | **0.92** | **0.86** | **0.86** | **0.92** | **0.78** | **0.868** |

![](../images/813495cee0fb547bbf46ded051c7535bc120010d10f3a4d7cf50c91e05ed6361.jpg)

*Figure 7: 两轮迭代的成功率对比。VLAW 在所有任务上均优于两个 baseline。*

> 💡 **结果解读：**
> - **DSRL 提升有限**（0.46 → 0.50）：多任务 RL 优化困难 + noise space 的表达能力受限。但 10 次评估不够说明问题。
> - **Ours-1 ≈ Filtered BC-2**（0.744 vs. 0.752）：**一轮 VLAW ≈ 两轮 Filtered BC**，说明合成数据可以替代一轮 real rollout，数据效率优势显著。
> - **Ours-2 达到 0.868**，比 Filtered BC-2（0.752）高 **+11.6 pp**：这 11.6% 完全来自 world model 合成数据的贡献。
> - **Drawing 任务最戏剧性**：base 0.22 → Ours-2 0.78，提升 +56 pp，精细运动控制从 world model 里获益最大。

**Ablations（Drawing 任务）：**

![](../images/b33f632f37b7d3500d5dff97058e4a69d7cbe69a41823d686dd69932546d32ed.jpg)

*Figure 9: 消融：减少合成数据量（500→250）或去掉 real rollout 数据，性能均下降。*

> 💡 **消融结论：**
> 1. 合成数据量越多越好（500 > 250），说明还有提升空间（为何不用 1000 条？论文未解释）
> 2. Real rollout 数据不可替代（去掉后性能下降），合成数据与真实数据**互补**而非替代
>
> **缺失的重要消融**：没有"直接用 pretrained Ctrl-World 生成数据（不做 fine-tune）"的 ablation。这个对照能直接量化 world model fine-tune 这一步的单独贡献——是本文最大的实验遗漏。
