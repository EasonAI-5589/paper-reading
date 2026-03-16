[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览
实验围绕四个 Research Questions 展开：RQ1 奖励精度、RQ2 策略性能/效率/泛化、RQ3 多视角融合的必要性、RQ4 Dopamine-RL 框架的必要性。覆盖 10 个仿真 + 8 个真实世界任务。

---

We evaluate Dopamine-Reward with GRM and Dopamine-RL on both simulation and real-world robotic platforms, covering a broad range of manipulation skills and deployment scenarios. This section summarizes our empirical findings and is organized around four questions:

- RQ1: How accurate is the GRM at perceiving task progress compared to VLMs and existing reward models?
- RQ2: How does Dopamine-RL perform in success rate, sample efficiency, and generalization against strong BC and RL baselines?
- RQ3: How critical is Multi-Perspective Progress Fusion for final performance?
- RQ4: How important is the Dopamine-RL framework for turning reward modeling into practical policy learning?

> 💡 **四个 RQ 的详细解读**：
>
> | RQ | 验证什么 | 对应方法 | 对应表格/图 |
> |----|---------|---------|-----------|
> | RQ1 | GRM 奖励精度：进度预测排序和真实进度有多一致 | Dopamine-Reward (3.1) | Table 1, Table 2 |
> | RQ2 | 端到端策略性能：成功率、样本效率、泛化 | 完整框架 | Table 3, Table 4 |
> | RQ3 | 三视角融合是否必要（消融） | 3.1.2 Multi-Perspective Fusion | Table 5 |
> | RQ4 | Dopamine-RL 框架各组件是否必要（消融） | 3.2 PBRS + One-shot | Table 5 |
>
> **RQ2 细节**：对比两类 baseline——
> - **BC（Behavior Cloning，行为克隆）**：最简单的模仿学习，直接用监督学习让模型模仿人类示教动作（状态→动作），不需要 reward，也不和环境交互。优点是简单稳定，缺点是上限受限于示教数据，且容易过拟合视觉表面特征
> - **RL baselines**：用其他 reward 设计（如 sparse reward、ConRFT）做 RL 训练
>
> RQ2 从三个维度评估：success rate（能不能完成任务）、sample efficiency（需要多少次交互才能学会）、generalization（换个场景还能不能 work）

---

## 4.1. Accurate Task Progress Perception (RQ1)

### 4.1.1. Video Frame Rank-Correlation

To quantitatively assess task progress perception, we follow the evaluation methodology of GVL [37] and measure the Value-Order Correlation (VOC) between the GRM's predicted progress and the ground-truth chronological order of shuffled frames. A higher VOC score ([-1, 1]) indicates a better understanding of temporal progress. We evaluate on a diverse suite of eight datasets spanning real-world robotics (DROID [24], AGIBOT-World [7], RoboBrain-X [17]), simulation (Libero [30], RoboCasa [40], RoboTwin2.0 [9]), and egocentric human manipulation (EgoDex [19]). To test robustness to temporal granularity, we test under three distinct sampling strategies: Sparse (S) using only major keyframes, Medium (M) using uniform samples between keyframes, and Dense (D) using uniform samples across the entire trajectory.

> 💡 **4.1.1 详解**：
>
> **VOC（Value-Order Correlation）是什么**：把视频帧打乱顺序，让 reward model 给每帧打进度分，然后看模型打分的排序和真实时间顺序有多一致。VOC ∈ [-1, 1]，1 = 完美一致，0 = 随机，负数 = 反着来。本质上测的是"模型能不能分辨哪个状态比哪个更接近完成"。
>
> **S/M/D 三种采样密度**：
> - **S（Sparse）**：只抽关键帧（如"抓起""放下"这些节点），帧数少，排序简单
> - **M（Medium）**：关键帧之间均匀插值，帧数适中
> - **D（Dense）**：整条轨迹均匀密采，帧数多，相邻帧差异很小，排序最难
>
> 测三种密度是为了验证鲁棒性——好的 reward model 不管帧抽得多密都能排对。
>
> **数据集（7 个，跨三类场景）**：
> - 真实机器人：DROID [24]、AGIBOT-World [7]、RoboBrain-X [17]
> - 仿真：LIBERO [30]、RoboCasa [40]、RoboTwin2.0 [9]
> - 人类自我视角：EgoDex [19]
>
> **对比方法**：GVL [37]（基于 VLM 的 reward model）、VLAC [59]（VLM-as-classifier）

---

![Table 1](../images/f50260d70e05b13d26875b20c3a6809f48c681e1ed0b3a91d1893264147898f8.jpg)
*Table 1. Video Frame Rank-Correlation (VOC) on Diverse Datasets. GRM 在所有 benchmark 和采样密度上都取得最优。*

> 💡 **Table 1 批读**：GRM-8B Multi-View 在 Average 上达到 0.96（S）/0.96（M）/0.94（D），碾压 GVL（0.20）和 VLAC（0.29）。Baselines 在 Dense 采样下明显退化，GRM 保持稳定。多视角比单视角高约 4%。

We compare our multi-view and single-view GRM against four state-of-the-art reward models: GVL [37], and VLAC [59]. As shown in Table 1, our multi-view GRM consistently achieves the highest VOC scores across all seven datasets and all sampling strategies. The performance of baseline models tends to degrade as sampling becomes denser, indicating a struggle with fine-grained temporal distinctions. In contrast, our model maintains exceptionally high performance, highlighting the robustness of our hop-based learning formulation and multi-perspective fusion. The performance gap is most significant in complex, long-horizon tasks (e.g., LIBERO [30], RoboBrain-X [17]), where our model's ability to accurately contextualize progress is paramount.

---

### 4.1.2. Task Completion Judgment

To assess the GRM's ability to make high-level judgments about task outcomes, we follow the protocol from SARM [8]. We collect 60 real-world rollouts for each of three tasks (stacking blocks, folding T-shirt, clearing desktop), with 20 successful (SE), 20 partially successful (PSE), and 20 failed (FE) episodes.

> 💡 **4.1.2 详解**：
>
> **SARM [8] 协议**：SARM（Self-Aligned Reward Model）是另一篇论文，这里借用了它的评测方法——给模型看一段 rollout 视频，让它判断任务结果属于三类中的哪一类（成功 SE / 部分成功 PSE / 失败 FE）。
>
> **测试任务**：3 个真实世界任务（叠积木、折 T 恤、清理桌面），每个任务 60 条 rollout（20 成功 + 20 部分成功 + 20 失败），总共 180 条。
>
> **为什么只比 VLM 和 RM，没有 VLA？** 因为这个实验测的是**感知能力**（能不能判断任务做到什么程度），不是控制能力。VLA 是用来输出动作的，不做任务完成判断。能做判断的只有通用 VLM（GPT-5、Gemini）和专用 RM（GVL、VLAC、GRM）。

---

![Table 2](../images/e2a86aaa6081874e6ad66918e7c2384ba99bb38c175c2730cfa42ee7620e10f1.jpg)
*Table 2. Task Completion Classification Accuracy (successes out of 60). GRM 比专门的奖励模型和大型通用模型都更准确。*

> 💡 **Table 2 批读**：GRM-8B Multi-View 达到 92.8%，超越 GPT-5（83.9%）和 Gemini-2.5-Pro（81.1%），说明专用微调 VLM 在任务理解上强于通用大模型。GVL/VLAC 低于 40%，几乎不可用。多视角比单视角高 9%——遮挡时一个视角丢失信息，另一个视角可以补上。

---

## 4.2. Performance, Efficiency, Generalization (RQ2)

We now evaluate the Dopamine-RL framework across 10 simulation tasks (from LIBERO [30] and RoboTwin2.0 [9]) and 8 real-world tasks, whose task setups and hardware platform are illustrated in Figure 4. In our simulation experiments, Dopamine-RL is evaluated under two distinct configurations: one leveraging the PPO [46] algorithm alongside the OpenVLA-OFT [26] model, and the other integrating the ReinFlow [61] algorithm with the $`\pi_0`$ [6] model. For real-world implementations, we pair Dopamine-RL with Cal-QL [39] and we employ a Human-in-the-Loop setup where we use just one single human demonstrations to adapt the GRM.

> 💡 **实验设置**:
> | 环境 | 任务数 | RL 算法 | Policy 架构 |
> |------|--------|---------|------------|
> | 仿真 (LIBERO + RoboTwin2.0) | 10 | PPO / ReinFlow | OpenVLA-OFT / $`\pi_0`$ |
> | 真实世界 | 8 | Cal-QL | — |
>
> Baselines: BC (50 demos), PPO+Sparse, ConRFT

---

![Table 3](../images/5076e53db6a62ef3c7ccf711d02927d5477c8b9283dbe8d4ec6d068d5ca3878c.jpg)
*Table 3. Policy Performance and Sample Efficiency. Dopamine-RL 用更少的 rollout 达到更高的成功率。*

> 💡 **Table 3 批读**：
>
> **表中没有标注但需要知道的信息**（来自正文 + Appendix C）：
> - 仿真任务：LIBERO-Goal 10 个任务，跑了两套配置——PPO + OpenVLA-OFT（Setting 1）和 ReinFlow + π₀（Setting 2），Table 3 未区分，Appendix C 中 ReinFlow+π₀ 明确报告了 81% 成功率
> - 仿真 RL+Sparse baseline：PPO [46] + sparse reward
> - 真实世界 RL+Sparse baseline：ConRFT [12]（Cal-QL [39] 的变体 + sparse reward）
>
> **表中方法解释**（三行对比的是同一个 VLA 上不同训练方式的差异）：
> - **BC (50 demos)**：行为克隆，用 50 条人类示教直接监督学习，不和环境交互（所以 Rollout 是横杠）
> - **RL + Sparse**：用同样的 RL 算法和 VLA，但 reward 只有完成任务时的 $r_{gold} = 1$，没有 GRM 提供的 dense 引导
> - **Dopamine-RL**：完整框架，同样的 RL 算法和 VLA，但用 GRM dense reward（PBRS）
>
> **Rollout (#)** = 达到该成功率需要的训练交互次数（agent 从头到尾尝试一次任务 = 1 rollout），衡量样本效率。
>
> **核心结论**：Dopamine-RL 在真实世界上 95.2% vs Sparse RL 68.0%，且只需 150 次 rollout（Sparse 需要 183 次）。真实世界提升远大于仿真，说明 dense reward 在探索困难、rollout 昂贵的场景下价值更大。

---

![Table 4](../images/b5a75f90d517491ab44e5bee8b18853635ba3ec41023b40fed70cc57324eb65a.jpg)
*Table 4. Generalization Performance Breakdown: ID vs. OOD. 对比 BC 和 Dopamine-RL 在 OOD 条件下的鲁棒性。*

> 💡 **Table 4 批读**:
>
> | 对比项 | 数据 | 说明 |
> |-------|------|------|
> | Ours ID 成功率 | 19-20/20 | 接近完美 |
> | Ours OOD 平均下降 | 8.3-19.3% | 仅轻微退化 |
> | BC OOD 平均下降 | 50-60% | 严重退化 |
> | Circuit: Ours vs BC (Layout) | 19/20 vs 1/20 | 布局变化下差距最大 |
>
> **三种 OOD 条件**:
> - Object Change（物体属性变化）
> - Layout Change（工作空间布局变化）
> - Background Change（背景视觉变化）
>
> **核心结论**: Dopamine-RL 学到了任务语义而非表面视觉特征，BC 严重过拟合

---

## 4.3. Ablation Studies (RQ3 & RQ4)

Finally, we conduct a series of ablation studies on a representative subset of three real-world tasks to validate the key design choices in the Dopamine-Reward framework.

---

![Table 5](../images/ea0427b24be246e423612dd7fb60bc658ca23a18a432c86ce050e0f0618e1b4b.jpg)
*Table 5. Ablation Study Results (Average Success Rate %). 每个组件都对最终性能至关重要。*

> 💡 **Table 5 批读 — 最重要的消融实验**:
>
> ### RQ3: 多视角融合
> | 消融项 | 成功率 | 下降 | 含义 |
> |-------|--------|------|------|
> | Full Framework | **85.0%** | — | 基线 |
> | w/o Fusion (Incremental Only) | 70.0% | -15.0% | 误差累积问题 |
> | w/o Fusion (Forward Only) | 65.7% | -19.3% | 长距离预测不准 |
> | w/o Fusion (Backward Only) | 62.5% | -22.5% | 远离目标时最差 |
>
> **洞察**: 三种视角的重要性排序 Incremental > Forward > Backward（但融合后效果最好）
>
> ### RQ4: Dopamine-RL 框架
> | 消融项 | 成功率 | 下降 | 含义 |
> |-------|--------|------|------|
> | w/o Policy-Invariant Shaping | 41.3% | **-43.7%** | **最关键组件！** |
> | w/o One-shot Adaptation | 63.2% | -21.8% | Zero-shot 在 OOD 任务不够 |
>
> **最 surprising 的发现**: 去掉 PBRS 后性能暴降 43.7%！agent 学到了到达高 reward 状态就停下来（semantic trap 的实证确认）

For RQ3, we ablate the Multi-Perspective Progress Fusion in Dopamine-Reward. Removing fusion and relying on a single progress estimator consistently hurts performance: the incremental-only, forward-anchored-only, and backward-anchored-only variants incur 15.0%, 19.3%, and 22.5% absolute drops, respectively. The incremental-only variant is particularly vulnerable to error drift over long horizons, confirming the importance of combining local and global progress perspectives.

For RQ4, the importance of the Dopamine-RL is evident. Removing policy-invariant reward shaping leads to a massive performance drop of 43.7%. The agent learns to reach "good-enough" states and stagnates, failing to complete the tasks, which confirms the "semantic trap" discussed in Section 3. Besides, relying solely on zero-shot GRM, it occasionally provides incorrect rewards for corner cases in out-of-distribution (OOD) tasks, such as assigning positive rewards to poor actions and negative rewards to good ones. This hinders the convergence of the policy, resulting in a 21.8% drop in success rate.

> 💡 **消融总结**:
> - **PBRS 是最关键设计** (-43.7%)：没有它，整个框架失效
> - **One-shot Adaptation 次之** (-21.8%)：预训练 GRM 有泛化能力但不够精准
> - **三视角融合是锦上添花** (-15~22.5%)：单一视角也能工作，但显著不如融合

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| VOC (GRM-8B MV, Avg Sparse) | 0.96 |
| VOC (GVL, Avg Sparse) | 0.20 |
| 任务完成判断准确率 | 92.8% (GRM) vs 83.9% (GPT-5) |
| 真实世界成功率 | 95.2% (Dopamine-RL) vs 9.8% (BC) |
| 真实世界 Rollout 数 | 150 (~1 hour) |
| OOD 性能下降 | 8-19% (Ours) vs 50-60% (BC) |
| PBRS 消融影响 | -43.7% |

### 核心洞察
1. GRM 在奖励精度上碾压所有 baseline（VOC: 0.96 vs 0.20-0.33）
2. 多视角 > 单视角 > 现有方法（+9% 在 task completion）
3. Dopamine-RL 在真实世界的优势远大于仿真——因为真实世界探索更昂贵
4. PBRS 是整个框架的命脉：没有它 = semantic trap = 失败
