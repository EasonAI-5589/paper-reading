[← 返回 README](../README.md)

# 3. Method

## 📌 预览
Method 由两大模块组成：(a) Dopamine-Reward——如何构建 GRM 并从中获取精确的进度估计；(b) Dopamine-RL——如何将进度估计转化为理论正确的 RL 奖励信号。核心公式集中在 hop-based 归一化、三视角融合、和 policy-invariant reward shaping。

---

Our approach is designed to address the core challenges in real-world robotic learning by introducing two synergistic components. First, we develop Dopamine-Reward that learns a general-purpose, step-aware process reward from multi-view inputs (Section 3.1). Second, we propose Dopamine-RL, a robust policy learning framework built upon Dopamine-Reward, resolving the theoretical flaws in conventional reward shaping (Section 3.2).

> 💡 **Section 概览**: 两个模块相辅相成——
> - Dopamine-Reward (3.1): 解决"奖励模型不够好"的问题
> - Dopamine-RL (3.2): 解决"奖励塑形有理论缺陷"的问题

---

![Figure 2](../images/7c21c299673b9579b5af5fd6f9d0a57c86fe70ba7fb52fb9096110598119332f.jpg)
*Figure 2. The overview of our method. (a) Dopamine-Reward: GRM 接收任务描述 + 多视角的 initial/goal/before/after 图像，预测 hop。Multi-Perspective Fusion 融合三种预测。(b) Dopamine-RL: One-Shot Adaptation + Policy-Invariant Reward Shaping。*

> 💡 **Figure 2 批读**:
> - **(a) 左半部分 — GRM 架构**: 输入是 6 张图（initial × 2 views + goal × 2 views + before × 2 views + after × 2 views）+ task description，输出一个 hop 值
> - **(a) 右半部分 — Fusion**: 三种推理模式（Incremental/Forward/Backward）取平均
> - **(b) 左 — One-Shot Adaptation**: 仅需一条 demo 做 SFT 微调
> - **(b) 右 — PBRS**: $`r = r_{gold} + \gamma\Phi(s_{t+1}) - \Phi(s_t)`$，telescoping sum 保证 policy invariance

---

## 3.1. Dopamine-Reward Modeling Method

### 3.1.1. General Reward Model (GRM) Construction

The core of our modeling method is to build the GRM, a vision-language model designed to estimate precise task progress. To ensure the model generalizes across diverse embodiments and tasks, we construct a large-scale dataset structured around relative temporal transitions. This section details the three-stage GRM training data construction pipeline, from raw video segmentation to a scientifically rigorous hop-based labeling strategy as follows:

> 💡 **3.1.1 要点预览**: GRM 数据构建的三阶段流水线：
> 1. Step-wise task progress discretization（任务进度离散化）
> 2. Hop-based relative progress normalization（相对进度归一化）
> 3. Sampling strategy and data balancing（采样与平衡）

---

**Step-wise task progress discretization.** We treat task progress itself as the supervision signal. Given raw multi-view video trajectories, we first segment each expert trajectory into sub-tasks using human-annotated multi-view keyframes $`\{K_0, K_1, \ldots, K_N\}`$, where $`K_0`$ is the initial observation, $`K_N`$ is the final success observation, and each $`K_j`$ is a set of synchronized multi-view keyframes. To obtain dense supervision, we perform adaptive sampling within each segment. For a trajectory with $`L`$ frames per view, we set a chunk size $`C`$ to determine the total number of sampled points and distribute them uniformly across the $`N`$ segments. The number of intermediate points $`m`$ within segment $`[K_j, K_{j+1}]`$ is:

$$
m = \left\lfloor \frac{1}{N} \left\lfloor \frac{L}{C} \right\rfloor \right\rfloor .
$$

> 💡 **Stage 1 — 进度离散化**:
> - **输入**: 多视角视频轨迹
> - **人工标注关键帧** $`K_0, K_1, ..., K_N`$（初始 → 各子任务完成点 → 最终成功）
> - **自适应采样**: 在每个 $`[K_j, K_{j+1}]`$ 段内均匀采 $`m`$ 个中间点
> - **公式直觉**: $`L/C`$ 是总采样数，$`1/N`$ 均分到每个 segment
> - **输出**: 状态序列 $`\mathcal{S} = \{s_0, s_1, ..., s_M\}`$，ground-truth 进度 $`\Phi(s_i) = i/M`$

This yields a sequence of states $`\mathcal{S} = \{s_0, s_1, \ldots, s_M\}`$ where each state $`s_i`$ is a set of synchronous multi-view visual observations. We then define the ground-truth global progress as $`\Phi(s_i) = i/M`$.

---

**Hop-based relative progress normalization.** A naive choice is to regress the progress gain $`\Phi_\delta(s_p, s_q) = \Phi(s_q) - \Phi(s_p)`$ between two states, but iterating such predictions accumulates error and can push the reconstructed $`\Phi^{\star}(s)`$ outside $`[0, 1]`$. Instead, we introduce a hop-based formulation that learns relative-relative progress. Each training sample is a tuple $`\mathcal{D}`$ containing a task description $`d_{task}`$, the initial state $`s_0`$, the goal state $`s_M`$, a "BEFORE" state $`s_p`$, an "AFTER" state $`s_q`$, and a hop label $`\mathcal{H}(s_p, s_q)`$ that normalizes the progress from $`s_p`$ to $`s_q`$ relative to the full task span from $`s_0`$ to $`s_M`$. Given $`\Phi(s_p)`$ and $`\Phi(s_q)`$, we define:

$$
\mathcal{H}(s_p, s_q) = \begin{cases} \dfrac{\Phi(s_q) - \Phi(s_p)}{\Phi(s_M) - \Phi(s_p)} & \text{if } q \geq p \text{ (PROGRESS)} \\ \dfrac{\Phi(s_q) - \Phi(s_p)}{\Phi(s_p) - \Phi(s_0)} & \text{if } q < p \text{ (REGRESS)} \end{cases}
$$

> 💡 **Stage 2 — Hop-based 归一化（核心创新）**:
>
> **为什么不直接回归进度差 $`\Phi(s_q) - \Phi(s_p)`$？**
> - 迭代预测会累积误差
> - 重建的 $`\Phi^*(s)`$ 可能超出 $`[0, 1]`$ 范围
>
> **Hop 公式直觉**:
> - **前进 (PROGRESS)**: 进度变化 / 剩余距离 → "你完成了剩余路程的多少比例？"
> - **后退 (REGRESS)**: 进度变化 / 已走距离 → "你倒退了已走路程的多少比例？"
> - **范围**: hop 值 $`\mathcal{H} \in [-1, 1]`$
>
> **关键理论优势**: 通过 hop 迭代重建的 $`\Phi^*(s)`$ **保证** 在 $`[0, 1]`$ 内（Appendix A.1 有数学归纳法证明）
>
> **类比**: 想象你从 A 走到 B，已走 60%。如果有人问"你前进了多少"，naive 方式说"10%"（绝对值），hop 方式说"你走完了剩余 40% 中的 25%"。后者更不容易累积误差。

This dynamically scales the supervision into $`[-1, 1]`$: for forward progress, the change is normalized by the remaining distance to the goal; for regression, by the distance already covered from the initial state. A key theoretical advantage is that, when global progress is reconstructed by iteratively applying predicted hops, the resulting $`\Phi^{\star}(s)`$ is guaranteed to remain strictly within [0, 1]. A detailed proof is provided in Appendix A.1.

---

**Sampling strategy and data balancing.** For each trajectory, we construct a balanced set of hop-based training samples. Continuous hop values are first discretized into $`N_{hop}`$ hop bins. The temporal distance between the "BEFORE" state $`s_p`$ and "AFTER" state $`s_q`$ in each pair is then chosen from $`N_{dis}`$ distance bins within each hop bin, yielding in total $`N_{hop} \times N_{dis}`$ non-trivial transitions. To reduce bias toward static segments, we further introduce an additional fraction $`\alpha`$ of samples explicitly labeled as zero-hop (i.e., $`\mathcal{H}(s_p, s_q) = 0`$), constructed by selecting pairs $`(s_p, s_q)`$ whose progress change is below a small threshold $`\epsilon`$:

$$
|\Phi(s_q) - \Phi(s_p)| \leq \epsilon.
$$

> 💡 **Stage 3 — 数据平衡策略**:
> - **双重分箱**: $`N_{hop}`$ 个 hop 大小的 bin × $`N_{dis}`$ 个时间距离的 bin → 确保各种进度变化幅度和时间跨度都有覆盖
> - **零 hop 样本**: 额外加入 $`\alpha`$ 比例的"无变化"样本（$`|\Delta\Phi| \leq \epsilon`$）→ 防止模型偏向总是预测有进展
> - **最终数据量**: 35M 样本，来自 3,400 小时视频、100K+ 轨迹

Applying this three-stage pipeline yields a dataset of 35M samples from about 3,400 hours of video and over 100K trajectories (see Appendix B). We train the GRM on this corpus to estimate hop-based relative progress between arbitrary state pairs, conditioned on the initial state, goal state, and task description.

> 💡 **3.1.1 小结**:
> | 阶段 | 输入 | 输出 | 关键操作 |
> |------|------|------|---------|
> | 1. 进度离散化 | 多视角视频 | 状态序列 $`\mathcal{S}`$ + $`\Phi(s_i)`$ | 关键帧标注 + 自适应采样 |
> | 2. Hop 归一化 | 状态对 $`(s_p, s_q)`$ | hop 标签 $`\mathcal{H} \in [-1,1]`$ | 相对进度归一化 |
> | 3. 数据平衡 | hop 标签集 | 35M 训练样本 | 双重分箱 + 零 hop 补充 |

---

### 3.1.2. Multi-Perspective Progress Fusion from GRM

To mitigate error accumulation and ensure consistent accuracy, we fuse predictions based on GRM from three complementary perspectives: incremental prediction, forward-anchored prediction, and backward-anchored prediction.

> 💡 **3.1.2 要点预览**: 单一预测方式都有缺陷，融合三种互补视角来消除误差。这是从"单模型多用法"中榨取更多精度的巧妙设计。

---

**Incremental Prediction** first offers a fine-grained, step-by-step assessment. Refer to Equation (2), the predicted global progress $`\Phi_I^{\star}(s_t)`$ is recursively computed from the preceding state's progress $`\Phi^{\star}(s_{t-1})`$ and the predicted hop $`\mathcal{H}^{\star}(s_{t-1}, s_t)`$. Let $`\Delta\Phi_{t-1,t}^{\star}`$ be the estimated progress hop:

$$
\Delta\Phi_{t-1,t}^{\star} = \begin{cases} [1 - \Phi^{\star}(s_{t-1})] \cdot \mathcal{H}^{\star} & \text{if } \mathcal{H}^{\star} \geq 0 \\ \Phi^{\star}(s_{t-1}) \cdot \mathcal{H}^{\star} & \text{if } \mathcal{H}^{\star} < 0 \end{cases}
$$

The incremental progress is then calculated as follow:

$$
\Phi_I^{\star}(s_t) = \Phi^{\star}(s_{t-1}) + \Delta\Phi_{t-1,t}^{\star},
$$

where $`\Phi_I^{\star}(s_t)`$ is accumulated along the trajectory, initialized with $`\Phi^{\star}(s_0) = 0`$.

> 💡 **视角 1 — Incremental（逐步递推）**:
> - **做法**: 从 $`s_0`$ 开始，每步用 GRM 预测相邻状态间的 hop，逐步累加
> - **优点**: 局部精度高，能捕捉细微变化
> - **缺点**: 误差累积——长轨迹上会逐渐偏离真实进度
> - **类比**: 像用步数计量距离，每步的小误差会叠加

---

While this method excels at capturing local dynamics, it is susceptible to the accumulation of prediction errors over long trajectories. To counteract this drift, we introduce extra two global perspectives. **Forward-Anchored Prediction** provides a stable global reference by anchoring to the initial state $`s_{init}`$, where progress is zero:

$$
\Phi_F^{\star}(s_t) = \mathcal{H}^{\star}(s_{init}, s_t).
$$

> 💡 **视角 2 — Forward-Anchored（前锚定）**:
> - **做法**: 直接问 GRM："从初始状态到当前状态完成了多少？"
> - **优点**: 全局稳定，不受中间步骤误差影响
> - **缺点**: 当 $`s_t`$ 距离 $`s_{init}`$ 很远时，单次预测精度下降
> - **类比**: 像用 GPS 测量总距离——稳定但精度受限

---

Conversely, **Backward-Anchored Prediction** is anchored to the goal state $`s_{goal}`$, where progress is one. This approach offers high sensitivity near task completion:

$$
\Phi_B^{\star}(s_t) = 1 + \mathcal{H}^{\star}(s_{goal}, s_t).
$$

> 💡 **视角 3 — Backward-Anchored（后锚定）**:
> - **做法**: 问 GRM："从目标状态到当前状态退了多少？" 然后 1 + (负数) = 当前进度
> - **优点**: 接近任务完成时精度最高（剩余距离短，hop 预测更准）
> - **缺点**: 远离目标时精度下降
> - **类比**: 像从终点倒着量——越近终点越准

---

These three methods offer complementary strengths: local precision (incremental), initial stability (forward), and goal sensitivity (backward). We fuse them via averaging to obtain a robust final progress estimate:

$$
\Phi^{\star}(s_t) = \frac{1}{3}\left(\Phi_I^{\star}(s_t) + \Phi_F^{\star}(s_t) + \Phi_B^{\star}(s_t)\right).
$$

> 💡 **三视角融合**:
> | 视角 | 优势区间 | 弱势区间 |
> |------|---------|---------|
> | Incremental | 局部/短时 | 长轨迹后段 |
> | Forward | 全程稳定 | 远距离精度降 |
> | Backward | 接近完成时 | 远离目标时 |
>
> 简单平均就能互补——Incremental 在开头准，Backward 在结尾准，Forward 全程兜底。
> 消融实验证明（Table 5）：去掉融合后性能分别下降 15%/19.3%/22.5%。

This fusion yields a more accurate and drift-resistant signal, which is critical for the subsequent reward shaping.

---

### 3.1.3. Progress Consistency Checking (Optional)

While the multi-perspective fusion via averaging (Equation (8)) serves as a baseline, its naive application in online RL faces the risk of Out-of-Distribution (OOD) hallucination. Due to the inherent limitations of data coverage, it is impossible for the training set to encompass every corner of the state space. During RL, the policy inevitably explores unseen regions where the reward model may yield spurious high signals, leading to "reward hacking." To address these, we propose a bi-directional consistency checking strategy that leverages consistency as a proxy for reliability, which is motivated by the observation that forward $`\Phi_F^{*}`$ and backward $`\Phi_B^{*}`$ predictions tend to exhibit significant divergence in OOD scenarios or observations, whereas they remain consistent in familiar states.

> 💡 **OOD Reward Hacking 问题**:
> - 训练数据无法覆盖所有状态
> - RL 探索会进入 OOD 区域 → GRM 可能给出虚高的 reward → "reward hacking"
> - **核心洞察**: Forward 和 Backward 预测在 OOD 状态下会**不一致**（因为从两个方向看同一个陌生状态，模型的幻觉模式不同）

---

**Consistency-Aware Weighting.** We first define the mean estimated progress $`\bar{\Phi}^{\ast}(s_t) = (\Phi_F^{\ast}(s_t) + \Phi_B^{\ast}(s_t)) / 2`$. To quantify uncertainty, we calculate a normalized discrepancy metric:

$$
\Delta_{norm}(s_t) = \frac{|\Phi_B^{\ast}(s_t) - \Phi_F^{\ast}(s_t)|}{\bar{\Phi}^{\ast}(s_t) + \epsilon},
$$

where $`\epsilon`$ is a small constant for numerical stability. Normalization by $`\bar{\Phi}^{*}`$ ensures that discrepancies are penalized more heavily during the early stages (where $`\Phi`$ is small), as precise guidance is critical initially. We then derive a confidence weight $`w_t \in (0, 1]`$ using a Gaussian kernel with sensitivity $`\alpha`$:

$$
w_t = \exp\left(-\alpha \cdot (\Delta_{norm}(s_t))^2\right).
$$

> 💡 **一致性权重**:
> - **不一致度** $`\Delta_{norm}`$: Forward 和 Backward 预测差异越大 → 越不可信
> - **除以 $`\bar{\Phi}`$**: 在任务早期（$`\Phi`$ 小）对不一致惩罚更重 → 早期引导很关键
> - **Gaussian kernel**: 将不一致度映射到 $`(0, 1]`$ 的置信权重
> - $`w_t \to 0`$: 不信这个预测；$`w_t \to 1`$: 完全信任

---

**Conservative State Update.** To prevent the policy from exploiting erroneous estimates in OOD scenarios, we employ a conservative update rule for the maintained progress state $`\Phi^{*}(s_t)`$ instead of Equation (8):

$$
\Phi^{*}(s_t) = \Phi^{*}(s_{t-1}) + \frac{w_t}{2} \cdot \left(\bar{\Phi}^{*}(s_t) - \Phi^{*}(s_{t-1}) + \Delta\Phi_{t-1,t}^{\star}\right).
$$

This mechanism acts as a semantic filter: it ignores uncertain updates when $`w_t \to 0`$ (retaining $`\Phi^{*}(s_{t-1})`$) and fully trusts the estimate when consistency is high $`w_t \to 1`$).

> 💡 **保守更新规则**:
> - 当 $`w_t \to 0`$（不可信）：$`\Phi^*(s_t) \approx \Phi^*(s_{t-1})`$，保持上一步的进度不变
> - 当 $`w_t \to 1`$（可信）：正常更新
> - 相当于一个**软门控**：不确定的时候宁可不更新，也不要被错误的 reward 带偏
> - 这个模块是 optional 的，但在 OOD 场景下能显著提升稳定性

---

## 3.2. Dopamine-RL Framework

Building upon Dopamine-Reward with GRM, we further introduce the Dopamine-RL framework, a reinforcement learning pipeline producing high-performance policy stimulated by Dopamine-Reward, featuring three key critical attributes: minimal downstream task effort for rapid progress alignment (Section 3.2.1), fast convergence with policy-invariant guarantees (Section 3.2.2) and seamless integration with diverse RL paradigms (Section 3.2.3).

> 💡 **Dopamine-RL 三大属性**:
> 1. 最小下游成本（one-shot 适配）
> 2. 快速收敛 + policy invariance 保证
> 3. 与任意 RL 算法兼容

---

### 3.2.1. One-shot GRM Adaptation

Dopamine-RL requires only one single human demonstration $`\mathcal{D}_{human}`$ to adapt the pre-trained GRM to novel or high-precision tasks, since the pre-trained GRM has already possessed a broad prior for assessing progress. Given a new task, we minimize the Mean Squared Error (MSE) between its predicted hop value, $`\mathcal{H}_\omega^{\star}`$, and the ground-truth, $`\mathcal{H}_{gt}`$:

$$
\mathcal{L}_{GRM}(\omega) = \mathbb{E}_{(s_p, s_q) \sim \mathcal{D}_{human}} \|\mathcal{H}_\omega^{\star} - \mathcal{H}_{gt}\|_2^2,
$$

where $`\omega`$ represents the GRM's parameters, initialized by pre-trained $`\text{GRM}_{\omega_0}`$. After SFT, we obtain a task-adapted $`\text{GRM}_{\omega_{\star}}`$, poised for efficient reinforcement learning.

> 💡 **One-Shot 适配**:
> - **只需 1 条人类示教**即可适配新任务
> - 原理：预训练 GRM 已有广泛的进度评估先验，只需在新任务上做简单的 SFT 微调
> - 损失函数：MSE on hop predictions
> - 这是 few-shot learning 的极端形式——真正的 one-shot
> - 消融实验（Table 5）：去掉 adaptation（zero-shot）性能下降 21.8%，说明 adaptation 很关键

---

![Figure 3](../images/5a7740b5e8a4698cf528bfaea884c741681778c79e476c372496f304348d867a.jpg)
*Figure 3. Reward profiles on a challenging real-world rollout. 对比人类标注参考 reward、VLAC baseline、和 GRM 在同一轨迹上的输出。*

> 💡 **Figure 3 批读**:
> - **参考信号（Human）**: 绿色线，在错误插入、低位置、错位时给低分，仅在接近成功时给高分
> - **VLAC**: 蓝色线，对错误操作不够敏感，曲线波动大
> - **GRM（Ours）**: 红色线，与人类参考高度吻合——准确惩罚错误操作，只在接近成功时给高 reward
> - 这是 GRM 优于现有方法的直观可视化证据

---

![Figure 4](../images/717701bbd83aed616fe8105db7d53090733500e4fec58e0d28a99c2e9f5c68c0.jpg)
*Figure 4. Real-world tasks and hardware setup. 左：8 个代表性长 horizon 操作任务。右：多视角硬件平台（Pika 遥操作 + ZED 相机）。*

> 💡 **Figure 4 批读**:
> - **8 个任务**: 插入、电路连接、折叠、拾放、组装等——都是 contact-rich 的精细操作
> - **硬件**: Pika 遥操作系统 + 标定的 ZED 相机（提供同步的腕部和第三人称视角）
> - 这些任务的共同特点：需要精确的空间感知 + 遮挡处理 → 正是多视角 GRM 的优势场景

---

### 3.2.2. Policy-Invariant Reward Shaping

A straightforward approach to defining the dense process reward function for policy learning is to use the direct increment of this progress: $`r(s_t, a_t, s_{t+1}) = \Phi^{\star}(s_{t+1}) - \Phi^{\star}(s_t)`$. However, optimizing the standard discounted return, $`J(\pi) = \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t, s_{t+1})]`$, with this reward is mathematically equivalent to maximizing a different objective: $`J'(\pi) \propto \mathbb{E}_\pi[\sum_{t=1}^{\infty} \gamma^{t-1} \Phi^{\star}(s_t) \mid s_0]`$, as detailed in Appendix A.2.

> 💡 **Semantic Trap 的数学证明**:
> - Naive reward: $`r = \Phi(s_{t+1}) - \Phi(s_t)`$（进度增量）
> - 看起来合理：奖励进步，惩罚后退
> - **但** 展开折扣回报后发现：最大化的其实是**各状态进度值的加权和**，而非任务完成
> - 这意味着 agent 会学到"快速到达高进度状态 → 停在那里不动 → 每步都收获高折扣 reward"
> - **这就是 semantic trap**：agent 在 90% 进度处原地打转，不愿冒险去完成最后 10%

This transformed objective creates a perverse incentive: it encourages the agent not to complete the task, but rather to seek and maintain states with high progress values. Consequently, the resulting policy is rewarded for stagnation, preferring a safe, suboptimal state over potentially risky trajectories that lead to true task completion. To resolve the misalignment, we formulate our GRM reward $`r_{GRM}`$ that adheres to three desiderata:

- **Optimal policy invariance.** The optimal policy learned with $`r_{GRM}`$ must coincide with that under the sparse gold reward $`r_{gold}`$ (1 at task completion, 0 otherwise), so shaping guides exploration without changing task objective.
- **Discount consistency**: $`r_{GRM}`$ must be compatible with the standard exponentially discounted return and TD or Bellman updates with factor $`\gamma`$ under a memoryless (Markov) reward assumption.
- **Locality.** At any step $`t`$, $`r_{GRM}`$ is efficiently computable from the single transition $`(s_t, a_t, s_{t+1})`$.

> 💡 **三个设计约束**:
> | 约束 | 含义 | 为什么重要 |
> |------|------|-----------|
> | Policy invariance | 加了 dense reward 后最优策略不变 | 不改变任务目标 |
> | Discount consistency | 兼容标准 $`\gamma`$-折扣 TD/Bellman | 与现有 RL 算法兼容 |
> | Locality | 只需当前 transition 即可计算 | 实时可用 |

---

Adherence to these desiderata uniquely determines the reward structure, we derive the reward from the continuous-time "discounted potential" $`e^{-\lambda t}\Phi^{\star}(s_t)`$. As detailed in Appendix A.4, the natural discrete-time, single-step increment that is consistent with this continuous form is:

$$
F(s_t, s_{t+1}) = \gamma\Phi^{\star}(s_{t+1}) - \Phi^{\star}(s_t),
$$

where $`\gamma = e^{-\lambda h}`$. To enable autonomous learning on real robots without the need for continuous human monitoring, we automate the determination of the sparse outcome reward $`r_{gold}`$. Specifically, we consider the task completed when the estimated progress falls within a close margin of the target (i.e., $`\Phi^{\star}(s_{t+1}) \geq 1 - \delta`$, with $`\delta = 0.05`$). Thus, $`r_{gold} = 1`$ if the completion threshold is met, and 0 otherwise. We add the shaping term $`F`$ to this automated gold-standard reward to define our final reward function:

$$
r_{GRM}(s_t, a_t, s_{t+1}) = r_{gold} + \gamma\Phi^{\star}(s_{t+1}) - \Phi^{\star}(s_t).
$$

> 💡 **Policy-Invariant Reward Shaping（核心公式）**:
>
> $$r_{GRM} = \underbrace{r_{gold}}_{\text{稀疏：完成=1}} + \underbrace{\gamma\Phi^*(s_{t+1}) - \Phi^*(s_t)}_{\text{PBRS 塑形项 F}}$$
>
> **与 naive 方法的关键区别**: 多了一个 $`\gamma`$ 系数！
> - Naive: $`\Phi(s_{t+1}) - \Phi(s_t)`$ → 改变最优策略（semantic trap）
> - PBRS: $`\gamma\Phi(s_{t+1}) - \Phi(s_t)`$ → **不改变**最优策略
>
> **自动成功检测**: $`\Phi^*(s_{t+1}) \geq 0.95`$ 时自动判定任务完成，无需人工监督
>
> 这直接来自 **Ng et al. 1999** 的 PBRS 理论，$`\Phi^*`$ 作为 potential function。

---

This form guarantees policy invariance: the cumulative discounted shaping term $`F`$ forms a telescoping sum that collapses to a constant boundary term depending only on the initial state $`s_0`$. Appendix A.5 shows that the discrete-time sum and the continuous-time integral of the discounted potential's derivative converge to the same constant:

$$
\underbrace{\sum_{t=0}^{\infty} \gamma^t (\gamma\Phi^{\star}(s_{t+1}) - \Phi^{\star}(s_t))}_{\text{Discrete PBRS Sum}} = \underbrace{-\Phi^{\star}(s_0)}_{\text{Boundary Term}}
$$

Since the shaping term telescopes to a state-dependent constant that is independent of the subsequent policy $`\pi`$, the shaped Q-function is simply a state-wise shift of the original one:

$$
Q_{GRM}^{\pi}(s, a) = Q_{gold}^{\pi}(s, a) - \Phi^{\star}(s).
$$

The shift $`-\Phi^{\star}(s)`$ is identical for all actions $`a`$ in a given state $`s`$, so the optimal action remains unchanged:

$$
\arg\max_a Q_{GRM}^{*}(s, a) = \arg\max_a Q_{gold}^{*}(s, a).
$$

> 💡 **Policy Invariance 证明直觉**:
> 1. PBRS 项求和后 telescope（相消）为常数 $`-\Phi^*(s_0)`$
> 2. 因此 $`Q_{GRM} = Q_{gold} - \Phi^*(s)`$——只是加了一个与 action 无关的偏移
> 3. $`\arg\max`$ 不变 → 最优策略不变
> 4. **结论**: dense reward 加速了探索，但不会把 agent 引向错误的目标
>
> 消融实验证实（Table 5）：去掉 PBRS 后性能**暴降 43.7%**——agent 掉入 semantic trap

This matches the standard Potential-Based Reward Shaping (PBRS) framework [41], with the GRM progress $`\Phi^{\star}`$ serving as the potential function.

---

### 3.2.3. Universal RL-Algorithm Compatibility

Dopamine-RL exhibits strong universality, seamlessly integrating with any RL algorithm, encompassing online RL, offline RL, and offline-to-online RL paradigms. It adapts effectively to both value-based methods and gradient-based approaches. By reshaping targeted reward functions to guide agent learning, Dopamine-RL is inherently agnostic to the specific RL algorithm employed. Experimental results confirm this flexibility. In simulations, we deploy under two settings: PPO [46] (Proximal Policy Optimization) algorithm and OpenVLA-OFT [26] model, and ReinFlow [61] algorithm with $`\pi_0`$ [6] model. In real-world settings, we combine with Cal-QL [39] (a offline-to-online Q-learning based RL algorithm) and it also delivers exceptional outcomes. Further details are shown in Appendix C.

> 💡 **RL 算法兼容性**:
>
> | 环境 | RL 算法 | Policy 架构 | 类型 |
> |------|---------|------------|------|
> | 仿真 | PPO [46] | OpenVLA-OFT [26] | Online RL |
> | 仿真 | ReinFlow [61] | $`\pi_0`$ [6] | Online RL (Flow) |
> | 真实世界 | Cal-QL [39] | — | Offline-to-Online |
>
> **关键**: Dopamine-RL 只改 reward 函数，不改 RL 算法本身 → 天然兼容一切 RL 方法。这是 PBRS 框架的固有优势。

---

## 🔖 Section 总结

### 整体 Pipeline 流程表

| 阶段 | 模块 | 输入 | 输出 | 关键操作 |
|-----|------|------|------|---------|
| 1 | 数据构建 | 多视角视频 | 35M hop 标签样本 | 关键帧标注 + 自适应采样 + hop 归一化 |
| 2 | GRM 预训练 | 35M 样本 | 通用 VLM 奖励模型 | 基于 RoboBrain 2.0 的 SFT |
| 3 | One-Shot 适配 | 1 条示教 + GRM | 任务适配的 GRM | MSE 微调 |
| 4 | 进度估计 | 当前观测 + GRM | $`\Phi^*(s_t)`$ | 三视角融合 (+ 一致性检查) |
| 5 | 奖励塑形 | $`\Phi^*(s_t)`$, $`\Phi^*(s_{t+1})`$ | $`r_{GRM}`$ | PBRS: $`r_{gold} + \gamma\Phi^* - \Phi^*`$ |
| 6 | RL 训练 | $`r_{GRM}`$ + 任意 RL 算法 | 高性能策略 | PPO / Cal-QL / ReinFlow |

### 关键设计选择及理由

**设计 1: Hop-based 相对进度（而非绝对进度）**
- **具体设计**: 预测归一化的相对进度变化 $`\mathcal{H} \in [-1,1]`$
- **设计理由**: 绝对进度迭代累积误差、可能超出 [0,1]
- **解决的问题**: 进度估计的稳定性和有界性

**设计 2: 三视角融合（而非单一预测）**
- **具体设计**: Incremental + Forward + Backward 简单平均
- **设计理由**: 三种方式互补——局部精度、全局稳定、目标敏感
- **解决的问题**: 长轨迹误差累积（消融: -15% ~ -22.5%）

**设计 3: PBRS 而非 naive reward shaping**
- **具体设计**: $`r = r_{gold} + \gamma\Phi - \Phi`$（多一个 $`\gamma`$）
- **设计理由**: 数学保证最优策略不变（telescoping sum）
- **解决的问题**: Semantic trap（消融: -43.7%）
