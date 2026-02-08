[← 返回 README](../README.md)

# 2 New Feature

## 📌 预览
本节详细描述 RoboBrain 2.5 的两大核心新特性：精确 3D 空间推理（2.1）和密集时间价值估计（2.2）。这是全文最核心的方法论部分。

---

Building upon the foundation of RoboBrain 2.0 [72] and utilizing the Qwen3-VL architecture [8], RoboBrain 2.5 introduces two core enhancements that further advance physical intelligence. Specifically, we first detail the concept of Precise 3D Spatial Reasoning (Section 2.1), which encompasses three metric-grounded competencies,- spatial referring, measuring, and tracing—derived solely from monocular RGB inputs [86]. We then describe Dense Temporal Value Estimation(Section 2.2), which learns a general-purpose, step-aware process modeling from multi-view RGB-only observations [67].

> 💡 **Section 概览**:
> - 架构基座：Qwen3-VL（8B）
> - 两个新特性都只需要 **RGB 输入**（单目 RGB for 3D / 多视角 RGB for temporal）
> - 空间推理参考 TraceSpatial [86]，时间估计参考 Robo-Dopamine [67]

---

## 2.1 Precise 3D Spatial Reasoning

> 💡 **2.1 要点预览**: 如何从单目 RGB 图像实现度量级 3D 空间推理？通过三个递进技能（定位→测量→轨迹生成）和解耦的 $(u,v,d)$ 表示。

For embodied agents to interact effectively with the physical world, they must accurately interpret and act upon spatial information. This necessitates a deep understanding of object locations, inter-object relationships, and precise metric quantities from visual observations. To address these fundamental requirements, we introduce a robust framework for Precise 3D Spatial Reasoning.

### 2.1.1 3D Spatial Referring, Measuring, and Tracing

Embodied robots usually have to execute actions based on increasingly complex, spatially constrained instructions [1, 10, 11, 34, 68, 70, 71, 85], such as "Water flowers from left to right with watering can hovering $_{1-5}$ cm above each one" in Figure 1, where recent data-scarce Vision-Language-Action (VLA) models fail to master. In this case, it would be beneficial to generate a 3D positional sequence, named as 3D spatial trace, as an intuitive bridge to interpret the instruction following procedure in 3D space and guide the generation of actual action trajectories for robots. However, this surrogate task (i.e., 3D spatial tracing) is inherently challenging as it requires multi-step, metric-grounded reasoning in complex 3D scenes. To be specific, each reasoning step requires two key components: (1) 3D spatial referring to resolve spatial relationships and accurately localize objects involved in the trace generation (e.g., identifying flowers with their from left to right order and locating them). (2) 3D spatial measuring to understand absolute, real-world metric quantities related to the trace in captured scene (e.g., quantifying each flower's physical height and 1–5 cm height above each). To this end, we equip RoboBrain 2.5 with these three key capabilities, enabling it to directly predict metric-grounded outputs from monocular images under spatial constraints for direct interaction with the 3D physical world.

> 💡 **批注**: 
> - **3D Spatial Trace** 是核心概念：一个有序的 3D 位置序列，作为指令到动作的中间桥梁
> - 生成 trace 需要两个前置能力：
>   1. **Referring**（定位）: 解析空间关系，定位对象（如"从左到右的花"）
>   2. **Measuring**（测量）: 理解绝对度量量（如"每朵花上方 1-5cm"）
> - 三个技能是递进关系：先定位 → 再测量 → 最后生成完整轨迹

---

### 2.1.2 3D Task Formulation

> 💡 **2.1.2 要点预览**: $(u, v, d)$ 解耦表示是整个 3D 空间推理的数学基础，巧妙利用相机内参将 2D+深度 转换为 3D。

We formalize 3D spatial tracing as the process of predicting an ordered sequence of 3D points $\tau = \{ p_{t} \}_{t=1}^{T}$ —each point $p_{t} = (u_{t}, v_{t}, d_{t})$ comprising image-plane coordinates $(u_{t}, v_{t})$ and absolute depth $d_{t}$ —from visual inputs (e.g., RGB images) and textual instructions via vision-language models. The resulting trace $\tau$ functions as a spatial plan for guiding entities (e.g., a robot end-effector or an object) to execute instructions. Crucially, these instructions typically encode both 3D spatial referring and 3D spatial measuring, often requiring multi-step compositional reasoning. For instance, in Figure 1, the instruction "Water flowers from left to right with watering can hovering 1-5 cm above each flower" necessitates determining the 3D positions and heights of all flowers in the scene. Although intermediate spatial cues (e.g., points identified through 3D spatial referring) may not coincide with the final keypoints used in the spatial trace, they provide essential evidence for multi-step reasoning—thereby enabling precise trace generation under spatial constraints at the start, the end, and along the trajectory.

> 💡 **批注**: 形式化定义：$\tau = \{p_t\}_{t=1}^T$，每个点 $p_t = (u_t, v_t, d_t)$
> - $u, v$: 图像平面坐标
> - $d$: 绝对深度
> - 中间推理点（如 referring 得到的位置）可能不在最终轨迹上，但提供多步推理的证据

---

At the core of our approach lies a task formulation designed to facilitate training and to leverage diverse data sources effectively. Rather than predicting 3D coordinates in the form $(x, y, z)$ in camera or world frame, we adopt a decoupled $(u, v, d)$ representation, which can be trivially projected to 3D coordinates using known camera intrinsics. This formulation is especially advantageous in embodied scenarios where camera parameters are readily accessible, as it obviates the need for vision-language models to learn camera geometry implicitly. Such an approach streamlines training and enhances accuracy. Furthermore, the $(u, v, d)$ representation can be straightforwardly projected into lower-dimensional subspaces. For instance, omitting $d$ yields a 2D visual trace (i.e., a sequence of points in the image plane), while retaining only the start and end points produces 3D or 2D spatial referring data (if depth is further removed). This flexibility not only promotes data reusability but also ensures compatibility with existing 2D datasets [23, 85], thereby boosting multi-task learning performance through co-training across complementary tasks and modalities.

> 💡 **批注 - $(u,v,d)$ 表示的优势**:
> 1. **不需要模型隐式学习相机几何**: 相机内参是已知的，直接 $(u,v,d) \to (x,y,z)$
> 2. **灵活降维**: 去掉 $d$ → 2D 轨迹；只保留首尾点 → referring 数据
> 3. **数据兼容**: 可以和现有 2D 数据集（如 Molmo/Pixmo）联合训练
> 
> 这个设计非常聪明——一个统一的表示同时服务于 referring、measuring、tracing 三个任务。

---

## 2.2 Dense Temporal Value Estimation

> 💡 **2.2 要点预览**: 如何从视觉观测中获得密集的执行进度反馈？通过 hop-based 标注策略 + 多视角融合 + 双向一致性检查。

Effective execution of long-horizon manipulation tasks demands more than just a final success signal; it requires continuous, granular feedback to guide the agent through complex intermediate states [3, 15, 52, 54, 80]. To address the limitations of sparse feedback, we introduce Dense Temporal Value Estimation, a vision-based mechanism that provides real-time, step-aware progress assessments as temporal value feedback, enabling robust closed-loop control and efficient RL.

> 💡 **批注**: 稀疏反馈（只有最终成功/失败）的问题在长时程任务中尤为突出。密集的时间价值估计本质上就是一个通用的 reward model。

---

### 2.2.1 Hop-wise Progress Construction

> 💡 **2.2.1 要点预览**: 三步流水线——轨迹分段 → hop-based 相对进度标注 → 数据平衡采样。核心创新是 hop normalization，保证迭代预测不会越界。

Central to our approach is the formulation of value estimation as task progress; thus, our model functions as a vision-language estimator designed to infer fine-grained, real-time progress from visual inputs. To guarantee generalizability across diverse embodiments and task families, we implement a three-stage data curation pipeline handling diverse data origins. This process spans from raw video segmentation to a systematic, hop-based labeling strategy, as detailed below:

**Step-wise task progress discretization.** Given raw multi-view video trajectories, we first segment each expert trajectory into sub-tasks using human-annotated multi-view keyframes $\{ K_{0}, K_{1}, \ldots, K_{N} \}$, where $K_{0}$ is the initial observation, $K_{N}$ is the final success observation, and each $K_{j}$ is a set of synchronized multi-view keyframes. To obtain dense supervision, we perform adaptive sampling within each segment. For a trajectory with $L$ frames per view, we set a chunk size $C$ to determine the total number of sampled points and distribute them uniformly across the $N$ segments. The number of intermediate points $m$ within segment $[K_{j}, K_{j+1}]$ is:

$$m = \left\lfloor \frac{1}{N} \left\lfloor \frac{L}{C} \right\rfloor \right\rfloor.$$

This yields a sequence of states $\mathcal{S} = \{ s_{0}, s_{1}, ..., s_{M} \}$, where each state $s_{i}$ is a set of synchronous multi-view visual observations. We then define the ground-truth global progress as $\Phi(s_{i}) = i / M$.

> 💡 **批注**: 
> - 用人工标注的关键帧将轨迹分段，然后在每段内均匀采样
> - 全局进度 $\Phi(s_i) = i/M$ 是线性的——假设每个采样点代表相等的进度增量
> - 这是一个简化假设，但配合后续的 hop normalization 可以工作得很好

---

**Hop-based relative progress normalization.** A naive choice is to regress the progress gain $\Phi_{\delta}(s_{p}, s_{q}) = \Phi(s_{q}) - \Phi(s_{p})$ between two states, but iterating such predictions accumulates error and can push the reconstructed $\Phi^{\star}(s)$ outside $[0, 1]$. Instead, we introduce a hop-based formulation that learns relative-relative progress and naturally supports dense temporal value estimation. Each training sample is a tuple $\mathcal{D}$ containing a task description $d_{\mathrm{task}}$, the initial state $s_{0}$, the goal state $s_{M}$, a "BEFORE" state $s_{p}$, an "AFTER" state $s_{q}$ and a hop label $\mathcal{H}(s_{p}, s_{q})$ that normalizes the progress from $s_{p}$ to $s_{q}$ relative to the full task span from $s_{0}$ to $s_{M}$. Given $\Phi(s_{p})$ and $\Phi(s_{q})$, we define:

$$\mathcal{H}(s_{p}, s_{q}) = \begin{cases} \frac{\Phi(s_{q}) - \Phi(s_{p})}{\Phi(s_{M}) - \Phi(s_{p})} & \text{if } q \geq p \text{ (PROGRESS)} \\ \frac{\Phi(s_{q}) - \Phi(s_{p})}{\Phi(s_{p}) - \Phi(s_{0})} & \text{if } q < p \text{ (REGRESS).} \end{cases}$$

This dynamically scales the supervision into $[-1, 1]$: for forward progress, the change is normalized by the remaining distance to the goal; for regression, by the distance already covered from the initial state. A key theoretical advantage is that, when global progress is reconstructed by iteratively applying predicted hops, the resulting $\Phi^{\star}(s)$ is guaranteed to remain strictly within $[0, 1]$. Please refer to Section B for the proof.

> 💡 **批注 - Hop Normalization 的精妙之处**:
> - **朴素做法**的问题：直接预测绝对进度差 $\Delta\Phi$，迭代累加会误差积累，可能越出 $[0,1]$
> - **Hop 做法**: 进度/退步 分别除以"到目标的剩余距离"/"已走过的距离"
> - 这保证 hop ∈ $[-1, 1]$，且迭代重建时 $\Phi^* \in [0, 1]$（有数学证明，见 Appendix B）
> - 直觉：越接近目标时，同样的绝对进度对应更大的 hop 值（放大信号），有助于精细控制

---

**Sampling strategy and data balancing.** For each trajectory, we construct a balanced set of hop-based training samples. Continuous hop values are first discretized into $N_{\mathrm{hop}}$ hop bins. The temporal distance between the "BEFORE" state $s_{p}$ and "AFTER" state $s_{q}$ in each pair is then chosen from $N_{\mathrm{dis}}$ distance bins within each hop bin, yielding in total $N_{\mathrm{hop}} \times N_{\mathrm{dis}}$ non-trivial transitions. To reduce bias toward static segments, we further introduce an additional fraction $\alpha$ of samples explicitly labeled as zero-hop (i.e., $\mathcal{H}(s_{p}, s_{q}) = 0$), constructed by selecting pairs $(s_{p}, s_{q})$ whose progress change is below a small threshold $\epsilon$:

$$| \Phi(s_{q}) - \Phi(s_{p}) | \leq \epsilon.$$

> 💡 **批注**: 数据平衡策略：
> - 离散化 hop 值为 $N_{\text{hop}}$ 个 bin，每个 bin 内按时间距离再分 $N_{\text{dis}}$ 个 bin
> - 额外加入 zero-hop 样本（几乎没有进度变化的帧对），防止模型偏向"总是预测有进展"

---

### 2.2.2 Multi-Perspective Progress Fusion

> 💡 **2.2.2 要点预览**: 三种互补的进度估计方式——增量式、前锚定、后锚定——取平均融合，抗漂移。

To mitigate error accumulation and ensure consistent accuracy, we fuse dense temporal value estimates from three complementary perspectives: incremental prediction, forward-anchored prediction, and backward-anchored prediction.

**Incremental Prediction** offers a fine-grained, step-by-step assessment. Refer to Equation (2), the predicted global progress $\Phi_{I}^{\star}(s_{t})$ is recursively computed from the preceding state's progress $\Phi^{\star}(s_{t-1})$ and the predicted hop $\mathcal{H}^{\star}(s_{t-1}, s_{t})$. Let $\Delta\Phi_{t-1,t}^{\star}$ be the estimated progress hop:

$$\Delta\Phi_{t-1,t}^{\star} = \begin{cases} [1 - \Phi^{\star}(s_{t-1})] \cdot \mathcal{H}^{\star} & \text{if } \mathcal{H}^{\star} \geq 0 \\ \Phi^{\star}(s_{t-1}) \cdot \mathcal{H}^{\star} & \text{if } \mathcal{H}^{\star} < 0. \end{cases}$$

The incremental progress is then calculated as follow:

$$\Phi_{I}^{\star}(s_{t}) = \Phi^{\star}(s_{t-1}) + \Delta\Phi_{t-1,t}^{\star},$$

where $\Phi_{I}^{\star}(s_{t})$ is accumulated along the trajectory, initialized with $\Phi^{\star}(s_{0}) = 0$. While this method excels at capturing local dynamics, it is susceptible to the accumulation of prediction errors over long trajectories.

> 💡 **批注**: 增量预测：逐帧累加 hop，擅长捕捉局部动态，但长轨迹会误差漂移。

---

To counteract this drift, we introduce two global perspectives. **Forward-Anchored Prediction** provides a stable global reference by anchoring to the initial state $s_{\mathrm{init}}$, where progress is zero:

$$\Phi_{F}^{\star}(s_{t}) = \mathcal{H}^{\star}(s_{\mathrm{init}}, s_{t}).$$

Conversely, **Backward-Anchored Prediction** is anchored to the goal state $s_{\mathrm{goal}}$, where progress is one. This approach offers high sensitivity near task completion:

$$\Phi_{B}^{\star}(s_{t}) = 1 + \mathcal{H}^{\star}(s_{\mathrm{goal}}, s_{t}).$$

These three methods offer complementary strengths: local precision (incremental), initial stability (forward), and goal sensitivity (backward). We fuse them via averaging to obtain a robust final progress estimate:

$$\Phi^{\star}(s_{t}) = \frac{1}{3} \left( \Phi_{I}^{\star}(s_{t}) + \Phi_{F}^{\star}(s_{t}) + \Phi_{B}^{\star}(s_{t}) \right).$$

This fusion yields a more accurate and drift-resistant value signal. Please also refer to [67] for how to apply this kind of value signal for RL process.

> 💡 **批注 - 三视角融合**:
> | 方式 | 锚点 | 优势 | 劣势 |
> |------|-------|------|------|
> | Incremental | 前一帧 | 局部精度高 | 长程漂移 |
> | Forward-Anchored | 初始状态 | 早期稳定 | 后期不敏感 |
> | Backward-Anchored | 目标状态 | 接近目标时敏感 | 远离目标时不稳定 |
> 
> 取平均是最简单的融合方式，但已经能显著提升鲁棒性。

---

### 2.2.3 Bi-directional Consistency Checking

> 💡 **2.2.3 要点预览**: 解决 RL 中的 OOD reward hacking 问题——用前向和后向预测的一致性作为可靠性代理。

While the multi-perspective fusion via averaging (Equation (8)) serves as a baseline, its naive application in online RL faces the risk of Out-of-Distribution (OOD) hallucination. Due to the inherent limitations of data coverage, it is impossible for the training set to encompass every corner of the state space. During RL, the policy inevitably explores unseen regions where dense temporal value estimation may yield spurious high signals, leading to "reward hacking." To address these, we propose a bi-directional consistency checking strategy that leverages consistency as a proxy for reliability. This design is motivated by the observation that forward $\Phi_{F}^{*}$ and backward $\Phi_{B}^{*}$ predictions tend to diverge significantly under OOD observations, whereas they remain consistent in familiar states.

> 💡 **批注**: OOD reward hacking 是 RL 中的经典问题——RL 策略会钻 reward model 的漏洞。这里的解决思路很优雅：如果前向和后向估计不一致，说明模型对这个状态不确定，应该保守更新。

---

**Consistency-Aware Weighting.** We first define the mean estimated progress $\bar{\Phi}^{*}(s_{t}) = (\Phi_{F}^{*}(s_{t}) + \Phi_{B}^{*}(s_{t})) / 2$. To quantify uncertainty, we calculate a normalized discrepancy metric:

$$\Delta_{\mathrm{norm}}(s_{t}) = \frac{|\Phi_{B}^{*}(s_{t}) - \Phi_{F}^{*}(s_{t})|}{\bar{\Phi}^{*}(s_{t}) + \epsilon},$$

where $\epsilon$ is a small constant for numerical stability. Normalization by $\bar{\Phi}^{*}$ ensures that discrepancies are penalized more heavily during the early stages (where $\Phi$ is small), as precise guidance is critical initially. We then derive a confidence weight $w_{t} \in (0, 1]$ using a Gaussian kernel with sensitivity $\alpha$:

$$w_{t} = \exp\left(-\alpha \cdot (\Delta_{\mathrm{norm}}(s_{t}))^{2}\right).$$

**Conservative State Update.** To prevent the policy from exploiting erroneous estimates in OOD scenarios, we employ a conservative update rule for the maintained progress state $\Phi^{*}(s_{t})$ instead of Equation (8):

$$\Phi^{*}(s_{t}) = \Phi^{*}(s_{t-1}) + \frac{w_{t}}{2} \cdot \left(\bar{\Phi}^{*}(s_{t}) - \Phi^{*}(s_{t-1}) + \Delta\Phi_{t-1,t}^{\star}\right).$$

This mechanism acts as a semantic filter: it ignores uncertain updates when $w_{t} \to 0$ (retaining $\Phi^{*}(s_{t-1})$) and fully trusts the estimate when consistency is high ($w_{t} \to 1$).

> 💡 **批注 - 保守更新机制**:
> - $w_t \to 0$ (前后不一致): 忽略当前估计，保持上一步的进度值
> - $w_t \to 1$ (前后一致): 完全信任当前估计
> - 归一化中除以 $\bar{\Phi}^*$ 的设计：任务早期（进度小）时对不一致惩罚更重，因为初期引导精度更关键
> - 这本质上是一个**不确定性感知的 reward shaping**

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 3D 表示 | 解耦 $(u, v, d)$，可转 3D |
| Hop 范围 | $[-1, 1]$，保证 $\Phi^* \in [0,1]$ |
| 融合方式 | 三视角取平均 |
| OOD 防护 | 双向一致性 + 保守更新 |

### 核心洞察
1. $(u,v,d)$ 表示是一个精妙的设计——统一了 referring/measuring/tracing，且兼容 2D 数据
2. Hop normalization 保证了数学上的有界性，是密集时间估计的理论基石
3. 三视角融合 + 双向一致性检查 = 鲁棒的 reward signal，可直接用于 RL
4. 整个方法论的设计理念：**让 VLM 既当感知器又当 reward model**
