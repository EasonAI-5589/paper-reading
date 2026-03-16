[← 返回 README](../README.md)

# 3. Problem Formulation and Challenges

## 📌 预览

本节先形式化定义具身策略（Embodied Policy）和五种建模分布，再提出 Motus 要解决的两大挑战：Challenge 1 是如何在一个框架里统一五种多模态生成能力；Challenge 2 是如何利用大规模异构数据（尤其是没有动作标签的视频）。

---

## Embodied Policies

We consider the task of language-conditioned robotic manipulation. For each embodiment, the task defines an action $`\mathbf{a} \in \mathcal{A}`$, an observation $`\mathbf{o} \in \mathcal{O}`$ (visual input), a language instruction $`\ell \in \mathcal{L}`$, and the proprioception of the robot $`\mathbf{p}`$, where $`\mathcal{A}`$, $`\mathcal{O}`$ and $`\mathcal{L}`$ denote the action space, the observation space, and the language instruction space respectively. The task typically provides an expert dataset $`\mathcal{D}_{\text{expert}} = \{\ell, p_1, o_1, a_1, \ldots, p_N, o_N, a_N\}`$, which contains robot proprioception, visual observations, and actions collected by an expert over $`N`$ timesteps, along with corresponding language annotations for each trajectory. We train a policy parameterized by $`\theta`$ on $`\mathcal{D}_{\text{expert}}`$. At each timestep $`t`$, the policy predicts the next $`k`$ actions (action chunking [59]) based on the current observation and proprioception, modeling the distribution $`p_\theta(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \mathbf{p}_t, \ell)`$ or $`p_\theta(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)`$. The policy $`p_\theta`$ is trained to maximize the likelihood objective:

$$
\max_\theta \ \mathbb{E}_{(o_t, p_t, a_{t+1:t+k}, \ell) \sim \mathcal{D}_{\text{expert}}} \log p_\theta(a_{t+1:t+k} \mid o_t, p_t, \ell).
$$

> **大白话解读**：
>
> **问题设定**：给机器人一句话指令（比如"把杯子放到桌上"），让它看着当前画面，输出一连串动作。
>
> **Action Chunking**：不是一次预测一个动作，而是一次预测未来 $`k`$ 步动作（比如 $`k=16`$）。好处是：
> - 减少累积误差（不用每步都重新推理）
> - 提高执行流畅度（一次规划一段连贯动作）
> - 降低推理延迟（一次前向传播出 $`k`$ 个动作）
>
> **Likelihood Objective**：训练目标就是让模型输出的动作分布尽可能接近专家数据中的动作。直觉理解——"看了专家怎么做的，学着模仿"，最大化"在看到这个画面和指令时，专家做出这些动作的概率"。

---

## Five Modeling Distributions

Furthermore, based on the symbolic definitions above, we can derive the probability distributions for the 5 modeling types of embodied intelligence, which can be integrated into a single model for training:

- **VLA**: $`p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)`$
- **WM**: $`p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \mathbf{a}_{t+1:t+k})`$
- **IDM**: $`p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_{t:t+k})`$
- **VGM**: $`p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \ell)`$
- **Video-Action Joint Prediction Model**: $`p(\mathbf{o}_{t+1:t+k}, \mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)`$

> **五种分布的直觉理解**：
>
> | 模型 | 输入 | 输出 | 直觉 |
> |------|------|------|------|
> | **VLA** | 当前画面 + 语言指令 | 动作序列 | "看到杯子，听到指令，决定怎么动" |
> | **WM** | 当前画面 + 动作序列 | 未来画面序列 | "如果我这样动，世界会变成什么样？" |
> | **IDM** | 当前 + 未来画面序列 | 动作序列 | "画面从 A 变到 B，中间做了什么动作？" |
> | **VGM** | 当前画面 + 语言指令 | 未来画面序列 | "根据指令，想象未来会是什么样" |
> | **Joint** | 当前画面 + 语言指令 | 动作 + 未来画面 | "同时想象未来场景并决定怎么做" |
>
> 关键观察：这五种分布的条件变量和生成变量都是 $`\mathbf{o}`$（视觉）、$`\mathbf{a}`$（动作）、$`\ell`$（语言）的不同组合。如果能用一个模型灵活切换"哪些作为条件、哪些需要生成"，就能统一五种范式。这正是 Motus 的 UniDiffuser 式调度器的设计动机。

---

## Challenge 1: Unifying Multimodal Generative Capabilities

A capable embodied agent must integrate a spectrum of cognitive functions—from understanding scenes and instructions, imagining possible futures, to predicting consequences and generating actions—to possess a human-like capacity, as a unified whole. Current models, however, are fragmented and fail to capture the full set of necessary capabilities within one system. This presents a challenge: how to unify the modeling of five key distributions—VLA, World Model, IDM, Video Generation Model, and Video-Action Joint Prediction Model—within a single framework. While prior work, such as UWMs [64], has made some progress, a critical limitation persists: these approaches are either trained from scratch, built upon smaller base models, or—even when incorporating some priors—invariably lack the full spectrum of knowledge, missing either visual understanding priors from VLMs or physical interaction priors from VGMs. Consequently, they lack the comprehensive world knowledge required for robust and generalizable embodied intelligence. Therefore, the nontrivial challenge of jointly modeling various distributions of vision, language, and action within a unified framework remains unaddressed, which is precisely the gap our work fills.

> **比喻理解**：
>
> 现有方法就像一个公司里**各自为政的部门**：
> - VLA 部门：擅长"听指令做事"，但不会想象后果
> - WM 部门：擅长"模拟推演"，但不会执行动作
> - VGM 部门：擅长"拍宣传片"（生成视频），但不懂任务指令
> - IDM 部门：擅长"事后分析"（看视频猜动作），但不能主动规划
>
> 这些部门各有专长但互不沟通。UWM 试图建了一个"联合办公室"，但它是白手起家（from scratch），没有引进任何有经验的专家。
>
> **Motus 要建的是"统一指挥中心"**：把三位资深专家（VLM 理解专家、VGM 生成专家、Action 动作专家）请进来，让他们共享信息（Tri-model Joint Attention），但各自保留专业技能（独立 FFN）。通过灵活的调度器（UniDiffuser 式），根据任务需要切换成上述任何一种工作模式。

---

## Challenge 2: Utilization of Heterogeneous Data

A central challenge in embodied intelligence is how to make effective use of large scale heterogeneous data. Action spaces vary widely between embodiments in dimension, range, and semantics, and robots differ in morphology, actuation, and sensing. As a result, control signals are not directly reusable and policies struggle to learn universal priors that transfer across embodiments. Existing approaches, including [8, 31, 43, 60], try to address this by using a general backbone with embodiment-specific information injection, or constructing high-dimensional action vectors that forcibly unify different embodiments. However, they still depend primarily on labeled robotic trajectories and cannot integrate these datasets with large-scale internet videos or egocentric human videos, which lack action annotations but contain abundant motion and physical interaction cues. This limitation prevents large-scale pretraining of the action expert and reduces the ability to learn general motion priors.

> **核心痛点解析**：
>
> 这个挑战的本质是**数据丰富但标注稀缺，且格式不统一**：
>
> **痛点 1：海量视频没有动作标注**
> - 互联网上有无穷无尽的视频（YouTube、人类操作视频、自我视角视频等）
> - 这些视频包含丰富的物理交互知识（怎么抓、怎么放、怎么推）
> - 但它们都没有 action label（没人给视频标注"此时机械臂关节角度是多少"）
> - 现有方法无法利用这些数据来预训练 action expert
>
> **痛点 2：不同机器人的 action space 完全不一样**
> - 7-DoF 机械臂的动作是 7 维向量（各关节角度）
> - 双臂机器人是 14 维
> - 人形机器人可能是 30+ 维
> - 维度、范围、物理含义全不一样，动作信号无法直接复用
>
> **现有方案的局限**：
> - $`\pi_{0.5}`$、X-VLA 等用 embodiment-specific head 来适配不同机器人
> - 但 backbone 的预训练仍然只能用有动作标注的机器人数据
> - 无法从海量无标注视频中学到通用的运动先验
>
> **Motus 的解法预告**：用**光流**作为跨 embodiment 的"通用运动语言"。无论是人手还是机械臂，移动物体时的像素位移模式是相似的。通过光流编码 latent action，可以在无动作标注的视频上预训练 action expert。

---

## 🔖 Section 总结

### 核心公式速查

| 符号 | 含义 |
|------|------|
| $`\mathbf{a} \in \mathcal{A}`$ | 动作，属于动作空间 |
| $`\mathbf{o} \in \mathcal{O}`$ | 观测（视觉输入），属于观测空间 |
| $`\ell \in \mathcal{L}`$ | 语言指令，属于语言空间 |
| $`\mathbf{p}`$ | 本体感受（关节角度等） |
| $`k`$ | action chunk 长度 |
| $`\mathcal{D}_{\text{expert}}`$ | 专家数据集 |

### 两大挑战对照表

| 挑战 | 问题 | 现有方案 | 为什么不够 | Motus 方案 |
|------|------|---------|-----------|-----------|
| Challenge 1: 统一建模 | 5 种分布分散在不同模型中 | UWM 统一但从零训练 | 缺预训练先验，世界知识不足 | MoT 融合三个预训练专家 + UniDiffuser 调度 |
| Challenge 2: 异构数据 | 海量视频无 action 标注 + 不同机器人 action space 不兼容 | embodiment-specific head | 仍依赖有标注数据，无法利用无标注视频 | 光流 latent action + 六层数据金字塔 |
