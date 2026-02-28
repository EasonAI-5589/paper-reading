[← 返回 README](../README.md)

# 2. Related Works

> 来源: VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model (arXiv 2602.12063)

---

## 📄 原文

> 💡 **Section 概览**: Related Work 分两个子节：① VLA Post-training（在线 RL/SFT 方法）；② 世界模型用于决策（从早期任务特定动力学模型到大规模视频扩散模型）。作者在每个子节末尾都点明了本文方法与已有工作的区别。

> 💡 **注意**: PDF 排版中 Figure 2 和 Figure 3 出现在本节页面，但根据 caption 内容（均描述 VLAW 方法流程），它们归属于 Section 4（Method）。本节正文不含图片。

---

### 2.1. Post-training Vision-Language-Action Models

Vision–language–action (VLA) models have achieved remarkable success in robotic manipulation tasks (Intelligence et al., 2025b; Pertsch et al., 2025; Liu et al., 2025a; Cui et al., 2025; Hu et al., 2024; Guo et al., 2024; Zhang et al., 2026). A common approach is to train the VLA on large-scale data and then perform supervised fine-tuning on target tasks (Zhang et al., 2025a; Black et al., 2024; Zhang et al., 2025b). Beyond supervised fine-tuning, improving VLA policies using online rollout data has emerged as a promising direction (Intelligence et al., 2025a; Guo et al., 2025b; Lu et al., 2025; Zang et al., 2025; Huang et al., 2024; Cheng et al., 2025). Some prior works adopt on-policy reinforcement learning methods, such as PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024), to improve VLA policies.

> 💡 **VLA Post-training 全景**:
> ```
> 路线 1: 大规模数据预训练 + 任务特定 SFT
> └── 代表: π₀.₅, OpenVLA, HiRobot, etc.
>
> 路线 2: Online rollout RL
> ├── on-policy RL: PPO / GRPO（主要在仿真中验证）
> └── offline RL: advantage-conditioned SL（π₀.₆*）
> ```

However, standard on-policy reinforcement learning typically requires a large number of rollouts and is therefore primarily validated in simulation environments (Li et al., 2025b;a; Liu et al., 2025b). Moreover, state-of-the-art VLA models are often trained with flow-matching objectives, which do not provide explicit policy likelihoods, making conventional policy-gradient methods difficult to apply. To enable policy learning in real-world settings, $\pi_{0.6}^{*}$ (Intelligence et al., 2025a) instead adopts an offline or batch reinforcement learning formulation with an advantage-conditioned supervised learning objective. Similarly, in our setting, we perform iterative policy improvement using batches of real-world rollout data together with world-model–generated synthetic data, and update the policy exclusively through stable supervised fine-tuning objectives.

> 💡 **本文与 π₀.₆* 的区别**:
> - π₀.₆*：advantage-conditioned SL，只用真实 rollout，无世界模型
> - VLAW：world model 生成合成 rollout + 真实 rollout，SFT 训练
> - **关键差异**：VLAW 引入了世界模型来扩增数据量（10x），这是本文相对 π₀.₆* 的核心卖点
> - **注意**：论文没有直接与 π₀.₆* 对比，这是实验设计的一个缺陷

---

### 2.2. World Models for Decision Making

Action-conditioned world models predict future outcomes given current observations and actions, and are also referred to as forward dynamics models. Many works leverage such models for model-based reinforcement learning (Hafner et al., 2020; Hansen et al., 2022; Oh et al., 2015; Wu et al., 2024) and visual planning (Finn & Levine, 2017; Ebert et al., 2018; Xie et al., 2019; Dasari et al., 2019; Yang et al., 2023). Among these, the most closely related approaches to ours are DayDreamer (Wu et al., 2023), SOLAR (Zhang et al., 2019) and World4rl (Jiang et al., 2025), which also operate in real-world visual model-based reinforcement learning settings. However, due to limited model capacity and data scale, these earlier methods often learned task-specific dynamics models.

> 💡 **世界模型用于决策的演进**:
> ```
> 第一代: 任务特定动力学模型
> ├── DayDreamer (Wu et al., 2023): 真实机器人 world model RL
> ├── SOLAR (Zhang et al., 2019): 结构化 latent space model-based RL
> └── 局限: 模型容量小、数据规模小、每个任务单独训练
>
> 第二代: 大规模视频扩散世界模型
> ├── Ctrl-World (Guo et al., 2025a): DROID 数据集上训练的多任务 WM
> ├── IRASim / IVP / AdaWorld: 多任务 action-conditioned 生成模型
> └── 问题: 仍然以演示数据为主训练 → 过度乐观偏差
> ```

With recent advances in video diffusion models (Ren et al., 2025; Ball et al., 2025; Mei et al., 2026), it has become feasible to train multi-task action-conditioned world models that can generate realistic future visual observations (Chen et al., 2024; Gao et al., 2025; Zhu et al., 2024; 2025; Sharma et al., 2026). Despite this progress, accurately modeling complex physical dynamics remains a fundamental challenge, as widely observed in prior world-model literature (Guo et al., 2025a), likely because these models are trained on offline robotics datasets usually consisting primarily of demonstrations. To address this challenge, we leverage online policy rollout data to ground a pretrained world model in new environments, thereby improving its accuracy around the policy's state–action distribution.

> 💡 **本文的创新点**：大规模视频扩散世界模型（Ctrl-World）已经足够强，但缺乏在线 rollout 数据的 grounding。VLAW 的关键创新是**用在线 rollout（含失败案例）微调世界模型**，而非从头训练新的世界模型。这是一个低成本、高效的 incremental 改进，而非颠覆性的架构创新。

---

## 🔖 Section 总结

### 核心洞察

1. **VLA post-training 的主流路线**：在线 rollout RL 是趋势，但 flow-matching 策略的 RL 难题（无 action log-prob）推动了 advantage-conditioned SL 的发展
2. **世界模型的演进瓶颈**：从任务特定到多任务，能力提升了，但「过度乐观偏差」是系统性问题
3. **本文定位**：站在大规模预训练世界模型（Ctrl-World）的肩膀上，用在线数据做精准修正，是工程上最实用的路径
