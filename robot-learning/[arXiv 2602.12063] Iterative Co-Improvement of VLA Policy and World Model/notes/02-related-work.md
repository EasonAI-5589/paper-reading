# VLAW 批读笔记 · Related Work

---

## 2. Related Works

### 2.1. Post-training Vision-Language-Action Models

Vision–language–action (VLA) models have achieved remarkable success in robotic manipulation tasks (Intelligence et al., 2025b; Pertsch et al., 2025; Liu et al., 2025a; Cui et al., 2025; Hu et al., 2024; Guo et al., 2024; Zhang et al., 2026). A common approach is to train the VLA on large-scale data and then perform supervised fine-tuning on target tasks (Zhang et al., 2025a; Black et al., 2024; Zhang et al., 2025b). Beyond supervised fine-tuning, improving VLA policies using online rollout data has emerged as a promising direction (Intelligence et al., 2025a; Guo et al., 2025b; Lu et al., 2025; Zang et al., 2025; Huang et al., 2024; Cheng et al., 2025). Some prior works adopt on-policy reinforcement learning methods, such as PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024), to improve VLA policies.

> 💡 **VLA post-training 的三条路**：
> 1. **SFT**（监督微调）：在目标任务上直接 fine-tune，简单但需要 demo 数据
> 2. **On-policy RL**（PPO/GRPO）：理论上最优，但需要大量 rollout，主要在仿真中验证
> 3. **Offline/Batch RL**（本文的方向）：用已有 rollout + weighted SFT，适合 real-world

However, standard on-policy reinforcement learning typically requires a large number of rollouts and is therefore primarily validated in simulation environments (Li et al., 2025b;a; Liu et al., 2025b). Moreover, state-of-the-art VLA models are often trained with flow-matching objectives, which do not provide explicit policy likelihoods, making conventional policy-gradient methods difficult to apply.

> 💡 **flow-matching VLA 的 RL 困境**：π₀ 系列没有 log π(a|o)，所以 REINFORCE/PPO 里的 log prob ratio 算不出来。这不只是工程问题，是理论层面的障碍。这也是为什么 π₀.₆* 和 VLAW 都选择了 weighted SFT 而非 policy gradient。

To enable policy learning in real-world settings, π₀.₆* (Intelligence et al., 2025a) instead adopts an offline or batch reinforcement learning formulation with an advantage-conditioned supervised learning objective. Similarly, in our setting, we perform iterative policy improvement using batches of real-world rollout data together with world-model–generated synthetic data, and update the policy exclusively through stable supervised fine-tuning objectives.

> 💡 **VLAW vs. π₀.₆* 的关系**：思路相似（都用 offline RL + SFT），核心区别是 VLAW 用 World Model 把数据放大了 10 倍（50 条 real rollout → 500 条合成轨迹），数据效率是主要优势。

---

### 2.2. World Models for Decision Making

Action-conditioned world models predict future outcomes given current observations and actions, and are also referred to as forward dynamics models. Many works leverage such models for model-based reinforcement learning (Hafner et al., 2020; Hansen et al., 2022; Oh et al., 2015; Wu et al., 2024) and visual planning (Finn & Levine, 2017; Ebert et al., 2018; Xie et al., 2019; Dasari et al., 2019; Yang et al., 2023). Among these, the most closely related approaches to ours are DayDreamer (Wu et al., 2023), SOLAR (Zhang et al., 2019) and World4rl (Jiang et al., 2025), which also operate in real-world visual model-based reinforcement learning settings. However, due to limited model capacity and data scale, these earlier methods often learned task-specific dynamics models.

> 💡 **早期 World Model 的局限**：DayDreamer（2023）是重要先驱，在真实机器人上用 Dreamer 做 MBRL，但当时模型容量有限，只能做单任务。VLAW 的 Ctrl-World 是在 DROID 全量数据上训练的多任务 world model，天然支持多任务设置。

With recent advances in video diffusion models (Ren et al., 2025; Ball et al., 2025; Mei et al., 2026), it has become feasible to train multi-task action-conditioned world models that can generate realistic future visual observations (Chen et al., 2024; Gao et al., 2025; Zhu et al., 2024; 2025; Sharma et al., 2026). Despite this progress, accurately modeling complex physical dynamics remains a fundamental challenge, as widely observed in prior world-model literature (Guo et al., 2025a), likely because these models are trained on offline robotics datasets usually consisting primarily of demonstrations.

> 💡 **Video Diffusion + Robot 的现状**：Genie 3、IRASim、WMPO 等近期工作用 video diffusion 做 robot world model，视觉质量大幅提升，但物理准确性仍然不足，根本原因还是训练数据（expert demo 为主，缺乏 failure 覆盖）。

To address this challenge, we leverage online policy rollout data to ground a pretrained world model in new environments, thereby improving its accuracy around the policy's state–action distribution.

> 💡 **Distribution Shift 视角**：Pretrained world model 在 DROID（expert demo）上训练，而 policy rollout 的状态-动作分布不同（尤其是 failure 时的状态）。用 online rollout fine-tune = 缩小 distribution shift，让 world model 贴近 policy 实际经历的状态。这是经典的 DAgger 思想在 world model 上的应用。
