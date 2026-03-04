[← 返回 README](../README.md)

# 2. Related Works

## 📌 预览

两条相关工作线：① VLA post-training（SFT → on-policy RL → offline RL，本文属于最后一类）；② World Model for Decision Making（从小容量单任务 MBRL 到大规模 video diffusion world model，本文是在后者基础上用 online rollout 来 ground）。

---

### 2.1. Post-training Vision-Language-Action Models

Vision–language–action (VLA) models have achieved remarkable success in robotic manipulation tasks (Intelligence et al., 2025b; Pertsch et al., 2025; Liu et al., 2025a; Cui et al., 2025; Hu et al., 2024; Guo et al., 2024; Zhang et al., 2026). A common approach is to train the VLA on large-scale data and then perform supervised fine-tuning on target tasks (Zhang et al., 2025a; Black et al., 2024; Zhang et al., 2025b). Beyond supervised fine-tuning, improving VLA policies using online rollout data has emerged as a promising direction (Intelligence et al., 2025a; Guo et al., 2025b; Lu et al., 2025; Zang et al., 2025; Huang et al., 2024; Cheng et al., 2025). Some prior works adopt on-policy reinforcement learning methods, such as PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024), to improve VLA policies.

> 💡 **VLA post-training 的三条路**：
> 1. **SFT**：在目标任务 demo 上直接 fine-tune，简单有效但需要 demo 数据
> 2. **On-policy RL**（PPO/GRPO）：理论最优，但需大量 rollout，主要在仿真中验证
> 3. **Offline/Batch RL + weighted SFT**（本文所属）：用已有 rollout 做加权监督学习，适合 real-world 低 rollout 预算

However, standard on-policy reinforcement learning typically requires a large number of rollouts and is therefore primarily validated in simulation environments (Li et al., 2025b;a; Liu et al., 2025b). Moreover, state-of-the-art VLA models are often trained with flow-matching objectives, which do not provide explicit policy likelihoods, making conventional policy-gradient methods difficult to apply.

> 💡 **Flow-matching VLA 的 RL 困境**：这是设计约束，不是工程问题。π₀ 系列没有显式 log π(a|o)，REINFORCE/PPO 里的 importance ratio 算不出来。这迫使 VLAW 和 π₀.₆* 都走 weighted SFT 路线。

To enable policy learning in real-world settings, $\pi_{0.6}^*$ (Intelligence et al., 2025a) instead adopts an offline or batch reinforcement learning formulation with an advantage-conditioned supervised learning objective. Similarly, in our setting, we perform iterative policy improvement using batches of real-world rollout data together with world-model–generated synthetic data, and update the policy exclusively through stable supervised fine-tuning objectives.

> 💡 **VLAW vs. π₀.₆* 的关键区别**：两者都用 offline RL + SFT，但 VLAW 额外引入了 World Model 把数据量放大 10 倍（50 条 real rollout → 500 条合成轨迹）。数据效率是 VLAW 相对 π₀.₆* 的核心优势。

---

### 2.2. World Models for Decision Making

Action-conditioned world models predict future outcomes given current observations and actions, and are also referred to as forward dynamics models. Many works leverage such models for model-based reinforcement learning (Hafner et al., 2020; Hansen et al., 2022; Oh et al., 2015; Wu et al., 2024) and visual planning (Finn & Levine, 2017; Ebert et al., 2018; Xie et al., 2019; Dasari et al., 2019; Yang et al., 2023). Among these, the most closely related approaches to ours are DayDreamer (Wu et al., 2023), SOLAR (Zhang et al., 2019) and World4rl (Jiang et al., 2025), which also operate in real-world visual model-based reinforcement learning settings. However, due to limited model capacity and data scale, these earlier methods often learned task-specific dynamics models.

> 💡 **早期 World Model MBRL 的局限**：DayDreamer（2023，Hafner + Goldberg + Abbeel）是真实机器人上做 MBRL 的重要先驱，但当时模型容量小（RSSM），只能做单任务，dynamics model 不能泛化。VLAW 的 base world model（Ctrl-World）是在 DROID 全量数据上训练的多任务 diffusion model，起点远高于这些早期工作。

With recent advances in video diffusion models (Ren et al., 2025; Ball et al., 2025; Mei et al., 2026), it has become feasible to train multi-task action-conditioned world models that can generate realistic future visual observations (Chen et al., 2024; Gao et al., 2025; Zhu et al., 2024; 2025; Sharma et al., 2026). Despite this progress, accurately modeling complex physical dynamics remains a fundamental challenge, as widely observed in prior world-model literature (Guo et al., 2025a), likely because these models are trained on offline robotics datasets usually consisting primarily of demonstrations.

> 💡 **Video Diffusion World Model 的现状**：Genie 3、IRASim、WMPO 等近期工作用 video diffusion 做 robot world model，视觉质量大幅提升，但物理准确性仍然不足。根本原因：训练数据以 offline demo 为主，缺乏多样物理交互的覆盖。这给 VLAW「用 online rollout fine-tune」的方案提供了清晰的 motivation。

To address this challenge, we leverage online policy rollout data to ground a pretrained world model in new environments, thereby improving its accuracy around the policy's state–action distribution.

> 💡 **Distribution Shift 视角**：Pretrained world model 在 DROID（expert demo）分布上训练，而 policy rollout 的状态-动作分布不同（尤其 failure 时的状态空间探索）。用 online rollout fine-tune = 缩小 train/deploy distribution gap，让 world model 在 policy 实际经历的状态附近更准确。这是 DAgger 思想在 world model 上的自然延伸。

---

## 🔖 Section 总结

### 核心洞察
1. VLA post-training 领域：on-policy RL 在仿真有效，real-world 更适合 offline/weighted SFT
2. Flow-matching VLA 的 RL 障碍促使 VLAW 和 π₀.₆* 都走 weighted SFT 路线
3. Video diffusion world model 视觉质量已经很好，但物理准确性不足——根源在训练数据分布
4. VLAW 用 online rollout fine-tune world model = 解决 distribution shift，是简单而有效的修复
