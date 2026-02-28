# Related Works

---

## 2.1. Post-training Vision-Language-Action Models

Vision–language–action (VLA) models have achieved remarkable success in robotic manipulation tasks (Intelligence et al., 2025b; Pertsch et al., 2025; Liu et al., 2025a; Cui et al., 2025; Hu et al., 2024; Guo et al., 2024; Zhang et al., 2026). A common approach is to train the VLA on large-scale data and then perform supervised fine-tuning on target tasks (Zhang et al., 2025a; Black et al., 2024; Zhang et al., 2025b). Beyond supervised fine-tuning, improving VLA policies using online rollout data has emerged as a promising direction (Intelligence et al., 2025a; Guo et al., 2025b; Lu et al., 2025; Zang et al., 2025; Huang et al., 2024; Cheng et al., 2025). Some prior works adopt on-policy reinforcement learning methods, such as PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024), to improve VLA policies.

However, standard on-policy reinforcement learning typically requires a large number of rollouts and is therefore primarily validated in simulation environments (Li et al., 2025b;a; Liu et al., 2025b). Moreover, state-of-the-art VLA models are often trained with flow-matching objectives, which do not provide explicit policy likelihoods, making conventional policy-gradient methods difficult to apply. To enable policy learning in real-world settings, π₀.₆* (Intelligence et al., 2025a) instead adopts an offline or batch reinforcement learning formulation with an advantage-conditioned supervised learning objective. Similarly, in our setting, we perform iterative policy improvement using batches of real-world rollout data together with world-model–generated synthetic data, and update the policy exclusively through stable supervised fine-tuning objectives.

> 💡 **VLA post-training 的两条路线**：
> 1. **SFT 路线**：直接在 target task 上微调（简单但不能突破演示质量上限）
> 2. **RL 路线**：用 rollout 做 online RL（PPO/GRPO），但需要大量 rollout 且不适配 flow-matching
>
> 💡 **π₀.₆* 的定位**：跟本文最接近的 baseline——也是用 offline batch RL 做 VLA post-training，但没有世界模型。本文是在 π₀.₆* 思路上加了世界模型来扩充数据量。
>
> 💡 **Flow-matching 的特殊性**：这个约束很现实。Flow-matching 没有显式 log-prob，standard REINFORCE 无法直接用。这也是为什么 VLAW 绕开 policy gradient，改用 weighted SFT。

---

## 2.2. World Models for Decision Making

Action-conditioned world models predict future outcomes given current observations and actions, and are also referred to as forward dynamics models. Many works leverage such models for model-based reinforcement learning (Hafner et al., 2020; Hansen et al., 2022; Oh et al., 2015; Wu et al., 2024) and visual planning (Finn & Levine, 2017; Ebert et al., 2018; Xie et al., 2019; Dasari et al., 2019; Yang et al., 2023). Among these, the most closely related approaches to ours are DayDreamer (Wu et al., 2023), SOLAR (Zhang et al., 2019) and World4rl (Jiang et al., 2025), which also operate in real-world visual model-based reinforcement learning settings. However, due to limited model capacity and data scale, these earlier methods often learned task-specific dynamics models.

With recent advances in video diffusion models (Ren et al., 2025; Ball et al., 2025; Mei et al., 2026), it has become feasible to train multi-task action-conditioned world models that can generate realistic future visual observations (Chen et al., 2024; Gao et al., 2025; Zhu et al., 2024; 2025; Sharma et al., 2026). Despite this progress, accurately modeling complex physical dynamics remains a fundamental challenge, as widely observed in prior world-model literature (Guo et al., 2025a), likely because these models are trained on offline robotics datasets usually consisting primarily of demonstrations. To address this challenge, we leverage online policy rollout data to ground a pretrained world model in new environments, thereby improving its accuracy around the policy's state–action distribution.

> 💡 **世界模型发展脉络**：
> - **早期**：DayDreamer、SOLAR——受限于模型容量，只能学任务特定的动力学，泛化性差
> - **近期**：视频扩散模型使得多任务世界模型成为可能（Genie 3、iRASimu 等），视觉保真度大幅提升
> - **本文的突破点**：不只是追求视觉保真度，而是解决物理保真度（能不能正确预测接触结果），这才是 policy learning 真正需要的
>
> 💡 **"grounding" 的含义**：把通用世界模型（在 DROID 全数据集上训练的 Ctrl-World）适配到特定下游任务的物理分布上。思路类似 domain adaptation——offline 预训练模型 + online fine-tuning。
>
> 💡 **竞争工作**：World-Gymnast（Sharma et al., 2026）、WMPO（Zhu et al., 2025）也在做类似的 world model + VLA RL，这是 2026 年初的热门方向。
