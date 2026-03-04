[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

三条相关工作线：2.1 在线 RL fine-tuning for VLA（现有方法为何不适用真实机器人）；2.2 World Model 本身（通用世界模型 vs. 具身操作世界模型的差异）；2.3 World Model as Simulator（最相关的前作，WoVR 在哪些地方超越）。

---

### 2.1 Online RL Fine-tuning for VLA Models

On-policy reinforcement learning has been increasingly adopted to fine-tune VLA models beyond imitation learning. However, directly transferring online on-policy fine-tuning to real robots remains impractical, as such methods require large-scale parallel rollouts, repeated environment resets, and tightly coupled policy–environment interaction, which are difficult to support under real-world hardware. To mitigate this, some off-policy approaches introduce offline data reuse or human intervention, but often suffer from limited scalability and performance degradation during online updates. An alternative direction builds large-scale real-robot infrastructures, yet existing systems still cannot practically support fully on-policy algorithms at scale. These limitations suggest that the challenge of online RL for VLA is systemic rather than algorithmic, motivating world-model-based approaches that decouple policy optimization from real-world interaction.

> 💡 **"Systemic rather than algorithmic"**：VLA 的在线 RL 问题不是算法不够好，而是基础设施层面的问题（真实机器人无法并行、重置成本高、需要人工监督）。这个判断为 world model-based RL 提供了清晰的 motivation——不是 PPO/GRPO 不行，而是根本无法在真实机器人上大规模运行。
>
> 顺带提到：当前能支持大规模实体 RL 的系统（SOP、RLinf-USER 等）也仍然无法满足完全 on-policy 算法的需求。

---

### 2.2 World Models

Recent progress in large-scale general-purpose world models has demonstrated strong long-horizon generation and spatial memory under large viewpoint changes. However, these models rely on complex Self-Forcing/DMD training pipelines, require massive pretraining data, cannot be trained from scratch, and are primarily designed for navigation-style tasks with mouse–keyboard control. In contrast, embodied manipulation exhibits fixed viewpoints, locally constrained dynamics, and fine-grained object interactions, leading to fundamentally different modeling objectives, data distributions, and inference requirements.

> 💡 **通用世界模型 vs. 具身操作世界模型的根本差异**：
> - 通用（游戏/导航）：大视角变化、连续状态空间、鼠标键盘控制
> - 具身操作：固定视角、局部约束动力学、精细物体交互（毫米级）
>
> 这说明直接把 Genie、DreamerV3 等通用世界模型套到机器人上是行不通的，需要专门设计。WoVR 选择 Wan 2.2-TI2V-5B 作为 backbone，然后做 action conditioning 适配。

To address embodied settings, prior works adapt pretrained video models into action-conditioned world models using projected end-effector position, AdaLN-based frame-wise action injection, cross-attention, and MoE-based conditioning. Despite improved action responsiveness, these approaches commonly suffer from slow inference, severe error accumulation in long-horizon autoregressive generation, and unstable modeling of fine-grained physical interactions, limiting their scalability for reinforcement learning.

> 💡 **现有 embodied world model 的三个问题**：推理慢（不适合 RL 采样）、长 horizon 误差积累、精细物理交互建模不稳定。这三个问题都是 WoVR 的 4.1 节要解决的，形成了很清晰的 problem → solution 对应关系。

---

### 2.3 World Models as Simulators

Many works have validated the correlation between VLA performance in real environments and in learned world models, demonstrating the potential of world models for out-of-distribution generalization, and exploring the use of WM-generated synthetic data to train VLAs. However, these approaches do not treat world models as true simulators.

World-Env and WMPO take an important step toward treating learned world models as simulators, aiming to avoid costly interaction with real environments during reinforcement learning. Despite these advances, both approaches largely treat the world model as a drop-in replacement for a standard simulator, mechanically coupling on-policy reinforcement learning with imagined rollouts. They do not explicitly address the fundamental challenge of reinforcement learning under hallucinated dynamics, where closed-loop prediction errors accumulate and incentivize policies to exploit model inaccuracies. As a result, these methods lack dedicated mechanisms to regulate rollout horizons, suppress post-success hallucinations, or align policy optimization with the reliability regime of the world model.

> 💡 **WoVR 相对于 WMPO / World-Env 的核心差异**：后两者把 world model 当作「drop-in replacement for a standard simulator」，没有考虑 world model 和标准仿真器的根本区别（会 hallucinate）。WoVR 是第一个明确把 hallucination 作为设计约束来考虑的框架：
> - 没有 rollout horizon 调控 → KIR
> - 没有 post-success hallucination 抑制 → Masked GRPO
> - 没有 policy-model 分布对齐 → PACE
>
> **与 VLAW 的比较**：VLAW 用 real rollout（含 failure）fine-tune world model 来减少 over-optimism，但 policy 优化用的是 binary-filtered BC（不是真正的 RL）。WoVR 用真正的 on-policy GRPO 在 world model 里做 RL，同时用三层机制控制 hallucination。两篇论文解决的是 world model-based policy improvement 的不同子问题。

---

## 🔖 Section 总结

### 核心洞察
1. VLA 在线 RL 的障碍是 systemic（基础设施）而非 algorithmic，world model 是解耦 policy 优化和真实交互的关键
2. 通用世界模型与具身操作世界模型的需求根本不同，不能直接复用
3. WMPO / World-Env 把 world model 当标准仿真器用，没有考虑 hallucination 的特殊性——这是 WoVR 的切入点
