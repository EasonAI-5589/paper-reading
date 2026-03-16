[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

三条相关工作线：2.1 在线 RL fine-tuning for VLA（现有方法为何不适用真实机器人）；2.2 World Model 本身（通用世界模型 vs. 具身操作世界模型的差异）；2.3 World Model as Simulator（最相关的前作，WoVR 在哪些地方超越）。

---

### 2.1 Online RL Fine-tuning for VLA Models

On-policy reinforcement learning has been increasingly adopted to fine-tune VLA models beyond imitation learning. However, directly transferring online on-policy fine-tuning to real robots remains impractical, as such methods require large-scale parallel rollouts, repeated environment resets, and tightly coupled policy–environment interaction, which are difficult to support under real-world hardware. To mitigate this, some off-policy approaches introduce offline data reuse or human intervention, but often suffer from limited scalability and performance degradation during online updates. An alternative direction builds large-scale real-robot infrastructures, yet existing systems still cannot practically support fully on-policy algorithms at scale. These limitations suggest that the challenge of online RL for VLA is systemic rather than algorithmic, motivating world-model-based approaches that decouple policy optimization from real-world interaction.

> 💡 **为什么不直接在真实机器人上跑 RL？**
>
> 答案是**系统性障碍**（不是算法不行）：
>
> | 障碍 | 说明 |
> |------|------|
> | **并行 rollout** | PPO/GRPO 需要同时跑几百上千个环境采数据，你不可能买几百个机械臂 |
> | **环境重置** | 每次试错后要把桌面恢复原样（杯子放回去等），真实世界重置很麻烦，经常需要人工介入 |
> | **紧耦合交互** | On-policy RL 要求"跑一批 → 更新策略 → 再跑一批"，循环很紧，真实机器人太慢跟不上 |
>
> 有人尝试过替代方案：
> - **Off-policy 方法**（复用旧数据）→ 性能会退化
> - **建大规模机器人农场**（买很多机器人）→ 还是不够大，仍然撑不起完全 on-policy 的训练
>
> **论文的结论：问题出在基础设施，不是算法。** 所以需要换个思路——用 world model 在"想象"里做 RL，完全脱离真实机器人交互。

---

### 2.2 World Models

Recent progress in large-scale general-purpose world models has demonstrated strong long-horizon generation and spatial memory under large viewpoint changes. However, these models rely on complex Self-Forcing/DMD training pipelines, require massive pretraining data, cannot be trained from scratch, and are primarily designed for navigation-style tasks with mouse–keyboard control. In contrast, embodied manipulation exhibits fixed viewpoints, locally constrained dynamics, and fine-grained object interactions, leading to fundamentally different modeling objectives, data distributions, and inference requirements.

> 💡 **通用世界模型 vs 机器人操作世界模型——两类完全不同的东西**
>
> | | 通用世界模型（游戏/导航） | 具身操作世界模型（机器人） |
> |---|---------|-----------------|
> | **视角** | 大幅变化（第一人称走动） | 固定不动（桌面俯视角） |
> | **控制方式** | 键盘鼠标（离散、粗粒度） | 连续关节角度/末端位置（精细到毫米级） |
> | **关注点** | 空间记忆、大场景 | 物体间精细交互（夹爪碰杯子会怎样？） |
>
> **结论：不能直接拿通用世界模型套到机器人上。** WoVR 选了 Wan 2.2-TI2V-5B（视频生成模型）作为骨干，专门做了 action conditioning 适配。

To address embodied settings, prior works adapt pretrained video models into action-conditioned world models using projected end-effector position, AdaLN-based frame-wise action injection, cross-attention, and MoE-based conditioning. Despite improved action responsiveness, these approaches commonly suffer from slow inference, severe error accumulation in long-horizon autoregressive generation, and unstable modeling of fine-grained physical interactions, limiting their scalability for reinforcement learning.

> 💡 **现有机器人专用 world model 的三个通病**：
>
> 1. **推理太慢** → RL 需要大量采样（比如一轮训练要生成 40,000 帧画面），world model 推理速度就是整个训练的瓶颈。7 FPS vs 23 FPS 直接决定训练是"几周跑完"还是"几个月跑不完"
> 2. **长 horizon 误差积累** → 每一帧基于上一帧生成，误差像滚雪球越来越大，步数够多画面就完全崩掉
> 3. **精细物理交互建模不稳定** → 夹爪抓取这种关键瞬间最容易出幻觉
>
> 这三个问题都是 WoVR 的 4.1 节要解决的。

---

### 2.3 World Models as Simulators

Many works have validated the correlation between VLA performance in real environments and in learned world models, demonstrating the potential of world models for out-of-distribution generalization, and exploring the use of WM-generated synthetic data to train VLAs. However, these approaches do not treat world models as true simulators.

World-Env and WMPO take an important step toward treating learned world models as simulators, aiming to avoid costly interaction with real environments during reinforcement learning. Despite these advances, both approaches largely treat the world model as a drop-in replacement for a standard simulator, mechanically coupling on-policy reinforcement learning with imagined rollouts. They do not explicitly address the fundamental challenge of reinforcement learning under hallucinated dynamics, where closed-loop prediction errors accumulate and incentivize policies to exploit model inaccuracies. As a result, these methods lack dedicated mechanisms to regulate rollout horizons, suppress post-success hallucinations, or align policy optimization with the reliability regime of the world model.

> 💡 **WMPO / World-Env 的问题：把 world model 当成了普通模拟器的"平替"**
>
> 它们的做法是：以前用 MuJoCo 模拟器跑 RL → 现在把 MuJoCo 换成 world model → 其他什么都不改。
>
> 但 world model 和物理模拟器有**本质区别**：物理模拟器不会幻觉，world model 会！不针对幻觉做特殊处理的后果：
>
> | WMPO/World-Env 缺什么 | 后果 | WoVR 怎么补 |
> |----------------------|------|------------|
> | 没有控制 rollout 长度 | 误差积累到崩 | **KIR**（从关键帧开始，缩短 rollout） |
> | 没有处理成功后继续生成的幻觉 | RL 从乱七八糟的画面里学到垃圾 | **Masked GRPO**（屏蔽成功后的步骤） |
> | 没有解决 policy 更新后的分布漂移 | world model 越来越不准 | **PACE**（定期微调 world model） |
>
> **结果就是 WMPO 在 LIBERO-Long 上提升 0 pp。** 长步骤任务里幻觉太严重，RL 根本学不到东西。

---

## 🔖 Section 总结

### 整个 Related Work 的逻辑链
```
真实机器人跑 RL → 不现实（2.1：系统性障碍）
    ↓
用 world model 替代 → 但不能直接套通用 WM（2.2：需求根本不同）
    ↓
已经有人试过在 WM 里跑 RL → 但没考虑幻觉，效果差（2.3：WMPO 提升 0 pp）
    ↓
WoVR：显式对抗幻觉的框架（本文）
```
