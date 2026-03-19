[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 从三个方向定位 Cosmos Policy：(1) Video-based robot policies — 多阶段 vs 统一模型；(2) VLA models — 不同的预训练 backbone 选择；(3) World models & value functions — 模块化 vs 统一架构。Cosmos Policy 在每个方向上都有明确的差异化定位。

---

## Video-based robot policies

Recent works have made great strides in leveraging video models for manipulation. Some methods first fine-tune video models on robot data and then train separate action modules to predict robot actions from generated video frames (Liang et al., 2025; Zhong et al., 2025; Hu et al., 2024; Liao et al., 2025; Unitree, 2025; Feng et al., 2025; Yang et al., 2025; Wang et al., 2025; He et al., 2024). Other works train unified video-action models that jointly predict future frames and actions (Li et al., 2025a; Zhu et al., 2025), but these approaches do not leverage pretrained video models and thus do not benefit from their spatiotemporal priors. In contrast to these works, we propose a single-stage fine-tuning approach that directly adapts pretrained video models to generate actions (as well as other modalities such as robot proprioceptive state and state values) within their native latent diffusion process.

> 💡 **批注**:
> 利用 video model 做 manipulation 的工作目前有两条路线：
> - **两阶段方法**（Video Policy, FlowVLA, VPP, Genie Envisioner, UnifolM-WMA, ViDAR, RoboEnVision 等）：先在 robot data 上微调视频模型，再训练独立的 action module 从生成的视频帧中预测动作。优势是利用了预训练视频模型的先验，但引入了多阶段训练和额外模块的复杂性
> - **统一模型**（UVA, UWM）：联合预测 future frames 和 actions，端到端简洁。但因为是自定义架构从头训练，没有利用预训练视频模型的 spatiotemporal priors
>
> Cosmos Policy 的定位：既利用预训练视频模型的先验，又保持端到端简洁——通过单阶段微调，直接在视频模型的 latent diffusion 过程中生成动作和其他模态。

---

## Vision-language-action models

State-of-the-art robotic manipulation policies increasingly leverage large pretrained backbones. Vision-language-action (VLA) models such as RT-2 (Brohan et al., 2023), OpenVLA (Kim et al., 2024), π₀.₅ (Intelligence et al., 2025), UniVLA (Bu et al., 2025), and CogVLA (Li et al., 2025b) fine-tune vision-language models on large-scale robotic imitation data, achieving strong performance across diverse manipulation tasks. While these methods exhibit strong generalization to various semantic concepts unseen in robotic interaction data, they leverage pretrained models that have mostly been trained on static image-text pairs rather than videos. In contrast to these VLAs, we leverage a pretrained video model that has learned spatiotemporal dynamics and implicit physics from predicting future frames for Internet-scale datasets. We hypothesize that this different pretrained backbone can serve as a strong foundation for low-level control policies.

> 💡 **批注**:
> SOTA 的 robotic manipulation policy 越来越多地用大型预训练模型做 backbone。VLA（RT-2, OpenVLA, π₀.₅, UniVLA, CogVLA）在大规模 robotic imitation data 上微调 VLM，在多种任务上表现很强，尤其擅长泛化到 robotic interaction data 中没见过的语义概念。但它们的 backbone 主要是在静态 image-text pairs 上训练的，缺乏视频的时序信息。
>
> Cosmos Policy 选择了不同的 backbone：预训练视频模型，它从互联网规模的视频数据中学到了 spatiotemporal dynamics 和 implicit physics。作者的假设是：这种不同的预训练 backbone 可以作为 low-level control policy 更好的基础。

---

## World models and value functions

World models have been used in various ways in robotics and reinforcement learning, from classical model-predictive control to modern neural approaches. Influential works such as Dyna (Sutton, 1991), MBPO (Janner et al., 2019), TD-MPC (Hansen et al., 2022; 2023), and the Dreamer family of works (Hafner et al., 2019; 2020; 2023) demonstrate the benefits of integrating planning with learning, using learned dynamics models to improve decision making in various control tasks. Recent works have explored different paradigms: FLARE (Zheng et al., 2025) adds learnable future tokens to diffusion transformer sequences to predict compact representations of future state, SAILOR (Jain et al., 2025) uses separate world and reward models with MPPI planning to iteratively search for better actions and refine the base policy, and Latent Policy Steering (Wang et al., 2025) pretrains world models using optical flow as an embodiment-agnostic action representation and subsequently trains a separate value function to steer the policy towards states with higher rewards. In contrast to these prior works that rely on separate modules for the policy, world model, and value function and typically train from models from scratch, we use a single unified architecture that serves simultaneously as the policy, world model, and value function and initialize from a pretrained video model.

> 💡 **批注**:
> World model 在 robotics 和 RL 中有很长的历史，从经典的 model-predictive control 到现代的神经网络方法。经典工作如 Dyna、MBPO、TD-MPC、Dreamer 系列都展示了将 planning 与 learning 结合的好处，用学到的 dynamics model 来改善决策。
>
> 近期的工作探索了不同范式：
> - **FLARE**：在 diffusion transformer 序列中加入 learnable future tokens 来预测紧凑的未来状态表示
> - **SAILOR**：用独立的 world model 和 reward model，配合 MPPI planning 迭代搜索更好的动作并优化 base policy
> - **Latent Policy Steering**：用 optical flow 作为 embodiment-agnostic 的动作表示来预训练 world model，再训练独立的 value function 把 policy 引导向高奖励状态
>
> 这些工作的共同点是：policy、world model、value function 是**分离的模块**，且通常**从头训练**。Cosmos Policy 的区别在于用**单一统一架构**同时做 policy、world model 和 value function，并且从**预训练视频模型**初始化。

---

## 🔖 Section 总结

### 核心洞察
1. Cosmos Policy 在 video-based policy 领域的定位是 "预训练 + 端到端"，避免了两阶段训练的复杂性
2. 与 VLA 的核心差异在于 backbone 的选择：video model vs VLM，对应不同的先验知识
3. 在 world model 方向的贡献是统一架构，用一个模型同时做 policy/world model/value function
4. 相比 FLARE、SAILOR 等最新方法，Cosmos Policy 更简洁且利用了预训练权重
