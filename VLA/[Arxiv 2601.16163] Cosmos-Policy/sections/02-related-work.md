[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 从三个方向定位 Cosmos Policy：(1) Video-based robot policies — 多阶段 vs 统一模型；(2) VLA models — 不同的预训练 backbone 选择；(3) World models & value functions — 模块化 vs 统一架构。Cosmos Policy 在每个方向上都有明确的差异化定位。

---

## Video-based robot policies

Recent works have made great strides in leveraging video models for manipulation. Some methods first fine-tune video models on robot data and then train separate action modules to predict robot actions from generated video frames (Liang et al., 2025; Zhong et al., 2025; Hu et al., 2024; Liao et al., 2025; Unitree, 2025; Feng et al., 2025; Yang et al., 2025; Wang et al., 2025; He et al., 2024). Other works train unified video-action models that jointly predict future frames and actions (Li et al., 2025a; Zhu et al., 2025), but these approaches do not leverage pretrained video models and thus do not benefit from their spatiotemporal priors. In contrast to these works, we propose a single-stage fine-tuning approach that directly adapts pretrained video models to generate actions (as well as other modalities such as robot proprioceptive state and state values) within their native latent diffusion process.

> 💡 **Video-based 方法分类**:
> 
> | 路线 | 方法 | 优势 | 劣势 |
> |------|------|------|------|
> | 两阶段 | Video Policy, FlowVLA, VPP, UnifolM, ViDAR, Genie Envisioner | 利用预训练视频模型 | 多阶段训练，需要额外 action decoder |
> | 统一但从头训练 | UVA, UWM | 端到端简洁 | 没有预训练先验 |
> | **Cosmos Policy** | 本文 | 利用预训练 + 端到端简洁 | — |
> 
> Cosmos Policy 的定位非常清晰：**既要预训练先验，又要端到端简洁**。

---

## Vision-language-action models

State-of-the-art robotic manipulation policies increasingly leverage large pretrained backbones. Vision-language-action (VLA) models such as RT-2 (Brohan et al., 2023), OpenVLA (Kim et al., 2024), π₀.₅ (Intelligence et al., 2025), UniVLA (Bu et al., 2025), and CogVLA (Li et al., 2025b) fine-tune vision-language models on large-scale robotic imitation data, achieving strong performance across diverse manipulation tasks. While these methods exhibit strong generalization to various semantic concepts unseen in robotic interaction data, they leverage pretrained models that have mostly been trained on static image-text pairs rather than videos. In contrast to these VLAs, we leverage a pretrained video model that has learned spatiotemporal dynamics and implicit physics from predicting future frames for Internet-scale datasets. We hypothesize that this different pretrained backbone can serve as a strong foundation for low-level control policies.

> 💡 **VLA vs Video Model 的哲学差异**:
> - **VLA 路线**（RT-2, OpenVLA, π₀, π₀.₅）：用 VLM 作为 backbone → 擅长语义泛化（"把红色的东西放到蓝色的东西旁边"）
> - **Video Model 路线**（Cosmos Policy）：用视频模型作为 backbone → 擅长动力学建模（"抓住滑块并拉开"）
> 
> 这两条路线不一定互斥，但 Cosmos Policy 的实验结果说明：对于 low-level manipulation，video model 的时空先验可能比 VLM 的语义先验更有用。这是一个重要的 empirical finding。

---

## World models and value functions

World models have been used in various ways in robotics and reinforcement learning, from classical model-predictive control to modern neural approaches. Influential works such as Dyna (Sutton, 1991), MBPO (Janner et al., 2019), TD-MPC (Hansen et al., 2022; 2023), and the Dreamer family of works (Hafner et al., 2019; 2020; 2023) demonstrate the benefits of integrating planning with learning, using learned dynamics models to improve decision making in various control tasks. Recent works have explored different paradigms: FLARE (Zheng et al., 2025) adds learnable future tokens to diffusion transformer sequences to predict compact representations of future state, SAILOR (Jain et al., 2025) uses separate world and reward models with MPPI planning to iteratively search for better actions and refine the base policy, and Latent Policy Steering (Wang et al., 2025) pretrains world models using optical flow as an embodiment-agnostic action representation and subsequently trains a separate value function to steer the policy towards states with higher rewards. In contrast to these prior works that rely on separate modules for the policy, world model, and value function and typically train from models from scratch, we use a single unified architecture that serves simultaneously as the policy, world model, and value function and initialize from a pretrained video model.

> 💡 **World Model 方法对比**:
> 
> | 方法 | 架构 | 初始化 | 特点 |
> |------|------|--------|------|
> | Dreamer 系列 | 单独的 world model + policy | 从头训练 | RL 导向，low-dim 任务 |
> | TD-MPC | dynamics model + value | 从头训练 | 在线优化动作序列 |
> | FLARE | DiT + learnable tokens | 从头训练 | 紧凑 future state 表示 |
> | SAILOR | 分离的 world + reward model | 分离训练 | MPPI 规划 |
> | **Cosmos Policy** | **统一架构** | **预训练视频模型** | **一个模型做三件事** |
> 
> Cosmos Policy 的独特之处：
> 1. **统一架构**：同一个 diffusion transformer 既是 policy，又是 world model，又是 value function
> 2. **预训练初始化**：从 Cosmos-Predict2 初始化，而非从头训练
> 3. 这种 "一体化" 设计的优势：参数共享、表示对齐、训练简洁

---

## 🔖 Section 总结

### 核心洞察
1. Cosmos Policy 在 video-based policy 领域的定位是 "预训练 + 端到端"，避免了两阶段训练的复杂性
2. 与 VLA 的核心差异在于 backbone 的选择：video model vs VLM，对应不同的先验知识
3. 在 world model 方向的贡献是统一架构，用一个模型同时做 policy/world model/value function
4. 相比 FLARE、SAILOR 等最新方法，Cosmos Policy 更简洁且利用了预训练权重
