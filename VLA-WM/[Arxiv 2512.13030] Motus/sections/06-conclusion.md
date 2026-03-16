[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览

总结 Motus 的核心贡献——在单一生成框架中统一五大具身建模范式，仿真 +15~45%、真实世界 +11~48%。未来方向指向更先进的统一架构、更通用的运动先验、以及从互联网规模视频中学习 latent action。

---

In this work, we present Motus, a unified latent-action world model that integrates mainstream capabilities of embodied foundation models into a single generative framework. Our approach bridges pretrained vision-language understanding and video generation experts through Mixture-of-Transformers architecture, while introducing optical-flow-driven latent actions to leverage internet-scale video data for cross-embodiment pretraining. Extensive experiments across simulation and real-world environments demonstrate that Motus consistently outperforms existing state-of-the-art embodied models (improved by +15~45% in simulation and +11~48% in real-world scenarios).

> 💡 **论文声称的三个层次贡献**：
>
> 1. **架构层**：MoT + Tri-model Joint Attention，统一 VLM + VGM + Action Expert
> 2. **数据层**：光流 latent action 解锁了无标注视频的预训练价值
> 3. **性能层**：仿真和真实世界双场景大幅超越 SOTA

---

In the future, we will continue to explore more advanced unified model architectures, pursue more universal motion priors, and learn latent actions from internet-scale general videos for embodied intelligence.

> 💡 **未来方向解读**：
>
> 论文提到三个方向，按重要性排序：
>
> | 方向 | 当前状态 | 潜在突破点 |
> |------|---------|-----------|
> | **更先进的统一架构** | MoT 是较新的选择，但不一定最优 | 更高效的跨模态融合方式（如 sparse MoE）|
> | **更通用的运动先验** | 目前用光流作为 latent action，仅编码 2D 运动 | 3D 光流、场景流、点追踪等更丰富的运动表示 |
> | **互联网规模视频的 latent action** | Stage 2 已做初步探索，但数据量有限 | 从 YouTube 级别的视频中学习运动先验 |
>
> **个人思考**：
> - 当前的光流 latent action 只编码 2D 像素运动，无法区分相机运动和物体运动——这是一个明显的局限
> - 如果能引入 3D 感知（深度估计 + 场景流），latent action 的表达力会大幅提升
> - 另一个未提到的方向：**在线学习 / test-time adaptation**——当前方法完全依赖离线训练，无法在部署时适应新环境

---

## 📊 全文总结

| 维度 | Motus 的回答 |
|------|-------------|
| **统一什么？** | VLA + WM + IDM + VGM + Joint，5 种分布 1 个模型 |
| **怎么统一？** | MoT 架构 + UniDiffuser 调度器 + Tri-model Joint Attention |
| **怎么预训练？** | 光流 latent action → 无标注视频也能训 action expert |
| **效果如何？** | 仿真 +15~45%，真实世界 +11~48%，跨平台一致领先 |
| **局限是什么？** | 2D 光流表达力有限、训练成本高（~18400 GPU-hrs）、未验证更大规模 |
