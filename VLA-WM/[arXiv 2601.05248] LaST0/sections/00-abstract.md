[← 返回 README](../README.md)

# 0. Abstract

## 📌 预览
摘要直接点出显式 CoT VLA 的两个瓶颈——推理延迟和语言空间的表征局限——并提出用 Latent Spatio-Temporal CoT 解决，同时亮出核心数字：10 个真机任务提升 13-14%。

---

> Vision-Language-Action (VLA) models have recently shown strong generalization, with some approaches seeking to explicitly generate linguistic reasoning traces or predict future observations prior to execution. However, explicit reasoning typically incurs non-negligible inference latency, which constrains the temporal resolution required for robotic manipulation. Moreover, such reasoning is confined to the linguistic space, imposing a representational bottleneck that struggles to faithfully capture ineffable physical attributes.

> 💡 **两个根本问题**：
> 1. **延迟问题**：显式生成文本/图像推理链 → 自回归开销大 → 控制频率受限（后面实验是 CoT-VLA 1.1 Hz）
> 2. **表征瓶颈**：语言空间无法精确描述物理属性（位姿精度、力的分布、3D 空间结构）

---

> To mitigate these limitations, we propose LaST₀, a framework that enables efficient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), capturing fine-grained physical and robotic dynamics that are often difficult to verbalize.

> 💡 **解决方案**：把 CoT 的推理空间从「语言/像素」换成「latent 向量」，又快又能表达物理信息。

---

> Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories.

> 💡 **Latent CoT 的三个维度**：
> | 维度 | 内容 | 编码器 |
> |------|------|--------|
> | 视觉语义 | 未来 RGB 帧 | SigLIP-Large |
> | 3D 几何 | 未来点云 | Uni3D |
> | 本体感知 | 机器人关节状态 | action tokenizer |
> 时序上按未来关键帧顺序展开，形成「时空」推理链。

---

> Furthermore, LaST₀ adopts a dual-system architecture implemented via a Mixture-of-Transformers design, where a reasoning expert conducts low-frequency latent inference and an acting expert generates high-frequency actions conditioned on robotics-oriented latent representations.

> 💡 **双系统架构**：慢想（低频 latent 推理）+ 快做（高频 action 生成），用 MoT 在同一模型里实现，两个 expert 通过共享 self-attention 交互。

---

> Across 10 real-world tasks spanning tabletop, mobile, and dexterous hand manipulation, LaST₀ improves mean success rates by 13%, 14% and 14% over prior SOTA VLA methods, respectively.

> 💡 **核心指标**：三类平台全部提升 13-14%，覆盖桌面/移动/灵巧手，说明方法的泛化性。
