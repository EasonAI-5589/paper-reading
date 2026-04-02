[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
Introduction 从 VLA + CoT 的近期进展出发，拆解显式 CoT 的两个根本限制，然后逐一介绍 LaST₀ 的三个创新点（LaST CoT、MoT 双系统、异步频率训练），最后给出贡献列表。

---

> By inheriting the semantic understanding and common-sense reasoning capabilities of Vision-Language Models (VLMs), Vision-Language-Action (VLA) models integrate rich pretrained knowledge with the low-level control capabilities of robotic policies.

> 💡 **VLA 的基本范式**：VLM 的语义理解能力 → 迁移到机器人操作控制。

---

> Rather than simply mapping observations to actions, recent advances in VLA models have been inspired by the Chain-of-Thought (CoT) reasoning paradigm... some approaches enhance manipulation stability and interpretability by explicitly generating linguistic reasoning traces or affordance representations... other studies seek to capture environmental dynamics by predicting future states.

> 💡 **显式 CoT VLA 的两条路线**：
> | 路线 | 做法 | 代表工作 |
> |------|------|---------|
> | 语言 CoT | 生成文本推理链再 act | π₀.₅, CoT-VLA |
> | 视觉预测 | 预测未来图像/观测再 act | CoT-VLA, WorldVLA |

---

> Despite their demonstrated benefits, explicit CoT VLA methods remain constrained by two fundamental challenges in robotic manipulation.
> On the one hand, explicit reasoning typically incurs non-negligible inference latency... limiting the VLA model's ability to achieve real-time responsiveness.
> On the other hand, explicit reasoning is often confined to the linguistic space, imposing a representational bottleneck that struggles to faithfully capture ineffable physical attributes.

> 💡 **两个根本瓶颈（这是 LaST₀ 的全部动机）**：
> ```
> 问题 1：延迟
>   显式生成文本/图像 → 自回归 token by token
>   → CoT-VLA 只有 1.1 Hz（后面会验证）
>   → 机器人控制需要 >10 Hz
>
> 问题 2：表征
>   自然语言 ≠ 物理量
>   "机械臂向右移动" 无法精确表达 Δx=3.2cm, roll=0.05rad
>   → 对精细操作（插销、折毛巾）影响尤其大
> ```

---

> In this paper, we propose LaST₀, a dual-system VLA model that enables efficient reason-before-act behavior through a Latent Spatio-Temporal Chain-of-Thought (CoT)... unlike prior explicit CoT-based VLA methods, LaST₀ performs reasoning in a compact latent space, enabling the capture of fine-grained physical and robotic dynamics that are difficult to verbalize, while supporting temporally coherent modeling.

> 💡 **核心思路**：推理空间从 token（离散/高维）换成 latent vector（连续/压缩），同时支持时序建模。

---

> Specifically, we introduce a token-efficient latent CoT space that autoregressively predicts future latent tokens of 2D images, 3D point clouds, and proprioceptive states.

> 💡 **三模态 latent**：图像（语义）+ 点云（几何）+ 本体（运动学），自回归预测未来 k 步。

---

> Meanwhile, the latent CoT space is extended across future keyframes, enabling temporally consistent causal reasoning, which improves action coherence in closed-loop robotic manipulation.

> 💡 **时序延伸**：不只预测下一帧，而是预测未来 H 个关键帧的 latent 序列，形成因果时序链。

---

> Therefore, leveraging temporally extended latent conditions, we further propose a dual-system architecture implemented via a Mixture-of-Transformers (MoT) design. Specifically, two experts are integrated within a single VLA model: a slow reasoning expert... and a fast acting expert...

> 💡 **MoT 双系统**：
> ```
> 同一个 DeepSeek-LLM 1.5B，拆成两套权重（FFN/Attn 投影/LayerNorm）：
>   慢 expert → 生成 latent CoT（低频，每 κ 步一次）
>   快 expert → 生成 action（高频，每步）
>
> 共享 self-attention KV → 快 expert 能看到 latent CoT 上下文
> ```

---

> For the training procedure, both the latent reasoning expert and the action expert are initialized from the same pretrained VLM (i.e., Janus-Pro)... large-scale pretraining on diverse robotic manipulation datasets... SFT jointly optimizes the two experts... action expert trained with heterogeneous fast-slow operating ratios.

> 💡 **训练要点**：
> - 基座：Janus-Pro（DeepSeek-LLM 1.5B + SigLIP）
> - 预训练：400K+ 轨迹（Open-X, DROID, ROBOMIND）
> - SFT：混合比例（1:1, 1:2, 1:4）随机训练 → 推理时自由选频率

---

## 贡献总结

1. **LaST₀ 框架**：首个用 Latent Spatio-Temporal CoT 实现 reason-before-act 的 VLA 模型
2. **时空 latent CoT 空间**：自回归建模未来语义/几何/本体信息，时序一致
3. **MoT 双系统架构**：低频 latent 推理 + 高频 action 生成，协调慎思与响应
