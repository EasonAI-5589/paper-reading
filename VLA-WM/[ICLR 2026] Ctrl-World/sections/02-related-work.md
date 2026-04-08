[← 返回 README](../README.md)

# 2. Related Work

## 两个主要相关方向

### 2.1 视频生成模型用于机器人学习

| 用法 | 代表工作 | 核心思路 |
|------|---------|---------|
| 生成合成轨迹 + 假 action | Gen2Act, etc. | 生成机器人视频，反推动作，扩充训练数据 |
| 视频模型直接作为 policy backbone | UniSim, SuSIE, etc. | 把 action 当作"视频的 token"输出 |
| 视频预测与 policy 联合训练 | DreamerV3, etc. | co-training，future prediction 作为辅助目标 |
| **Ctrl-World（本文）** | — | action-conditioned 预测未来观测，专供 policy 评估+改进 |

### 2.2 Action-Conditioned World Models

| 工作 | 特点 | 局限 |
|------|------|------|
| WPE (Quevedo et al., 2025) | 最近的 action-conditioned WM | 单视角，长时一致性差 |
| IRASim (Zhu et al., 2024) | 帧级动作条件的早期工作 | 单视角 |
| DreamerV3 (Hafner et al.) | 低维 state space WM | 不处理图像级别预测 |
| **Ctrl-World** | 多视角 + 帧级控制 + 记忆检索 | — |

---

## 💡 批读注解

本节相对常规，记一个关键信息：

**IRASim 是 Ctrl-World 的直接前身**，用了类似的"帧级 cross-attention 注入动作"思路，但只有单视角且没有记忆机制。Ctrl-World 在 IRASim 基础上加了多视角 + 记忆，是直接的扩展改进。

论文 Table 1 中有 IRASim vs Ctrl-World 的量化对比，Ctrl-World FVD **97.4 vs IRASim 138.1**，优势明显。
