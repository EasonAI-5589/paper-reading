[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

三条研究路线：① 生成式世界模型 → ② JEPA 世界模型（核心）→ ③ Latent Dynamics 规划。

---

## 2.1 生成式世界模型

> World Models aim to learn predictive models of environment dynamics from data. A prominent class consists of generative approaches that explicitly model environment dynamics in pixel space.

代表工作：IRIS [3], DIAMOND [6], OASIS [8], DreamerV4 [4], Genie [9], HunyuanWorld [10]

> 💡 生成式 WM 在像素空间建模动力学，本质上是"学一个模拟器"。优点是直观，缺点是计算开销大（需要在高维像素空间做生成）。很多还依赖奖励信号，不是 task-agnostic。LeWM 属于 JEPA 路线，在紧凑 latent space 而非像素空间预测。

---

## 2.2 JEPA 世界模型

两大分支：

### 2.2.1 自监督表征学习

> One prominent line applies JEPA to self-supervised representation learning by predicting the latent embeddings of masked input patches: I-JEPA for images, V-JEPA for videos.

这些方法通常用 EMA + Stop-Gradient 稳定训练。

> 💡 EMA/SG 的问题：理论理解有限——"它们通常不对应一个良定义的目标函数的最小化" [17]。换句话说，EMA/SG 有效但"不知道为什么有效"。

### 2.2.2 动作条件 latent 世界建模

| 方法 | 策略 | 问题 |
|------|------|------|
| DINO-WM, OSVI-WM | 冻结预训练编码器 | 表征被预训练限死 |
| PLDM | VICReg + 端到端 | 7 项损失，训练不稳定 |
| 其他 | 辅助信号（本体感知、动作解码器） | 增加复杂度 |

> 💡 **LeWM 的定位**: 端到端 + 仅两项损失 + 可证明防坍缩。在 JEPA 世界模型这条线上，这是最"干净"的方案。

---

## 2.3 Latent Dynamics 规划

两种范式：

| 范式 | 描述 | WM 角色 |
|------|------|---------|
| **Imagination-based Policy** | 在 WM 中做 RL，训练策略 | 训练时用，推理时不用 |
| **MPC (Model Predictive Control)** | 推理时在 latent space 做规划 | 推理时持续使用 |

LeWM 采用 **MPC + CEM** 范式：推理时在 latent space 用 Cross-Entropy Method 优化动作序列。

> 💡 MPC 范式的优势：不需要额外训练策略网络，WM 本身就是控制器的一部分。缺点是推理时计算量更大——这正是 LeWM 的 48× 加速如此重要的原因。
