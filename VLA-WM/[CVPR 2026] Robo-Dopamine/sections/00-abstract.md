[← 返回 README](../README.md)

# Abstract

## 📌 预览
论文定义了两大核心问题（奖励模型不够好 + 奖励塑形有理论缺陷），对应提出两大方案（Dopamine-Reward + Dopamine-RL）。

---

The primary obstacle for applying reinforcement learning (RL) to real-world robotics is the design of effective reward functions. While recently learning-based Process Reward Models (PRMs) are a promising direction, they are often hindered by two fundamental limitations: their reward models lack step-aware understanding and rely on single-view perception, leading to unreliable assessments of fine-grained manipulation progress; and their reward shaping procedures are theoretically unsound, often inducing a semantic trap that misguides policy optimization.

> 💡 **问题定义**: 两大根本性局限——
> 1. **奖励模型本身不行**: 缺乏 step-aware 理解 + 依赖单视角 → 无法精确评估细粒度操作进度
> 2. **奖励塑形有理论缺陷**: 理论上不够 sound，会产生 "semantic trap"（语义陷阱）误导策略优化
>
> "semantic trap" 是本文的核心概念之一：agent 学会了到达高 reward 状态然后**停滞不前**，而不是完成任务。

---

To address these, we introduce Dopamine-Reward, a novel reward modeling method for learning a general-purpose, step-aware process reward model from multi-view inputs. At its core is our General Reward Model (GRM), trained on a vast 3,400+ hour dataset, which leverages Step-wise Reward Discretization for structural understanding and Multi-Perspective Reward Fusion to overcome perceptual limitations.

> 💡 **方案一 — Dopamine-Reward**:
> - 核心是 GRM（通用奖励模型），用 VLM 架构
> - 两个关键技术：Step-wise Reward Discretization（步级奖励离散化）+ Multi-Perspective Reward Fusion（多视角奖励融合）
> - 训练数据量巨大：3,400+ 小时

---

Building upon Dopamine-Reward, we propose Dopamine-RL, a robust policy learning framework that employs a theoretically-sound Policy-Invariant Reward Shaping method, which enables the agent to leverage dense rewards for efficient self-improvement without altering the optimal policy, thereby fundamentally avoiding the semantic trap.

> 💡 **方案二 — Dopamine-RL**:
> - Policy-Invariant Reward Shaping：数学上保证加入 dense reward 后最优策略不变
> - 这直接对应了 Ng et al. 1999 的 Potential-Based Reward Shaping (PBRS) 框架
> - 从根本上避免 semantic trap

- 💡 **Dense Reward 是什么？** RL 的奖励信号分两种：
  - **Sparse reward（稀疏奖励）**：只在任务完成时给 1，其他时刻都是 0。比如机器人把杯子放到盘子上，只有最后放好了才给奖励。问题是 agent 在漫长的操作过程中完全没有反馈，不知道自己是在靠近目标还是远离目标，探索极其困难。
  - **Dense reward（密集奖励）**：**每一步都给奖励信号**。比如 GRM 估计当前进度 Φ\*(s\_t)，然后每步计算 r = Φ\*(s\_{t+1}) - Φ\*(s\_t)——靠近目标就是正奖励，远离就是负奖励。这样 agent 每一步都有方向指引，学习效率大幅提升。
  - 本文的 dense reward 就是 GRM 提供的**逐步进度信号**。但 naive 的 dense reward 会导致 semantic trap（agent 停在高进度处不动），所以需要 PBRS 来保证理论正确性。

---

Extensive experiments across diverse simulated and real-world tasks validate our approach. GRM achieves state-of-the-art accuracy in reward assessment, and Dopamine-RL built on GRM significantly improves policy learning efficiency. For instance, after GRM is adapted to a new task in a one-shot manner from a single expert trajectory, the resulting reward model enables Dopamine-RL to improve the policy from near-zero to 95% success with only 150 online rollouts (approximately 1 hour of real robot interaction), while retaining strong generalization across tasks.

> 💡 **关键结果**:
> - 奖励精度：92.8% 准确率，VOC 0.953
> - 效率：one-shot 适配 → 150 rollouts (~1h) → 95% 成功率
> - 泛化：跨布局、背景、物体变化都保持性能

---

## 🔖 Section 总结

### 核心洞察
1. 现有 PRM 的两个根本缺陷：模型能力不足 + 奖励塑形理论有漏洞
2. 解法一一对应：GRM 解决模型问题，PBRS 解决理论问题
3. 极高的样本效率：150 次交互 = 1 小时真实机器人操作
