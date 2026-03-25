[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

JEPA 的核心挑战是表征坍缩（representation collapse）——模型把所有输入映射到相同表征来"作弊"满足预测目标。现有方案要么靠 heuristics（EMA、SG、多项损失），要么靠冻结预训练编码器回避问题。LeWM 提出一个原理清晰的解法。

---

## 1.1 世界模型的愿景

> A central goal of artificial intelligence is to develop agents that acquire skills across diverse tasks and environments using a single, unified learning paradigm—one that operates directly from sensory inputs of its surroundings.

> World Models (WMs) are a powerful family of methods that learn to predict the consequences of actions in the environment. When successful, WMs allow agents to plan and to improve themselves solely from their model of the world, i.e., in imagination space.

> 💡 世界模型的核心价值：学好一个世界模型后，智能体可以在"想象空间"中规划和自我提升，不需要与真实环境交互。这在离线学习场景尤为重要。

---

## 1.2 JEPA 的坍缩问题

> A recent popular approach for learning world models is the Joint Embedding Predictive Architecture (JEPA). Instead of attempting to model every aspect of the environment, JEPA focuses on capturing the most relevant features needed to predict future states.

> However, despite their conceptual simplicity, existing JEPA methods are **highly prone to collapse**. In this failure mode, the model maps all inputs to nearly identical representations to trivially satisfy the temporal prediction objective.

> 💡 **坍缩是 JEPA 的"原罪"**: 如果把所有输入编码成同一个向量，预测下一步就是平凡的（永远预测那个常量向量）。预测损失为零，但学到的表征完全无用。

---

## 1.3 现有防坍缩方案的问题

现有方法按 Figure 2 分为三类：

| 类别 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| **端到端** | PLDM | 从像素端到端学习 | 7 项损失、6 个超参数、训练不稳定、无正式防坍缩保证 |
| **基于基础模型** | DINO-WM | 冻结编码器避免坍缩 | 不是端到端、表征受限于预训练编码器的知识边界 |
| **任务特定** | Dreamer, TD-MPC | 强 RL 性能 | 需要奖励信号或特权状态信息 |

> 💡 **每条路线都有"死穴"**:
> - PLDM: VICReg 启发的 7 项损失 → 超参调不好就坍缩，调好了也不稳定
> - DINO-WM: DINOv2 编码器是冻结的 → 永远学不到超出预训练知识的新表征
> - Dreamer/TD-MPC: 需要奖励 → 不是 task-agnostic，不符合通用世界模型的愿景

---

## 1.4 LeWM 的定位（Figure 2 的"全满"）

> We propose LeWorldModel (LeWM), the first method to learn a stable JEPA end-to-end from raw pixels without heuristic, principled, and simple.

LeWM 同时满足：
- ✅ 端到端学习（End-to-End）
- ✅ 从像素学（Pixels Based）
- ✅ 任务无关（Task Agnostic）
- ✅ 无需重建（Reconstruction Free）
- ✅ 无需奖励（Reward Free）
- ✅ 仅 1 个超参数
- ✅ 可证明的防坍缩保证

> 💡 这是一个"全都要"的方案，而且每一项都有清晰的技术支撑（不是 wishful thinking）。

---

## 1.5 三个贡献

1. **端到端 JEPA + 两项损失**: 稳定、鲁棒，超参搜索复杂度从 O(n⁶) 降到 O(log n)
2. **竞争性控制性能**: 15M 参数，规划快 48×，在 PushT 等任务上超过 PLDM 和 DINO-WM
3. **物理理解评估**: latent probing + Violation-of-Expectation，证明 latent space 编码了有意义的物理结构
