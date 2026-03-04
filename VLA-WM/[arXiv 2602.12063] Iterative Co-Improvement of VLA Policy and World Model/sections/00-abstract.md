[← 返回 README](../README.md)

# Abstract

## 📌 预览

本文提出 VLAW 框架：用少量真实机器人 rollout（含 failure）来改进 World Model 的物理保真度，再用改进后的 World Model 生成大量合成数据来提升 VLA policy 性能。两轮迭代后在 5 类 contact-rich 任务上取得 39.2% 绝对成功率提升。

---

The goal of this paper is to improve the performance and reliability of vision-language-action (VLA) models through iterative online interaction. Since collecting policy rollouts in the real world is expensive, we investigate whether a learned simulator—specifically, an action-conditioned video generation model—can be used to generate additional rollout data.

> 💡 **核心动机**：真实机器人 rollout 成本高（需要人工 reset、监视），因此研究是否能用学到的「模拟器」（action-conditioned video generation model，即 World Model）来生成额外的 rollout 数据替代真实收集。这个动机清晰直接，是当前 robot learning 领域真实存在的瓶颈。

Unfortunately, existing world models lack the physical fidelity necessary for policy improvement: they are predominantly trained on demonstration datasets that lack coverage of many different physical interactions (particularly failure cases) and struggle to accurately model small yet critical physical details in contact-rich object manipulation.

> 💡 **现有 World Model 的两个核心问题**：
> 1. **过度乐观（Over-optimism）**：训练数据以成功 demo 为主，缺乏 failure case 覆盖，导致 World Model 对物理交互结果预测过于乐观
> 2. **contact-rich 建模差**：堆叠、擦写、挖取等任务中，微小的物理细节（有没有真的抓住、有没有真的接触到）很难精确建模，且容易产生模糊预测
>
> 注意：这两个问题被前人工作（Quevedo et al. 2025、Guo et al. 2025a）所验证，不只是本文 claim。

We propose a simple iterative improvement algorithm that uses real-world roll-out data to improve the fidelity of the world model, which can then, in turn, be used to generate supplemental synthetic data for improving the VLA model.

> 💡 **VLAW 核心思路（一句话）**：Real rollout（含 failure）→ 改进 World Model → World Model 生成合成数据 → 改进 VLA Policy → 循环。这是一个**互利的正反馈循环**，解决思路极其简洁——核心修复是**训练数据的分布变化**，而不是 model architecture 的改动。

In our experiments on a real robot, we use this approach to improve the performance of a state-of-the-art VLA model on multiple downstream tasks. We achieve a $39.2\%$ absolute success rate improvement over the base policy and $11.6\%$ improvement from training with the generated synthetic rollouts. Videos can be found at this anonymous website: https://sites.google.com/view/vlaw-arxiv.

> 💡 **两个关键数字要分清**：
> - **39.2%** = 整个 pipeline 的总提升（base 46% → VLAW-2 87%），包含 real rollout fine-tune 和 world model 合成数据的综合效果
> - **11.6%** = 合成数据单独贡献的提升（Filtered BC-2 75.2% → VLAW-2 86.8%），这才是 world model 真正新增的价值
>
> 实验在**真实机器人**（不是仿真）上进行，是本文说服力的核心来源。

---

## 🔖 Section 总结

### 关键数字速查

| 指标 | 数值 |
|------|------|
| 总绝对成功率提升 | +39.2% |
| 合成数据单独贡献 | +11.6% |
| 实验平台 | 真实机器人（DROID） |

### 核心洞察
1. World Model 的物理保真度不足是核心障碍，根源在于训练数据只有 demo（成功轨迹）
2. 加入 failure rollout 来 fine-tune World Model，可以消除 over-optimism
3. 方法设计极简：不改 architecture，只改数据分布
