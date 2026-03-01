# VLAW 批读笔记 · Abstract

---

## Abstract

The goal of this paper is to improve the performance and reliability of vision-language-action (VLA) models through iterative online interaction. Since collecting policy rollouts in the real world is expensive, we investigate whether a learned simulator—specifically, an action-conditioned video generation model—can be used to generate additional rollout data.

> 💡 **核心动机**：真实机器人 rollout 成本高（需要人工 reset、监视），所以想用 World Model 替代真实环境来生成额外数据。这个动机清晰、直接。

Unfortunately, existing world models lack the physical fidelity necessary for policy improvement: they are predominantly trained on demonstration datasets that lack coverage of many different physical interactions (particularly failure cases) and struggle to accurately model small yet critical physical details in contact-rich object manipulation.

> 💡 **现有 World Model 的两个核心问题**：
> 1. **过度乐观（Over-optimism）**：只在成功 demo 上训练，对物理交互结果预测太乐观
> 2. **接触动力学建模差**：堆叠、擦写、挖取等 contact-rich 任务里，小的物理细节难以精确建模（且容易产生模糊预测）
>
> 这一段是为方法铺垫：只要让 World Model 也看到 failure case，问题就能缓解——核心修复思路非常直白。

We propose a simple iterative improvement algorithm that uses real-world roll-out data to improve the fidelity of the world model, which can then, in turn, be used to generate supplemental synthetic data for improving the VLA model.

> 💡 **VLAW 一句话**：Real rollout（含 failure）→ 改进 World Model → World Model 生成合成数据 → 改进 VLA Policy → 循环。这是一个**互利的正反馈循环**，结构很优雅。

In our experiments on a real robot, we use this approach to improve the performance of a state-of-the-art VLA model on multiple downstream tasks. We achieve a **39.2% absolute success rate improvement** over the base policy and **11.6% improvement** from training with the generated synthetic rollouts.

> 💡 **数字要分清楚**：
> - **39.2%** = 总提升（real rollout fine-tune + world model 合成数据的综合效果），即 base 46% → VLAW-2 87%
> - **11.6%** = 合成数据单独贡献（相对于只用 real rollout 的 Filtered BC 基线，75.2% → 86.8%）
>
> 后者才是 world model 真正贡献的增量，前者是整个 pipeline 的总效果。
