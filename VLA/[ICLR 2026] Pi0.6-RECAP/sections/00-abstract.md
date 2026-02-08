[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文提出 RECAP（RL with Experience and Corrections via Advantage-conditioned Policies），一种让 VLA 模型通过真实世界部署经验进行强化学习自我提升的通用方法。核心思路：advantage conditioning + 异构数据（demo + on-policy + 人工干预）。

---

We study how vision-language-action (VLA) models can improve through real-world deployments via reinforcement learning (RL). We present a general-purpose method, RL with Experience and Corrections via Advantage-conditioned Policies (RECAP), that provides for RL training of VLAs via advantage conditioning. Our method incorporates heterogeneous data into the self-improvement process, including demonstrations, data from on-policy collection, and expert teleoperated interventions provided during autonomous execution. RECAP starts by pretraining a generalist VLA with offline RL, which we call $\pi_{0.6}^{*}$ that can then be specialized to attain high performance on downstream tasks through on-robot data collection. We show that the $\pi_{0.6}^{*}$ model trained with the full RECAP method can fold laundry in real homes, reliably assemble boxes, and make espresso drinks using a professional espresso machine. On some of the hardest tasks, RECAP more than doubles task throughput and roughly halves the task failure rate.

> 💡 **Abstract 批读**:
> - **问题**: VLA 模型如何通过真实世界部署进行 RL 自我提升？
> - **方法**: RECAP = advantage-conditioned policy + 异构数据（demonstrations + on-policy rollouts + expert interventions）
> - **模型**: π*0.6 = π0.6 + advantage conditioning，先 offline RL 预训练，再 on-robot 数据微调
> - **任务**: 叠衣服、组装箱子、制作浓缩咖啡
> - **关键数字**: 最难任务 throughput 翻倍，failure rate 减半
> - **核心卖点**: 这是一个 general-purpose recipe，不是针对某个特定任务的 trick

---

## 🔖 Section 总结

### 核心洞察
1. RECAP 是第一个将通用 RL recipe（human reward feedback + interventions）应用于 VLA 并在真实世界复杂任务上显著提升的工作
2. 三种数据来源缺一不可：demonstrations 提供基础、autonomous rollouts 提供探索、interventions 提供纠错
3. Advantage conditioning 是关键技术选择——比 PPO/AWR 更适合大规模 VLA
