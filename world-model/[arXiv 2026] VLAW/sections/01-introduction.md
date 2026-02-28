[← 返回 README](../README.md)

# 1. Introduction

> 来源: VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model (arXiv 2602.12063)

---

## 📄 原文

> 💡 **Section 概览**: Introduction 分四段：① VLA 成功但 online rollout 贵；② 现有世界模型的物理保真度问题；③ VLAW 方案总览；④ 贡献总结（物理保真世界模型 + 奖励模型 + 迭代提升 VLA）。

Vision-language-action (VLA) models have achieved great success in robot manipulation by training on large-scale demonstration data (Intelligence et al., 2025b; Kim et al., 2024; Shi et al., 2025; Guo et al., 2025b; Zhang et al., 2024; Chen et al., 2025). Recent studies further show that VLA models can benefit substantially from post-training on online interaction rollouts (Intelligence et al., 2025a). However, in real-world robotic settings, collecting online policy rollout trajectories requires significant human labor, such as resetting the environment and monitoring robot execution, which is expensive and time-consuming (Atreya et al., 2025; Jain et al., 2025). As a result, the number of online rollouts available for VLA models is often limited, restricting the effectiveness and scalability of post-training.

> 💡 **问题背景**: VLA post-training（在线 rollout 微调）有效，但代价高。关键约束是**人工成本**（重置环境 + 监督机器人），不是计算成本。这与 LLM RLHF 的瓶颈（标注成本）类似，但机器人场景的代价更大。

![Figure 1](../images/1a0b6496130bf42cbf9395cfa1b9bd88a87f16bb485a255a1ea5b855941ee029.jpg)
*Figure 1. VLA model roll-outs in the real world are time-consuming and unscalable. In VLAW, we first learn an action-conditioned world model using limited real-world online rollouts, which in turn generates large-scale synthetic data in imagination.*

> 💡 **Figure 1 批读**: 直观展示核心动机——真实 rollout 昂贵（左），用世界模型在想象中生成大量合成数据（右）。这是本文所有设计的出发点。注意图示的流向：少量真实 rollout → 修正世界模型 → 大量想象 rollout → 提升策略。

Instead of relying solely on real-world policy rollouts, learning an action-conditioned world model to generate synthetic rollouts in imagination offers a promising alternative (Team et al., 2025; Li et al., 2024; Team, 2025b). However, we find that existing world models lack the physical fidelity required for effective policy improvement. As noted in prior works, these models tend to be overly optimistic about predicted trajectories, as they are trained predominantly on demonstration datasets that lack coverage of diverse physical interactions, especially failure cases (Quevedo et al., 2025). Moreover, they struggle to accurately model small yet critical physical details in contact-rich manipulation and can produce blurry visual predictions (Guo et al., 2025a). Consequently, existing action-conditioned world models have largely focused on relatively simple pick-and-place motions and often fail to generate reliable synthetic data for complex tasks involving frequent collisions or deformable objects.

> 💡 **现有世界模型的两大问题**:
> ```
> 问题 1: 过度乐观偏差（Overoptimism Bias）
> ├── 原因: 训练数据以演示为主，缺乏失败案例
> └── 症状: 世界模型预测轨迹时总倾向于"成功"
>
> 问题 2: 接触丰富操作建模差
> ├── 原因: 小的物理细节难以建模（抓握、碰撞、形变）
> └── 症状: 模糊的视觉预测，无法准确模拟接触动力学
> ```
> 这两个问题导致世界模型生成的合成数据质量低，甚至有害。

In this paper, we propose a simple yet scalable framework, VLAW, that iteratively improves VLA models via world-model rollouts, as shown in Figure 2. We first learn a physically-grounded world model by finetuning on online rollout data, which includes many failure cases. We find that after training on online rollout data, the world model learns to capture the complex dynamics encountered during policy execution, substantially improving its ability to model both success and failure cases. The improved world model is subsequently used to generate large-scale, high-fidelity synthetic trajectories, which are automatically annotated using a vision–language reward model (Lee et al., 2026). During policy optimization, we only use stable supervised learning objectives that can easily scale to large expressive models (e.g., flow-matching policies with intractable action probabilities (Intelligence et al., 2025b)), as opposed to dynamic programming/bootstrapping or policy gradients.

> 💡 **VLAW 的三个设计选择（每个都有深意）**:
> 1. **在线 rollout 修正世界模型**：直接对症下药，解决过度乐观偏差
> 2. **VLM 奖励模型**：用 Qwen3-VL 自动标注成功/失败，避免手动标注
> 3. **只用 SFT（flow-matching）不用 RL**：稳定、可扩展，不依赖显式 action likelihood（这很重要，因为 flow-matching 策略没有易于计算的 action log-prob）

The core contribution of this paper is a simple and scalable world-model-based reinforcement learning framework for improving state-of-the-art VLA policies in the real world. In our experiments, we use the widely used real-robot platform DROID (Khazatsky et al., 2024). We start from a pretrained VLA policy, $\pi_{0.5}$ (Intelligence et al., 2025b) and an action-conditioned world model, Ctrl-World (Guo et al., 2025a). We first verify that, using policy online rollout data, we learn a physically grounded generative world model that can accurately model both success and failure trajectories, which is essential for generating useful synthetic data. In addition, to obtain a reward model for robot tasks, we fine-tune Qwen3-VL (Team, 2025a; Lee et al., 2026) on real-robot rollout data. Finally, using the synthetic data generated by the world model, we improve the pretrained $\pi_{0.5}$ across many downstream contact-rich manipulation tasks that involve deformable objects in a multi-task setup, outperforming baseline with $11.6\%$.

> 💡 **贡献总结（3点）**:
> ```
> 1. 物理保真世界模型
>    └── 用 online rollout（含失败案例）微调 Ctrl-World
>        验证：success/failure 建模均准确
>
> 2. 机器人奖励模型
>    └── 微调 Qwen3-VL-4B-Instruct
>        作用：自动过滤世界模型生成的成功轨迹
>
> 3. 迭代 VLA 提升
>    └── 在 DROID 5 类接触丰富任务上 +11.6%（合成数据）
>        总提升 +39.2%（vs base policy）
> ```

---

## 🔖 Section 总结

### 关键数字速查

| 项目 | 内容 |
|------|------|
| 基础 VLA | π₀.₅ (Physical Intelligence) |
| 基础世界模型 | Ctrl-World (Guo et al., 2025a) |
| 奖励模型 | Qwen3-VL-4B-Instruct (微调) |
| 实验平台 | DROID (Franka Panda) |

### 核心洞察

1. **方案最小化原则**：不改模型架构，不用复杂 RL，只用微调 + SFT，简单有效
2. **关键洞察**：世界模型的「过度乐观偏差」是可修复的——只需加入在线 rollout（含失败案例）微调
3. **flow-matching + SFT 的选择**：避开了 RLHF/PPO 的不稳定性，但代价是无法直接优化奖励，需要依赖奖励模型过滤
