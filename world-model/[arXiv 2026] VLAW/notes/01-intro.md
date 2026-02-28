# Abstract + Introduction

---

## Abstract

The goal of this paper is to improve the performance and reliability of vision-language-action (VLA) models through iterative online interaction. Since collecting policy rollouts in the real world is expensive, we investigate whether a learned simulator—specifically, an action-conditioned video generation model—can be used to generate additional rollout data. Unfortunately, existing world models lack the physical fidelity necessary for policy improvement: they are predominantly trained on demonstration datasets that lack coverage of many different physical interactions (particularly failure cases) and struggle to accurately model small yet critical physical details in contact-rich object manipulation. We propose a simple iterative improvement algorithm that uses real-world roll-out data to improve the fidelity of the world model, which can then, in turn, be used to generate supplemental synthetic data for improving the VLA model. In our experiments on a real robot, we use this approach to improve the performance of a state-of-the-art VLA model on multiple downstream tasks. We achieve a 39.2% absolute success rate improvement over the base policy and 11.6% improvement from training with the generated synthetic rollouts.

> 💡 **核心问题表述**：现有世界模型不够用，因为它们只在演示数据上训练，没见过失败案例，没见过接触丰富的动态。
>
> 💡 **解题思路**：先用真实 rollout（含失败）修正世界模型保真度 → 再让世界模型生成合成数据 → 提升 VLA。这是个"鸡生蛋/蛋生鸡"的迭代框架。
>
> 💡 **数字要记住**：+39.2% 总提升，其中世界模型合成数据贡献 +11.6%（剩下来自真实 rollout 直接用于微调）。

---

## 1. Introduction

Vision-language-action (VLA) models have achieved great success in robot manipulation by training on large-scale demonstration data (Intelligence et al., 2025b; Kim et al., 2024; Shi et al., 2025; Guo et al., 2025b; Zhang et al., 2024; Chen et al., 2025). Recent studies further show that VLA models can benefit substantially from post-training on online interaction rollouts (Intelligence et al., 2025a). However, in real-world robotic settings, collecting online policy rollout trajectories requires significant human labor, such as resetting the environment and monitoring robot execution, which is expensive and time-consuming (Atreya et al., 2025; Jain et al., 2025). As a result, the number of online rollouts available for VLA models is often limited, restricting the effectiveness and scalability of post-training.

> 💡 **动机清晰**：VLA 的 post-training 已经证明有效（π₀.₆*），但真实 rollout 太贵了（每次都要人去 reset 环境、监督执行）。这是现实约束，不是技术问题。每个任务类别只用了 50 个 rollout，在工业实验室里算很少。

Instead of relying solely on real-world policy rollouts, learning an action-conditioned world model to generate synthetic rollouts in imagination offers a promising alternative (Team et al., 2025; Li et al., 2024; Team, 2025b). However, we find that existing world models lack the physical fidelity required for effective policy improvement. As noted in prior works, these models tend to be overly optimistic about predicted trajectories, as they are trained predominantly on demonstration datasets that lack coverage of diverse physical interactions, especially failure cases (Quevedo et al., 2025). Moreover, they struggle to accurately model small yet critical physical details in contact-rich manipulation and can produce blurry visual predictions (Guo et al., 2025a). Consequently, existing action-conditioned world models have largely focused on relatively simple pick-and-place motions and often fail to generate reliable synthetic data for complex tasks involving frequent collisions or deformable objects.

> 💡 **问题诊断精准**："过度乐观"是关键词。演示数据只有成功轨迹，世界模型根本没学过失败时物理世界长什么样，自然就会把所有轨迹都预测成成功。
>
> 💡 **任务难度定位**：接触丰富的任务（frequent collisions）和可变形物体——这类任务在传统仿真里也很难建模（需要精确的物理引擎），让神经网络世界模型来搞更难。这也是为什么不直接在 IsaacGym/MuJoCo 里做。

In this paper, we propose a simple yet scalable framework, VLAW, that iteratively improves VLA models via world-model rollouts, as shown in Figure 2. We first learn a physically-grounded world model by finetuning on online rollout data, which includes many failure cases. We find that after training on online rollout data, the world model learns to capture the complex dynamics encountered during policy execution, substantially improving its ability to model both success and failure cases. The improved world model is subsequently used to generate large-scale, high-fidelity synthetic trajectories, which are automatically annotated using a vision–language reward model (Lee et al., 2026). During policy optimization, we only use stable supervised learning objectives that can easily scale to large expressive models (e.g., flow-matching policies with intractable action probabilities (Intelligence et al., 2025b)), as opposed to dynamic programming/bootstrapping or policy gradients.

> 💡 **方法设计选择**：刻意不用 PPO/GRPO 之类的 on-policy RL——因为 π₀.₅ 用 flow-matching，没有显式 action log-prob，policy gradient 没法直接用。用加权 SFT（只训练成功轨迹）规避了这个问题。这是工程上的务实选择，作者在 4.3 节给了理论解释。
>
> 💡 **奖励模型是关键基础设施**：合成轨迹里哪些是"成功"的，靠 VLM 来判断，不需要人工标注。这才使得规模化成为可能。

The core contribution of this paper is a simple and scalable world-model-based reinforcement learning framework for improving state-of-the-art VLA policies in the real world. In our experiments, we use the widely used real-robot platform DROID (Khazatsky et al., 2024). We start from a pretrained VLA policy, π₀.₅ (Intelligence et al., 2025b) and an action-conditioned world model, Ctrl-World (Guo et al., 2025a). We first verify that, using policy online rollout data, we learn a physically grounded generative world model that can accurately model both success and failure trajectories, which is essential for generating useful synthetic data. In addition, to obtain a reward model for robot tasks, we fine-tune Qwen3-VL (Team, 2025a; Lee et al., 2026) on real-robot rollout data. Finally, using the synthetic data generated by the world model, we improve the pretrained π₀.₅ across many downstream contact-rich manipulation tasks that involve deformable objects in a multi-task setup, outperforming baseline with 11.6%.

> 💡 **Figure 1 & 2 的核心信息**：
> - Figure 1：真实 rollout 贵且不可扩展；想象中的 rollout 便宜且可无限生成
> - Figure 2：真实 rollout → 世界模型 grounding → 大量合成数据 → 策略提升
>
> 💡 **基础模型选择**：π₀.₅（Physical Intelligence 开发的当前最强开放 VLA）+ Ctrl-World（同组 Guo et al. 的世界模型）。这是典型的"把自己组的工作串起来做 paper"的操作——但效果是真实的。
