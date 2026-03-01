# VLAW 批读笔记 · Abstract & Introduction

> 论文：**VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model**
> 作者：Yanjiang Guo, Tony Lee, Lucy Xiaoyang Shi, Jianyu Chen, Percy Liang, Chelsea Finn
> arXiv: 2602.12063

---

## Abstract

The goal of this paper is to improve the performance and reliability of vision-language-action (VLA) models through iterative online interaction. Since collecting policy rollouts in the real world is expensive, we investigate whether a learned simulator—specifically, an action-conditioned video generation model—can be used to generate additional rollout data.

> 💡 **核心动机清晰**：真实机器人 rollout 太贵（需要人工 reset + 监视），所以想用 World Model 模拟来生成额外数据。这是一个很自然的想法，之前也有人做过（DayDreamer 等），但都有一个老问题——World Model 的物理保真度不够。

Unfortunately, existing world models lack the physical fidelity necessary for policy improvement: they are predominantly trained on demonstration datasets that lack coverage of many different physical interactions (particularly failure cases) and struggle to accurately model small yet critical physical details in contact-rich object manipulation.

> 💡 **指出问题的根源非常准确**：现有 World Model 的两个致命弱点：
> 1. **过度乐观（Over-optimism）**：只在 demo 数据（基本全是成功轨迹）上训练，导致对交互结果的预测太乐观
> 2. **接触动力学建模差**：堆叠、擦写、挖取这类 contact-rich 任务里，小的物理细节（抓没抓住、是否真的接触）很难建模
> 
> 这一段其实是在替自己的方法铺垫：只要让 World Model 也看到 failure case，问题就能缓解。

We propose a simple iterative improvement algorithm that uses real-world roll-out data to improve the fidelity of the world model, which can then, in turn, be used to generate supplemental synthetic data for improving the VLA model.

> 💡 **VLAW 的核心思路（用一句话总结）**：Real rollout → 改进 World Model → World Model 生成合成数据 → 改进 VLA Policy → （循环）。这是一个**互利的正反馈循环**，很优雅。

In our experiments on a real robot, we use this approach to improve the performance of a state-of-the-art VLA model on multiple downstream tasks. We achieve a **39.2% absolute success rate improvement** over the base policy and **11.6% improvement** from training with the generated synthetic rollouts.

> 💡 **数字解读**：
> - 39.2% 是总提升（包含 world model 数据 + real rollout fine-tune 的综合效果）
> - 11.6% 是 synthetic rollout 单独贡献的提升（相对于只用 real rollout 的 Filtered BC 基线）
> 这两个数字要分清楚，后面 Table 2 有细节。

---

## 1. Introduction

Vision-language-action (VLA) models have achieved great success in robot manipulation by training on large-scale demonstration data [...]. Recent studies further show that VLA models can benefit substantially from post-training on online interaction rollouts. However, in real-world robotic settings, collecting online policy rollout trajectories requires significant human labor, such as resetting the environment and monitoring robot execution, which is expensive and time-consuming.

> 💡 **背景铺垫**：VLA 后训练（post-training）的必要性已经被验证（π₀.₆* 等工作），但 real rollout 成本高，这是当前的主要瓶颈。这给了 World Model-based synthetic data 一个明确的 motivation。

![](../images/1a0b6496130bf42cbf9395cfa1b9bd88a87f16bb485a255a1ea5b855941ee029.jpg)

*Figure 1: Real-world VLA rollout 成本高、难以规模化。VLAW 先用少量真实 rollout 学 World Model，再在想象中生成大规模合成数据。*

Instead of relying solely on real-world policy rollouts, learning an action-conditioned world model to generate synthetic rollouts in imagination offers a promising alternative. However, we find that existing world models lack the physical fidelity required for effective policy improvement. As noted in prior works, these models tend to be overly optimistic about predicted trajectories, as they are trained predominantly on demonstration datasets that lack coverage of diverse physical interactions, especially failure cases. Moreover, they struggle to accurately model small yet critical physical details in contact-rich manipulation and can produce blurry visual predictions.

> 💡 **文献综述嵌入 intro**：这里提到 Quevedo et al. 2025 和 Guo et al. 2025a（Ctrl-World）对这个问题的观察，说明这不只是本文的 claim，而是 community 的共识。
> 
> 注意：blurry visual predictions 在 contact-rich 场景里特别致命——机器人有没有真正抓住物体、纸巾有没有碰到白板，这类细节一旦模糊就没法用来训练 policy。

Consequently, existing action-conditioned world models have largely focused on relatively simple pick-and-place motions and often fail to generate reliable synthetic data for complex tasks involving frequent collisions or deformable objects.

> 💡 **现有工作的局限**：pick-and-place 是 robotics 领域的"简单题"，而本文专门挑了 stacking、erasing、scooping、drawing 这些有 contact 的任务来验证。这是一个有意识的 benchmark 选择，值得注意。

In this paper, we propose a simple yet scalable framework, VLAW, that iteratively improves VLA models via world-model rollouts. We first learn a physically-grounded world model by finetuning on online rollout data, which includes many failure cases. We find that after training on online rollout data, the world model learns to capture the complex dynamics encountered during policy execution, substantially improving its ability to model both success and failure cases.

> 💡 **关键 insight**：让 World Model 同时看到 success 和 failure，这样它就不再过度乐观了。**训练数据分布的改变**是核心贡献，而不是 model architecture 的改变——这让方法非常简洁（"simple yet scalable"）。

The improved world model is subsequently used to generate large-scale, high-fidelity synthetic trajectories, which are automatically annotated using a vision–language reward model (Lee et al., 2026).

> 💡 **自动标注奖励**：用 VLM（Qwen3-VL-4B）来判断合成轨迹是成功还是失败，避免了人工标注。这是整个 pipeline 能 scale 的关键——否则每条合成轨迹还需要人来看。

During policy optimization, we only use stable supervised learning objectives that can easily scale to large expressive models (e.g., flow-matching policies with intractable action probabilities), as opposed to dynamic programming/bootstrapping or policy gradients.

> 💡 **方法设计的重要决策**：不用 RL（PPO/GRPO 等），而是用简单的 supervised fine-tuning（SFT）。原因：
> 1. π₀.₅ 用的是 flow-matching，没有显式 log-likelihood，所以 policy gradient 很难用
> 2. SFT 更稳定，更容易 scale
> 这个选择牺牲了理论上的最优性，但换来了工程上的可行性——很务实。

![](../images/070ac6acefcb65430e2abb56f6111926f4a3411c0d30e91fe02d6869d3377a90.jpg)

*Figure 2: Policy online rollout 数据让 World Model "接地气"（grounded），之后 World Model 可以生成大量数据用于 policy 训练。*

The core contribution of this paper is a simple and scalable world-model-based reinforcement learning framework for improving state-of-the-art VLA policies in the real world. We start from a pretrained VLA policy, π₀.₅ and an action-conditioned world model, Ctrl-World. We first verify that, using policy online rollout data, we learn a physically grounded generative world model that can accurately model both success and failure trajectories. In addition, to obtain a reward model for robot tasks, we fine-tune Qwen3-VL on real-robot rollout data. Finally, using the synthetic data generated by the world model, we improve the pretrained π₀.₅ across many downstream contact-rich manipulation tasks that involve deformable objects in a multi-task setup, outperforming baseline with 11.6%.

> 💡 **贡献拆解**：
> 1. 物理保真度更高的 World Model（online rollout fine-tune）
> 2. 针对机器人任务的 VL 奖励模型（Qwen3-VL fine-tune）
> 3. 迭代的 VLA + World Model 协同改进 pipeline
> 
> 三个模块（World Model、Reward Model、Policy）各自都有 fine-tune，整个 pipeline 的训练开销是比较大的。后文应该关注具体的资源消耗。
