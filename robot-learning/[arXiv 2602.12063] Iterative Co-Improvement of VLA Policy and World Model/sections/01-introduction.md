[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

Introduction 分三层铺垫：① VLA post-training 有价值但 real rollout 成本高；② World Model 是替代方案但现有 World Model 物理保真度不够；③ VLAW 的解法：用 online rollout（含 failure）ground World Model，再生成合成数据改进 VLA，迭代进行。贡献清单在最后。

---

Vision-language-action (VLA) models have achieved great success in robot manipulation by training on large-scale demonstration data (Intelligence et al., 2025b; Kim et al., 2024; Shi et al., 2025; Guo et al., 2025b; Zhang et al., 2024; Chen et al., 2025). Recent studies further show that VLA models can benefit substantially from post-training on online interaction rollouts (Intelligence et al., 2025a). However, in real-world robotic settings, collecting online policy rollout trajectories requires significant human labor, such as resetting the environment and monitoring robot execution, which is expensive and time-consuming (Atreya et al., 2025; Jain et al., 2025). As a result, the number of online rollouts available for VLA models is often limited, restricting the effectiveness and scalability of post-training.

> 💡 **第一段结构**：VLA 很成功（多文献支撑）→ post-training on online rollout 很有用（π₀.₆* 等工作）→ 但 real rollout 成本高是瓶颈。引用密度高说明这个方向当前非常活跃，是竞争激烈的领域。

![Figure 1](../images/1a0b6496130bf42cbf9395cfa1b9bd88a87f16bb485a255a1ea5b855941ee029.jpg)
*Figure 1: VLA model roll-outs in the real world are time-consuming and unscalable. In VLAW, we first learn an action-conditioned world model using limited real-world online rollouts, which in turn generates large-scale synthetic data in imagination.*

> 💡 **Figure 1 批读**：两列对比（real world vs. world model imagination）。左侧：真实 rollout 每条都需要人工 reset + 监视，不可规模化。右侧：world model 里可以并行生成大量轨迹。这张图直观说明了规模化潜力——但前提是 world model 要足够准确。

Instead of relying solely on real-world policy rollouts, learning an action-conditioned world model to generate synthetic rollouts in imagination offers a promising alternative (Team et al., 2025; Li et al., 2024; Team, 2025b). However, we find that existing world models lack the physical fidelity required for effective policy improvement. As noted in prior works, these models tend to be overly optimistic about predicted trajectories, as they are trained predominantly on demonstration datasets that lack coverage of diverse physical interactions, especially failure cases (Quevedo et al., 2025). Moreover, they struggle to accurately model small yet critical physical details in contact-rich manipulation and can produce blurry visual predictions (Guo et al., 2025a). Consequently, existing action-conditioned world models have largely focused on relatively simple pick-and-place motions and often fail to generate reliable synthetic data for complex tasks involving frequent collisions or deformable objects.

> 💡 **World Model 的两个已知问题**（前人工作已证实）：
> - **Over-optimism**：只在 demo（成功轨迹）上训练 → 对失败的物理交互预测不准
> - **Physical fidelity 差**：blurry prediction 在 contact-rich 场景里特别致命——机器人有没有真的抓住物体，细节一旦模糊就无法用来训练 policy
>
> 「**pick-and-place 是简单题**」这句话隐含了本文的 benchmark 定位——专门挑有频繁 contact 和 deformable objects 的难任务来验证。

In this paper, we propose a simple yet scalable framework, VLAW, that iteratively improves VLA models via world-model rollouts, as shown in Figure 2. We first learn a physically-grounded world model by finetuning on online rollout data, which includes many failure cases. We find that after training on online rollout data, the world model learns to capture the complex dynamics encountered during policy execution, substantially improving its ability to model both success and failure cases.

> 💡 **核心 insight**：让 World Model 同时看到 success 和 failure，它就不再 over-optimistic 了。这个改动的代价极小（只是额外收集一些 rollout 数据，不改模型结构），效果却显著——训练数据分布的改变是本文的核心贡献。

The improved world model is subsequently used to generate large-scale, high-fidelity synthetic trajectories, which are automatically annotated using a vision–language reward model (Lee et al., 2026).

> 💡 **自动标注的重要性**：用 VLM（Qwen3-VL-4B）自动判断合成轨迹是否成功，免去人工标注。这是整个 pipeline 能 scale 的关键——否则生成 500 条合成轨迹每条还需要人看。

During policy optimization, we only use stable supervised learning objectives that can easily scale to large expressive models (e.g., flow-matching policies with intractable action probabilities (Intelligence et al., 2025b)), as opposed to dynamic programming/bootstrapping or policy gradients.

> 💡 **不用 RL 的理由**：π₀.₅ 用 flow-matching，没有显式 log π(a|o)，所以 REINFORCE/PPO 的 log prob ratio 根本算不出来。这不是工程问题，是理论障碍。因此选择 weighted SFT（在成功轨迹上做 BC）——牺牲理论最优性，换来工程可行性和训练稳定性。

![Figure 2](../images/070ac6acefcb65430e2abb56f6111926f4a3411c0d30e91fe02d6869d3377a90.jpg)
*Figure 2: Policy online rollout data can help ground the pretrained world model in downstream tasks. Once the world model is grounded, we can generate massive data for policy learning.*

> 💡 **Figure 2 批读**：两阶段示意图。Phase 1：真实 rollout 数据（含 failure，绿色 + 红色）fine-tune world model，让它「接地气」。Phase 2：接地气的 world model 生成大量数据用于 policy 训练。注意图里 world model 生成的轨迹多样性——可以从同一初始状态生成大量不同轨迹来搜索成功的。

The core contribution of this paper is a simple and scalable world-model-based reinforcement learning framework for improving state-of-the-art VLA policies in the real world. In our experiments, we use the widely used real-robot platform DROID (Khazatsky et al., 2024). We start from a pretrained VLA policy, $\pi_{0.5}$ (Intelligence et al., 2025b) and an action-conditioned world model, Ctrl-World (Guo et al., 2025a). We first verify that, using policy online rollout data, we learn a physically grounded generative world model that can accurately model both success and failure trajectories, which is essential for generating useful synthetic data. In addition, to obtain a reward model for robot tasks, we fine-tune Qwen3-VL (Team, 2025a; Lee et al., 2026) on real-robot rollout data. Finally, using the synthetic data generated by the world model, we improve the pretrained $\pi_{0.5}$ across many downstream contact-rich manipulation tasks that involve deformable objects in a multi-task setup, outperforming baseline with $11.6\%$.

> 💡 **三个模块 + 三个验证**：
> 1. **World Model 接地气**（fine-tune on online rollout）→ 验证：Table 1 的 video quality + confusion matrix
> 2. **Reward Model 自动标注**（fine-tune Qwen3-VL）→ 验证：Appendix C 的混淆矩阵
> 3. **VLA Policy 提升**（training on synthetic data）→ 验证：Table 2 的成功率
>
> 注意：第一作者 Yanjiang Guo 同时也是 Ctrl-World 的作者，所以本文是在自己前作的基础上的延伸工作。

---

## 🔖 Section 总结

### 核心洞察
1. VLA post-training 需要 online rollout，但 real rollout 成本高 → World Model 是替代方案
2. 现有 World Model 因为只在 demo 上训练，over-optimistic，不足以用来改进 policy
3. VLAW 的解法：用 online rollout（含 failure）fine-tune World Model → 消除 over-optimism → 生成有效合成数据 → 迭代改进 VLA
4. 方法设计核心：不改 architecture，只改训练数据分布；不用 RL，用稳定的 weighted SFT
