[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
Introduction 回答三个问题：(1) 为什么视频模型适合做策略？(2) 现有方法有什么问题？(3) Cosmos Policy 怎么解决？核心论点是视频模型学到的时空动力学先验是做控制的天然好底子，而且可以用极简方式（不改架构）适配成策略。

---

Large pretrained video generation models have shown impressive ability to generate physically plausible and temporally coherent videos (NVIDIA et al., 2025; Wan et al., 2025; Yang et al., 2024; Bao et al., 2024; Kong et al., 2024; Zheng et al., 2024). Unlike pretrained vision-language models— which learn semantic concepts from static image-text pairs and have been popularized as robot policy backbones by recent vision-language-action (VLA) model research ((Brohan et al., 2023; Kim et al., 2024; Intelligence et al., 2025; Li et al., 2025b))—pretrained video generation models learn temporal causality, implicit physics, and motion patterns from millions of videos. These spatiotemporal priors hold significant value for robotics applications. In this work, we explore how to effectively leverage video models for robotic control and how they can incorporate policy rollout data to refine their world models and enable more effective planning.

> 💡 **批注**:
> 这段的逻辑线：video generation model 已经展现出生成物理合理、时序连贯视频的能力 → 而目前主流的 robot policy backbone 是 VLM（从静态 image-text pairs 学语义概念）→ 相比之下，预训练视频模型能学到 temporal causality（时序因果）、implicit physics（隐式物理）、motion patterns（运动模式）这些 spatiotemporal priors（时空先验），对机器人应用更有价值 → 本文探索如何利用 video model 做 robotic control，以及如何通过 policy rollout data 优化 world model 并实现更有效的 planning。

---

Prior works have made significant progress on adapting video models for robotic manipulation, leveraging both robot action data and "action-less" Internet video data to train generalizable policies and perform new tasks with small amounts of demonstrations (Liang et al., 2025; Zhong et al., 2025; Hu et al., 2024; Liao et al., 2025; Unitree, 2025; Feng et al., 2025; Yang et al., 2025; Wang et al., 2025). However, these works often require multiple training stages (e.g., video fine-tuning followed by action module training) and introduce new architectural components, such as separate action diffusers or inverse dynamics models. Other works avoid these complexities by training unified video-action models (Li et al., 2025a; Zhu et al., 2025), but they do not leverage pretrained video models due to their custom design, limiting their ability to capitalize on the spatiotemporal priors.

> 💡 **批注**:
> 之前的工作已经在把 video model 用于 robotic manipulation，利用 robot action data 和 action-less 的互联网视频数据来训练有泛化能力的 policy，使其能根据少量 demo 学会执行新任务。但这些工作通常需要多个训练阶段（比如先微调视频模型，再训练 action module），并且引入新的架构组件：
> - **Action diffuser**：独立的扩散模型，专门用来从视频模型的特征中生成动作序列，相当于在视频模型后面接了一个单独的动作生成器
> - **Inverse dynamics model**：给定当前帧和下一帧，反推出中间需要什么动作（即从状态转移反推动作），也是一个额外的模块
>
> 另一种思路是训练 unified video-action model，避免了多阶段和额外模块的复杂性，但因为是从头训的自定义架构，没有利用预训练视频模型的 spatiotemporal priors。
>
> → 两类方法各有缺陷：前者复杂，后者浪费先验。Cosmos Policy 的定位就是取两者之长。

---

In this work, we address these limitations with Cosmos Policy: an effective robot policy that is adapted from a pretrained video model (Cosmos-Predict2-2B (NVIDIA et al., 2025)) through a single stage of post-training on robot demonstrations. Unlike prior works which carefully design separate action modules and algorithms, Cosmos Policy makes no architectural modifications and instead leverages the pretrained model's core learning mechanism to capture action distributions. Since video models are effective at modeling complex, high-dimensional, multimodal distributions and can generate temporally coherent videos with hundreds of frames, we hypothesize that their learning algorithms are well-suited for representing actions alongside other modalities. Following this reasoning, we directly fine-tune a video model to simultaneously generate robot actions, future state images, and future state values (expected total cumulative rewards), all of which we encode as latent frames within the model's latent diffusion sequence. With future state and value predictions, Cosmos Policy can use best-of-N sampling to plan by generating candidate actions, imagining their resulting future states, ranking these states by predicted value, and executing the highest-value action. This search process produces trajectories that are more likely to succeed at the task.

> 💡 **批注**:
> Cosmos Policy 从 Cosmos-Predict2-2B 出发，不改架构、不加额外 action module，单阶段后训练就变成 robot policy。核心假设是：视频模型已经能建模复杂的高维多模态分布（视频生成已验证），那它的学习算法（latent diffusion）也应该能建模动作——动作不过是另一种 modality。
>
> 具体做法：直接微调视频模型，同时生成 action、future state image、future state value（期望累计奖励），全部编码为 latent frames 塞进扩散序列。有了 future state 和 value 的预测能力后，就可以做 best-of-N planning：生成多组候选动作 → 想象每组动作的未来状态 → 按 value 排序 → 执行价值最高的动作，提高任务成功率。
>
> 本质上是 model-based planning，但 policy、world model、value function 全部共享同一个模型。

---

Our main contribution is the Cosmos Policy approach for fine-tuning pretrained video models to incorporate different modalities that enable visuomotor control and planning. We evaluate our method in two modes: first as a direct policy (without planning) and then with model-based planning using the future state and value predictions. As a direct policy, Cosmos Policy achieves a new state of the art in both the LIBERO and RoboCasa simulation benchmarks (98.5% and 67.1% average success rates, respectively), outperforming diffusion-based policies trained from scratch, video-based policies (e.g., UVA, Video Policy), and even fine-tuned VLAs (e.g., π₀.₅, OpenVLA-OFT, CogVLA, UniVLA, DP-VLA, GR00T-N1.5). It also achieves the highest average success rate (93.6%) among state-of-the-art policies in challenging real-world bimanual manipulation tasks. Further, when enhanced with model-based planning, we observe a 12.5 percent higher task completion rate on average in two challenging real-world manipulation tasks. In these experiments, we show that Cosmos Policy can incorporate past experiences from policy rollouts to refine its world model and value function and plan more effectively. Lastly, we compare our model-based planning approach to a model-free variant and study their relative advantages.

> 💡 **批注**:
> 主要贡献是提出 Cosmos Policy：微调预训练视频模型，整合不同模态（action、future state、value），实现视觉运动控制和规划。在两种 setting 下评估：
>
> 1. **Direct policy**（不用 planning）：LIBERO 98.5%、RoboCasa 67.1%、真实世界双臂任务 93.6%，全面超过 diffusion policy、video-based policy 和 fine-tuned VLA
> 2. **Model-based planning**：通过 policy rollout 收集经验数据，优化 world model 和 value function，然后用 best-of-N 做 planning（生成候选动作 → world model 预测未来状态 s' → V(s') 评估 → 选最优），额外提升 12.5%
>
> 最后（Lastly），还对比了 model-based 和 model-free 两种 planning 方式。两者用的是**同一个模型**，区别在于 conditioning mask 不同：model-based 先用 world model 预测未来状态 s'，再以 s' 为条件预测 V(s')（mask 掉 s, a）；model-free 跳过 world model，直接以 (s, a) 为条件预测 Q(s, a)（mask 掉 s'）。实验表明 V(s') 更好，因为预测出的未来状态为价值评估提供了更丰富的信息。

---

## 🔖 Section 总结

### 核心洞察
1. **视频模型 vs VLM**：视频模型学到的 temporal dynamics 比 VLM 的 semantic understanding 更适合 low-level 控制
2. **统一 vs 模块化**：Cosmos Policy 证明不需要设计单独的 action module，视频扩散过程本身就能生成动作
3. **假设验证**：视频模型的学习算法（latent diffusion）确实能同时建模动作、图像、标量值等多种模态
4. **数据效率**：预训练视频模型的 spatiotemporal priors 带来了显著的数据效率提升
