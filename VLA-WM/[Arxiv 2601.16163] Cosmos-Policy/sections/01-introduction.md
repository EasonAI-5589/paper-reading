[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
Introduction 回答三个问题：(1) 为什么视频模型适合做策略？(2) 现有方法有什么问题？(3) Cosmos Policy 怎么解决？核心论点是视频模型学到的时空动力学先验是做控制的天然好底子，而且可以用极简方式（不改架构）适配成策略。

---

Large pretrained video generation models have shown impressive ability to generate physically plausible and temporally coherent videos (NVIDIA et al., 2025; Wan et al., 2025; Yang et al., 2024; Bao et al., 2024; Kong et al., 2024; Zheng et al., 2024). Unlike pretrained vision-language models— which learn semantic concepts from static image-text pairs and have been popularized as robot policy backbones by recent vision-language-action (VLA) model research ((Brohan et al., 2023; Kim et al., 2024; Intelligence et al., 2025; Li et al., 2025b))—pretrained video generation models learn temporal causality, implicit physics, and motion patterns from millions of videos. These spatiotemporal priors hold significant value for robotics applications. In this work, we explore how to effectively leverage video models for robotic control and how they can incorporate policy rollout data to refine their world models and enable more effective planning.

> 💡 **动机**: 这段点出了视频模型 vs VLM 的核心差异：
> - **VLM**（如 LLaVA, Qwen-VL）：从静态 image-text pairs 学习 → 语义理解强，但缺乏时序动力学知识
> - **视频模型**（如 Cosmos, Wan, Sora）：从视频学习 → 学到了 temporal causality（时序因果）、implicit physics（隐式物理）、motion patterns（运动模式）
> - 对于机器人控制来说，后者显然更 "对口"——控制的本质就是理解 action 如何导致 state 变化

---

Prior works have made significant progress on adapting video models for robotic manipulation, leveraging both robot action data and "action-less" Internet video data to train generalizable policies and perform new tasks with small amounts of demonstrations (Liang et al., 2025; Zhong et al., 2025; Hu et al., 2024; Liao et al., 2025; Unitree, 2025; Feng et al., 2025; Yang et al., 2025; Wang et al., 2025). However, these works often require multiple training stages (e.g., video fine-tuning followed by action module training) and introduce new architectural components, such as separate action diffusers or inverse dynamics models. Other works avoid these complexities by training unified video-action models (Li et al., 2025a; Zhu et al., 2025), but they do not leverage pretrained video models due to their custom design, limiting their ability to capitalize on the spatiotemporal priors.

> 💡 **现有方法的两类问题**:
> 
> | 方法路线 | 代表工作 | 问题 |
> |---------|---------|------|
> | 多阶段训练 + 额外模块 | Video Policy, FlowVLA, Genie Envisioner | 先微调视频模型，再训练单独的 action decoder/inverse dynamics → 复杂、多阶段 |
> | 统一视频-动作模型 | UVA, UWM | 不用预训练视频模型，从头训 → 没有利用 spatiotemporal priors |
> 
> Cosmos Policy 取两者之长：既用预训练视频模型（利用先验），又统一建模（不加额外模块）。

---

In this work, we address these limitations with Cosmos Policy: an effective robot policy that is adapted from a pretrained video model (Cosmos-Predict2-2B (NVIDIA et al., 2025)) through a single stage of post-training on robot demonstrations. Unlike prior works which carefully design separate action modules and algorithms, Cosmos Policy makes no architectural modifications and instead leverages the pretrained model's core learning mechanism to capture action distributions. Since video models are effective at modeling complex, high-dimensional, multimodal distributions and can generate temporally coherent videos with hundreds of frames, we hypothesize that their learning algorithms are well-suited for representing actions alongside other modalities. Following this reasoning, we directly fine-tune a video model to simultaneously generate robot actions, future state images, and future state values (expected total cumulative rewards), all of which we encode as latent frames within the model's latent diffusion sequence. With future state and value predictions, Cosmos Policy can use best-of-N sampling to plan by generating candidate actions, imagining their resulting future states, ranking these states by predicted value, and executing the highest-value action. This search process produces trajectories that are more likely to succeed at the task.

> 💡 **Cosmos Policy 的核心假设**:
> - 视频模型能建模复杂的高维多模态分布（已经在视频生成中验证）
> - 因此，其学习算法也应该能建模动作分布（动作不过是另一种 "modality"）
> - **Latent Frame Injection**：把动作、未来状态、价值都编码为 latent frames，插入视频模型的扩散序列 → 不改架构
> 
> **规划机制**：
> 1. 生成多个候选动作 (best-of-N)
> 2. 用 world model 预测每个动作的未来状态
> 3. 用 value function 给每个未来状态打分
> 4. 选分数最高的动作执行
> 
> 这本质上是 **model-based planning**，但 world model 和 value function 与 policy 共享同一个模型！

---

Our main contribution is the Cosmos Policy approach for fine-tuning pretrained video models to incorporate different modalities that enable visuomotor control and planning. We evaluate our method in two modes: first as a direct policy (without planning) and then with model-based planning using the future state and value predictions. As a direct policy, Cosmos Policy achieves a new state of the art in both the LIBERO and RoboCasa simulation benchmarks (98.5% and 67.1% average success rates, respectively), outperforming diffusion-based policies trained from scratch, video-based policies (e.g., UVA, Video Policy), and even fine-tuned VLAs (e.g., π₀.₅, OpenVLA-OFT, CogVLA, UniVLA, DP-VLA, GR00T-N1.5). It also achieves the highest average success rate (93.6%) among state-of-the-art policies in challenging real-world bimanual manipulation tasks. Further, when enhanced with model-based planning, we observe a 12.5 percent higher task completion rate on average in two challenging real-world manipulation tasks. In these experiments, we show that Cosmos Policy can incorporate past experiences from policy rollouts to refine its world model and value function and plan more effectively. Lastly, we compare our model-based planning approach to a model-free variant and study their relative advantages.

> 💡 **贡献总结**:
> 1. **方法**：Latent Frame Injection — 在视频扩散序列中注入非图像模态，不改架构
> 2. **直接策略 SOTA**：LIBERO 98.5%, RoboCasa 67.1%, ALOHA 93.6%
> 3. **Model-based planning**：world model + value function 做 best-of-N，额外提升 12.5%
> 4. **经验学习**：从 policy rollout 数据中学习，改进 world model 和 value function
> 
> **值得注意**：在 RoboCasa 上只用 50 个 demo 就超过了用 300 个 demo 的方法（如 Video Policy 66.0%, GR00T-N1.5 64.1%），数据效率极高！

---

## 🔖 Section 总结

### 核心洞察
1. **视频模型 vs VLM**：视频模型学到的 temporal dynamics 比 VLM 的 semantic understanding 更适合 low-level 控制
2. **统一 vs 模块化**：Cosmos Policy 证明不需要设计单独的 action module，视频扩散过程本身就能生成动作
3. **假设验证**：视频模型的学习算法（latent diffusion）确实能同时建模动作、图像、标量值等多种模态
4. **数据效率**：预训练视频模型的 spatiotemporal priors 带来了显著的数据效率提升
