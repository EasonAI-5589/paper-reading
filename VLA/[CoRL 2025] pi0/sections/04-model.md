[← 返回 README](../README.md)

# IV. The π₀ Model

## 📌 预览
本节详细描述 π₀ 的模型架构：VLM backbone + action expert（MoE 风格），使用 conditional flow matching 建模连续动作分布，支持 50Hz 高频控制。

---

The $\pi _ { 0 }$ model, illustrated in Figure 3, consists primarily of a language model transformer backbone. Following the standard late fusion VLM recipe [3, 11, 30], image encoders embed the robot's image observations into the same embedding space as language tokens. We further augment this backbone with robotics-specific inputs and outputs — namely, proprioceptive state and robot actions. $\pi _ { 0 }$ uses conditional flow matching [28, 32] to model the continuous distribution of actions. Flow matching provides our model with high precision and multimodal modeling capability, making it especially well suited to high-frequency dexterous tasks. Our architecture is inspired by Transfusion [59], which trains a single transformer using multiple objectives, with tokens corresponding to continuous outputs supervised via a flow matching loss and tokens corresponding to discrete outputs supervised via a cross-entropy loss. Building on Transfusion, we additionally found that using a separate set of weights for the robotics-specific (action and state) tokens led to an improvement in performance. This design is analogous to a mixture of experts [45, 25, 12, 16] with two mixture elements, where the first element is used for image and text inputs, and the second is used for robotics-specific inputs and outputs. We refer to the second set of weights as the action expert.

> 💡 **批注**:
> - **Late fusion VLM**: 图像通过 encoder 映射到与语言 token 相同的 embedding 空间
> - **Flow matching 的优势**: 高精度 + 多模态建模 → 适合高频灵巧任务
> - **Transfusion 启发**: 同一 transformer 中不同 token 用不同 loss（cross-entropy vs flow matching）
> - **关键创新 — Action Expert**: 对机器人专用 token 使用独立权重 → MoE 风格（2 个 expert）
>   - Expert 1: VLM backbone → 处理图像 + 文本
>   - Expert 2: Action Expert → 处理机器人状态 + 动作

---

Formally, we want to model the data distribution $p ( \mathbf { A } _ { t } | \mathbf { o } _ { t } )$ , where $\mathbf { A } _ { t } = [ \mathbf { a } _ { t } , \mathbf { a } _ { t + 1 } , . . . , \mathbf { a } _ { t + H - 1 } ]$ corresponds to an action chunk of future actions (we use $H = 50$ for our tasks), and $\mathbf { o } _ { t }$ is an observation. The observation consists of multiple RGB images, a language command, and the robot's proprioceptive state, such that $\mathbf o _ { t } = [ \mathbf I _ { t } ^ { 1 } , . . . , \mathbf I _ { t } ^ { n } , \boldsymbol { \ell } _ { t } , \mathbf q _ { t } ]$ , where $\mathbf { I } _ { t } ^ { i }$ is $i ^ { \mathrm { th } }$ image (with 2 or 3 images per robot), $\ell _ { t }$ is a sequence of language tokens, and $\mathbf { q } _ { t }$ is a vector of joint angles. The images $\mathbf { I } _ { t } ^ { i }$ and state $\mathbf { q } _ { t }$ are encoded via corresponding encoders and then projected via a linear projection layer into the same embedding space as the language tokens.

> 💡 **批注**:
> - **Action chunk**: H=50 步的未来动作序列 → 50Hz 下对应 1 秒的动作
> - **观测构成**: 多张 RGB 图像 + 语言指令 + 关节角度
> - 所有模态统一映射到同一 embedding 空间 → transformer 统一处理

---

For each action $\mathbf { a } _ { t ^ { \prime } }$ in the action chunk ${ \bf A } _ { t }$ , we have a corresponding action token that we feed through the action expert. During training, we supervise these action tokens using a conditional flow matching loss [28, 32],

$$
L ^ { \tau } ( \boldsymbol { \theta } ) = \mathbb { E } _ { p ( \mathbf { A } _ { t } | \mathbf { o } _ { t } ) , q ( \mathbf { A } _ { t } ^ { \tau } | \mathbf { A } _ { t } ) } | | \mathbf { v } _ { \boldsymbol { \theta } } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } ) - \mathbf { u } ( \mathbf { A } _ { t } ^ { \tau } | \mathbf { A } _ { t } ) | | ^ { 2 } ,
$$

where subscripts denote robot timesteps and superscripts denote flow matching timesteps, with $\tau \in [ 0 , 1 ]$ . Recent work in high-resolution image [14] and video [38] synthesis has shown that flow matching can achieve strong empirical performance when combined with a simple linear-Gaussian (or optimal transport) probability path [28], given by $q ( \mathbf { A } _ { t } ^ { \tau } | \mathbf { A } _ { t } ) = \mathcal { N } ( \tau \mathbf { A } _ { t } , ( 1 - \tau ) \mathbf { I } )$ . In practice, the network is trained by sampling random noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , computing the "noisy actions" $\mathbf { A } _ { t } ^ { \tau } = \tau \mathbf { A } _ { t } + ( 1 - \tau ) \epsilon$ , and then training the network outputs $\mathbf { v } _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } )$ to match the denoising vector field ${ \bf u } ( { \bf A } _ { t } ^ { \tau } | { \bf A } _ { t } ) = { \bf A } _ { t } - \epsilon$ . The action expert uses a full bidirectional attention mask, so that all action tokens attend to each other. During training, we sample the flow matching timestep $\tau$ from a beta distribution that emphasizes lower (noisier) timesteps. See Appendix B for more details.

> 💡 **批注 — Flow Matching 核心原理**:
> - **目标**: 学习一个向量场 $v_\theta$，将噪声（τ=0）逐步变换为真实动作（τ=1）
> - **训练过程**:
>   1. 采样噪声 ε ~ N(0,I)
>   2. 构造含噪动作 $A_t^\tau = \tau A_t + (1-\tau)\epsilon$（线性插值）
>   3. 训练网络预测去噪方向 $u = A_t - \epsilon$
> - **vs Diffusion**: Flow matching 用线性路径（optimal transport），比 DDPM 的路径更直接
> - **τ 采样**: 用 beta 分布偏向低 τ（高噪声），因为预测高噪声下的去噪方向更有价值
> - **Action token 间**: 双向注意力（不是因果的）→ 动作之间互相看得到

---

At inference time, we generate actions by integrating the learned vector field from $\tau = 0$ to $\tau = 1$ , starting with random noise $\mathbf { A } _ { t } ^ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . We use the forward Euler integration rule:

$$
\mathbf { A } _ { t } ^ { \tau + \delta } = \mathbf { A } _ { t } ^ { \tau } + \delta \mathbf { v } _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } ) ,
$$

where $\delta$ is the integration step size. We use 10 integration steps (corresponding to $\delta = 0.1$) in our experiments. Note that inference can be implemented efficiently by caching the attention keys and values for the prefix $\mathbf { o } _ { t }$ and only recomputing the suffix corresponding to the action tokens for each integration step. We provide more details regarding the inference procedure, including the inference time for each part of the model, in Appendix D.

> 💡 **批注 — 推理过程**:
> - 从纯噪声开始，**10 步 Euler 积分**生成动作 → 比典型 diffusion (100+ 步) 快很多
> - **KV Cache 优化**: 观测部分只算一次，10 步积分只需重复计算 action token 部分
> - 推理时间：~73ms on-board (RTX 4090) → 完全满足实时控制需求

---

**Non-VLM baseline model.** In addition to our main VLA model, we also trained a similar baseline model that did not use a VLM initialization for ablation experiments. This model, which we refer to as $\pi _ { 0 }$ -small, has 470M parameters, does not use VLM initialization, and has a number of small differences that we found to be helpful for training on our data without VLM initialization, which are summarized in Appendix C. This model is used in our comparisons to evaluate the benefits of incorporating VLM pre-training.

> 💡 **批注**: 
> - π₀-small (470M) 作为 ablation baseline → 评估 VLM 预训练的价值
> - 架构差异：DistilBERT 编码语言、DiT 风格 action expert、encoder-decoder 交互（vs MoE）
> - 核心对比：**有 VLM 预训练 vs 无 VLM 预训练**

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| Action chunk 长度 H | 50 步 |
| Flow matching 积分步数 | 10 步 (δ=0.1) |
| π₀ 总参数量 | 3.3B (3B VLM + 300M action expert) |
| π₀-small 参数量 | 470M |
| 推理时间 (on-board) | ~73ms |
| 推理时间 (off-board) | ~86ms |

### 核心洞察
1. **MoE 风格架构**: VLM backbone 处理视觉-语言，Action Expert 处理机器人输入输出 → 通过 self-attention 交互
2. **Flow matching > 自回归离散化**: 支持连续动作、action chunking、高频控制
3. **高效推理**: KV cache + 10 步积分 → 73ms 内完成，支持 50Hz 控制
4. **τ 采样策略**: 偏向低 τ（高噪声）→ 动作预测任务的特殊需求（vs 图像生成偏向中间 τ）
