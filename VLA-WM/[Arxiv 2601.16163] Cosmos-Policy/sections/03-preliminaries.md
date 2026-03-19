[← 返回 README](../README.md)

# 3. Preliminaries

## 📌 预览
这个 section 介绍两个基础背景：(1) Cosmos 视频模型的架构和训练目标；(2) MDP + 模仿学习的形式化。这些是理解后续 Section 4 方法设计的前提。

---

## Cosmos video model

The pretrained video model that serves as the initialization for Cosmos Policy is Cosmos-Predict2-2B-Video2World (NVIDIA et al., 2025), a latent video diffusion model that receives a starting image and textual description as input and predicts subsequent frames to create a short video. The model operates over continuous tokens encoded by the Wan2.1 spatiotemporal VAE tokenizer (Wan et al., 2025) and is trained using the EDM denoising score matching formulation (Karras et al., 2022). The core training objective for the denoiser network $D_\theta$ at noise level $\sigma$ is: $\mathcal{L}(D_\theta, \sigma) = \mathbb{E}_{\mathbf{x}_0, \mathbf{c}, \mathbf{n}} \left[ \| D_\theta(\mathbf{x}_0 + \mathbf{n}; \sigma, \mathbf{c}) - \mathbf{x}_0 \|_2^2 \right]$, where $\mathbf{x}_0$ is a clean VAE-encoded image sequence, c represents the textual description encoded as T5-XXL embeddings (Raffel et al., 2020), $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ is i.i.d. Gaussian noise used to corrupt $\mathbf{x}_0$, and $D_\theta$ is a diffusion transformer (Peebles & Xie, 2023) that learns to recover the clean sample given the corrupted one. $D_\theta$ conditions on c via cross-attention and on $\sigma$ via adaptive layer normalization (Perez et al., 2018; Peebles & Xie, 2023). The Wan2.1 tokenizer compresses a video sequence of size $(1 + T) \times H \times W \times 3$ into a latent sequence of size $(1 + T') \times H' \times W' \times 16$, where $T' = T/4$, $H' = H/8$, $W' = W/8$; these resulting latent frames compose $\mathbf{x}_0$ above. The first frame undergoes no temporal compression to allow for conditioning on a single input image. During training, a conditioning mask is used to ensure that the first latent frame corresponding to the input image remains clean (without noise) while subsequent frames are corrupted with noise.

> 💡 **Cosmos-Predict2-2B 关键技术细节**:
>
> **模型基本信息**：
> - 模型名：Cosmos-Predict2-2B-Video2World
> - 参数量：2B
> - 类型：Latent Video Diffusion Model
> - 输入：起始图像 + 文本描述 → 输出：后续视频帧
>
> **核心训练目标**（EDM denoising score matching）：
> $$\mathcal{L}(D_\theta, \sigma) = \mathbb{E}_{\mathbf{x}_0, \mathbf{c}, \mathbf{n}} \left[ \| D_\theta(\mathbf{x}_0 + \mathbf{n}; \sigma, \mathbf{c}) - \mathbf{x}_0 \|_2^2 \right]$$
> - **输入**：$\mathbf{x}_0 + \mathbf{n}$，即 clean latent sequence $\mathbf{x}_0$ 加上高斯噪声 $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$
> - **条件**：$\sigma$（噪声级别，通过 adaptive layer norm 注入）和 $\mathbf{c}$（T5-XXL 文本 embedding，通过 cross-attention 注入）
> - **目标**：让去噪网络 $D_\theta$（一个 DiT）从加噪的输入中恢复出 clean sample $\mathbf{x}_0$
> - **直觉**：模型学会"给定一张噪声图，还原出原始的干净视频帧序列"，这个去噪能力后续被直接复用来生成动作
>
> **核心组件**：
> - **VAE Tokenizer**：Wan2.1 spatiotemporal VAE
>   - 空间压缩：8× (H/8, W/8)
>   - 时间压缩：4× (T/4)，但第一帧不做时间压缩（用于条件输入）
>   - 通道数：16（latent channel dimension）
> - **去噪网络**：Diffusion Transformer (DiT)
>   - 文本条件：T5-XXL embeddings → cross-attention
>   - 噪声级别条件：$\sigma$ → adaptive layer normalization
>
> **为什么这个设计适合做 policy**：
> - VAE 的 latent space 是连续的 → 适合表示连续动作
> - DiT 能处理序列中的任意 token → 可以自然地加入新的 modality
> - 条件生成机制（conditioning mask）→ 可以控制哪些帧是输入、哪些是要生成的

---

## MDP formulation and imitation learning

We frame robotic manipulation tasks as finite-horizon Markov decision processes (MDPs) defined by the tuple $\langle S, A, T, R, H \rangle$, where $S$ is a set of states, $A$ is a set of actions, $T: S \times A \to \Pi(S)$ is the state transition function, $R: S \times A \to \mathbb{R}$ is the reward function, and $H \in \mathbb{N}$ is the time horizon, with time steps $t \in \{1, 2, \ldots, H\}$. We train a policy $\pi: S \to \Pi(A)$ to maximize rewards, using sparse rewards where $R(s_t, a_t) = 0$ for $t < H$ and terminal rewards $R(s_H, a_H) \in [0, 1]$. We train policies via imitation learning on expert demonstrations containing state-action pairs. Following Zhao et al. (2023), all policies predict action chunks—sequences of actions for multiple timesteps—to improve motion smoothness and success rates.

> 💡 **批注**:
> 把机器人操作任务形式化为有限时间步的 MDP $\langle S, A, T, R, H \rangle$：S 是状态集（相机图像、关节角度等），A 是动作集，T 是状态转移函数，R 是奖励函数，H 是最大步数。
>
> 几个关键设计选择：
> - **Sparse reward**：中间步骤奖励为 0，只在最后一步给 $R(s_H, a_H) \in [0, 1]$。模型在执行过程中完全没有中间信号，只有最后才知道成败。这也是 value function 很重要的原因——需要从中间状态"预判"最终会不会成功
> - **Imitation learning**：从专家示范（人类遥操作的 state-action 对）学习，不做 RL 探索。好处是不需要设计 dense reward，坏处是只能学到示范覆盖到的行为
> - **Action chunks**（来自 ACT, Zhao et al., 2023）：不是逐步预测单个动作，而是一次预测未来多步的动作序列。动作更连贯平滑，也减少了 policy 调用次数。Cosmos Policy 在 ALOHA 上用 50 步 = 2 秒 @25Hz

---

## World models and value functions

A world model $\hat{T}: S \times A \to \Pi(S)$ learns to predict the future state given current state and action, approximating the true environment dynamics. The value function for a policy $\pi$ at state $s$ represents expected discounted returns from $s$ under $\pi$. It is defined as $V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{k=t}^{H} \gamma^{k-t} R(s_k, a_k) \mid s_t = s \right] = \mathbb{E}_{\tau \sim \pi} \left[ \gamma^{H-t} R(s_H, a_H) \mid s_t = s \right]$ in the sparse reward setting, where $\gamma$ is a discount factor that backpropagates the terminal reward through time. We simply use Monte Carlo returns in this work, labeling each transition in a rollout with the observed return $\gamma^{H-t} R(s_H, a_H)$. Note: To be precise, we acknowledge that the true state is not fully observable, and the world model predicts future observations (robot proprioception and camera images). However, for notational simplicity and readability, we opt to use the term "state" and treat observations as approximations of the state.

> 💡 **批注**:
> **World model** $\hat{T}: S \times A \to \Pi(S)$：给定当前状态和动作，预测下一个状态会是什么。本质上是学习环境的动力学——"我在这个状态下做这个动作，世界会变成什么样"。
>
> **Value function** 的通用定义：
> $$V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{k=t}^{H} \gamma^{k-t} R(s_k, a_k) \mid s_t = s \right]$$
> 含义是：从状态 s 出发，按照 policy π 执行，未来所有奖励的折扣累加和的期望。$\gamma \in (0, 1)$ 是折扣因子，越远的奖励权重越小。
>
> 因为本文用的是 **sparse reward**（只有最后一步 $R(s_H, a_H) \in [0,1]$，中间全是 0），所以上面的求和简化为：
> $$V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \gamma^{H-t} R(s_H, a_H) \mid s_t = s \right]$$
> 直觉理解：value 就是"从当前状态出发，最终成功完成任务的折扣概率"。$\gamma^{H-t}$ 表示离终点越远，value 越小——鼓励模型尽快完成任务。
>
> **Monte Carlo returns**：本文用最简单的方式估计 value——直接用 rollout 的实际结果标注。一条轨迹跑完后，如果成功了（$R=1$），就把轨迹中每个 transition 标注为 $\gamma^{H-t} \times 1$；如果失败了（$R=0$），全部标注为 0。不需要 TD learning 之类的复杂方法。
>
> **简化假设**：严格来说状态不完全可观测（POMDP），world model 预测的其实是 observation（相机图像 + 本体感知）而非真正的 state。但为了表述简洁，论文统一用 "state" 指代。

---

## 🔖 Section 总结

### 关键技术组件
| 组件 | 具体实现 |
|------|---------|
| 基础模型 | Cosmos-Predict2-2B (DiT + Wan2.1 VAE) |
| 空间压缩 | 8× |
| 时间压缩 | 4×（第一帧除外）|
| Latent 通道 | 16 |
| 训练目标 | EDM denoising score matching |
| 文本编码 | T5-XXL |
| 奖励 | Sparse terminal reward ∈ [0,1] |
| Value 估计 | Monte Carlo returns |

### 核心洞察
1. Cosmos-Predict2 的 latent space 设计天然适合注入新模态：连续、高维、DiT 处理
2. Sparse reward + MC returns 是最简单的 value 学习方式，但对于长 horizon 任务可能不够准确
3. Figure 2 是理解整个方法的关键图——latent frame injection 是核心创新
