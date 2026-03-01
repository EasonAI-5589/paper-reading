# VLAW 批读笔记 · Method

---

## 4. Co-Improvement of VLA and World Model

In this section, we describe the details of our method. The overall pipeline consists of the following steps:

1. **World model post-training (Sec. 4.1):** Finetune the world model $M$ using real-world rollout data $\mathcal{D}_{\mathrm{real}}$, jointly training it with the original DROID dataset $\mathcal{D}_{\mathrm{DROID}}$ to maintain broad coverage. Also finetune the vision-language reward model $R$ on $\mathcal{D}_{\mathrm{real}}$.
2. **VLA policy post-training (Sec. 4.2):** Generate synthetic dataset $\mathcal{D}_{\mathrm{syn}}$, filter with reward model to get $\mathcal{D}_{\mathrm{syn}}^+$, then finetune the VLA policy.
3. **Alternate between Steps 1 and 2** iteratively.

> 💡 **算法结构总览**：整个 pipeline 有三个可学习组件——World Model、Reward Model、VLA Policy——它们轮流被 real rollout 数据改进。这是一个典型的 EM 风格的迭代优化：固定其他，更新一个。

![](../images/4c94b6400adfaca4ef97e0ccaa83e70632d70a670bdc9b7a7e9be76efe32bbd3.jpg)

*Figure 3: VLAW 完整 pipeline：(1) 真实 rollout → (2) World Model fine-tune → (3) 大规模合成轨迹生成 → (4) Reward Model 筛选 → Policy 更新。*

---

### 4.1. World Model Learning with Real Roll-outs

**Real World Policy Roll-outs.** Previous work has identified two major challenges in learning effective world models: (1) over-optimism, as training data is dominated by successful demonstrations; and (2) limited physical fidelity, particularly when modeling complex dynamics involving frequent contacts or deformable objects.

To address these issues, we get $K$ trajectories by rolling out the policy in the real world, forming a dataset $\mathcal{D}_{\mathrm{real}} = \{\tau_{\mathrm{real}}^1, ..., \tau_{\mathrm{real}}^K\}$, we also assign a sparse reward $r_\tau \in \{0, 1\}$ to each trajectory to indicate success or not every time we reset robot.

> 💡 **$K=50$ per task**（见 Sec 5.1）：每个任务类别收集 50 条真实 rollout，5 类任务共 250 条。这是相对较少的数据量，能用这么少的 real rollout 撬动如此大的提升，是本文的卖点之一。

**Training Objective.** We initialize from the pretrained Ctrl-World model, a strong diffusion-based world model trained on the full DROID dataset $\mathcal{D}_{\mathrm{DROID}}$. Finetuning on the online rollout dataset $\mathcal{D}_{\mathrm{real}}$ follows the original diffusion objective:

$$\mathcal{L}_{\mathcal{D}_{\mathrm{real}}} = \mathbb{E}_{x_0, \epsilon, t'} \left\| \hat{x}_0(x_{t'}, t', c) - x_0 \right\|^2$$

where the prediction target $x_0 = o_{t+1:t+H}$ is sampled from $\mathcal{D}_{\mathrm{real}}$, $x_{t'} = \sqrt{\bar{\alpha}_{t'}} x_0 + \sqrt{1-\bar{\alpha}_{t'}} \epsilon_{t'}$ denotes the noised future at diffusion step $t'$, and $c$ represents conditioning inputs including the action chunk $a_{t:t+H}$ and current observation $o_t$.

> 💡 **Loss 很标准**：就是 Stable Video Diffusion（Blattmann et al. 2023）的 diffusion loss，没有任何花哨改动。关键的改变完全来自训练数据（加入 failure rollout），而不是 loss 设计。这是一个"数据为王"的论点。

**Progressively Growing Dataset and Co-training.** During successive iterations, we continuously append newly collected trajectories: $\mathcal{D}_{\mathrm{real}} = \mathcal{D}_{\mathrm{real}} \cup \tau_{\mathrm{real}}^i$. To prevent overfitting to the limited online rollout data, we also co-train with the original DROID dataset:

$$\mathcal{L} = \mathcal{L}_{\mathcal{D}_{\mathrm{real}}} + \lambda \mathcal{L}_{\mathcal{D}_{\mathrm{DROID}}}$$

> 💡 **Co-training 的必要性**：直接只在 50 条 rollout 上 fine-tune 肯定会 overfit，加入 DROID 作为正则化是合理的。但 $\lambda$ 怎么调？论文没有给 ablation 说明这个超参的敏感度——这是一个小遗漏。

**Finetuning Reward Model.** We leverage Qwen3-VL-4B-Instruct to assess whether a trajectory succeeds or not. We find that the zero-shot VLM is not accurate enough, so in the first iteration, we fine-tune the VLM with the success labels $r_\tau$ in $\mathcal{D}_{\mathrm{real}}$.

The reward model takes as input a trajectory video $\tau_{\mathrm{real}}^i$ together with a query asking whether the task instruction $I^i$ is successfully completed. We classify a trajectory as successful if the probability assigned to the 'yes' token exceeds a threshold $\alpha$:

$$R(\tau^i) = \mathbf{1}\left[P(\textsf{yes} \mid \tau^i, I^i) > \alpha\right]$$

> 💡 **Threshold 的作用**：直接输出 yes/no 会过于乐观（Appendix C 的混淆矩阵显示 FP 很多），设 threshold=0.8 后 FP 从 8 降到 2。这个细节非常重要——如果 reward model 把太多失败轨迹标为成功，合成数据就会有噪声，损害 policy 训练。

---

### 4.2. Iterative Improvement for VLA Policy

**Scalable Training Pipeline.** We generate $N$ trajectories by rolling out the policy in imagination: $\mathcal{D}_{\mathrm{syn}} = \{\tau_{syn}^1, ..., \tau_{syn}^N\}$. We then apply the reward model to identify successful trajectories and construct a filtered dataset containing only success cases: $\mathcal{D}_{\mathrm{syn}}^+$.

> 💡 **$N=500$ per task**：World Model 生成 500 条合成轨迹，是 real rollout（50 条）的 10 倍。假设 reward model 筛选率约 50%（粗估），实际用于训练的合成 success 轨迹约 250 条。这是一个 10x 的数据放大效应。

**Policy Learning Objective.** We update the π₀.₅ policy using a weighted flow-matching objective over both real-world rollouts and world-model–generated data. After filtering for successful trajectories, we assign a binary weight $w(o,a)=1$ to transitions from successful trajectories and $w(o,a)=0$ to transitions from failed trajectories:

$$\mathcal{L} = \mathbb{E}_{(o,a) \sim \mathcal{D}_{\mathrm{syn}}^+ \cup \mathcal{D}_{\mathrm{real}}^+} \mathcal{L}_{\mathrm{FM}}(\theta; o, a)$$

> 💡 **本质是 Behavior Cloning on successes**：这个目标就是在成功轨迹上做 BC，没有负样本的 contrastive 项，也没有 Q-value 加权。虽然理论上不如 advantage-weighted regression（AWR），但实践中很稳定。
>
> **和 Filtered BC 基线的区别**：基线只用 50 条 real rollout 里的 success 轨迹做 BC，VLAW 多了 500 条合成 success 轨迹——差异仅在于数据量，方法是一样的。所以 11.6% 的提升完全归功于 World Model 合成数据的质量。

---

### 4.3. Relation to Regularized Reinforcement Learning

Under the regularized RL setting, the optimal improved policy admits a closed-form solution:

$$\pi^*(a \mid o) \propto w(o,a) \pi_{\mathrm{ref}}(a \mid o), \quad w(o,a) = \exp\left(\frac{A^{\pi_{\mathrm{ref}}}(o,a)}{\beta}\right)$$

Since the VLA policy uses flow-matching (no explicit log-likelihood), we define a surrogate divergence:

$$D_{\mathrm{FM}}(\pi^*(\cdot \mid o), \pi_\theta(\cdot \mid o)) \triangleq \mathbb{E}_{a \sim \pi^*(\cdot \mid o)} \left[ \mathcal{L}_{\mathrm{FM}}(\theta; o, a) \right]$$

By setting $\gamma \to 1$ and assigning large negative reward to failure trajectories, the policy update reduces to Eq. 4.

> 💡 **这一节的作用**：把看起来"只是在 success 轨迹上做 BC"的方法，和正规的 regularized RL 框架联系起来，给方法一个理论背书。
>
> **但需要注意的近似**：
> 1. 把 $\gamma \to 1$（discount factor = 1，无限视野）
> 2. 用 binary reward 近似 advantage
> 3. 用 offline 数据近似 on-policy 采样
>
> 这三个近似叠在一起，理论保证就比较弱了。这一节更像是"事后解释"，而不是"从理论推导方法"。和 AWR（Peng et al. 2019）的连接是真实存在的，本质上就是 AWR 在 flow-matching 上的适配版本。

---

## Algorithm 1: VLAW

```
Require: π_θ, M_φ, R, K (real rollout budget), N (synthetic budget), K_iter
Initialize D_real = ∅
for i = 1 to K_iter do:
  (1) Roll out π_θ in real world → collect K trajectories → append to D_real
  (2) Fine-tune M_φ on D_real + D_DROID (Eq 1, 2)
      Fine-tune R on D_real
  (3) Roll out π_θ in M_φ → generate D_syn = N synthetic trajectories
      Apply R with threshold α → get D_syn+
  (4) Fine-tune π_θ on D_real+ ∪ D_syn+ (flow-matching, Eq 4)
end for
```

> 💡 **超参数总结**：
> - $K = 50$（每次 real rollout 条数 per task）
> - $N = 500$（每次合成轨迹条数 per task）
> - $K_{\mathrm{iter}} = 2$（迭代次数）
> - $\alpha = 0.8$（reward model 阈值）
> - World model fine-tune: 50K steps
> - Policy fine-tune: 2K steps, batch size 256
>
> **计算开销估算**：每次迭代需要 (1) 真实机器人跑 250 条轨迹 + (2) 训练 world model 50K steps + (3) 生成 2500 条合成轨迹 + (4) 训练 policy 2K steps。没有报告具体训练时间，这是论文的一个缺失信息。
