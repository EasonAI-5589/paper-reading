# VLAW 批读笔记 · Method

---

## 4. Co-Improvement of VLA and World Model

In this section, we describe the details of our method. The overall pipeline consists of the following steps:

1. **World model post-training (Sec. 4.1):** Finetune world model $M$ on $\mathcal{D}_{\mathrm{real}}$, co-train with $\mathcal{D}_{\mathrm{DROID}}$. Finetune reward model $R$ on $\mathcal{D}_{\mathrm{real}}$.
2. **VLA policy post-training (Sec. 4.2):** Generate $\mathcal{D}_{\mathrm{syn}}$, filter with $R$ → $\mathcal{D}_{\mathrm{syn}}^+$, finetune VLA.
3. Alternate between Steps 1 and 2 iteratively.

> 💡 **三个可学习组件**：World Model、Reward Model、VLA Policy——三者轮流被 real rollout 数据改进，是一个 EM 风格的迭代优化。

![](../images/4c94b6400adfaca4ef97e0ccaa83e70632d70a670bdc9b7a7e9be76efe32bbd3.jpg)

*Figure 3: VLAW 完整 pipeline：① 真实 rollout → ② World Model fine-tune → ③ 大规模合成轨迹生成 → ④ Reward Model 筛选 → Policy 更新。*

---

### 4.1. World Model Learning with Real Roll-outs

**Real World Policy Roll-outs.** Previous work has identified two major challenges: (1) over-optimism, as training data is dominated by successful demonstrations; and (2) limited physical fidelity in contact-rich or deformable-object tasks.

To address these issues, we get $K$ trajectories by rolling out the policy in the real world, forming $\mathcal{D}_{\mathrm{real}} = \{\tau_{\mathrm{real}}^1, ..., \tau_{\mathrm{real}}^K\}$, with sparse reward $r_\tau \in \{0, 1\}$.

> 💡 **$K = 50$ per task**（见 Sec 5.1），5 类任务共 250 条。这是相对较少的真实数据，能用这么少的 real rollout 撬动如此大的提升是本文的核心卖点。

**Training Objective.** Initialize from pretrained Ctrl-World, finetuning with the standard diffusion objective:

$$\mathcal{L}_{\mathcal{D}_{\mathrm{real}}} = \mathbb{E}_{x_0, \epsilon, t'} \left\| \hat{x}_0(x_{t'}, t', c) - x_0 \right\|^2$$

where $x_0 = o_{t+1:t+H}$ is the future observation chunk, $c$ includes action chunk $a_{t:t+H}$ and current observation $o_t$.

> 💡 **Loss 完全标准**：就是 Stable Video Diffusion 的 diffusion loss，没有任何特殊设计。核心贡献全部来自**训练数据的改变**（加入 failure rollout），而非 loss 或 architecture 的创新。这是一个"数据为王"的论点。

**Progressively Growing Dataset and Co-training.** Continuously append new trajectories: $\mathcal{D}_{\mathrm{real}} = \mathcal{D}_{\mathrm{real}} \cup \tau_{\mathrm{real}}^i$. Co-train with DROID to prevent overfitting:

$$\mathcal{L} = \mathcal{L}_{\mathcal{D}_{\mathrm{real}}} + \lambda \mathcal{L}_{\mathcal{D}_{\mathrm{DROID}}}$$

> 💡 **Co-training 的必要性**：只在 50 条 rollout 上 fine-tune 必然 overfit，加入 DROID 作为正则化是合理的。但 $\lambda$ 怎么调、对结果有多敏感，论文未给 ablation——是一个小遗漏。

**Finetuning Reward Model.** Use Qwen3-VL-4B-Instruct as reward model. Zero-shot 不够准，所以在第一次迭代中用 $\mathcal{D}_{\mathrm{real}}$ 的 success 标签 fine-tune。

A trajectory is considered successful if P('yes' token) > threshold $\alpha$:

$$R(\tau^i) = \mathbf{1}\left[P(\textsf{yes} \mid \tau^i, I^i) > \alpha\right]$$

> 💡 **Threshold 的作用**：直接输出 yes/no 会 FP 过多（Appendix C：FP=8）；设 $\alpha=0.8$ 后 FP 降到 2。如果 reward model 把太多失败轨迹标为成功，合成数据就会污染 policy 训练——所以宁可保守，牺牲一些 recall（FN 增加）。

---

### 4.2. Iterative Improvement for VLA Policy

**Scalable Training Pipeline.** Generate $N$ synthetic trajectories $\mathcal{D}_{\mathrm{syn}}$, filter to $\mathcal{D}_{\mathrm{syn}}^+$（only successes）.

> 💡 **$N = 500$ per task**：是 real rollout（50条）的 **10倍**。假设成功率约 50%，实际用于训练的合成数据约 250 条，相当于"免费"多了 250 条成功示范。

**Policy Learning Objective.** Weighted flow-matching loss over $\mathcal{D}_{\mathrm{syn}}^+ \cup \mathcal{D}_{\mathrm{real}}^+$：

$$\mathcal{L} = \mathbb{E}_{(o,a) \sim \mathcal{D}_{\mathrm{syn}}^+ \cup \mathcal{D}_{\mathrm{real}}^+} \mathcal{L}_{\mathrm{FM}}(\theta; o, a)$$

> 💡 **本质是"在成功轨迹上做 BC"**：binary weight（成功=1，失败=0），没有连续的 advantage 加权，没有 contrastive 项。理论上不如 AWR，但实践中稳定且有效。
>
> **和 Filtered BC 基线的唯一差别**：基线只有 50 条 real success 轨迹，VLAW 多了 500 条合成 success 轨迹。11.6% 的提升完全来自这些合成数据的质量。

---

### 4.3. Relation to Regularized Reinforcement Learning

Under regularized RL, the optimal policy:

$$\pi^*(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o) \exp\left(\frac{A^{\pi_{\mathrm{ref}}}(o,a)}{\beta}\right)$$

Since flow-matching VLA 没有显式 log-likelihood，定义 surrogate FM divergence：

$$D_{\mathrm{FM}}(\pi^*, \pi_\theta) \triangleq \mathbb{E}_{a \sim \pi^*(\cdot \mid o)} \left[ \mathcal{L}_{\mathrm{FM}}(\theta; o, a) \right]$$

令 $\gamma \to 1$，binary reward 近似 advantage → 化简为 Eq.4（实际训练目标）。

> 💡 **这一节的价值与局限**：
> - **价值**：把看起来"只是在 success 上做 BC"的方法，与正规 regularized RL 框架（AWR, Peng et al. 2019）联系起来，给方法理论背书
> - **局限**：三个近似叠加（binary weight ≈ exp(A/β)、γ→1、offline ≈ on-policy）后，理论保证已很弱。这一节更像"事后解释"而非"从理论推导方法"。

---

### Algorithm 1: VLAW

```
Require: π_θ, M_φ, R, K=50/task, N=500/task, K_iter=2, α=0.8
Initialize D_real = ∅
for i = 1 to K_iter:
  (1) 真实 rollout：roll out π_θ → 收集 K 条轨迹 → append to D_real
  (2) 模型更新：
      - Fine-tune M_φ on D_real + D_DROID (50K steps)
      - Fine-tune R on D_real（仅第一次迭代）
  (3) 合成数据生成：
      - Roll out π_θ in M_φ → 生成 D_syn（N 条）
      - Apply R with α → filter → D_syn+
  (4) Policy 更新：
      - Fine-tune π_θ on D_real+ ∪ D_syn+ (2K steps, bs=256)
return π_θ, M_φ
```

> 💡 **关键超参数一览**：K=50, N=500, K_iter=2, α=0.8, World model fine-tune=50K steps, Policy fine-tune=2K steps (bs=256)。计算开销未报告，是本文的一个缺失信息。
