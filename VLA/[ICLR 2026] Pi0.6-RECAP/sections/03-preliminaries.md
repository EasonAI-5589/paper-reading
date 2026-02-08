[← 返回 README](../README.md)

# III. Preliminaries

## 📌 预览
标准 RL 符号定义 + Regularized RL 的理论基础。核心结论：advantage-conditioned policy 可以保证 policy improvement，且通过"improvement indicator" $I$ 可以避免显式表示 improvement probability。

---

**Reinforcement learning.** We consider the standard RL setting in which an agent, given by a policy $\pi(\mathbf{a}_t | \mathbf{o}_t)$, selects actions $\mathbf{a}_t$ given an observation $\mathbf{o}_t \in \mathcal{O}$. We define a trajectory as $\tau = (\mathbf{o}_0, \mathbf{a}_0, \cdots, \mathbf{o}_T) \in \mathcal{O} \times \mathcal{A} \cdots \mathcal{O}$. A distribution over trajectories $\rho_\pi(\tau)$ is induced by the policy $\pi(\mathbf{a}_t | \mathbf{o}_t)$ and the stochastic dynamics $p(\mathbf{o}_{t+1} | \mathbf{o}_t, \mathbf{a}_t)$:

$$\rho_\pi(\tau) = p(\mathbf{o}_0) \prod_{t=0}^{T-1} \pi(\mathbf{a}_t | \mathbf{o}_t) p(\mathbf{o}_{t+1} | \mathbf{o}_t, \mathbf{a}_t)$$

The reward function is given by $r(\mathbf{o}_t, \mathbf{a}_t)$, and we abbreviate it to $r_t$ to shorten notation, where $r_T$ is the terminal reward. We can define the discounted cumulative reward, or return, as $R(\tau) = \sum_{t=0}^{T} r_t$ (we do not use a discount factor, though one could easily be added). The goal of RL is to maximize the cumulative reward (or return), learning a policy that maximizes $\mathcal{I}(\pi) = \mathbb{E}_{\tau \sim \rho_\pi} [R(\tau)] = \mathbb{E}_{\tau \sim \rho_\pi} [\sum_{t=0}^{T} r_t]$. The value function for a policy $\pi$ is then defined as $V^\pi(\mathbf{o}_t) = \mathbb{E}_{\tau_{t+1:T}} [\sum_{t'=t}^{T} r_{t'}]$. The advantage of action $\mathbf{a}_t$ is $A^\pi(\mathbf{o}_t, \mathbf{a}_t) = \mathbb{E}_{\rho_\pi(\tau)} [\sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(\mathbf{o}_{t+N})] - V^\pi(\mathbf{o}_t)$, corresponding to an n-step estimate.

> 💡 **标准 RL 符号速查**:
> | 符号 | 含义 |
> |------|------|
> | $\pi(\mathbf{a}_t \| \mathbf{o}_t)$ | Policy：观测→动作 |
> | $V^\pi(\mathbf{o}_t)$ | Value function：当前状态的期望回报 |
> | $A^\pi(\mathbf{o}_t, \mathbf{a}_t)$ | Advantage：这个动作比平均好多少 |
> | $R(\tau)$ | Return：整个轨迹的总奖励 |
>
> 注意这里用的是 n-step advantage estimate，不是 GAE（λ）

---

**Regularized reinforcement learning.** Instead of maximizing $\mathcal{I}(\pi)$, it is common to use regularization in RL, optimizing for a policy that maximizes reward while remaining close to some reference policy $\pi_\text{ref}$ [66–70]. This is important, for example, when we want to train for many gradient steps on the same data, in which case $\pi_\text{ref}$ typically corresponds to the behavior policy that collected the training data. This can be formalized via the objective $\mathcal{I}(\pi, \pi_\text{ref}) = \mathbb{E}_{\tau \sim \rho_{\pi_\theta}} [\sum_{t=0}^{T} \gamma^t r_t] - \beta \mathbb{E}_{\mathbf{o} \sim \rho_{\pi_\theta}} [D(\pi(\cdot|\mathbf{o}) || \pi_\text{ref}(\cdot|\mathbf{o}))]$, where $D$ denotes some divergence metric. For the case where $D$ is the KL divergence, we have the well-known result that $\hat{\pi}(\mathbf{a}|\mathbf{o}) \propto \pi_\text{ref}(\mathbf{a}|\mathbf{o}) \exp(A^{\pi_\text{ref}}(\mathbf{o}, \mathbf{a}) / \beta)$ is the solution to $\max_\pi J(\pi, \pi_\text{ref})$, with Lagrange multiplier $\beta$ [67–70].

> 💡 **Regularized RL 的直觉**:
> - 不只是最大化 reward，还要和 reference policy 保持接近（防止跑偏）
> - KL 正则化的闭式解：$\hat{\pi} \propto \pi_\text{ref} \cdot \exp(A/\beta)$ → advantage 高的动作概率增大
> - 这就是 RLHF 中 DPO 等方法的理论基础

---

Our advantage-conditioned policy extraction method is based on a closely related but less well-known result: if we define the policy $\hat{\pi}(\mathbf{a}|\mathbf{o}) \propto \pi_\text{ref}(\mathbf{a}|\mathbf{o}) p(I|A^{\pi_\text{ref}}(\mathbf{o}, \mathbf{a}))^\beta$, where $p(I|A^{\pi_\text{ref}}(\mathbf{o}, \mathbf{a})) = g(A^{\pi_\text{ref}}(\mathbf{o}, \mathbf{a})) / \int g(A^{\pi_\text{ref}}(\mathbf{o}, \mathbf{a}')) d\mathbf{a}'$ is the probability of any action a improving over $\pi_\text{ref}$ as measured by a monotonically increasing function $g$, then $\hat{\pi}$ is guaranteed to improve over $\pi_\text{ref}$, i.e., $\mathcal{I}(\hat{\pi}) \geq \mathcal{I}(\pi_\text{ref})$ [4, 71]. We will use this property in deriving our policy extraction method in Section IV-B. Using this definition we can then obtain a parametric policy from the closed form definition of $\hat{\pi}$ by solving the following minimization problem: $\min_\theta \mathbb{E}_{s \sim \rho_{\pi_\text{ref}}} [KL(\hat{\pi}, \pi_\theta)]$.

> 💡 **RECAP 的理论基石**:
> - $p(I|A)$ = "improvement probability"：advantage 越高，这个动作越可能是"改进"
> - 关键定理：$\hat{\pi} \propto \pi_\text{ref} \cdot p(I|A)^\beta$ 保证 policy improvement
> - 和 KL-regularized 解的区别：用 improvement probability 替代了 $\exp(A/\beta)$
> - 最终通过 KL 投影到参数化 policy $\pi_\theta$
> - 这个理论来自 CFGRL [4] 和 Wang et al. [71]

---

## 🔖 Section 总结

### 核心洞察
1. **Advantage conditioning 的理论保证**：只要 improvement indicator 和 advantage 正相关，conditioned policy 就能保证改进
2. **和 RLHF 的联系**：KL-regularized RL 的闭式解 $\propto \exp(A/\beta)$ 是 DPO 的基础；RECAP 用 improvement probability 替代，更容易实现
3. **为什么 advantage conditioning 比 PPO 更适合 VLA**：不需要 log-likelihood（flow matching 没有 tractable likelihood），只需要 conditional supervised learning
