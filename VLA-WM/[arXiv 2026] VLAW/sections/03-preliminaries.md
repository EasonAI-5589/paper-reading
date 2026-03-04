[← 返回 README](../README.md)

# 3. Preliminaries

> 来源: VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model (arXiv 2602.12063)

---

## 📄 原文

> 💡 **Section 概览**: 定义了本文的问题设定——多任务机器人操作 MDP，以及两类轨迹来源：真实 rollout 和世界模型生成轨迹。这节很短，主要是符号定义，为 Section 4 的方法描述做铺垫。

**Problem Setting.** We study a multi-task robotic manipulation problem, where each task is specified by a language instruction $I$ and is modeled as a Markov decision process (MDP) $\mathcal{M}_I = (\mathcal{S}, \mathcal{A}, P, R_I, \gamma)$. Here, $s$ denotes the state space, $\mathcal{A}$ the action space, $P(s_{t+1} \mid s_t, a_t)$ the transition dynamics, $R_I$ the task-dependent reward function, and $\gamma$ the discount factor. At the beginning of training, we are given a pretrained vision–language–action (VLA) policy $\pi_\theta$ and an action-conditioned world model $M_\phi$. The policy maps the current state and instruction to an action distribution, $a_t \sim \pi_\theta(\cdot \mid s_t, I)$, while the world model predicts the next state conditioned on the current state and action, $\hat{s}_{t+1} \sim M_\phi(\cdot \mid s_t, a_t)$, where $\hat{s}_{t+1}$ denotes the predicted next state.

> 💡 **符号对照表**:
> | 符号 | 含义 |
> |------|------|
> | $I$ | 语言任务指令 |
> | $\mathcal{M}_I$ | 任务 MDP |
> | $\pi_\theta$ | VLA 策略（参数为 θ） |
> | $M_\phi$ | Action-conditioned 世界模型（参数为 φ） |
> | $s_t$ | 真实状态（观测） |
> | $\hat{s}_{t+1}$ | 世界模型预测的下一状态 |
> | $r_\tau \in \{0,1\}$ | 轨迹级别稀疏奖励（成功/失败） |

The policy is allowed to collect online roll-outs in the real environment, resulting in trajectories $\tau_{\mathrm{real}}^i = \{s_0, a_0, \dots, a_{T-1}, s_T\}$. Each trajectory is labeled with a task-level reward $r_i$ indicating success or failure. Our goal is to leverage online interaction to iteratively improve the policy so that it performs well across all tasks.

> 💡 **稀疏奖励设计**：每条轨迹只有一个 0/1 reward，而非逐步 reward。这是真实机器人任务的常见设定（只看最终结果），避免了 dense reward shaping 的复杂性，但也意味着信号稀疏，需要奖励模型来扩增信号。

**World Model Generated Trajectories.** In addition to real-world interaction, we can roll out the policy inside the world model. Starting from an initial state $s_0$ sampled from a real trajectory, the policy and world model interact in a closed loop via $a_t \sim \pi_\theta(\cdot \mid \hat{s}_t, I)$ and $\hat{s}_{t+1} \sim M_\phi(\cdot \mid \hat{s}_t, a_t)$. By iterating this process, we auto-regressively generate a complete imagined trajectory $\tau_{\mathrm{syn}}^j = \{s_0, a_0, \hat{s}_1, a_1, \dots, a_{T-1}, \hat{s}_T\}$.

> 💡 **Policy-in-the-loop 的关键细节**：
> - 初始帧 $s_0$ 从**真实轨迹**中采样——这保证了想象轨迹从真实的物理状态出发，避免 distribution shift 累积
> - 闭环交互：策略 → 动作 → 世界模型 → 下一帧 → 策略 → ...
> - 注意：想象轨迹保留真实初始帧 $s_0$，但后续帧都是 $\hat{s}$（预测值）——整条轨迹的质量依赖于世界模型的长程稳定性

---

## 🔖 Section 总结

### 关键数字速查

| 项目 | 说明 |
|------|------|
| 奖励类型 | 轨迹级稀疏奖励 $r_\tau \in \{0,1\}$ |
| 想象轨迹初始帧 | 从真实轨迹采样 $s_0$ |
| 后续帧 | 全部由世界模型预测 $\hat{s}$ |

### 核心洞察

1. **MDP 是多任务的**：一个策略要处理 5 类任务，由语言指令 $I$ 区分——这比单任务设定难得多
2. **世界模型的角色**：不是真实环境的替代品，而是**数据扩增器**——用少量真实 rollout 的起始帧，生成大量多样化的想象轨迹
3. **closed-loop vs open-loop**：Policy-in-the-loop 是闭环（策略输出的 action 影响下一帧），比 action replay（预录制动作序列）更难，但更能反映策略的实际行为
