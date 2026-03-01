[← 返回 README](../README.md)

# 3. Preliminaries

## 📌 预览

定义问题设定（MDP + multi-task）和两个核心模型（VLA policy π_θ、World Model M_φ）的角色，以及 World Model 生成轨迹（closed-loop imagination）的工作方式。这一节是方法推导的符号基础。

---

**Problem Setting.** We study a multi-task robotic manipulation problem, where each task is specified by a language instruction $I$ and is modeled as a Markov decision process (MDP) $\mathcal{M}_I = (\mathcal{S}, \mathcal{A}, P, R_I, \gamma)$. Here, $s$ denotes the state space, $\mathcal{A}$ the action space, $P(s_{t+1} \mid s_t, a_t)$ the transition dynamics, $R_I$ the task-dependent reward function, and $\gamma$ the discount factor. At the beginning of training, we are given a pretrained vision–language–action (VLA) policy $\pi_\theta$ and an action-conditioned world model $M_\phi$. The policy maps the current state and instruction to an action distribution, $a_t \sim \pi_\theta(\cdot \mid s_t, I)$, while the world model predicts the next state conditioned on the current state and action, $\hat{s}_{t+1} \sim M_\phi(\cdot \mid s_t, a_t)$, where $\hat{s}_{t+1}$ denotes the predicted next state.

> 💡 **符号澄清**：这里的 state $s$ 实际上是 **observation（图像帧）**，不是真正的物理环境状态。World model 做的是未来图像的生成（video prediction），不依赖物理引擎或任何 3D 状态表示。这是整个方法完全在「像素空间」运作的前提——无需仿真器，只需真实视频数据。
>
> **两个核心模型的角色**：
> - **π_θ（VLA policy）**：输入 (当前图像 + 语言指令) → 输出动作
> - **M_φ（World model）**：输入 (当前图像 + 动作) → 输出下一帧图像
>
> 两者在 closed-loop 下交替运行，即可在「想象」中模拟一段完整的机器人轨迹。

The policy is allowed to collect online roll-outs in the real environment, resulting in trajectories $\tau_{\mathrm{real}}^i = \{s_0, a_0, \dots, a_{T-1}, s_T\}$. Each trajectory is labeled with a task-level reward $r_i$ indicating success or failure. Our goal is to leverage online interaction to iteratively improve the policy so that it performs well across all tasks.

> 💡 **Sparse Reward 设定**：只有 task-level 的 success/failure 标签（0/1），没有 dense reward。这贴近真实场景——人工只需在任务结束时判断「成没成」，不需要对每个时间步打分。这也是为什么需要 VLM reward model 来自动判断合成轨迹的成败（而不是用 dense reward signal）。

**World Model Generated Trajectories.** In addition to real-world interaction, we can roll out the policy inside the world model. Starting from an initial state $s_0$ sampled from a real trajectory, the policy and world model interact in a closed loop via $a_t \sim \pi_\theta(\cdot \mid \hat{s}_t, I)$ and $\hat{s}_{t+1} \sim M_\phi(\cdot \mid \hat{s}_t, a_t)$. By iterating this process, we auto-regressively generate a complete imagined trajectory $\tau_{\mathrm{syn}}^j = \{s_0, a_0, \hat{s}_1, a_1, \dots, a_{T-1}, \hat{s}_T\}$.

> 💡 **Closed-loop vs. Open-loop 的区别（很重要）**：
> - **Open-loop replay**（Section 5.2 用于评估 world model 质量）：固定真实动作序列 → world model 预测图像。误差只来自视频预测，不含 policy 决策误差。
> - **Closed-loop rollout**（实际生成合成数据用的）：policy 看着 world model 生成的图像实时决策下一步动作。更接近真实执行，但 world model 的误差会累积（compounding error）——每一帧的预测误差会影响下一步 policy 的决策。
>
> Closed-loop 能在 20 秒的长视野内保持稳定（Section 5.2 Figure 5），说明 post-trained world model 的误差累积还算可控——这是整个「在想象中搜索成功轨迹」方案能奏效的关键前提。

---

## 🔖 Section 总结

### 符号速查

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | VLA policy（图像 + 语言 → 动作） |
| $M_\phi$ | Action-conditioned world model（图像 + 动作 → 下一帧图像） |
| $\tau_{\mathrm{real}}^i$ | 真实 rollout 轨迹，带 success/failure 标签 $r_i$ |
| $\tau_{\mathrm{syn}}^j$ | World model 生成的合成轨迹（closed-loop） |

### 核心洞察
1. State = observation（图像），整个方法在像素空间运作，不需要物理仿真器
2. Sparse reward 设定贴近真实场景，是 VLM reward model 存在的 motivation
3. Closed-loop rollout 的 compounding error 是整个 pipeline 能否 work 的关键风险点，Section 5.2 对此有验证
