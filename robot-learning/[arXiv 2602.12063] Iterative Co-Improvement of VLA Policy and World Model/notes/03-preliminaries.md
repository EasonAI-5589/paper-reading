# VLAW 批读笔记 · Preliminaries

---

## 3. Preliminaries

**Problem Setting.** We study a multi-task robotic manipulation problem, where each task is specified by a language instruction $I$ and is modeled as a Markov decision process (MDP) $\mathcal{M}_I = (\mathcal{S}, \mathcal{A}, P, R_I, \gamma)$. Here, $\mathcal{S}$ denotes the state space, $\mathcal{A}$ the action space, $P(s_{t+1} \mid s_t, a_t)$ the transition dynamics, $R_I$ the task-dependent reward function, and $\gamma$ the discount factor. At the beginning of training, we are given a pretrained vision–language–action (VLA) policy $\pi_\theta$ and an action-conditioned world model $M_\phi$.

> 💡 **符号澄清**：这里的 state $s$ 实际上是 observation（图像），不是真正的物理环境状态。World model 做的是未来图像帧 $o_{t+1}$ 的生成，而不是物理仿真。这个区分很重要——整个方法都在图像空间运作，不依赖任何物理引擎。

The policy maps the current state and instruction to an action distribution, $a_t \sim \pi_\theta(\cdot \mid s_t, I)$, while the world model predicts the next state conditioned on the current state and action, $\hat{s}_{t+1} \sim M_\phi(\cdot \mid s_t, a_t)$, where $\hat{s}_{t+1}$ denotes the predicted next state.

> 💡 **两个核心模型的角色**：
> - **π_θ（VLA policy）**：输入（图像 + 语言指令）→ 输出动作
> - **M_φ（World model）**：输入（当前图像 + 动作）→ 输出下一帧图像
>
> 两者在 closed-loop 下交替运行，就能在"想象"中模拟一段机器人轨迹。

The policy is allowed to collect online roll-outs in the real environment, resulting in trajectories $\tau_{\mathrm{real}}^i = \{s_0, a_0, \dots, a_{T-1}, s_T\}$. Each trajectory is labeled with a task-level reward $r_i$ indicating success or failure. Our goal is to leverage online interaction to iteratively improve the policy so that it performs well across all tasks.

> 💡 **Sparse Reward 设定**：只有 task-level 的 success/failure 标签（0/1），没有 dense reward。这非常贴近真实场景——人只需要在任务结束时判断"成没成"，不需要对每个时间步打分。这也是为什么需要 VLM reward model 来自动判断合成轨迹的成败。

**World Model Generated Trajectories.** In addition to real-world interaction, we can roll out the policy inside the world model. Starting from an initial state $s_0$ sampled from a real trajectory, the policy and world model interact in a closed loop via $a_t \sim \pi_\theta(\cdot \mid \hat{s}_t, I)$ and $\hat{s}_{t+1} \sim M_\phi(\cdot \mid \hat{s}_t, a_t)$. By iterating this process, we auto-regressively generate a complete imagined trajectory $\tau_{\mathrm{syn}}^j = \{s_0, a_0, \hat{s}_1, a_1, \dots, a_{T-1}, \hat{s}_T\}$.

> 💡 **Closed-loop 想象 vs. Open-loop Replay 的区别**：
> - **Open-loop replay**（Section 5.2 的评估）：固定真实动作序列，看 world model 能不能正确预测图像。误差只来自视频预测。
> - **Closed-loop rollout**（实际生成合成数据用的）：Policy 看着 world model 生成的图像来决策下一步动作。更接近真实执行，但对 world model 的误差累积（compounding error）更敏感。
>
> Closed-loop 能稳定跑 20 秒（Figure 5 展示），说明 post-trained world model 的误差累积还算可控，这是整个 pipeline 能奏效的前提。
