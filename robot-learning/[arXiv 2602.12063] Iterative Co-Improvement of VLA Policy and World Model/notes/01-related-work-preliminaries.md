# VLAW 批读笔记 · Related Work & Preliminaries

---

## 2. Related Works

### 2.1. Post-training Vision-Language-Action Models

Vision–language–action (VLA) models have achieved remarkable success in robotic manipulation tasks. A common approach is to train the VLA on large-scale data and then perform supervised fine-tuning on target tasks. Beyond supervised fine-tuning, improving VLA policies using online rollout data has emerged as a promising direction. Some prior works adopt on-policy reinforcement learning methods, such as PPO or GRPO, to improve VLA policies.

> 💡 **VLA post-training 的三条路**：
> 1. **SFT**（监督微调）：在目标任务上直接 fine-tune，简单但需要 demo 数据
> 2. **On-policy RL**（PPO/GRPO）：理论上最优，但需要大量 rollout，主要在仿真环境里验证
> 3. **Offline/Batch RL**（本文的方向）：用已有 rollout 数据 + weighted SFT，适合 real-world

However, standard on-policy reinforcement learning typically requires a large number of rollouts and is therefore primarily validated in simulation environments. Moreover, state-of-the-art VLA models are often trained with flow-matching objectives, which do not provide explicit policy likelihoods, making conventional policy-gradient methods difficult to apply.

> 💡 **flow-matching VLA 的 RL 困境**：这是本文方法设计的核心约束。π₀ 系列用的 flow-matching 没有 log π(a|o)，所以 REINFORCE/PPO 里的 log prob ratio 算不出来。这不只是工程问题，是理论层面的障碍。
> 
> 这也是为什么 π₀.₆* 选择了 advantage-conditioned SFT，VLAW 选择了 weighted flow-matching loss——都是在绕过这个障碍。

To enable policy learning in real-world settings, π₀.₆* instead adopts an offline or batch reinforcement learning formulation with an advantage-conditioned supervised learning objective. Similarly, in our setting, we perform iterative policy improvement using batches of real-world rollout data together with world-model–generated synthetic data, and update the policy exclusively through stable supervised fine-tuning objectives.

> 💡 **与 π₀.₆* 的关系**：VLAW 和 π₀.₆* 思路相似（都用 offline RL + SFT），区别在于 VLAW 用 World Model 大幅扩充了数据量（从 50 条 real rollout → 500 条合成轨迹 per task），这是杠杆效应的核心所在。

---

### 2.2. World Models for Decision Making

Action-conditioned world models predict future outcomes given current observations and actions. Many works leverage such models for model-based reinforcement learning and visual planning. Among these, the most closely related approaches are DayDreamer, SOLAR, and World4rl, which also operate in real-world visual model-based reinforcement learning settings. However, due to limited model capacity and data scale, these earlier methods often learned task-specific dynamics models.

> 💡 **早期 World Model 的局限**：DayDreamer（2023）是一个重要的先驱工作（Hafner + Goldberg + Abbeel），在真实机器人上用 Dreamer 做 MBRL，但当时的模型容量有限，只能做单任务。VLAW 的 Ctrl-World 是在 DROID 全量数据上训练的，天然支持多任务。

With recent advances in video diffusion models, it has become feasible to train multi-task action-conditioned world models that can generate realistic future visual observations. Despite this progress, accurately modeling complex physical dynamics remains a fundamental challenge, as widely observed in prior world-model literature, likely because these models are trained on offline robotics datasets usually consisting primarily of demonstrations.

> 💡 **Video Diffusion + Robot = Ctrl-World 的背景**：近两年 Genie、IRASim、WMPO 等工作都在用 video diffusion 做 robot world model，质量提升很大（视觉逼真），但物理准确性仍然不足。本文的贡献是：用少量 online rollout 数据（含 failure）来"校准"（ground）这个 pretrained diffusion world model。

To address this challenge, we leverage online policy rollout data to ground a pretrained world model in new environments, thereby improving its accuracy around the policy's state–action distribution.

> 💡 **Distribution Shift 视角**：Pretrained world model 是在 DROID（expert demo）上训练的，而 policy rollout 的状态-动作分布和 expert demo 不同（尤其是 failure 时的状态）。用 online rollout 来 fine-tune = 减少 distribution shift，让 world model 更贴近 policy 实际跑到的状态。这是经典的 DAgger 思想在 world model 上的应用。

---

## 3. Preliminaries

**Problem Setting.** We study a multi-task robotic manipulation problem, where each task is specified by a language instruction $I$ and is modeled as a Markov decision process (MDP) $\mathcal{M}_I = (\mathcal{S}, \mathcal{A}, P, R_I, \gamma)$. The policy maps the current state and instruction to an action distribution, $a_t \sim \pi_\theta(\cdot \mid s_t, I)$, while the world model predicts the next state conditioned on the current state and action, $\hat{s}_{t+1} \sim M_\phi(\cdot \mid s_t, a_t)$.

> 💡 **符号澄清**：这里的 state $s$ 实际上是 observation（图像），不是真正的环境状态。World model 做的是 $o_{t+1}$ 的生成，而不是真正的状态转移。这个区分很重要——后面的实验用的是图像级别的 replay 和生成，而不是物理仿真。

The policy is allowed to collect online roll-outs in the real environment, resulting in trajectories $\tau_{\mathrm{real}}^i = \{s_0, a_0, \dots, a_{T-1}, s_T\}$. Each trajectory is labeled with a task-level reward $r_i$ indicating success or failure. Our goal is to leverage online interaction to iteratively improve the policy so that it performs well across all tasks.

> 💡 **Sparse Reward 设定**：只有 task-level 的 success/failure 标签，没有 dense reward。这非常贴近真实场景——机器人做完一个任务之后，人只需要说"成功"或"失败"，不需要在每一步都给分。

**World Model Generated Trajectories.** Starting from an initial state $s_0$ sampled from a real trajectory, the policy and world model interact in a closed loop [...] By iterating this process, we auto-regressively generate a complete imagined trajectory $\tau_{\mathrm{syn}}^j = \{s_0, a_0, \hat{s}_1, a_1, \dots, a_{T-1}, \hat{s}_T\}$.

> 💡 **Closed-loop 想象**：这里区别于 open-loop replay（固定动作序列放进 world model）。Closed-loop 是 policy 看着 world model 生成的图像来决策下一步动作——这更接近真实执行，但对 world model 的误差累积（compounding error）更敏感。后面 Figure 5 展示的 20 秒 long-horizon rollout 能保持稳定，说明 post-trained world model 的 error 积累还算可控。
