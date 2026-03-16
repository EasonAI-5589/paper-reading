[← 返回 README](../README.md)

# Abstract

## 📌 预览

WoVR 的核心问题：用 World Model 做 VLA 的 RL post-training 时，closed-loop rollout 的 hallucination（幻觉）会腐蚀优化信号，让 policy 利用模型错误而非真实进展。三层解决方案：① 更稳定的 World Model（dual-channel action + first-frame anchor）；② KIR（Keyframe-Initialized Rollouts）缩短有效误差深度；③ PACE 维持 policy-model 分布对齐。

---

Reinforcement learning (RL) promises to unlock capabilities beyond imitation learning for Vision–Language–Action (VLA) models, but its requirement for massive real-world interaction prevents direct deployment on physical robots. Recent work attempts to use learned world models as simulators for policy optimization, yet closed-loop imagined rollouts inevitably suffer from hallucination and long-horizon error accumulation. Such errors do not merely degrade visual fidelity—they corrupt the optimization signal, encouraging policies to exploit model inaccuracies rather than genuine task progress.

> 💡 **第一段：问题是什么——三句话层层递进**
>
> **① RL 对 VLA 有用，但需要大量真实交互。** 真实机器人太慢太贵，不现实。
>
> **② 用 world model 做模拟器是个方案，但有幻觉问题。** 视频生成模型会出错，误差随时间步积累（closed-loop error accumulation）。
>
> **③ 关键洞察：幻觉的危害不是"画面不好看"，而是"RL 学歪了"。** 当 world model 在第 N 步幻觉时（比如杯子突然消失），RL 会**exploit（利用）world model 的 bug**，学到的是"怎么触发幻觉拿奖励"而不是真正的技能。这就是 "corruption of optimization signal"——优化信号被污染了。
>
> **与 VLAW 的视角对比**（两者互补，不矛盾）：
> | | WoVR | VLAW |
> |---|------|------|
> | **认为问题是什么** | 闭环 rollout 的幻觉（推理时误差积累） | WM 训练数据分布偏差（缺 failure case → over-optimism） |
> | **解决思路** | 三层机制让 RL 跟不完美 WM 共处 | 用在线数据微调 WM + filtered BC |

We propose WoVR, a reliable world-model-based reinforcement learning framework for post-training VLA policies. Instead of assuming a faithful world model, WoVR explicitly regulates how RL interacts with imperfect imagined dynamics. It improves rollout stability through a controllable action-conditioned video world model, reshapes imagined interaction to reduce effective error depth via Keyframe-Initialized Rollouts, and maintains policy–simulator alignment through World Model-Policy co-evolution.

> 💡 **三层结构一览**：
> 1. **Simulator level**：Dual-channel action + first-frame anchoring → 提升 rollout 稳定性
> 2. **Interaction level**：KIR（从关键帧初始化 rollout）→ 缩短有效误差深度
> 3. **Alignment level**：PACE（policy-world model 协同进化）→ 维持分布对齐
>
> 注意关键词：**"regulates how RL interacts with imperfect imagined dynamics"**——不是修复 world model，而是设计让 RL 和不完美 world model 更好共处的机制。这是一个从 RL 框架角度出发的设计思路，而不只是 world model 改进。

Extensive experiments on LIBERO benchmarks and real-world robotic manipulation demonstrate that WoVR enables stable long-horizon imagined rollouts and effective policy optimization, improving average LIBERO success from $39.95\%$ to $69.2\%$ (+29.3 points) and real-robot success from $61.7\%$ to $91.7\%$ (+30.0 points). These results show that learned world models can serve as practical simulators for reinforcement learning when hallucination is explicitly controlled.

> 💡 **关键数字**：
> - LIBERO 均值：39.9% → 69.2%（**+29.3 pp**）
> - 真实机器人：61.7% → 91.7%（**+30.0 pp**）
>
> **与 VLAW 的数字对比**：
> - VLAW：真实机器人 5 类任务 46% → 87%（+41 pp，但比较是 base 到 VLAW，LIBERO 更难）
> - WoVR：LIBERO 仿真 +29.3 pp + 真实机器人 +30 pp
>
> 两篇论文的实验平台不同（VLAW 用 DROID，WoVR 用 LIBERO + Franka），直接比较数字意义不大，但都展示了 world model 做 RL 的巨大潜力。

---

## 🔖 Section 总结

| 指标 | 数值 |
|------|------|
| LIBERO 平均提升 | +29.3 pp（39.9% → 69.2%） |
| 真实机器人平均提升 | +30.0 pp（61.7% → 91.7%） |

### 核心洞察
1. World Model 做 RL 的根本障碍是 hallucination 腐蚀 optimization signal，不只是视觉质量差
2. 解决路径：不修复 world model 本身，而是设计让 RL 与不完美 world model 更好共处的机制
3. 三层控制（simulator / interaction / alignment）各自针对 hallucination 的不同来源
