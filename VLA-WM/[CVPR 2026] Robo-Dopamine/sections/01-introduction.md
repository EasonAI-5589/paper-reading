[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
Introduction 分四层递进：IL 的根本局限 → RL 奖励难题 → 现有 PRM 的两大缺陷 → 本文方案。Figure 1 是全文的路线图。

---

While large-scale imitation learning (IL) has substantially advanced embodied intelligence [1, 4, 5, 15, 21, 25], its reliance on static, expert-curated datasets imposes fundamental limitations [7, 24, 43, 52, 64, 65], which exhibits sub-optimal sample efficiency, poor generalization to out-of-distribution (OOD) scenarios, and also struggles to acquire precise and contact-rich manipulation skills [23, 63]. In contrast, reinforcement learning (RL) offers a compelling alternative [11, 29, 33, 35, 56, 58, 60]. Through continuous environmental interaction, RL enables agents to transcend the limitations of static expert data, facilitating superior generalization and the mastery of high-precision tasks.

> 💡 **IL vs RL 的根本矛盾**:
> - IL 三大局限：样本效率差、OOD 泛化差、精细操作学不好
> - RL 的优势：通过持续交互突破静态数据限制
> - 这里引用了一大批 VLA 工作（pi0, GR00T, Helix 等），说明 IL 已经到了瓶颈

---

However, the primary obstacle for applying RL to real-world robotics is the design of effective reward functions. Conventional approaches falter at two extremes: sparse, binary outcome rewards [11, 33, 35, 56, 60] make exploration in long-horizon, contact-rich tasks prohibitively difficult, while handcrafted dense rewards [16, 44, 54, 55] require significant domain expertise, limiting scalability and general applicability. This dichotomy has motivated the shift towards learning-based Process Reward Models (PRMs) [2, 8, 36, 37, 59].

> 💡 **奖励函数的两难困境**:
> | 类型 | 优点 | 缺点 |
> |------|------|------|
> | Sparse (binary) | 简单、无偏 | 长horizon探索困难 |
> | Handcrafted dense | 信号丰富 | 需要领域知识，不可扩展 |
> | **Learned PRM** | 可学习、可扩展 | **本文要解决的问题** |

---

Despite their promise, current PRMs are hindered by two fundamental limitations. First, the underlying reward models often exhibit critical deficiencies: their task-specific design [8, 18] inherently limits generalization; uniform reward distributions [37, 59] fail to capture the varying salience of crucial sub-steps; and a reliance on single-view observations [2, 8, 36, 37, 59] fails in manipulation scenes where occlusions obscure fine-grained progress only visible from wrist-level views. Second, the reward shaping algorithms utilizing these dense signals are often theoretically flawed. Naively incorporating dense rewards can induce a semantic trap [41] that misguides policy optimization by inadvertently altering the optimal policy, causing the agent to prioritize high proxy rewards from intermediate steps over the true task objective.

> 💡 **PRM 两大根本缺陷（核心 motivation）**:
>
> **缺陷 1 — 奖励模型能力不足**:
> - task-specific 设计 → 泛化差（SARM [8]）
> - uniform reward 分配 → 忽略关键子步骤的重要性差异（GVL [37], VLAC [59]）
> - 单视角 → 遮挡时失效（几乎所有现有工作）
>
> **缺陷 2 — 奖励塑形理论有漏洞**:
> - naive dense reward ($r = \Phi(s_{t+1}) - \Phi(s_t)$) 会改变最优策略
> - 产生 "semantic trap"：agent 学会到达高 reward 状态后**原地不动**
> - 引用 Ng et al. 1999 [41] 的经典工作

---

To address these, we introduce Dopamine-Reward, a novel dense reward modeling method for learning a general-purpose, step-aware process reward from multi-view inputs. Dopamine-Reward directly tackles the first limitation by leveraging two key techniques: Hop-based Step-wise General Reward Model (GRM) Construction for a fine-grained, structural understanding of task progression from various viewpoints, and Multi-Perspective Reward Fusion via GRM to integrate bidirectional global reward and state-wise incremental reward for more precise reward estimation, which are made possible by a meticulous annotation pipeline encompassing over 3,400 hours of data, 100K trajectories, and more than 350 daily tasks, offering broad coverage, fine-grained labels, and well-balanced distributions across real robots, simulations, and egocentric human videos.

> 💡 **Dopamine-Reward 方案**:
> - **Hop-based GRM**: 不直接回归绝对进度，而是预测相对进度 "hop"——归一化到 [-1, 1] 的相对变化
> - **Multi-Perspective Fusion**: 融合 incremental + forward + backward 三个视角
> - 数据规模：3,400h / 100K trajectories / 350+ tasks / 三种来源（真实机器人 + 仿真 + 人类自我视角视频）

---

Building upon GRM via Dopamine-Reward, we propose a robust and unified policy learning framework Dopamine-RL to resolve the second limitation. Dopamine-RL employs a theoretically-sound Policy-Invariant Reward Shaping method, which enables the agent to leverage the dense rewards from our GRM for highly efficient self-improvement without altering the underlying optimal policy, thereby fundamentally avoiding the semantic trap.

> 💡 **Dopamine-RL 方案**:
> - Policy-Invariant Reward Shaping: $r_{GRM} = r_{gold} + \gamma\Phi^*(s_{t+1}) - \Phi^*(s_t)$
> - 这是 PBRS 框架的直接应用，数学上保证 $\arg\max_a Q^*_{GRM}(s,a) = \arg\max_a Q^*_{gold}(s,a)$
> - 与 RL 算法无关：兼容 PPO、Cal-QL、ReinFlow 等

---

Extensive experiments on over 10 simulation and 8 real-world tasks demonstrate the superiority of our methods: (1) State-of-the-art Reward Accuracy. The GRM achieves over 92.8% accuracy in progress assessment, with a Value-Order Consistency (VOC) score of 0.953 on rank-correlation benchmarks, outperforming established baselines. (2) High Training Efficiency. After GRM is adapted to a new task in a one-shot manner from a single expert demonstration, the resulting reward model enables Dopamine-RL to improve a policy from near-zero to 95% success rate within approximately 150 online rollouts (about one hour of real robot interaction), with some tasks reaching 100% success rate. (3) Improved Generalization. By combining step-wise structural modeling, reward fusion, and multi-view perception for robust estimation under occlusion and fine-grained state changes, our GRM provides more reliable learning signals, enabling Dopamine-RL to generalize more effectively to unseen layouts, backgrounds, and object variations.

> 💡 **三大实验亮点**:
> 1. **奖励精度 SOTA**: 92.8% 准确率 + VOC 0.953（碾压 GVL 的 0.20、VLAC 的 0.33）
> 2. **极高样本效率**: 150 rollouts ≈ 1 小时 → 95% 成功率（部分任务 100%）
> 3. **强泛化**: OOD 下仅 8-19% 性能下降 vs BC 的 50-60%

---

![Figure 1](../images/6654deab10733ca7e3661248811122a6bf1a2c9cb3e9610986da7297de71f311.jpg)
*Figure 1. Overview of Robo-Dopamine. (Left) GRM 架构：35M 样本训练，多视角输入预测 hop-based 相对进度。(Bottom Right) Dopamine-RL：one-shot 适配 + policy-invariant reward shaping。(Top Right) 雷达图：奖励精度 SOTA；柱状图：仿真和真实世界的策略提升。*

> 💡 **Figure 1 批读**:
> - **左侧**: GRM 的数据来源（真实机器人 + 仿真 + 人类视频）和模型架构（VLM 输入初始/目标/before/after 四帧）
> - **右下**: Dopamine-RL 流程——one-shot 适配 → reward shaping → RL 训练
> - **右上雷达图**: 在所有 benchmark 上 GRM 都显著优于 GVL 和 VLAC
> - **右上柱状图**: Dopamine-RL 在仿真和真实世界都大幅提升成功率

---

An overview of Dopamine-Reward and Dopamine-RL, together with our empirical gains in reward accuracy and policy performance, is shown in Figure 1. In summary, our main contributions are as follows:

- We propose Dopamine-Reward, a novel reward modeling method built around a General Reward Model (GRM) that provides step-aware, fine-grained, and occlusion-resilient process rewards for precise robotic manipulation.

- We introduce Dopamine-RL, a robust policy learning framework with a theoretically grounded Policy-Invariant Reward Shaping scheme, which effectively exploits dense GRM rewards to accelerate policy optimization while avoiding the semantic trap.

- We curate a large-scale, 3,400-hour multi-view dataset with over 100K trajectories and 350 daily manipulation tasks across real robots, simulation, and egocentric human videos, offering broad coverage, fine-grained annotations, and balanced supervision for training GRM.

- Extensive experiments validate our framework as follow: GRM achieves state-of-the-art reward assessment (over 92.8% progress accuracy and a 0.953 Value-Order Consistency score), while on 10 simulated and 8 real-world tasks, Dopamine-RL, after one-shot GRM adaptation, raises policy success from near-zero to 95% within 150 online rollouts (about one hour of robot interaction), with some tasks reaching 100% success and generalizing to unseen layouts, backgrounds, and object variations.

> 💡 **贡献总结**:
> | 贡献 | 解决的问题 |
> |------|-----------|
> | Dopamine-Reward (GRM) | PRM 缺乏 step-aware + 单视角局限 |
> | Dopamine-RL (PBRS) | 奖励塑形的 semantic trap |
> | 大规模数据集 | 数据覆盖不足、标注不均衡 |
> | 实验验证 | 端到端有效性证明 |

---

## 🔖 Section 总结

### 核心洞察
1. IL → RL 的转变是必然趋势，但奖励设计是核心瓶颈
2. 现有 PRM 的两大问题高度互补：模型能力 + 理论正确性缺一不可
3. Dopamine 的命名寓意：多巴胺 = 奖励信号，无论在人脑还是机器人中
