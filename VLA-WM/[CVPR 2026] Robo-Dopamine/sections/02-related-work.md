[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 分两个方向：RL for Robotic Skills（RL 算法 + 架构）和 Learned Process Reward Models（从 ORM 到 PRM 的演进）。本文定位在 PRM 的三个空白上。

---

## 2.1 Reinforcement Learning for Robotic Skills

Reinforcement Learning (RL) has demonstrated the potential to create policies that surpass the capabilities of imitation learning [11, 28, 29, 31, 33, 35, 56, 58, 60], enabling the discovery of novel and robust strategies for complex, contact-rich and dexterous tasks. Research in this area has progressed along two principal directions. The first direction investigates various policy optimization strategies, including offline RL [20, 34, 35], online RL [13, 29, 31, 32, 58], and mixed variants [11, 28, 38]. The second direction explores the efficient application of RL to different model architectures, such as small fully-connected models [35, 38], autoregressive models [11, 33, 56], and diffusion/flow-based models [31, 58, 62]. Independent of the chosen optimization algorithm or policy architecture, a more fundamental bottleneck is to design a reward function that is effective and scalable in real-world RL, which has driven a broad shift away from manual reward engineering [16, 44, 50, 54, 55] toward learning-based reward models.

> 💡 **RL for Robotics 两个研究方向**:
>
> | 方向 | 代表工作 | 关键词 |
> |------|---------|--------|
> | **优化策略** | Offline RL [20,34,35], Online RL [13,29,31], Mixed [11,28,38] | PPO, Cal-QL, GRPO |
> | **模型架构** | FC [35,38], Autoregressive [11,33,56], Diffusion/Flow [31,58,62] | VLA, pi0, ReinFlow |
>
> 关键论点：**无论选哪种算法或架构，奖励函数设计才是更根本的瓶颈**。这为本文的 GRM 方案提供了动机——与算法/架构解耦的通用奖励模型。

---

## 2.2 Learned Process Reward Models

In real-world RL, a common practice is to train a success classifier as Outcome Reward Models (ORMs) to provide a binary reward signal [11, 35], which renders exploration prohibitively difficult in complex, long-horizon tasks. To mitigate sparsity, recent work leverages vision-language models (VLMs) as Process Reward Models (PRMs) [2, 8, 36, 37, 59], providing denser feedback by, for example, predicting progress deltas between paired observations [59] or assigning per-frame progress scores with respect to a language goal [37]. While several methods introduce additional structure by decomposing tasks into steps [8, 18], some open challenges remain (Section 1). First, task-specific designs may limit generalization across diverse activities [8, 18]. Second, many approaches adopt nearly uniform reward allocations, which may underweight the salience of critical sub-steps [37, 59]. In addition, current PRMs typically rely on single-view observations [2, 8, 36, 37, 59], which can impede multi-perspective state estimation and increase sensitivity to occlusions. In contrast, our method, Dopamine-Reward, aims to address these issues by learning a general-purpose, step-aware reward model that explicitly fuses multi-view inputs, enabling a more robust and fine-grained reward estimation.

> 💡 **从 ORM 到 PRM 的演进**:
>
> | 工作 | 核心方法 | 优势 | 局限 |
> |------|---------|------|------|
> | ORM (ConRFT [11], HIL-RL [35]) | 二分类：成功/失败 | ✓ 简单无偏 | ✗ 稀疏信号，长horizon探索难 |
> | LIV [36] | 语言-图像表征做奖励 | ✓ 语言条件化 | ✗ 单视角，task-specific |
> | GVL [37] | VLM 做 in-context value 学习 | ✓ 利用大模型能力 | ✗ uniform reward 分配 |
> | VLAC [59] | VLM 预测 paired progress delta | ✓ 相对进度比绝对更鲁棒 | ✗ 单视角，无 step-aware |
> | SARM [8] | stage-aware 分解 | ✓ 引入结构 | ✗ task-specific 设计 |
> | **Ours (GRM)** | hop-based + multi-view + fusion | ✓ 通用 + step-aware + 多视角 | 需大规模数据 |
>
> **一句话小结**: 现有 PRM 共性问题是"三缺"——缺通用性、缺步级感知、缺多视角。

---

## 2.3 本文定位

- **最相关工作**: VLAC [59]——也是预测相对进度变化，但 GRM 引入了 hop-based 归一化（保证 [0,1] 有界）+ 多视角融合
- **主要 baseline**: GVL [37]（VOC 评测标准提出者）和 VLAC [59]（PRM SOTA）
- **填补的空白**: 现有工作都无法同时满足"通用 + step-aware + 多视角 + 理论正确的奖励塑形"，本文是第一个

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| GVL Average VOC | 0.12-0.20 |
| VLAC Average VOC | 0.24-0.33 |
| GRM Average VOC | **0.93-0.96** |

### 核心洞察
1. RL for Robotics 的研究已经从"哪个算法好"转向"怎么设计奖励"
2. PRM 是趋势，但现有方法都有明显短板（单视角、uniform reward、task-specific）
3. 本文的关键差异化：hop-based 相对进度 + 多视角 + PBRS 理论保证
