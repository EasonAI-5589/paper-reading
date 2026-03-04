# VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model

**作者**: Yanjiang Guo*, Tony Lee*, Lucy Xiaoyang Shi*, Jianyu Chen, Percy Liang, Chelsea Finn  
**机构**: Stanford University, Tsinghua University  
**年份**: 2026 | **arXiv**: [2602.12063](https://arxiv.org/abs/2602.12063)  
**链接**: [PDF](https://arxiv.org/pdf/2602.12063) · [Project Page](https://sites.google.com/view/vlaw-arxiv) · [paper.pdf](paper.pdf)

---

## 一句话总结

用少量真实机器人 rollout（含 failure）fine-tune World Model 来消除过度乐观偏差，再让 World Model 生成 10× 的合成成功轨迹训练 VLA policy。两轮迭代后，5 类 contact-rich 任务平均成功率 46% → 87%（+39.2 pp），其中合成数据贡献 +11.6 pp。

---

## 核心贡献

1. **发现并解决 World Model 的 over-optimism 问题**：用包含 failure case 的 online rollout 数据 fine-tune pretrained Ctrl-World，FP 从 11 降到 1（↓91%），FVD 从 225 降到 64
2. **VLAW 迭代协同优化 pipeline**：World Model ↔ VLA Policy 相互增强的正反馈循环，每轮 50 条 real rollout 撬动 500 条合成轨迹
3. **Flow-matching VLA 的 RL 理论框架**：将 binary-filtered BC 与 AWR / regularized RL 正式联系起来
4. **真实机器人验证**：5 类 contact-rich 任务（stacking、open book、erase marks、scooping、drawing），每类 50 次评估

---

## 📖 批读导航

| Section | 文件 | 内容 |
|---------|------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 | 动机、方法一句话、关键数字（39.2% / 11.6%） |
| [01 - Introduction](sections/01-introduction.md) | 引言 | 三层铺垫 + 贡献清单 + Figure 1, 2 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作 | VLA post-training 三条路 + World Model for MBRL 演进 |
| [03 - Preliminaries](sections/03-preliminaries.md) | 预备知识 | MDP 设定 + π_θ / M_φ 符号 + closed-loop 想象机制 |
| [04 - Method](sections/04-method.md) | 方法 | 4.1 WM fine-tune + 4.2 Policy fine-tune + 4.3 AWR 理论联系 + Algorithm 1 |
| [05 - Experiments](sections/05-experiments.md) | 实验 | 5.1 设置 + 5.2 WM 质量（Table 1 + confusion matrix） + 5.3 Policy 提升（Table 2 + ablation） |
| [06 - Conclusions](sections/06-conclusions.md) | 结论 | 总结 + 总体评价（优缺点） |
| [07 - Appendix](sections/07-appendix.md) | 附录 | A: AWR 推导 + B: 任务细节 + C: Reward model 混淆矩阵 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| Base policy 平均成功率 | 0.460 |
| VLAW-2 平均成功率 | **0.868** |
| 总提升 | +39.2 pp |
| 合成数据贡献（vs. Filtered BC-2） | +11.6 pp |
| 最难任务（Drawing）提升 | +56 pp（0.22 → 0.78） |
| World model FP 减少（加入 online rollout 后） | 11 → 1（↓91%） |
| Real rollout / task / iteration | 50 条 |
| Synthetic rollout / task / iteration | 500 条（10× 放大） |
| 迭代次数 | 2 轮 |

---

## 📊 Citation Landscape

> 数据来源：[Semantic Scholar](https://www.semanticscholar.org/paper/affb9be3dedba7952dfad2fb44a3ccdae909c60d) · [Connected Papers](https://www.connectedpapers.com/main/2602.12063)

### TLDR (Semantic Scholar)

> A simple iterative improvement algorithm is proposed that uses real-world roll-out data to improve the fidelity of the world model, which can then be used to generate supplemental synthetic data for improving the VLA model.

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 62 |
| 被引次数 | 0（新论文，2026-02） |
| Influential Citations | 0 |

### 参考文献分组（Top 5 per category，按引用量排序）

#### VLA / Robot Policy

| 论文 | 年份 | 引用 |
|------|------|------|
| OpenVLA: An Open-Source Vision-Language-Action Model | 2024 | 1,607 |
| π₀: A Vision-Language-Action Flow Model for General Robot Control | 2024 | 1,150 |
| π₀.₅: A Vision-Language-Action Model with Open-World Generalization | 2025 | 523 |
| DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | 2024 | 551 |
| AWR: Simple and Scalable Off-Policy Reinforcement Learning | 2019 | 735 |

#### World Model / Video Generation

| 论文 | 年份 | 引用 |
|------|------|------|
| Stable Video Diffusion | 2023 | 2,080 |
| Mastering Atari with Discrete World Models (Dreamer v2) | 2020 | 1,092 |
| DayDreamer: World Models for Physical Robot Learning | 2022 | 428 |
| Action-Conditional Video Prediction using Deep Networks in Atari Games | 2015 | 890 |
| Deep visual foresight for planning robot motion | 2016 | 840 |

#### RL / Optimization

| 论文 | 年份 | 引用 |
|------|------|------|
| PPO: Proximal Policy Optimization Algorithms | 2017 | 25,333 |
| TRPO: Trust Region Policy Optimization | 2015 | 7,653 |
| Dream to Control: Learning Behaviors by Latent Imagination (Dreamer v1) | 2019 | 1,715 |
| DeepSeekMath (GRPO) | 2024 | 4,690 |
| Visual Foresight: Model-Based Deep RL for Vision-Based Robotic Control | 2018 | 440 |

#### Metrics / Evaluation

| 论文 | 年份 | 引用 |
|------|------|------|
| SSIM: Image quality assessment | 2004 | 54,843 |
| GANs Trained by a Two Time-Scale Update Rule (FID) | 2017 | 17,037 |
| Unreasonable Effectiveness of Deep Features (LPIPS) | 2018 | 16,035 |
| PSNR vs. SSIM | 2010 | 4,617 |
| Towards Accurate Generative Models of Video (FVD) | 2018 | 1,085 |

### 推荐论文（Semantic Scholar Recommendations）

| 论文 | 年份 | 引用 | arXiv |
|------|------|------|-------|
| World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy | 2026 | 0 | 2602.06508 |
| WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL | 2026 | 0 | 2602.13977 |
| Beyond Imitation: RL-Based Sim-Real Co-Training for VLA Models | 2026 | 0 | 2602.12628 |
| World-Gymnast: Training Robots with RL in a World Model | 2026 | 1 | 2602.02454 |
| RISE: Self-Improving Robot Policy with Compositional World Model | 2026 | 0 | 2602.11075 |
| GigaBrain-0.5M*: a VLA That Learns From World Model-Based RL | 2026 | 0 | 2602.12099 |
| On-the-Fly VLA Adaptation via Test-Time RL | 2026 | 1 | 2601.06748 |
| Self-Correcting VLA: Online Action Refinement via Sparse World Imagination | 2026 | 0 | 2602.21633 |
| RoboCurate: Harnessing Diversity with Action-Verified Neural Trajectory | 2026 | 0 | 2602.18742 |
| SOP: A Scalable Online Post-Training System for VLA Models | 2026 | 5 | 2601.03044 |

> 💡 推荐列表清一色 2026 年的 VLA + World Model 工作，说明这是一个正在快速发展的研究方向。World-VLA-Loop、WoVR、World-Gymnast 和 GigaBrain 都在做类似的事情（world model + VLA policy 联合优化），是直接竞品。
