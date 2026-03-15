[← 返回 README](../README.md)

# Abstract

## 📌 预览
Cosmos Policy 的核心主张：把预训练的视频生成模型（Cosmos-Predict2）直接微调成机器人策略，**不改架构、单阶段训练**，同时生成动作、未来状态和价值函数，实现 SOTA 性能。

---

# COSMOS POLICY: FINE-TUNING VIDEO MODELS FOR VISUOMOTOR CONTROL AND PLANNING

Moo Jin Kim¹² Yihuai Gao¹² Tsung-Yi Lin¹ Yen-Chen Lin¹ Yunhao Ge¹ Grace Lam¹ Percy Liang² Shuran Song¹² Ming-Yu Liu¹ Chelsea Finn² Jinwei Gu¹

¹NVIDIA ²Stanford University
https://research.nvidia.com/labs/dir/cosmos-policy/

> 💡 **作者团队**: NVIDIA + Stanford 的豪华阵容。Chelsea Finn（机器人学习大佬）和 Ming-Yu Liu（NVIDIA 视觉生成领域核心人物）共同指导。Moo Jin Kim 之前做了 OpenVLA 和 OpenVLA-OFT，现在转向视频模型路线。

![Figure 1](../images/a058f0f4361ff8b4637f60793836b1e1e56298a0527bcbf107cc0152d732bc75.jpg)
*Figure 1: Cosmos Policy 概览。基于 NVIDIA Cosmos-Predict2-2B 视频基础模型微调而成的 SOTA 机器人策略。处理多模态输入和多视角相机图像，预测 (1) 动作序列，(2) 未来状态（机器人本体感知 + 图像观测），(3) 价值（未来状态的期望奖励）。不修改基础视频模型架构，所有模态通过视频扩散学习目标联合建模。*

> 💡 **Figure 1 批读**:
> - 这张图展示了 Cosmos Policy 的整体架构：输入是多视角相机图像 + 语言指令 + 机器人本体感知状态，输出是动作、未来状态和价值
> - **关键洞察**：所有输出都被编码为 latent frames，复用视频模型的扩散生成管线，这是 "no architectural changes" 的核心
> - 与 VLA 方法（如 π₀, OpenVLA）的本质区别：VLA 基于 vision-language model（学的是语义概念），Cosmos Policy 基于 video model（学的是时序动力学）

---

Recent video generation models demonstrate remarkable ability to capture complex physical interactions and scene evolution over time. To leverage their spatiotemporal priors, robotics works have adapted video models for policy learning but introduce complexity by requiring multiple stages of post-training and new architectural components for action generation. In this work, we introduce Cosmos Policy, a simple approach for adapting a large pretrained video model (Cosmos-Predict2) into an effective robot policy through a single stage of post-training on the robot demonstration data collected on the target platform, with no architectural modifications. Cosmos Policy learns to directly generate robot actions encoded as latent frames within the video model's latent diffusion process, harnessing the model's pretrained priors and core learning algorithm to capture complex action distributions. Additionally, Cosmos Policy generates future state images and values (expected cumulative rewards), which are similarly encoded as latent frames, enabling test-time planning of action trajectories with higher likelihood of success. In our evaluations, Cosmos Policy achieves state-of-the-art performance on the LIBERO and RoboCasa simulation benchmarks (98.5% and 67.1% average success rates, respectively) and the highest average score in challenging real-world bimanual manipulation tasks, outperforming strong diffusion policies trained from scratch, video model-based policies, and state-of-the-art vision-language-action models fine-tuned on the same robot demonstrations. Furthermore, given policy rollout data, Cosmos Policy can learn from experience to refine its world model and value function and leverage model-based planning to achieve even higher success rates in challenging tasks. We release code, models, and training data at https://research.nvidia.com/labs/dir/cosmos-policy/.

> 💡 **Abstract 批读**:
> 
> **核心贡献三句话概括**：
> 1. **简单统一**：把视频模型直接微调成策略，不改架构、不加额外模块、单阶段训练
> 2. **多模态生成**：动作、未来状态、价值函数都编码为 latent frames，通过同一个扩散过程生成
> 3. **可规划**：用 world model + value function 做 best-of-N 搜索，进一步提升成功率
> 
> **关键数字**：
> - LIBERO: 98.5% 平均成功率（SOTA）
> - RoboCasa: 67.1%（SOTA，且仅用 50 个 demo，别人用 300+）
> - 真实世界 ALOHA 双臂任务：93.6% 平均分
> - 规划带来额外 12.5% 提升
> 
> **与 STAR-Pro 的关系**：这篇论文聚焦于利用视频模型的 spatiotemporal prior 做 low-level control，与 STAR-Pro 的多智能体规划是不同层面的工作，但视频模型作为 policy backbone 的思路值得关注。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 基础模型 | Cosmos-Predict2-2B |
| LIBERO 成功率 | 98.5% |
| RoboCasa 成功率 | 67.1% |
| ALOHA 平均分 | 93.6% |
| 规划增益 | +12.5% |

### 核心洞察
1. 视频模型的时空先验比 VLM 的语义先验更适合 low-level 控制
2. 不需要设计单独的动作模块——视频扩散的生成能力足以建模动作分布
3. 统一架构同时做策略、世界模型、价值函数
