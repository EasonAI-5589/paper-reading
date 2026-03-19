# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

**作者**: Moo Jin Kim¹², Yihuai Gao¹², Tsung-Yi Lin¹, Yen-Chen Lin¹, Yunhao Ge¹, Grace Lam¹, Percy Liang², Shuran Song¹², Ming-Yu Liu¹, Chelsea Finn², Jinwei Gu¹  
**机构**: ¹NVIDIA ²Stanford University  
**年份**: 2026 | **ArXiv**: [2601.16163](https://arxiv.org/abs/2601.16163)  
**项目主页**: [research.nvidia.com/labs/dir/cosmos-policy/](https://research.nvidia.com/labs/dir/cosmos-policy/)

---

## 一句话总结

把预训练视频模型（Cosmos-Predict2-2B）直接微调成机器人策略——不改架构、单阶段训练——通过将动作/未来状态/价值编码为 latent frames 注入视频扩散序列，实现三大 benchmark SOTA。

## 核心贡献

1. **Latent Frame Injection**：将非图像模态（动作、本体感知、价值函数）编码为 latent frames，直接注入视频扩散序列，零架构修改
2. **统一架构三合一**：同一个 DiT 同时作为 policy、world model、value function，通过 conditioning mask 切换训练目标
3. **数据效率极高**：RoboCasa 上只用 50 个 demo 超过所有用 300+ demo 的方法
4. **Model-based Planning**：从 rollout 经验学习，用 best-of-N 搜索在困难任务上额外提升 12.5%
5. **全面 SOTA**：LIBERO 98.5%、RoboCasa 67.1%、ALOHA 93.6%

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 论文概览 + Figure 1（系统架构） |
| [01 - Introduction](sections/01-introduction.md) | 动机：视频模型 vs VLM 的先验差异 + 贡献总结 |
| [02 - Related Work](sections/02-related-work.md) | Video policy / VLA / World model 三条路线的定位 |
| [03 - Preliminaries](sections/03-preliminaries.md) | Cosmos-Predict2 架构 + MDP + Figure 2（latent 序列） |
| [04 - Method](sections/04-method.md) | ⭐ 核心方法：Latent Injection + Joint Training + Planning |
| [05 - Experiments](sections/05-experiments.md) | ⭐ LIBERO/RoboCasa/ALOHA 全面评估 + 消融 + Planning |
| [06 - Discussion](sections/06-discussion.md) | 局限性：推理延迟、rollout 数据需求、搜索深度 |
| [07 - Appendix](sections/07-appendix.md) | 噪声调度调整、训练超参、详细评估设置、额外消融 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 基础模型 | Cosmos-Predict2-2B (DiT + Wan2.1 VAE) |
| LIBERO 成功率 | **98.5%** (SOTA, 超 CogVLA 97.4%) |
| RoboCasa 成功率 | **67.1%** (SOTA, 仅 50 demo，其他方法用 300+) |
| ALOHA 平均分 | **93.6%** (超 π₀.₅ 88.6%) |
| Planning 增益 | +12.5% (困难真实任务) |
| 预训练贡献 | +3.9% (LIBERO ablation) |
| Future state 预测贡献 | +22.7% (RoboCasa, 去掉后从 67.1%→44.4%) |
| 推理延迟 (5步) | 0.61s / action chunk |
| 推理延迟 (1步) | 0.16s (仅损失 0.7%) |
| Planning 延迟 | 4.9s (8× H100) |

## 方法概览

```
输入: 多视角图像 + 本体感知 + 语言指令
  ↓
[VAE Tokenization] → Latent Frames
  ↓
[Latent Frame Injection] → 插入 action/proprio/value 的 latent frames
  ↓
[DiT Denoising] → 条件扩散生成
  ↓
输出: Action chunk + Future state images + Value

Planning (可选):
  Policy Model → N 个候选动作
  Planning Model → 预测每个动作的未来状态 + 价值
  → 选择最高价值的动作执行
```

## 📝 批读总结

### 核心思路

**用视频模型生成动作，这就是 policy。** 视频模型天然理解物理世界的运动规律，把动作编码成 latent frames 注入视频扩散序列，模型在生成"未来视频"的同时也生成了动作。

### 方法三件套

| 组件 | 做了什么 | 为什么这样设计 |
|------|---------|-------------|
| **Latent Frame Injection**（4.1） | 把动作、本体感知、价值等非图像模态编码成 latent frames，插入视频扩散序列 | 不改架构，对 DiT 来说新插入的 frames 和图像 frames 长一样，无法区分 |
| **Joint Training**（4.2） | 同一个模型通过 conditioning mask 同时训练 policy + world model + value function | 三个任务共享权重，互相受益（auxiliary losses 贡献 +1.5%） |
| **Model-based Planning**（4.3） | 用 rollout 数据微调出 planning model，推理时 best-of-N 搜索选最优动作 | 让模型从"闭眼执行"变成"先想后做" |

### 关于 Planning（4.3）的理解

这是论文最不好懂的部分。关键点：

1. **为什么需要 rollout 数据？** 只在成功 demo 上训练的 world model 和 value function 没见过失败，给啥都打高分，无法区分好坏动作
2. **Dual Deployment**：原始 checkpoint 当 policy（提候选），微调后的 checkpoint 当 planning model（WM + VF，评估候选）。分开是为了保证 on-policy
3. **V(s') vs Q(s,a)**：V(s') 先用 WM 预测未来状态再打分（model-based），Q(s,a) 直接给动作打分（model-free）。V(s') 远优于 Q(s,a)，因为把一个难问题拆成了两个简单问题

### 实验亮点

| 实验 | 结论 | 意义 |
|------|------|------|
| **LIBERO**（Table 1） | 98.5% SOTA，所有方法统一 50 demo/task | 控制数据量比纯模型能力 |
| **RoboCasa**（Table 2） | 67.1%，只用 50 demo，其他方法用 300~3000 | **数据效率极高**，视频预训练先验 > 暴力加数据 |
| **ALOHA**（Figure 4） | 93.6%，超过 π₀.₅ 和 OpenVLA-OFT+ | 超越大规模动作数据预训练的 VLA |
| **消融**（Table 4） | 预训练 -3.9%，auxiliary losses -1.5% | 预训练先验是核心优势 |
| **Planning**（Figure 7） | 难任务 +12.5%，V(s') >> Q(s,a) | planning 有效但昂贵（8 GPU × 5s） |

### 对比方法的失败模式

- **π₀.₅**：高精度任务（ziploc bag）抓不住滑块 → VLM backbone 空间分辨率不足
- **OpenVLA-OFT+**：多模态任务（candies）手伸向两颗糖果中间 → L1 regression 的 mode averaging 问题
- **Cosmos Policy**：diffusion process 天然建模多模态分布 + 视频模型保留高分辨率空间信息，两个问题都避免了

### 局限性

- Planning 需要 8 GPU 并行 + ~5s/决策，**不适合实时控制**
- 需要额外收集 rollout 数据做后训练
- 基础模型 2B 参数，推理成本不低
- RoboCasa 上只测了 50 demo 设定，没测更多数据是否还能进一步提升

---

## 📊 Citation Landscape

**TLDR** (Semantic Scholar): *Cosmos Policy is a simple approach for adapting a large pretrained video model (Cosmos-Predict2) into an effective robot policy through a single stage of post-training on the robot demonstration data collected on the target platform, with no architectural modifications.*

**引用统计**: 参考文献 48 篇 | 被引 17 次 | Influential Citations: 2

### 参考文献分组 (Top by citations)

#### 🎬 Video Generation Models
| 论文 | 年份 | 引用 |
|------|------|------|
| CogVideoX | 2024 | 1,524 |
| Wan 2.1 | 2025 | 1,127 |
| HunyuanVideo | 2024 | 995 |
| Open-Sora | 2024 | 542 |
| Vidu | 2024 | 125 |

#### 🤖 Robot Policy / VLA
| 论文 | 年份 | 引用 |
|------|------|------|
| Diffusion Policy | 2023 | 2,623 |
| RT-2 | 2023 | 2,541 |
| OpenVLA | 2024 | 1,741 |
| ACT (ALOHA) | 2023 | 1,406 |
| π₀ | 2024 | 1,261 |
| π₀.₅ | 2025 | 596 |
| GR00T-N1 | 2025 | 542 |
| OpenVLA-OFT | 2025 | 322 |

#### 🌍 World Models & RL
| 论文 | 年份 | 引用 |
|------|------|------|
| Dreamer (v1) | 2019 | 1,748 |
| MBPO | 2019 | 1,138 |
| Dyna | 1991 | 1,127 |
| Dreamer (v2) | 2020 | 1,116 |
| Dreamer (v3) | 2023 | 947 |
| TD-MPC | 2022 | 367 |
| TD-MPC2 | 2023 | 340 |

#### 🏠 Benchmarks & Datasets
| 论文 | 年份 | 引用 |
|------|------|------|
| LIBERO | 2023 | 622 |
| MimicGen | 2023 | 270 |
| RoboCasa | 2024 | 246 |

#### 🔧 Foundation Models
| 论文 | 年份 | 引用 |
|------|------|------|
| T5 | 2019 | 24,758 |
| DiT | 2022 | 4,942 |
| FiLM | 2017 | 3,179 |
| EDM | 2022 | 3,039 |

### 🔗 相关链接

- [Connected Papers](https://www.connectedpapers.com/main/2601.16163)
- [Semantic Scholar](https://www.semanticscholar.org/paper/e1042da4004da8b246737df2ee1ea96012b17824)
- [arXiv](https://arxiv.org/abs/2601.16163)
- [Project Page](https://research.nvidia.com/labs/dir/cosmos-policy/)
