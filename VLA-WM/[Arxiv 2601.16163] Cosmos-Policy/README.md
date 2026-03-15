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
