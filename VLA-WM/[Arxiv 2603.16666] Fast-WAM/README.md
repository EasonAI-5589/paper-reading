# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

**作者**: Tianyuan Yuan, Zibin Dong, Yicheng Liu, **Hang Zhao**（赵行）
**来源**: arXiv 2603.16666 (Mar 2026) | **机构**: 清华大学交叉信息学院 (IIIS) + 星海图 (Galaxea AI)
**链接**: [arXiv](https://arxiv.org/abs/2603.16666) | [Project Page](https://yuantianyuan01.github.io/FastWAM/)

## 一句话总结

Fast-WAM 通过将训练阶段的视频建模与推理阶段的未来生成**解耦**，证明了 WAM 的性能增益主要来自训练时的视频联合训练（学到更好的世界表征），而非推理时的显式未来想象。推理时仅需单次前向传播，时延 190ms，比 imagine-then-execute WAM 快 4 倍以上。

## 核心贡献

1. **提出核心问题**: WAM 的增益到底来自训练时的视频建模，还是推理时的未来想象？这个问题此前从未被系统研究
2. **Fast-WAM 架构**: 基于 MoT (Mixture-of-Transformer) 共享注意力，Video DiT + Action Expert DiT；训练时保留视频联合训练，推理时跳过未来视频生成，直接从世界表征预测动作
3. **控制变量实验设计**: 构建 Fast-WAM-Joint / Fast-WAM-IDM / Fast-WAM w.o. video co-train 三个变体，共享骨干/tokenization/训练配方，系统隔离训练目标 vs 推理方式的贡献
4. **关键发现**: 去掉推理时未来想象 → 性能几乎不变（差距 <2%）；去掉训练时视频联合训练 → 性能断崖式下跌（RoboTwin -8%, 真实世界 85%→10%）

## 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：核心问题 + 方法 + 关键发现 |
| [01 - Introduction](sections/01-introduction.md) | 动机：WAM 两个增益来源的纠缠 + Fast-WAM 解耦思路 + 三个贡献 |
| [02 - Related Work](sections/02-related-work.md) | VLA 策略 + WAM 和视频机器人策略 + 与 VPP/UVA 的区别 |
| [03 - Method](sections/03-method.md) | 问题形式化 + MoT 架构 + 结构化注意力掩码 + Flow Matching 训练 + 控制变体设计 |
| [04 - Experiments](sections/04-experiments.md) | ⭐ RoboTwin + LIBERO + 真实世界毛巾折叠 + 控制变量对比（全文核心） |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性分析 + 总体评价 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 总参数量 | ~6B（5B Video DiT + 1B Action Expert DiT） |
| Video 骨干 | Wan2.2-5B（预训练 DiT） |
| Action Expert 隐层维度 | $`d_a = 1024`$ |
| 动作 horizon | h = 32 |
| 视频帧数/chunk | 9 帧（4× 时间下采样） |
| 推理去噪步数 | 10 步，CFG scale=1.0 |
| **推理时延** | **190 ms**（单 NVIDIA RTX 5090D V2 32GB） |
| **RoboTwin 成功率** | **91.8%**（无 embodied pretraining） |
| **LIBERO 平均成功率** | **97.6%**（无 embodied pretraining） |
| 真实世界平台 | Galaxea R1 Lite（毛巾折叠） |
| 真实世界训练数据 | 60 小时遥操作 |
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01, cosine annealing |

## ⭐ 控制变量实验结果（全文核心）

| 变体 | 训练视频目标 | 推理未来想象 | RoboTwin Avg | LIBERO Avg | 推理时延 |
|------|:---:|:---:|:---:|:---:|:---:|
| **Fast-WAM** | ✅ | ❌ | **91.8%** | 97.6% | **190 ms** |
| Fast-WAM-Joint | ✅ | ✅ (联合) | 90.6% | 98.5% | 580 ms |
| Fast-WAM-IDM | ✅ | ✅ (先视频后动作) | 91.3% | 98.0% | 810 ms |
| **Fast-WAM w.o. video co-train** | ❌ | ❌ | **83.8%** ⬇️ | **93.5%** ⬇️ | 190 ms |

> 💡 **核心结论**: 三个有视频联合训练的变体性能接近（差距 <2%），去掉视频联合训练后性能断崖式下跌（-8% ~ -81%）。**视频训练的价值 >> 推理时未来想象的价值**。

---

## BibTeX

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Yuan, Tianyuan and Dong, Zibin and Liu, Yicheng and Zhao, Hang},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026}
}
```

## 相关论文

| 论文 | 年份 | 关系 |
|------|------|------|
| Motus (Bi et al.) | 2025 | 统一 latent action world model，MoT 架构先驱，Fast-WAM 的 Joint 变体与之对应 |
| LingBot-VA (Li et al.) | 2026 | Causal world modeling，video-then-action 范式，Fast-WAM-IDM 变体与之对应 |
| WAM (Ye et al.) | 2026 | 定义了 WAM 概念，joint modeling 范式 |
| π0 / π0.5 (Physical Intelligence) | 2024/2025 | VLA 基线，flow model |
| UVA (Li et al.) | 2025 | 同样探索跳过推理时视频解码，但缺少控制变量实验 |
| VPP (Hu et al.) | 2024 | 用 video diffusion model 的表征做策略条件，不做显式视频生成 |
| LeWorldModel (Maes et al.) | 2026 | JEPA 世界模型，LeCun 组，谢赛宁建议与本文一起看 |
