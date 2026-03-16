# Motus: A Unified Latent Action World Model

**作者**: Hongzhe Bi, Hengkai Tan (共同项目负责人), Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, **Hang Su**, **Jun Zhu**
**来源**: arXiv 2512.13030 (Dec 2025) | **机构**: 清华大学 + 北京大学 + 地平线机器人
**链接**: [arXiv](https://arxiv.org/abs/2512.13030) | [Project Page](https://motus-robotics.github.io/motus) | [GitHub](https://github.com/thu-ml/Motus)

## 一句话总结

Motus 通过 Mixture-of-Transformer (MoT) 架构统一 VLM + VGM + Action Expert 三个预训练专家，配合 UniDiffuser 式调度器和光流驱动的 latent action，将 5 种具身建模范式（VLA、WM、IDM、VGM、联合预测）融合在单一模型中，并通过三阶段训练 + 六层数据金字塔实现大规模跨具身预训练。

## 核心贡献

1. **统一五大建模范式**: 在单一生成框架中统一 VLA、World Model、IDM、Video Generation Model 和 Video-Action Joint Prediction Model，无需牺牲通用多模态先验
2. **MoT + Tri-model Joint Attention**: 三个专家各自保留独立 FFN，共享多头自注意力层，融合 VLM（Qwen3-VL-2B）、VGM（Wan 2.2 5B）和 Action Expert 的互补知识
3. **UniDiffuser 式调度器**: 为视频和动作分配不同的 timestep/噪声尺度，灵活切换 5 种推理模式
4. **光流驱动的 Latent Action**: 用光流编码像素级 "delta action"，使 action expert 能在无动作标签的视频数据上预训练，桥接视觉动态与控制信号
5. **可扩展训练方案**: 三阶段训练（视频预训练 → latent action 预训练 → 具身 SFT）+ 六层数据金字塔（web → egocentric → synthetic → task-agnostic → multi-robot → target-robot）

## 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1 架构图 + 五大统一分布 |
| [01 - Introduction](sections/01-introduction.md) | 两大挑战 + 解决方案概述 |
| [02 - Related Work](sections/02-related-work.md) | 统一多模态模型 + Latent Action 模型 |
| [03 - Problem Formulation](sections/03-problem-formulation.md) | 问题定义 + 五种分布 + 两大挑战详述 |
| [04 - Methodology](sections/04-method.md) | MoT 架构 + Latent Action VAE + 训练流程 |
| [05 - Experiments](sections/05-experiments.md) | RoboTwin 2.0 仿真 + 真实世界实验 + 消融 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + 局限性 + 未来方向 |
| [07 - Appendix](sections/07-appendix.md) | 补充实验 + 实现细节 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 模型参数 | ~1B（Action Expert 与 Wan 同深度） |
| VGM 骨干 | Wan 2.2 5B |
| VLM 骨干 | Qwen3-VL-2B |
| 仿真提升 (vs X-VLA) | **+15%** (88.66% vs 72.80%, RoboTwin 2.0 Clean) |
| 仿真提升 (vs $`\pi_{0.5}`$) | **+45%** (88.66% vs 42.98%, RoboTwin 2.0 Clean) |
| 真实世界提升 | **+11~48%**（跨 AC-One 和 Agilex-Aloha-2 两平台） |
| 仿真任务数 | 50 个 RoboTwin 2.0 操作任务 |
| 真实世界平台 | AC-One、Agilex-Aloha-2（双臂） |
| 训练阶段 | 3 阶段（Video Gen → Latent Action → SFT） |
| 数据金字塔 | 6 层（Web → Egocentric → Synthetic → Task-agnostic → Multi-robot → Target-robot） |
| Latent Action 维度 | 14 维（匹配典型机器人动作空间） |
| Latent Action 训练数据配比 | 90% 无标签 + 10% 有标签 |
| SFT 微调步数 | 40k steps |

---

## 📊 Citation Landscape

**TLDR** (Semantic Scholar): Motus is a unified latent action world model that leverages existing general pretrained models and rich, sharable motion information to enable large-scale action pretraining and achieves superior performance against state-of-the-art methods in both simulation and real-world scenarios.

| 指标 | 数值 |
|------|------|
| 参考文献数 | 62 |
| 被引次数 | 19 |
| Influential Citations | 2 |

**🔗 外部链接**: [Semantic Scholar](https://www.semanticscholar.org/paper/c9f926f18886a1e4107f91125aac209a4565bfa0) | [Connected Papers](https://www.connectedpapers.com/main/2512.13030)

### 推荐论文 (Semantic Scholar)

| 论文 | 年份 | 引用 | arXiv |
|------|------|------|-------|
| BagelVLA: Long-Horizon Manipulation via Interleaved Generation | 2026 | 1 | 2602.09849 |
| LDA-1B: Scaling Latent Dynamics Action Model | 2026 | 0 | 2602.12215 |
| UniLACT: Depth-Aware RGB Latent Action Learning | 2026 | 0 | 2602.20231 |
| VLA-JEPA: Enhancing VLA with Latent World Models | 2026 | 0 | 2602.10098 |
| Causal World Modeling for Robot Control | 2026 | 6 | 2601.21998 |
| BridgeV2W: Bridging Video Generation to Embodied WM | 2026 | 0 | 2602.03793 |
| HALO: Unified VLA Model for Embodied Manipulation | 2026 | 0 | 2602.21157 |
| World Guidance: WM in Condition Space for Action | 2026 | 0 | 2602.22010 |
| Chain of World: WM Thinking in Latent Motion | 2026 | 0 | 2603.03195 |

---

## BibTeX

```bibtex
@article{bi2025motus,
  title={Motus: A Unified Latent Action World Model},
  author={Bi, Hongzhe and Tan, Hengkai and Xie, Shenghao and Wang, Zeyuan and Huang, Shuhe and Liu, Haitian and Zhao, Ruowen and Feng, Yao and Xiang, Chendong and Rong, Yinze and Zhao, Hongyan and Liu, Hanyu and Su, Zhizhong and Ma, Lei and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.13030},
  year={2025}
}
```
