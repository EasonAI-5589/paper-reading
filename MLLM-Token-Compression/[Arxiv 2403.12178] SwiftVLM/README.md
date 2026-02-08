# SwiftVLM: Efficient Vision-Language Model Inference via Cross-Layer Token Bypass

**作者**: Chen Qian*, Xinran Yu*, Danyang Li, Guoxuan Chi, Zheng Yang, Qiang Ma, Xin Miao (Tsinghua University)  
**来源**: Arxiv 2403.12178 (2026.02)  
**链接**: [arXiv](https://arxiv.org/abs/2403.12178)

## 一句话总结
提出 **bypass** 剪枝范式——不丢弃低排名 visual token，而是保留并转发到后续剪枝层重新评估，结合 DP 选择最优剪枝层，实现 training-free 的高效视觉 token 剪枝。

## 核心贡献
1. **发现层间 token 重要性差异**：浅层认为不重要的 token 在深层可能变得高度相关
2. **Bypass 范式**：第三种剪枝策略（与 merge/drop 并列），保留未选中 token 并通过 offset alignment 在后续层重新评估
3. **DP 选层**：发现各层 token 判别能力非单调，用动态规划选择最优剪枝层
4. **SwiftVLM**：training-free 方法，在 2 个 VLM、9 个 benchmark 上全面超越现有方法

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：bypass 范式概述 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 层间 token 重要性分析 + 贡献 (Figure 1-4) |
| [02 - Related Work](sections/02-related-work.md) | Text-agnostic vs Text-aware 方法分类 |
| [03 - Method](sections/03-method.md) | Attention 基础 + DP 选层 + Bypass 架构 + 对齐分析 + FLOPs (Figure 5, Eq.1-19) |
| [04 - Experiments](sections/04-experiments.md) | 主实验 + 效率 + 消融 + 分析 + 泛化 (Table 1-4, Figure 6-8) |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性分析 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LLaVA-1.5-7B 最优剪枝层 | 3, 11, 15 |
| 192 tokens, Localization 相对准确率 | **86.9%** (vs FEATHER 66.9%) |
| 128 tokens, Localization 相对准确率 | **69.8%** (vs FEATHER 50.8%) |
| 192 tokens, Non-localization 相对准确率 | **99.0%** |
| 128 tokens, Prefill 加速 | **2.04×** |
| LLaVA-NeXT, 22.2% tokens, 相对准确率 | **97.1%** |

## 方法速览

```
Input Image → Visual Encoder → 576 visual tokens
                                    ↓
              Layer 1-2: 全部保留
                                    ↓
              Layer 3 (剪枝层 x): T-V attention 排序
                 ├── Top tokens → 直接保留
                 └── Low tokens → 分组合并(代理) + bypass(保留原始)
                                    ↓
              Layer 4-10: 用 top + merged tokens 推理
                                    ↓
              Layer 11 (剪枝层 y): offset alignment → 恢复 bypass tokens → 重新排序
                 └── 最终保留 top tokens
                                    ↓
              Layer 12-32: 用最终选中的 tokens 推理
                                    ↓
              Output
```
