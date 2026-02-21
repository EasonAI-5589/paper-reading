# HiDivDrop: Vision Token Reduction in MLLMs via Late Injection and Differentiable Top-K

> **ICLR 2026** (Submission #25145)
> OpenReview: https://openreview.net/forum?id=2baJBgfr9S
> arXiv: 可能为 2503.14075 或尚未公开 (搜索结果显示 OpenReview-only)
> Keywords: MLLMs, Vision Token Pruning, Efficiency and Compression, Interpretability and Analysis

## 一句话总结

现有 progressive vision token pruning 方法**误解了浅层的作用**（以为浅层做 fusion，实际是 passive 的）并使用**过于僵硬的 pruning schedule**。HiDivDrop 通过 **Late Injection**（跳过浅层）和 **Concave Pyramid Pruning**（动态自适应 pruning rate + Early Exit）解决这两个问题，~90% compression 下保持原始性能。

## 核心贡献

1. **Late Injection Strategy**: 绕过 passive 浅层，直接在 active fusion 层注入 visual tokens，避免过早丢弃
2. **Concave Pyramid Pruning**: 动态调整 middle/deep layers 的 pruning rate，early exit 机制
3. **Differentiable Top-K Operator**: 可微分的 token 选择算子，支持端到端训练优化
4. **Inter-layer Similarity Measure**: 层间相似性度量用于优化 pruning schedule
5. **SOTA Results**: ~90% visual token compression，98.3% performance retention @ 88.9% pruning，training acceleration 1.72×

## 方法概述

```
Visual Encoder → [Skip Shallow Layers] → Late Injection at Active Fusion Layer
                                        → Concave Pyramid Pruning:
                                          - Accelerated early reduction
                                          - Differentiable Top-K selection
                                          - Early Exit when similarity saturates
                                        → Reduced tokens for remaining layers
```

### 1. Late Injection
- 观察：MLLM 浅层对 visual tokens 是 "passive" 的（不做真正的 multimodal fusion）
- 方案：不在 layer 0 注入 visual tokens，而是在 active fusion 开始的层注入
- 好处：减少浅层的无效计算 + 兼容 FlashAttention + 解决动态 pruning 的 position ID mismatch

### 2. Concave Pyramid Pruning
- 不同于 PyramidDrop 的线性/均匀 schedule
- 使用凹形（concave）曲线：前期快速裁剪，后期逐渐减缓
- Early Exit: 当层间表示相似度饱和时停止 pruning

### 3. Differentiable Top-K
- 标准 Top-K 不可导，无法做端到端训练
- 使用可微分近似，使 pruning 决策可以被 gradient-based optimization 优化
- 使得 pruning schedule 可以与模型训练联合优化

## 关键实验结果 (LLaVA-1.5-7B, 11 benchmarks)

| Pruning Ratio | Avg. Performance Retention | vs PDrop |
|---|---|---|
| 66.7% | ~99%+ | — |
| 88.9% | **98.3%** | +4.1% |
| 91.7% | **96.5%** | PDrop fails at this ratio |

- Training acceleration: **1.72×**
- Prefill latency: 63.6ms → 28.8ms

## 局限性（推测）

- 需要训练（vs DART 的 training-free）
- Late Injection 的最优层需要确定（可能 model-specific）
- Differentiable Top-K 的训练稳定性可能需要额外调参

## 与其他工作的关系

- **vs PyramidDrop/PDrop**: HiDivDrop 纠正了 PDrop 对浅层角色的误解，且用更灵活的 concave schedule
- **vs DART/StopLooking**: DART 是 training-free + duplication-based；HiDivDrop 是 training-based + schedule-optimized。两者关注不同维度（what to prune vs when/how much to prune），可能互补
- **vs FastV**: FastV 在固定层做 one-shot pruning；HiDivDrop 做 progressive 但 with better schedule
- **vs VisionTrim**: 另一篇 ICLR 2026 token compression 工作，可能关注不同方面

---

## Citation Landscape

*Note: 由于 Semantic Scholar 429 rate limit，暂无 citation 数据。HiDivDrop 作为 ICLR 2026 接收论文，预计会有较高影响力。*

### 核心参考文献

| Paper | Relation |
|---|---|
| PyramidDrop (CVPR 2025) | 直接 baseline，HiDivDrop 纠正其设计缺陷 |
| FastV (ECCV 2024) | Attention-based one-shot pruning |
| ToMe (ICLR 2023) | Token merging 先驱 |
| FlashAttention (NeurIPS 2022) | HiDivDrop 兼容 FA |
| LLaVA-1.5 | 主要实验平台 |
