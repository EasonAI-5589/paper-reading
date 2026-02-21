# 1. Introduction

## 原文要点

MLLMs 在 image captioning、VQA、video understanding 等任务上表现出色，但计算开销巨大，主要来自大量 vision tokens。现有方法通过 token pruning 来加速，核心是定义 token importance score 后裁掉 "不重要" 的 tokens。

本文指出 importance-based methods 存在四大严重问题：

**(I) 忽略 token 间的交互**：Importance 是静态计算的，但当某个 token 被裁掉后，剩余 token 的 importance 应该发生变化。例如两个相似 token，裁掉一个后另一个应该变得更重要。现有方法完全忽略这种动态交互。

**(II) 与 FlashAttention 不兼容**：FlashAttention 不暴露 attention scores，而大多数 importance-based 方法恰恰依赖 attention scores。关闭 FlashAttention 会显著增加延迟和内存。

**(III) Position bias**：Attention scores 存在位置偏差——靠近最后 token 的位置倾向于获得更高 attention score，这不能真正反映 token 的价值。

**(IV) 精度下降严重**：最令人惊讶的是，一些有影响力的 importance-based 方法在 88.9% reduction ratio 下，精度甚至**不如随机裁剪** (Figure 2)。

这些观察启发了 DART：用 **token duplication** 替代 importance 作为裁剪标准。直觉是：多个 token 表示相同信息时，保留一个即可。

DART 两步流程：
1. 选少量 pivot tokens (≤2%)
2. 计算每个 token 与 pivot 的 cosine similarity，保留低相似度（即非重复）的 token

> 💡 **这是这篇论文最核心的 insight**：importance ≠ 该不该保留。两个高 importance 的 token 如果高度相似，保留一个就够了。这从 information theory 角度是非常自然的——我们要 maximize retained information，而不是 maximize retained importance sum。

> 💡 Figure 2 是全文最具说服力的证据：FastV 和 SparseVLM 在 88.9% reduction 下居然不如 random pruning。这说明 attention-based importance 在极端压缩下不仅没用，还有害——因为 position bias 导致保留的 token 集中在特定区域。

> 💡 与 FlashAttention 的兼容性是一个很实际的工程优势。现实部署中 FlashAttention 几乎是标配，任何需要关闭 FA 才能工作的方法都会面临实际 speedup 大打折扣的问题。SparseVLM 的实际加速只有 1.56×，远低于理论值，就是因为这个原因。

## 三大贡献总结

1. **Rethink Token Importance**：实证证明 attention score 不适合做 token pruning 的指标
2. **Token Duplication as Key Factor**：Training-free, plug-and-play, FlashAttention 兼容
3. **Superior Performance with Extreme Compression**：4 个 MLLM, 10+ benchmarks 全面领先
