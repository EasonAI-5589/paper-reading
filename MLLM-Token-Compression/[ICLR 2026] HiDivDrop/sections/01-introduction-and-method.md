# HiDivDrop: Introduction & Method

> ⚠️ 本节基于 OpenReview abstract + web search 信息整理。PDF 为二进制格式无法直接解析（无 MinerU token），详细 section-by-section 批读待 PDF 解析后补充。

## 问题定义

MLLM 的计算瓶颈在于 vision tokens 的二次方复杂度。Progressive vision token pruning 是有前景的方案，但现有方法有两个关键限制：

1. **误解浅层的作用**：现有方法（如 PDrop）认为浅层对 multimodal fusion 至关重要，因此在浅层保留大量 visual tokens。但实际上浅层是 "passive" 的——不做真正的跨模态融合。
2. **僵硬的 pruning schedule**：线性或均匀的 pruning rate 分配，不能适应不同层的实际需求。

> 💡 这两个观察非常精辟。PyramidDrop 的 "pyramid" schedule（浅层保留多，深层保留少）看似直觉合理，但如果浅层本身不做 fusion，那么在浅层保留大量 visual tokens 是浪费计算。HiDivDrop 的 Late Injection 直接跳过这些浪费。

## Late Injection Strategy

核心 insight：浅层是 passive 的，visual tokens 在浅层不会被有效利用。

- 不在 layer 0 注入 visual tokens
- 识别 "active fusion" 开始的层（通过层间行为分析）
- 在该层直接注入 visual tokens
- 好处：
  - 避免浅层的无效计算
  - 解耦 visual KV projection 和 prefill bottleneck
  - 兼容 FlashAttention-style kernels
  - 修复动态 pruning 带来的 position ID mismatch

> 💡 Late Injection 是一个非常 elegant 的设计。它的哲学是：与其在浅层精心 prune（像 FastV 在 layer 2 后 prune），不如直接不给浅层 visual tokens。这是 "最极端的 pruning"——100% pruning in shallow layers。

> 💡 Position ID mismatch 是一个容易被忽视的工程细节。动态 pruning 会改变 token 数量，导致 position embedding 不匹配。HiDivDrop 通过 Late Injection 优雅地解决了这个问题。

## Concave Pyramid Pruning

替代 PyramidDrop 的线性 schedule：

- **凹形曲线**：前期（刚注入后）快速裁剪，后期逐渐减缓
- **直觉**：刚注入的 visual tokens 有最多冗余（还没被 LLM "消化"），应该大幅裁剪；深层已经是精炼后的 tokens，应该谨慎裁剪
- **Early Exit 机制**：监控层间 token 表示的相似度，当相似度饱和（变化很小）时停止 pruning，避免过度裁剪

> 💡 Concave 比 linear 更合理的原因：信息熵在 pruning 过程中是递减的。前几次 pruning 去掉的是最冗余的 tokens（信息损失小），后续 pruning 开始触及有用信息（信息损失大）。所以应该 "前快后慢"。

> 💡 Early Exit 机制类似于 training 中的 early stopping——用 validation metric（层间相似度）监控 overfitting（过度裁剪）。

## Differentiable Top-K Operator

- 问题：标准 argmax/top-K 不可导，无法通过 gradient descent 优化 pruning 决策
- 方案：使用可微分近似（如 Gumbel-Softmax 的变体或 soft top-K）
- 好处：pruning schedule 和 token selection 可以与模型训练联合优化
- 这是 HiDivDrop 相比 training-free 方法的核心优势来源

> 💡 Differentiable Top-K 使得 HiDivDrop 可以做 end-to-end optimization，这是 DART 等 training-free 方法做不到的。代价是需要训练，但 HiDivDrop 同时还能加速训练 (1.72×)，因为 pruned tokens 在 forward/backward 中都不参与计算。

## Inter-layer Similarity Measure

用于指导 Concave Pyramid 的具体形状和 Early Exit 的触发：
- 衡量相邻层 token 表示的变化程度
- 变化大 → 该层正在做有意义的 processing → 保留更多 tokens
- 变化小 → 已经饱和 → 可以更激进地 prune 或 exit

> 💡 这个 measure 类似于 CKA (Centered Kernel Alignment) 用于分析 layer representations 的思路。它为 pruning schedule 提供了数据驱动的指导，而非手工设定。
