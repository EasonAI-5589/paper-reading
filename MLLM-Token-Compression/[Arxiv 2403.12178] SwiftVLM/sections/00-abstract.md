[← 返回 README](../README.md)

# Abstract

## 📌 预览
SwiftVLM 提出了一种新的视觉 token 剪枝范式——**bypass**，通过保留未选中的 token 并转发到后续剪枝层重新评估，避免早期剪枝导致的不可逆信息丢失。

---

Visual token pruning is a promising approach for reducing the computational cost of vision–language models (VLMs), and existing methods often rely on early pruning decisions to improve efficiency. While effective on coarse-grained reasoning tasks, they suffer from significant performance degradation on tasks requiring fine-grained visual details. Through layer-wise analysis, we reveal substantial discrepancies in visual token importance across layers, showing that tokens deemed unimportant at shallow layers can later become highly relevant for text-conditioned reasoning. To avoid irreversible critical information loss caused by premature pruning, we introduce a new pruning paradigm, termed bypass, which preserves unselected visual tokens and forwards them to subsequent pruning stages for re-evaluation. Building on this paradigm, we propose SwiftVLM, a simple and training-free method that performs pruning at model-specific layers with strong visual token selection capability, while enabling independent pruning decisions across layers. Experiments across multiple VLMs and benchmarks demonstrate that SwiftVLM consistently outperforms existing pruning strategies, achieving superior accuracy–efficiency trade-offs and more faithful visual token selection behavior.

> 💡 **Abstract 批读**:
> - **问题**: 现有视觉 token 剪枝方法依赖早期剪枝决策，在需要细粒度视觉细节的任务上性能下降严重
> - **核心发现**: 视觉 token 的重要性在不同层之间差异显著——浅层认为不重要的 token 在深层可能变得高度相关
> - **核心方案**: 提出 **bypass** 范式——不丢弃未选中 token，而是保留并转发到后续剪枝层重新评估
> - **方法特点**: training-free，在具有强 token 选择能力的特定层执行剪枝，各层独立决策
> - **关键词**: visual token pruning, bypass, cross-layer, training-free

---

## 🔖 Section 总结

### 核心洞察
1. 早期剪枝导致不可逆信息丢失是现有方法的根本问题
2. Bypass 范式的关键创新：保留 → 转发 → 重新评估，而非一次性丢弃
3. Training-free 是重要卖点，无需额外训练开销
