[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
按 attention-based / similarity-based / joint 三类梳理 training-free visual token pruning 方法。

---

### Attention-based Pruning
Attention-based pruning estimates token importance from attention statistics, either inside the LLM decoder or within the vision encoder.

**LLM-side methods:**
- **FastV** (ECCV 2024): prune visual tokens by cross attention scores in shallow layers
- **LLaVA-PruMerge**: attention-based pruning + token merging to compress redundant tokens
- **SparseVLM** (ICML 2025): text-guided attention scoring + token recycling for progressive sparsification
- **PyramidDrop** (PDrop): layer-wise progressive dropping aligned with model depth
- **TopV**: visual token pruning during prefilling, compatible with FlashAttention
- **FitPrune**: budget-aware pruning by minimizing attention-distribution divergence

**Vision-encoder-side methods:**
- **FasterVLM** / **HiRED**: rank tokens using [CLS]-based attention for early or region-aware pruning

> 💡 **批注**: Attention-based 方法的系统性问题：importance estimates biased toward salient regions，可能遗漏 subtle but critical 的全局上下文。这恰好是 FSR 要解决的问题。

### Similarity-based Pruning
- **DivPrune** (CVPR 2025): max-min diversity selection — 代表性和多样性子集
- **DART** (EMNLP 2025): 基于 duplication 剪枝，保留与 pivots 不相似的 token

> 💡 **批注**: Similarity-based 方法偏向 global coverage，但忽略 fine-grained local details。与 attention-based 形成互补关系——这正是 joint 方法和 FSR 试图结合的。

### Joint Attention-Similarity-based Pruning
- **VisionZip** (CVPR 2025): attention importance + redundancy reduction
- **VisPruner** (ICCV 2025): 同上
- **CDPruner** (NeurIPS 2025): instruction relevance + DPP-style conditional diversity
- **HoloV** (NeurIPS 2025): partition-wise allocation + connectivity-aware selection

> 💡 **批注**: CDPruner 是 FSR 最直接的竞品——同样使用 instruction relevance（CLIP text encoder）+ diversity。区别在于：
> - CDPruner: DPP formulation（全局优化，但计算开销较大）
> - FSR: 显式分阶段（Focus → Scan → Refine），贪心但有 2-approximation 理论保证
> - FSR 的 Refine 阶段额外做了 token aggregation，CDPruner 没有

---

## 🔖 Section 总结

### Citation Landscape
| 类别 | 方法 | 信号 | 局限 |
|------|------|------|------|
| Attention | FastV, PruMerge, SparseVLM, PDrop, TopV, FitPrune, FasterVLM, HiRED | Attention scores | 偏 local salient |
| Similarity | DivPrune, DART | Token diversity | 偏 global, 忽略 local |
| Joint | VisionZip, VisPruner, CDPruner, HoloV | Both | 极端压缩下仍难平衡 |
| **FSR** | **本文** | **Dual-pathway + conditional sampling + aggregation** | **动态 local/global 分配** |
