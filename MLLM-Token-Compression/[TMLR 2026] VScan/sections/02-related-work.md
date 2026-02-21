[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
两部分：(1) 高效 LVLM 的架构级优化（Q-Former, Perceiver, FlashAttention），(2) Token Reduction 两类方法。

---

## Efficient Large Vision-Language Models

Building on powerful auto-regressive LLMs [57, 15], recent LVLMs typically adopt an encoder-projector-decoder architecture, where visual inputs are encoded into tokens and jointly processed with language sequences [38, 36, 4, 13, 56, 11]. However, as image resolution increases or the input scales to multi-image/video, the number of visual tokens grows proportionally, leading to a quadratic increase in computation cost and runtime due to the self-attention mechanism [58, 7, 14, 67], which limits the scalability of LVLMs in real-world applications [6, 12, 49, 31, 69, 68, 9]. To mitigate this issue, several LVLMs introduced specialized modules to enhance efficiency—such as the Q-Former in InstructBLIP [16] and the perceiver resampler [26] in OpenFlamingo [1]—that distill dense visual inputs into a compact set of features before LLM decoding. Orthogonal to these architectural strategies, FlashAttention [18, 17] has emerged as a widely adopted, hardware-aware optimization that accelerates attention computation by minimizing redundant memory access, offering substantial speedups without compromising performance.

> 💡 **架构级效率优化**：
> - **Q-Former** (InstructBLIP): learnable queries 压缩 visual features，但需要训练
> - **Perceiver Resampler** (OpenFlamingo): 类似思路
> - **FlashAttention**: 硬件级优化，与 token reduction 正交且兼容
> - VScan 属于 token reduction，与这些方法互补

---

## Vision Token Reduction in LVLMs

Another line of work aims to improve model efficiency on the sequence dimension—pioneering works such as ToMe [6] and FastV [12] have explored strategies like visual token merging and text-guided pruning to improve the efficiency of LVLMs. Building on these advances, subsequent approaches can be broadly divided into two main categories: (1) Text-agnostic pruning approaches [52, 3, 70, 64, 59, 60], which identify and remove redundant or uninformative visual tokens during the visual encoding stage. For instance, VisionZip [64] selects dominant tokens based on [CLS] attention scores, while FOLDER [59] introduces token merging with reduction overflow in the final blocks of the visual encoder. (2) Text-aware pruning approaches [71, 62, 41, 65, 55], which aim to remove visual tokens that are irrelevant to the text query during the LLM decoding stage. For instance, SparseVLM [71] proposes an iterative sparsification strategy that selects visual-relevant text tokens to rate the significance of vision tokens, and PyramidDrop [62] performs progressive pruning at multiple decoding layers to balance efficiency and context preservation. In this work, we present a comprehensive analysis of how LVLMs process visual tokens during both the visual encoding and language decoding stages, and propose a corresponding two-stage approach, VScan, to effectively improve the inference efficiency of LVLMs while maintaining robust performance.

> 💡 **Token Reduction 方法分类**：
> 
> | 类别 | 代表方法 | 压缩位置 | 核心机制 |
> |------|----------|----------|----------|
> | Text-agnostic | VisionZip, FOLDER, ToMe | Visual Encoder | [CLS] attention / self-attention |
> | Text-aware | FastV, SparseVLM, PyramidDrop | LLM Decoder | Text-guided attention pruning |
> | **两阶段** | **VScan** | **Encoder + LLM** | **Global+Local scan + Middle pruning** |
> 
> VScan 是第一个系统性地在两个阶段都做 reduction 的 training-free 方法。

---

## 🔖 Section 总结

### 核心洞察
1. 现有方法要么只在 encoder 做（text-agnostic），要么只在 LLM 做（text-aware），都是单阶段
2. VScan 的定位：两阶段互补，且基于实证分析选择了更优的操作位置
