# Abstract

> 来源: SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference (ICML 2025)

---

## 📄 原文

In vision-language models (VLMs), visual tokens usually bear a significant amount of computational overhead despite sparsity of information in them when compared to text tokens. To address this, most existing methods learn a network to prune redundant visual tokens using certain training data. Differently, we propose a text-guided training-free token optimization mechanism dubbed SparseVLM that eliminates the need of extra parameters or fine-tuning costs. Given that visual tokens complement text tokens in VLM's linguistic reasoning, we select relevant text tokens to rate the significance of visual tokens using self-attention matrices and, then, prune visual tokens using the proposed strategy to maximize sparsity while retaining information. In particular, we introduce a rank-based strategy to adaptively determine the sparsification ratio for each layer, alongside a token recycling method that compresses pruned tokens into more compact representations. Experimental results show that SparseVLM increases the efficiency of various VLMs in a number of image and video understanding tasks.

> 💡 **一句话总结**: 提出 SparseVLM，一个 **training-free** 的 text-guided visual token 稀疏化方法，通过 self-attention 矩阵让文本引导视觉 token 的剪枝，不需要额外参数或微调。

> 💡 **核心卖点**:
> 1. **Training-free** — 不像 FastV 等需要训练，直接复用 self-attention 矩阵
> 2. **Text-guided** — 根据问题 prompt 自适应决定哪些视觉 token 重要（不同问题关注不同区域）
> 3. **Rank-based 自适应比例** — 用 attention 矩阵的秩来决定每层剪多少
> 4. **Token recycling** — 被剪掉的 token 不直接丢弃，而是聚类压缩成紧凑表示
> 5. **效果**: LLaVA 上比 FastV 好 11-17%，视频任务好 14.7%

---

## 💡 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 压缩率 | 4.5× (576→128 tokens) |
| 性能保留 | 97% |
| 延迟降低 | 37% |
| vs FastV (LLaVA) | +11.2-17.3% |
| vs FastV (MiniGemini) | +9.2-20.4% |
| vs FastV (VideoLLaVA) | +14.7% |
