[← 返回 README](../README.md)

# Abstract

## 📌 预览
SparseVLM 提出了一种 text-guided、training-free 的视觉 token 稀疏化方法，通过自注意力矩阵评估视觉 token 重要性，自适应裁剪冗余 token 并回收压缩信息。

---

In vision-language models (VLMs), visual tokens usually bear a significant amount of computational overhead despite sparsity of information in them when compared to text tokens. To address this, most existing methods learn a network to prune redundant visual tokens using certain training data. Differently, we propose a text-guided training-free token optimization mechanism dubbed SparseVLM that eliminates the need of extra parameters or fine-tuning costs. Given that visual tokens complement text tokens in VLM's linguistic reasoning, we select relevant text tokens to rate the significance of visual tokens using self-attention matrices and, then, prune visual tokens using the proposed strategy to maximize sparsity while retaining information. In particular, we introduce a rank-based strategy to adaptively determine the sparsification ratio for each layer, alongside a token recycling method that compresses pruned tokens into more compact representations. Experimental results show that SparseVLM increases the efficiency of various VLMs in a number of image and video understanding tasks. Our code is available at https://github.com/Gumpest/SparseVLMs.

> 💡 **Abstract 批读**:
> - **问题**: VLM 中视觉 token 数量多但信息稀疏，带来巨大计算开销
> - **现有方案局限**: 需要训练额外网络来剪枝，有额外参数和微调成本
> - **SparseVLM 方案**: Text-guided + Training-free，三大核心组件：
>   1. **文本引导的重要性评估** — 用自注意力矩阵中文本-视觉交互来评分
>   2. **Rank-based 自适应稀疏化** — 每层自动决定裁剪比例
>   3. **Token Recycling** — 被裁剪的 token 不直接丢弃，而是压缩重构
> - **关键词**: training-free, text-guided, adaptive sparsification, token recycling

---

## 🔖 Section 总结

### 核心洞察
1. 视觉 token 信息密度远低于文本 token，存在大量冗余
2. 文本 token 应当指导视觉 token 的裁剪（text-aware），而非独立裁剪
3. Training-free 方案避免了额外训练开销，可即插即用
