[← 返回 README](../README.md)

# Abstract

## 📌 预览
Abstract 概述了当前 MLLM 中视觉 token 长度过长带来的计算开销问题，指出基于 importance 的 token pruning 存在多个缺陷，提出 DART（Duplication-Aware Reduction of Tokens）方法，通过 token 重复度而非重要性来指导剪枝。

---

Vision tokens in multimodal large language models often dominate huge computational overhead due to their excessive length compared to linguistic modality. Abundant recent methods aim to solve this problem with token pruning, which first defines an importance criterion for tokens and then prunes the unimportant vision tokens during inference. However, in this paper, we show that the importance is not an ideal indicator to decide whether a token should be pruned. Surprisingly, it usually results in inferior performance than random token pruning and leading to incompatibility to efficient attention computation operators. Instead, we propose DART (Duplication-Aware Reduction of Tokens), which prunes tokens based on its duplication with other tokens, leading to significant and training-free acceleration. Concretely, DART selects a small subset of pivot tokens and then retains the tokens with low duplication to the pivots, ensuring minimal information loss during token pruning. Experiments demonstrate that DART can prune $8 8 . 9 \%$ vision tokens while maintaining comparable performance, leading to a $\mathbf { 1 . 9 9 } \times$ and $2 . 9 9 \times$ speed-up in total time and prefilling stage, respectively, with good compatibility to efficient attention operators 1.

> 💡 **批注**: DART 的核心思想一句话概括：**不要找"重要"的 token，而是删掉"重复"的 token**。这是一个范式转变——从 importance-based pruning 转向 duplication-based pruning。关键数字：88.9% 的 vision token 可以被裁掉，总推理加速 1.99×，prefill 加速 2.99×。更令人惊讶的是，传统 importance-based 方法居然比 random pruning 还差，这直接动摇了 FastV/SparseVLM 等方法的理论根基。
