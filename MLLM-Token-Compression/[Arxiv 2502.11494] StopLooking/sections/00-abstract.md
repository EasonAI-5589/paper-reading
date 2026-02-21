[← 返回 README](../README.md)

# Abstract

## 📌 预览
DART 提出用 token duplication（而非 importance）来指导 vision token pruning。核心发现：importance-based 方法常不如 random pruning。DART 选少量 pivot tokens，保留与 pivot 低重复度的 token，88.9% 压缩下仍保持可比性能，1.99×/2.99× 加速，兼容 FlashAttention。

---

Vision tokens in multimodal large language models often dominate huge computational overhead due to their excessive length compared to linguistic modality. Abundant recent methods aim to solve this problem with token pruning, which first defines an importance criterion for tokens and then prunes the unimportant vision tokens during inference. However, in this paper, we show that the importance is not an ideal indicator to decide whether a token should be pruned. Surprisingly, it usually results in inferior performance than random token pruning and leading to incompatibility to efficient attention computation operators. Instead, we propose DART (Duplication-Aware Reduction of Tokens), which prunes tokens based on its duplication with other tokens, leading to significant and training-free acceleration. Concretely, DART selects a small subset of pivot tokens and then retains the tokens with low duplication to the pivots, ensuring minimal information loss during token pruning. Experiments demonstrate that DART can prune 88.9% vision tokens while maintaining comparable performance, leading to a 1.99× and 2.99× speed-up in total time and prefilling stage, respectively, with good compatibility to efficient attention operators.

> 💡 **摘要批注**: 论文的核心 claim 很清晰：(1) importance-based pruning 不如 random——这是一个很 provocative 的发现；(2) duplication 比 importance 更重要；(3) DART 方法简单高效，training-free，兼容 FlashAttention。88.9% 压缩 + 1.99× 实际加速是很强的数字。

---
