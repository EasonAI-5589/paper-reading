[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 阐述 MLLM 的 vision token 计算瓶颈，指出 importance-based token pruning 的四大问题（忽略交互、不兼容 FA、position bias、不如 random），提出 DART 基于 duplication 的替代方案。

---

Multimodal large language models (MLLMs) exhibit remarkable capabilities across a diverse range of multimodal tasks, including image captioning, visual question answering (VQA), video understanding (Wang et al., 2024b), and multimodal reasoning (Wang et al., 2024c; Kang et al., 2025). However, such impressive performance is always accompanied by huge computation costs, which are mainly caused by massive vision tokens in the input data, especially for high-resolution images (Li et al., 2024d) and multi-frame video (Tang et al., 2023), leading to challenges in their applications.

> 💡 **开篇**: 标准的问题引出——MLLM 能力强但计算贵，核心瓶颈在 vision tokens 数量。高分辨率图像和多帧视频让问题更严重。

---

To solve this problem, abundant recent methods introduce token pruning to remove the vision tokens in a training-free manner, which usually first defines the importance score of each token, and then prunes the most unimportant tokens during the inference phrase (Chen et al., 2024; Zhang et al., 2024c; Liu et al., 2024e). The key to a token pruning method is the definition of the importance of vision tokens, where most existing methods are based on the attention scores between vision-only tokens and vision-language tokens. However, this paper argues that these importance-based methods have several serious problems.

> 💡 **现有范式**: 现有方法都遵循 "定义 importance → 删不重要的 token" 这一范式。本文要挑战的就是这个根本假设。

---

(I) Ignoring interactions between tokens during pruning: Although the interaction between different tokens is considered in attention scores, however, importance-based methods directly remove the most unimportant tokens, ignoring the truth that the importance of each token should be adjusted when other tokens are pruned or preserved. For instance, for two similar tokens, if one of both is determined to be pruned, then the importance of the other token should be improved and vice versa. Unfortunately, previous importance-based token pruning methods fail to model such interaction.

> 💡 **问题 I — 静态 vs 动态**: 这是一个关于 combinatorial optimization 的经典问题。贪心地按静态 score 选 top-k，忽略了 token 之间的条件依赖。类似于信息论中的 greedy feature selection 问题：单独看每个 feature 的 MI 和联合选择 k 个 feature 的最优子集是不同的。

---

(II) Incompatibility to efficient attention: Efficient attention operators such as FlashAttention (Dao et al., 2022) have become the default configure in neural networks, which accelerates attention computation by around 2× and reduce the memory costs from O(N²) to O(N). However, these efficient attention operators make attention scores not accessible during computation, indicating conflicts with most previous importance-based token pruning methods. Disabling FlashAttention for accessing attention scores significantly improves the overall latency and memory footprint.

> 💡 **问题 II — FA 不兼容**: 这是一个非常实际的工程约束。FlashAttention 的 IO-aware 实现不在 HBM 中 materialize attention matrix，所以无法提取 attention scores。要用 attention-based pruning 就必须关 FA，导致实际加速大打折扣。SparseVLM 的 1.56× vs DART 的 1.99× 就是这个差距。

---

(III) Bias in token positions: As claimed by abundant recent works (Endo et al., 2024; Zhang et al., 2024b) and shown in Figure 1, attention scores have position bias, where the tokens are positionally close to the last token tend to have a higher attention score, making attention score does not truly reveal the value of this token.

> 💡 **问题 III — Position bias**: Attention sink 现象的另一面。序列末尾的 token 倾向于获得更高 attention，这在 LLM 中是已知现象。对 vision token 来说，这意味着图像右下角的 token 被不合理地 "偏爱"。

---

(IV) Significant accuracy drop: Although the aforementioned three problems have reminded us of the ineffectiveness of importance-based token pruning, however, it is still extremely surprising to find that some influential importance-based token pruning methods show inferior accuracy than random token pruning, (i.e., randomly selecting the tokens for pruning), as shown in Figure 2.

> 💡 **问题 IV — 不如 random**: 这是全文最 provocative 的发现。Figure 2 显示在 LLaVA-1.5-7B 上 88.9% reduction 时，FastV 和 SparseVLM 的 2/3 benchmark 结果不如 random pruning。这直接否定了 "importance-based pruning 至少比 random 好" 这个 assumption。

---

The above observations demonstrates the disadvantages of importance-based token pruning methods, while also introducing the expectation for the ideal alternative: The expected method should consider both the individual value of a token and its interaction to other tokens. It should be cheap in computation and friendly to hardware, and shows no bias in the positions of tokens.

> 💡 **理想方法的四个要求**: (1) 考虑 token 间交互，(2) 计算廉价，(3) 硬件友好（兼容 FA），(4) 无 position bias。DART 逐一满足这些要求。

---

These insights inspire us to incorporate token duplication into the token reduction. Intuitively, when multiple tokens exhibit identical or highly similar representations, it is natural to retain only one of them for the following computation, thereby maintaining efficiency without harming accuracy. Building upon this idea, we introduce a simple but effective token pruning pipeline referred to as DART (Duplication-Aware Reduction of Tokens) with the following two steps.

> 💡 **核心 insight**: 从 "哪个 token 重要" 转向 "哪些 token 重复"。这在信息论角度非常自然——我们要最大化保留的信息量，而不是保留 importance score 之和。两个高 importance 但高度相似的 token，保留一个就够了。

---

Firstly, we begin by selecting a small subset of tokens as pivot tokens, which comprise no more than 2% of the total tokens. Such pivot tokens can be selected based on the norm of tokens or even randomly selected, which does not introduce notable computations. Secondly, we then calculate the cosine similarity between pivot tokens and the remaining image tokens. Since the pivot tokens are fewer than 2%, such computation is efficient in both computing and memory. With a desired token reduction ratio, we retain only those vision tokens with the lowest cosine similarity to pivot tokens and remove the similar ones. The entire process is simple and highly efficient, completing in no more than 0.08 seconds, friendly to efficient attention operators, and leading to significantly higher accuracy than previous methods.

> 💡 **DART 流程**: (1) 选 ≤2% pivot tokens（K-norm / random 均可），(2) 算 cosine similarity，保留与 pivot 最不相似的 token。关键优势：不需要 attention scores → 兼容 FA；只需一次矩阵乘法 → ≤0.08s overhead；pivot 选择方式不敏感 → robust。

---

In summary, our contributions are three-fold:

• Rethink Token Importance. Through empirical analysis, we demonstrate the suboptimality of relying on attention scores to measure token importance to guide the token reduction paradigm.

• Token Duplication as a Key Factor. Building on token duplication, we introduce a training-free, plug-and-play token reduction method that seamlessly integrates with Flash Attention.

• Superior Performance with Extreme Compression. Extensive experiments across four diverse MLLMs and over 10 benchmarks demonstrate the clear superiority of DART. For instance, our method outperforms the second-best method by 2.2% (93.7% vs. 91.5%) on LLaVA-1.5-7B with an 88.9% reduction ratio.

> 💡 **三大贡献**: Rethink（importance 不 work）→ Propose（duplication-based DART）→ Validate（4 MLLMs, 10+ benchmarks, 2.2% 领先）。93.7% vs 91.5% 在极端 88.9% 压缩下是显著的差距。

---
