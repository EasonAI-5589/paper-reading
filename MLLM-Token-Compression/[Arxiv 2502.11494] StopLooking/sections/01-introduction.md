[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 详细阐述了 importance-based token pruning 的四大问题（忽略 token 交互、与 FlashAttention 不兼容、位置偏差、精度下降），并提出 DART 的核心思路：通过选取少量 pivot token 并计算余弦相似度来移除高重复度 token。

---

Multimodal large language models (MLLMs) exhibit remarkable capabilities across a diverse range of multimodal tasks, including image captioning, visual question answering (VQA), video understanding (Wang et al., 2024b), and multimodal reasoning (Wang et al., 2024c; Kang et al., 2025). However, such impressive performance is always accompanied by huge computation costs, which are mainly caused by massive vision tokens in the input data, especially for high-resolution images (Li et al., 2024d) and multi-frame video (Tang et al., 2023), leading to challenges in their applications.

> 💡 **批注**: 标准开篇，点明 MLLM 的能力与计算开销的矛盾。特别强调高分辨率图像和多帧视频场景下 vision token 数量爆炸的问题。

---

![Figure 1](../images/9f41f65d855f18b38aab8af29749a424cbc1bbcc5ad07747833d495a3e6f6550.jpg)
*Figure 1: Comparison between DART and FastV. Red text indicates hallucination from vanilla LLaVA-1.5-7B, green text represents hallucination from DART, and blue text represents hallucination from FastV.*

> 💡 **Figure 1 批注**: 这张对比图非常直观：FastV（基于 attention score 的方法）保留的 token 集中在图像右下角（位置偏差），导致比原始模型产生更多幻觉；而 DART 保留的 token 分布更均匀，反而减少了幻觉。这是全文最有说服力的可视化之一。

---

To solve this problem, abundant recent methods introduce token pruning to remove the vision tokens in a training-free manner, which usually first defines the importance score of each token, and then prunes the most unimportant tokens during the inference phrase (Chen et al., 2024; Zhang et al., 2024c; Liu et al., 2024e). The key to a token pruning method is the definition of the importance of vision tokens, where most existing methods are based on the attention scores between vision-only tokens and vision-language tokens. However, this paper argues that these importance-based methods have several serious problems.

> 💡 **批注**: 引出核心论点——importance-based paradigm 有严重问题。这里提到的 Chen et al., 2024 就是 FastV，Zhang et al., 2024c 是 SparseVLM。

---

(I) Ignoring interactions between tokens during pruning: Although the interaction between different tokens is considered in attention scores, however, importance-based methods directly remove the most unimportant tokens, ignoring the truth that the importance of each token should be adjusted when other tokens are pruned or preserved. For instance, for two similar tokens, if one of both is determined to be pruned, then the importance of the other token should be improved and vice versa. Unfortunately, previous importance-based token pruning methods fail to model such interaction.

> 💡 **批注（问题 I）**: 这是一个非常好的洞察——importance score 是**静态**的，但 pruning 过程是**动态**的。当一个 token 被删除后，与它相似的 token 的重要性应该上升，因为它现在承担了额外的信息负担。这个论点直接指向了 DART 的设计动机：关注 token 之间的**关系**（duplication）而非单个 token 的**属性**（importance）。

---

![Figure 2](../images/90095a07321cd99f0145d1ac6943f6acf1e2685c151aa6255630d2329c554766.jpg)
*Figure 2: Performance of FastV and SparseVLM compared with random token pruning on the LLaVA1.5-7B, with a $8 8 . 9 \%$ token reduction ratio.*

> 💡 **Figure 2 批注**: 这张图是全文的"炸弹"——FastV 和 SparseVLM 在多个 benchmark 上**不如随机剪枝**！这直接挑战了整个 importance-based pruning 范式。注意这是在 88.9% 的激进剪枝比例下，说明在高压缩率下 importance-based 方法的偏差被放大了。

---

$\mathbf { \Pi } ^ { ( \mathbf { I I } ) }$ Incompatibility to efficient attention: Efficient attention operators such as FlashAttention (Dao et al., 2022) have become the default configure in neural networks, which accelerates attention computation by around $2 \times$ and reduce the memory costs from $O ( N ^ { 2 } )$ to $O ( N )$ . However, these efficient attention operators make attention scores not accessible during computation, indicating conflicts with most previous importancebased token pruning methods. Disabling FlashAttention for accessing attention scores significantly improves the overall latency and memory footprint. (III) Bias in token positions: As claimed by abundant recent works (Endo et al., 2024; Zhang et al., 2024b) and shown in Figure 1, attention scores have position bias, where the tokens are positionally close to the last token tend to have a higher attention score, making attention score does not truly reveal the value of this token.

> 💡 **批注（问题 II & III）**: 
> - **FlashAttention 不兼容**：这是一个非常实际的工程问题。FlashAttention 不暴露 attention map，而 FastV/SparseVLM 恰恰需要 attention score。关闭 FlashAttention 来获取 attention score → 加速方法反而导致减速，讽刺。
> - **位置偏差**：靠近序列末尾的 token attention score 偏高（attention sink 的变体），这意味着 importance score 并不真正反映 token 的信息价值。

---

(IV) Significant accuracy drop: Although the aforementioned three problems have reminded us of the ineffectiveness of importance-based token pruning, however, it is still extremely surprising to find that some influential importance-based token pruning methods show inferior accuracy than random token pruning, (i.e., randomly selecting the tokens for pruning), as shown in Figure 2.

> 💡 **批注（问题 IV）**: 最致命的一击：importance-based < random。这说明 attention score 作为 importance 指标不仅不准，而且有害——它引入的偏差比随机还糟糕。

---

The above observations demonstrates the disadvantages of importance-based token pruning methods, while also introducing the expectation for the ideal alternative: The expected method should consider both the individual value of a token and its interaction to other tokens. It should be cheap in computation and friendly to hardware, and shows no bias in the positions of tokens.

> 💡 **批注**: 理想方法的四个要求：(1) 考虑 token 间交互，(2) 计算廉价，(3) 硬件友好（兼容 FlashAttention），(4) 无位置偏差。DART 恰好满足所有四条。

---

These insights inspire us to incorporate token duplication into the token reduction. Intuitively, when multiple tokens exhibit identical or highly similar representations, it is natural to retain only one of them for the following computation, thereby maintaining efficiency without harming accuracy. Building upon this idea, we introduce a simple but effective token pruning pipeline referred to as DART (Duplication-Aware Reduction of Tokens) with the following two steps.

> 💡 **批注**: 从 importance 转向 duplication 的动机非常自然：重复的 token 携带冗余信息，删除它们的信息损失最小。这比"删不重要的"更有理论保障。

---

Firstly, we begin by selecting a small subset of tokens as pivot tokens, which comprise no more than $2 \%$ of the total tokens. Such pivot tokens can be selected based on the norm of tokens or even randomly selected, which does not introduce notable computations. Secondly, we then calculate the cosine similarity between pivot tokens and the remaining image tokens. Since the pivot tokens are fewer than $2 \%$ , such computation is efficient in both computing and memory. With a desired token reduction ratio, we retain only those vision tokens with the lowest cosine similarity to pivot tokens and remove the similar ones. The entire process is simple and highly efficient, completing in no more than 0.08 seconds, friendly to efficient attention operators, and leading to significantly higher accuracy than previous methods.

> 💡 **批注**: DART 的核心算法极其简洁：
> 1. 选 ≤2% 的 pivot token（K-norm / random 均可）
> 2. 计算 pivot 与其余 token 的余弦相似度
> 3. 保留与 pivot 相似度最低的 token（即最不重复的）
> 
> 计算开销仅 0.08s，且完全不需要 attention score → 天然兼容 FlashAttention。

---

In summary, our contributions are three-fold:

• Rethink Token Importance. Through empirical analysis, we demonstrate the suboptimality of relying on attention scores to measure token importance to guide the token reduction paradigm.

• Token Duplication as a Key Factor. Building on token duplication, we introduce a trainingfree, plug-and-play token reduction method that seamlessly integrates with Flash Attention.

• Superior Performance with Extreme Compression. Extensive experiments across four diverse MLLMs and over 10 benchmarks demonstrate the clear superiority of DART. For instance, our method outperforms the second-best method by $2 . 2 \%$ $9 3 . 7 \%$ vs. $9 1 . 5 \%$ ) on LLaVA1.5-7B with an $8 8 . 9 \%$ reduction ratio.

> 💡 **批注**: 三大贡献清晰：(1) 挑战 importance paradigm，(2) 提出 duplication paradigm + FlashAttention 兼容，(3) 88.9% 压缩下仍超越 SOTA 2.2%。第三点的 93.7% vs 91.5% 是在 64 token 设定下（原始 576 token），这个压缩率非常激进。
