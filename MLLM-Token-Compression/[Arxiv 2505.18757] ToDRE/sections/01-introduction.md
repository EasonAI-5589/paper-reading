[← 返回 README](../README.md)

# 1 Introduction

Leveraging the superior reasoning capability of large language models (LLMs), large vision-language models (LVLMs) have achieved impressive performance in various multimodal understanding tasks such as visual question answering and video understanding. LVLMs convert visual inputs into visual tokens and align the converted visual tokens with text tokens for various multimodal understanding tasks. However, the inference of LVLMs often incurs prohibitive computational and memory costs due to the massive number of visual tokens involved, significantly restricting LVLM applicability in various downstream tasks.

Two representative approaches have recently been explored for improving the LVLM inference efficiency. The first approach is **model-centric**: it speeds up the inference via knowledge distillation, parameter quantization, or transformer replacement. However, this approach requires model retraining which incurs significant computational resources. The second approach is **data-centric**: it works by token pruning or block skipping, and has attracted increasing attention due to its training-free and architecture-agnostic nature.

> 💡 **批注**：Model-centric vs Data-centric 分类清晰。ToDRE 属于 data-centric，training-free 是核心优势。

Most existing token pruning techniques compress visual tokens by estimating "redundancy" from a single metric, such as cross-modal attention, visual token similarity, or the divergence of LLM's outputs before and after token pruning. However:
- Attention scores exhibit clear **positional bias** that tends to discard informative tokens erroneously
- Similarity-based approach **merges** similar visual tokens whose performance is often clearly lower than direct token pruning
- Using output divergence requires a held-out **calibration set** and model-specific distribution matching

> 💡 **批注**：三种现有指标的局限性总结得很精炼。Positional bias 是 attention-based 方法的核心痛点，DART 也指出了这一点。

Moreover, we observe an **"information migration" phenomenon**: cross-modal attention (both visual-to-text and text-to-visual) is strong in early layers but fades in deeper layers, suggesting that visual information is progressively absorbed into text representations within the first half of the LLM decoder. Given that output tokens exhibit near-zero attention to visual tokens during decoding, most existing work passes all remaining visual tokens from the prefilling stage into decoding, thereby incurring unnecessary computations.

> 💡 **批注**：Information migration 是 Stage 2 的理论基础——既然深层 cross-modal attention 已经很弱，说明 visual info 已经被"吸收"到 text representations 中了，此时可以安全删除 visual tokens。这个观察与 VTW 的 decoding-stage 发现互补，ToDRE 进一步将其推广到 prefilling stage。

We design ToDRE, a simple yet effective token pruning technique that incorporates both visual token diversity and task-specific token relevance. ToDRE performs token pruning in the embedding space prior to LLM input and during the LLM prefilling stage:
1. **Stage 1**: Greedy max-sum diversification algorithm that iteratively identifies and preserves visual tokens with minimal cumulative similarity to the selected tokens → circumvents positional bias
2. **Stage 2**: Leverages "information migration" by adaptively selecting one layer in the latter half of the LLM decoder and drops all visual tokens → continuous inference-time efficiency gains as decoding length increases

> 💡 **批注**：Stage 2 的效率增益随 decoding length 增加而增大，这在长文本生成场景下尤为重要。但现有 benchmark 多为短答案，所以实验中 Stage 2 的效率提升较小（Table 5 中 Stage 2 only 仅 8.8% 时间减少）。

### Contributions

1. **Revisit redundancy indicators**: Prove that inter-token diversity and token-task relevance are orthogonal factors; treating them separately enables more effective token pruning
2. **Training-free and plug-and-play framework**: Two-stage, fully compatible with FlashAttention, no additional training
3. **Extensive empirical validation**: 4 LVLMs × 12 benchmarks
