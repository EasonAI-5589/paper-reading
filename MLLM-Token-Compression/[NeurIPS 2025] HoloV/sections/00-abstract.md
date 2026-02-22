[← 返回 README](../README.md)

# Abstract

## 📌 预览

HoloV 提出了一种即插即用的视觉 token 剪枝框架，核心洞察是：基于 attention 的剪枝方法倾向于保留语义相似的 token，在高剪枝率下性能急剧下降。HoloV 通过在不同空间 crop 之间自适应分配剪枝预算来保留全局视觉上下文。

---

Despite their powerful capabilities, Multimodal Large Language Models (MLLMs) suffer from considerable computational overhead due to their reliance on massive visual tokens. Recent studies have explored token pruning to alleviate this problem, which typically uses text-vision cross-attention or [CLS] attention to assess and discard redundant visual tokens. In this work, we identify a critical limitation of such attention-first pruning approaches, i.e., they tend to preserve semantically similar tokens, resulting in pronounced performance drops under high pruning ratios. To this end, we propose HoloV, a simple yet effective, plug-and-play visual token pruning framework for efficient inference. Distinct from previous attention-first schemes, HoloV rethinks token retention from a holistic perspective. By adaptively distributing the pruning budget across different spatial crops, HoloV ensures that the retained tokens capture the global visual context rather than isolated salient features. This strategy minimizes representational collapse and maintains task-relevant information even under aggressive pruning. Experimental results demonstrate that our HoloV achieves superior performance across various tasks, MLLM architectures, and pruning ratios compared to SOTA methods. For instance, LLaVA1.5 equipped with HoloV preserves $95.8\%$ of the original performance after pruning $88.9\%$ of visual tokens, achieving superior efficiency-accuracy trade-offs.

> 💡 **摘要批注**: HoloV 的核心贡献可以归纳为三点：
> 1. **问题发现**：现有 attention-first 剪枝方法在高剪枝率下性能骤降，因为它们倾向保留语义相似的 token（信息冗余）
> 2. **方法设计**：提出 crop-wise 自适应分配策略，从全局视角（holistic perspective）而非局部 attention 来决定保留哪些 token
> 3. **实验亮点**：LLaVA-1.5 剪掉 88.9% token 仍保留 95.8% 性能，且方法是 plug-and-play、model-agnostic 的
>
> 关键词：representational collapse（表征坍塌）— 这是本文要解决的核心问题。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 最高剪枝率 | 88.9% |
| 性能保留率 | 95.8% |
| 方法特点 | plug-and-play, model-agnostic |
