[← 返回 README](../README.md)

# Abstract

## 📌 预览

摘要交代了三件事：(1) 问题——MLLM 因大量视觉 token 有推理瓶颈；(2) 现有方法的不足——各自聚焦重要性或多样性，缺乏原则性整合框架；(3) IDPruner 的解法——用 MMR 算法做 Pareto 最优平衡，无需注意力图，兼容 FlashAttention。

---

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities, yet they encounter significant computational bottlenecks due to the massive volume of visual tokens. Consequently, visual token pruning, which substantially reduces the token count, has emerged as a critical technique for accelerating MLLM inference. Existing approaches focus on token importance, diversity, or an intuitive combination of both, without a principled framework for their optimal integration. To address this issue, we first conduct a systematic analysis to characterize the trade-off between token importance and semantic diversity. Guided by this analysis, we propose the Importance and Diversity Pruner (IDPruner), which leverages the Maximal Marginal Relevance (MMR) algorithm to achieve a Pareto-optimal balance between these two objectives. Crucially, our method operates without requiring attention maps, ensuring full compatibility with FlashAttention and efficient deployment via one-shot pruning. We conduct extensive experiments across various model architectures and multimodal benchmarks, demonstrating that IDPruner achieves state-of-the-art performance and superior generalization across diverse architectures and tasks. Notably, on Qwen2.5-VL-7B-Instruct, IDPruner retains 95.18% of baseline performance when pruning 75% of the tokens, and still maintains 86.40% even under an extreme 90% pruning ratio. Our code is available at https://github.com/Tencent/AngelSlim.

> 💡 **核心主张**: 论文的核心洞察是：现有方法做的是重要性或多样性的单目标优化，或者直觉式的两者叠加。IDPruner 将其表述为信息检索中的 MMR 问题，在数学上有 Pareto 最优保证。这比「直觉组合」高了一个层次。
>
> 💡 **关键数字**: Qwen2.5-VL-7B 上，保留 25% token → 95.18% baseline 性能；保留 10% token → 86.40% baseline 性能。这是非常强的结果。
>
> 💡 **工程亮点**: 无需注意力图 + 兼容 FlashAttention + one-shot 剪枝 → 可以直接集成 vLLM 等推理引擎，这是实际部署的关键。
>
> 💡 **与 STAR-Pro 的关系**: IDPruner 是 STAR-Pro 的直接竞品。值得注意的是，IDPruner **不是 training-free** 的——它的重要性估计依赖 VisionSelector（需要端到端训练的可学习模块）。这是 STAR-Pro 相比 IDPruner 的潜在优势点（如果 STAR-Pro 是 training-free 的话）。

## 🔖 Abstract 总结

- **问题**: MLLM 视觉 token 太多，推理效率低
- **现状不足**: 现有方法缺乏原则性框架整合重要性和多样性
- **方法**: MMR 算法 + 无注意力图 + one-shot 剪枝
- **结果**: Qwen2.5-VL-7B, 75% 剪枝 → 95.18%，90% 剪枝 → 86.40%
