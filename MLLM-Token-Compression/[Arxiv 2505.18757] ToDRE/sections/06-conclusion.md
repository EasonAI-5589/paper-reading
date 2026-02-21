[← 返回 README](../README.md)

# 6 Conclusion

We systematically analyze redundancy in LVLM inference and identify two key inefficiencies:
1. Redundant visual tokens that inflate intra-modal computation
2. Tokens that contribute little cross-modal information during decoding

To address these inefficiencies, we propose ToDRE, a training-free, architecture-agnostic framework that first selects a maximally diverse subset of visual tokens via a greedy max-sum diversification algorithm, then removes all remaining visual tokens once cross-modal attention fades. Experiments on twelve image- and video-language benchmarks show that ToDRE prunes up to 90% of visual tokens while preserving 95.0% of the original performance, achieving 2.6× faster inference and 14.5% lower memory usage than uncompressed baselines.

> 💡 **批注**：ToDRE 的设计哲学简洁有力：分而治之。Diversity 和 Relevance 分开处理，各司其职。局限性方面，作者未讨论：(1) max-sum diversification 是否能保证最优解（实际上是 NP-hard 问题的贪心近似）；(2) 在需要精细空间推理的任务中，diversity-based 选择是否会遗漏关键的局部细节；(3) Stage 2 的阈值 τ 和检查层 7L/8 的选择是否对不同模型需要调整。
