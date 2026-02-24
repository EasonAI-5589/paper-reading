[← 返回 README](../README.md)

# 6. Conclusion

## 📌 预览

Conclusion 简洁地总结了 IDPruner 的动机、方法和贡献：hybrid 策略正在成为新标准，但缺乏系统分析框架；IDPruner 用 MMR 实现最优平衡；SOTA + 跨架构鲁棒性。

---

Recent progress in visual token pruning shows that hybrid strategies are surpassing methods that rely only on importance or diversity, becoming the new standard in this field. However, there is a lack of systematic analysis on how to effectively harmonize these two objectives. In this study, we provide a framework to analyze this trade-off and demonstrate that the Maximal Marginal Relevance (MMR) mechanism is an effective strategy to achieve an optimal balance. Based on this insight, we propose IDPruner, a method that explicitly balances token importance and semantic redundancy. Extensive evaluations show that our method achieves state-of-the-art performance and remains robust across different model architectures. We believe this work offers a solid foundation for systematically balancing importance and diversity, enabling more efficient MLLMs.

> 💡 **Conclusion 的定位**: Conclusion 非常简短（一段），这是会议论文的典型风格。核心信息：(1) 背景——hybrid 是新趋势但缺乏分析框架；(2) 贡献——MMR 分析框架 + IDPruner 算法；(3) 结果——SOTA + 跨架构。没有过度夸大，表述准确。
>
> 💡 **"Solid Foundation" 的含义**: 作者认为这项工作不仅提供了一个好算法，还提供了一个分析框架（Pareto 前沿分析、Hopkins Statistic、重要性保留率），后续研究可以用这个框架评估新的 hybrid 策略。这是论文的学术价值所在。
>
> 💡 **对 STAR-Pro 的启示**: IDPruner 的 Conclusion 承认了「how to harmonize importance and diversity」是一个值得深入研究的问题，并宣称自己提供了 solid foundation。STAR-Pro 如果有不同的（更优或互补的）方案，可以在 Related Work 中引用 IDPruner 的这个框架，讨论 STAR-Pro 与 IDPruner 的异同，并在实验中直接比较。

## 🔖 Section 总结

Conclusion 精炼地总结了三点：(1) Hybrid 方法是趋势但缺分析框架；(2) MMR 是最优平衡机制；(3) IDPruner 实现了 SOTA + 鲁棒泛化。

## 📊 整体论文评价

**优点**:
- 分析框架清晰（Hopkins Statistic + Pareto Frontier）
- MMR 迁移优雅，有信息检索理论支撑
- 工程实用性强（FlashAttention 兼容 + One-shot + O(KN)）
- 跨架构实验充分

**缺点/局限**:
- 依赖 VisionSelector（需要训练），不是 training-free
- 未在长视频 benchmark 上评测
- λ 超参数未做精细搜索（但默认 0.5 表现稳定）
- 论文写作时间紧（作者在 Appendix F 承认用了 AI 辅助写作）
