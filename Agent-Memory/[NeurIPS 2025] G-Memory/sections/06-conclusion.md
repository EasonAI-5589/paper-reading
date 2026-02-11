[← 返回 README](../README.md)

# 6 Conclusion & Limitation

## 📌 预览
总结 G-Memory 的贡献和局限性。

---

In this paper, we conduct a thorough examination of existing memory architectures designed for multi-agent systems (MAS) and identify that their overly simplified designs fundamentally hinder the systems' capacity for self-evolution. To bridge this gap, we propose G-Memory, a hierarchical memory framework that organizes the complex and extended interaction trajectories of MAS into a three-tier graph hierarchy: the insight, query, and interaction graphs. G-Memory provides each agent with customized and hierarchical memory cues, ranging from abstract, generalizable insights to fine-grained, task-critical collaborative segments, and dynamically evolves its knowledge base across episodes. Extensive experiments demonstrate that G-Memory can be seamlessly integrated into state-of-the-art MAS frameworks, significantly enhancing their self-evolution capability, e.g., up to 20.89% ↑ improvement on embodied action tasks.

**Limitations:** Although G-Memory has been evaluated across three domains and five benchmarks, further validation on more diverse tasks (e.g., medical QA) would strengthen its soundness, which we leave for future work.

> 💡 **局限性分析**:
> - 作者自己提到了 **medical QA** 作为未来方向——这与我们的研究方向高度相关！
> - 其他潜在局限：
>   1. **LLM 依赖**: Graph sparsifier 和 insight 生成都依赖 LLM，如果 LLM 质量不好，记忆质量也会差
>   2. **可扩展性**: insight graph 随任务增长会不会爆炸？论文没讨论 insight 的合并/遗忘机制（Appendix C 有 merge 但没深入分析）
>   3. **冷启动**: 前几个任务没有记忆可用，需要一定的 warm-up 期
>   4. **评估局限**: 所有 benchmark 都是结构化任务，open-ended 场景（如对话）的效果未知
>
> 💡 **对我们的启发**:
> - G-Memory 在 medical QA 上还未验证 → 这是一个可以做的方向
> - 我们的多图医学记忆可以借鉴三层图结构，但需要处理医学领域的特殊性（如时间序列、多模态检查结果）
> - Insight 的合并/遗忘机制值得深入研究——在医学场景中，旧的诊断经验可能被新指南推翻

---

## 🔖 Section 总结

### 核心洞察
1. G-Memory 的核心价值：让 MAS 从「无状态工具」变成「经验型团队」
2. Medical QA 是作者明确提到的未来方向，与我们的研究高度契合
3. 可扩展性和冷启动是实际部署时需要解决的问题
