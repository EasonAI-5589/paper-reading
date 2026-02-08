[← 返回 README](../README.md)

# 5. Conclusion

In this paper, we introduce Mem-T, a comprehensive hierarchical memory framework, and MoT-GRPO, a novel RL paradigm for memory agents. By decomposing sparse terminal rewards into dense, step-wise supervision via memory operation trees, MoT-GRPO enables the joint optimization of memory construction and retrieval policies. The extensive experiments demonstrate that Mem-T not only achieves state-of-the-art performance across in-domain and out-of-domain benchmarks but also realizes a superior Pareto efficiency between task accuracy and inference overhead. We believe Mem-T represents a shift from heuristic-based storage to fully learnable, attribution-centric memory systems, paving the way for the development of self-evolving agents capable of lifelong learning.

> 💡 **批注**: 结论简洁有力。"from heuristic-based storage to fully learnable, attribution-centric memory systems" 是一个很好的定位。不过论文没有讨论局限性，比如：(1) training 需要 evidence annotation（Evidence Alignment Gate 依赖 ground-truth evidence），这在实际部署中可能不可用；(2) 4B 模型的记忆容量上限——当信息流持续增长时，memory database 会无限膨胀吗？(3) 多模态记忆的扩展性。这些都是未来工作的方向。
