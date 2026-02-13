[← 返回 README](../README.md)

# 5 Conclusion

## 📌 预览
总结 + Limitations。

---

In this work, we propose Agentic Memory (AgeMem), a unified memory management framework that enables LLM-based agents to jointly control long-term and short-term memory through learnable, tool-based actions. By integrating memory operations directly into the agent's policy and training them with a progressive reinforcement learning strategy, AgeMem replaces heuristic memory pipelines with an end-to-end optimized solution. Extensive experiments across diverse long-horizon benchmarks show that AgeMem improves both task performance and memory quality while maintaining efficient context usage. These results highlight the importance of unified, agent-centric memory policies and suggest a promising direction for building scalable and adaptive LLM agents capable of long-term reasoning.

> 💡 **结论要点**：
> - 核心贡献：统一 LTM+STM → tool-based → 渐进 RL → 端到端优化
> - 实验验证：多 benchmark + 多 backbone + task performance + memory quality + context efficiency
> - 前瞻：unified agent-centric memory policy 是构建可扩展、自适应 agent 的有前途方向

---

# Limitations

While AgeMem demonstrates strong performance across multiple settings, there remain opportunities for further extension. The current implementation adopts a fixed set of memory management tools, which provides a clear and effective abstraction but could be extended to support more fine-grained control in future work. In addition, although we evaluate our approach on several representative long-horizon benchmarks, broader coverage of tasks and environments may further strengthen the empirical understanding of the framework.

> 💡 **Limitations 批读**:
> - **固定 tool 集合**：目前 6 个 tool 是预定义的，不能动态扩展 → 对比 MemSkill 的 Designer 可以自动设计新 skill
> - **Benchmark 覆盖**：5 个 benchmark 虽然多样但都偏文本，没有多模态场景
> - **未提到的潜在问题**：
>   - 训练只在 HotpotQA 上做 → 如果换到其他训练集效果如何？
>   - 8× RTX 4090 的训练成本 → 是否可以更高效？
>   - 只用了 7B 和 4B 的小模型 → 大模型是否还需要这种训练？

---

## 🔖 Section 总结

### 核心洞察
1. AgeMem 的核心哲学：memory 管理不该是外挂模块，应该是 agent 自身的决策能力
2. 主要局限在于 tool 集合固定和 benchmark 覆盖 → 这两个方向都值得进一步探索
