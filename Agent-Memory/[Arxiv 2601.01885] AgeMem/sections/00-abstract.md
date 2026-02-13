[← 返回 README](../README.md)

# Abstract

## 📌 预览
AgeMem 的核心 pitch：统一 LTM+STM 管理 → tool-based actions → 三阶段渐进 RL + step-wise GRPO → 5 个 benchmark SOTA。

---

Large language model (LLM) agents face fundamental limitations in long-horizon reasoning due to finite context windows, making effective memory management critical. Existing methods typically handle long-term memory (LTM) and short-term memory (STM) as separate components, relying on heuristics or auxiliary controllers, which limits adaptability and end-to-end optimization. In this paper, we propose Agentic Memory (AgeMem), a unified framework that integrates LTM and STM management directly into the agent's policy. AgeMem exposes memory operations as tool-based actions, enabling the LLM agent to autonomously decide what and when to store, retrieve, update, summarize, or discard information. To train such unified behaviors, we propose a three-stage progressive reinforcement learning strategy and design a step-wise GRPO to address sparse and discontinuous rewards induced by memory operations. Experiments on five long-horizon benchmarks demonstrate that AgeMem consistently outperforms strong memory-augmented baselines across multiple LLM backbones, achieving improved task performance, higher-quality long-term memory, and more efficient context usage.

> 💡 **Abstract 批读**:
> - **问题**：现有方法 LTM 和 STM 分开管理，依赖 heuristic/辅助控制器
> - **方案**：AgeMem 把 memory 操作暴露为 tool action，agent 自主决策
> - **训练**：三阶段渐进 RL + step-wise GRPO 解决稀疏奖励
> - **结果**：5 个 long-horizon benchmark，多个 LLM backbone，全面超越 baseline
> - **亮点**：同时提升了 task performance、memory quality、context efficiency 三个维度

---

## 🔖 Section 总结

### 核心洞察
1. LTM 和 STM 的分离管理是现有方法的根本瓶颈 → 需要统一框架
2. Memory 操作天然适合建模为 tool action → 可以用 RL 端到端优化
3. Memory 操作导致的稀疏/不连续奖励是训练的核心技术挑战 → step-wise GRPO
