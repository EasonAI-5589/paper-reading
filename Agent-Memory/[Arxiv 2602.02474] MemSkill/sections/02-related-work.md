[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
两条线：(1) LLM Agent Memory Systems —— 从 pipeline 到 RL 优化；(2) Self-Evolving LLM Agents —— 从经验蒸馏到 self-play 进化。

---

## 2.1. LLM Agent Memory Systems

Prior work on agent memory focuses on constructing external memories from interaction histories and leveraging them to support downstream reasoning and decision making. Typical pipelines periodically extract salient information into a memory store, retrieve relevant entries for a new query, and update the store via consolidation or pruning (Kang et al., 2025; Zhong et al., 2024; Xu et al., 2025; Packer et al., 2023; Chhikara et al., 2025; Fang et al., 2025). More recently, learning-based approaches such as Memory-R1 (Yan et al., 2025) and Mem-α (Wang et al., 2025a) optimize memory management with reinforcement learning using downstream task signals. Despite this progress, memory management is still largely governed by static, hand-crafted routines for extraction, consolidation, and pruning.

> 💡 **Agent Memory 脉络**:
> 
> | 阶段 | 代表方法 | 特点 |
> |------|----------|------|
> | Pipeline 式 | MemoryBank, A-MEM, Mem0, MemoryOS | 固定提取→检索→更新流程 |
> | RL 优化 | Memory-R1, Mem-α | 用下游任务奖励优化 memory 操作选择 |
> | **MemSkill** | 本文 | 操作本身也可进化 |
> 
> Memory-R1 和 Mem-α 是最接近的前人工作，但它们的 action space 仍然是固定的 {add, update, delete, skip}。MemSkill 把 action space 本身也变成可学习的。

---

Several concurrent works also explore self-evolving memory in agent settings, but differ fundamentally from our focus. Evo-Memory provides a streaming benchmark and evaluation framework for test-time memory evolution (Wei et al., 2025), while MemEvolve meta-optimizes memory architectures within a predefined modular design space (Zhang et al., 2025). By contrast, we target the evolution of memory skills themselves, enabling the system to refine and grow its reusable memory operations over time.

> 💡 **跟 concurrent works 的区分**:
> - **Evo-Memory**: 做的是 benchmark/评估框架，不是方法
> - **MemEvolve**（张冠宇等）: 在预定义的模块 design space 里搜最优架构组合
> - **MemSkill**: 进化的是 skill（操作策略），粒度更细
> 
> 一个直觉类比：MemEvolve 是 NAS（搜网络架构），MemSkill 是学 policy（搜行为策略）。

---

## 2.2. Self-Evolving LLM Agents

Recent work on self-evolving LLM agents studies how agents can improve from interaction experience with minimal manual supervision. ExpeL (Zhao et al., 2024) distills trajectories into editable natural-language insights and retrieves relevant experiences to guide future decisions, while EvolveR (Wu et al., 2025) formalizes an experience lifecycle that consolidates interactions into reusable principles and closes the loop with reinforcement learning updates. A complementary line reduces reliance on curated data via self-play style curricula: Absolute Zero Reasoner (Zhao et al., 2025) trains a proposer and solver with verifiable rewards from a code executor, and Multi-Agent Evolve (Chen et al., 2025) extends this to a proposer solver judge triad with LLM-based evaluation; R-Zero (Huang et al., 2025) follows a similar challenger solver co-evolution pattern. Beyond curricula, systems such as AgentEvolver (Zhai et al., 2025) and RAGEN (Wang et al., 2025b) study efficient agent learning dynamics and stabilization in multi-turn RL settings, while ADAS (Hu et al., 2024) and AlphaEvolve (Novikov et al., 2025) explore automated discovery and evolutionary improvement of agent designs. Finally, SkillWeaver (Zheng et al., 2025) shows that agents can discover and refine reusable skills for web interaction. In contrast, our focus is on self-evolving memory skills that govern how agents construct and revise memories over time.

> 💡 **Self-Evolving Agents 谱系**:
> 
> 按路线分：
> 1. **经验蒸馏**: ExpeL（轨迹→insight）、EvolveR（经验生命周期）
> 2. **Self-play/Co-evolution**: Absolute Zero Reasoner、Multi-Agent Evolve、R-Zero
> 3. **Agent 架构搜索**: ADAS、AlphaEvolve、AgentEvolver
> 4. **Skill 发现**: SkillWeaver（web interaction 的 skill）
> 
> MemSkill 属于 (4) 的 memory 特化版本：发现和进化的是 memory 操作的 skill，而非 web 操作。
> 
> 注意 RAGEN（Wang et al., 2025b）也是多轮 RL agent 训练，但关注的是训练稳定性，不是 memory skill 进化。

---

## 🔖 Section 总结

### 核心洞察
1. Agent Memory: pipeline → RL 优化固定操作 → MemSkill 进化操作本身
2. Self-Evolving Agents: 经验蒸馏 / self-play / 架构搜索 / skill 发现 四条线
3. MemSkill 的独特定位：把 "self-evolving" 聚焦到 memory skill 上，是 skill discovery 在 memory 领域的特化
