[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
动机：LTM/STM 分治导致碎片化 → 三大挑战 → AgeMem 统一方案。Figure 1 是全文最重要的图。

---

In long-horizon agentic tasks involving multi-step reasoning and complex workflows (Chang et al., 2024), the effectiveness of large language model (LLM) agents is fundamentally constrained by the information they can attend to at any given time, which we collectively refer to as the agent's memory (Xiong et al., 2025; Goodyear et al., 2025). Memory typically falls into two categories: long-term memory (LTM), which persistently stores user- or task-specific knowledge (Zhong et al., 2024; Jiang et al., 2024), and short-term memory (STM), which comprises the information contained in the current input context (Wu et al., 2025b; Gao et al., 2025b). High-quality LTM supports efficient retrieval of accumulated knowledge, while effective STM management reduces redundancy and preserves salient context. Together, they mitigate the limitations of finite context windows, making their joint management crucial for improving agent performance in complex reasoning settings.

> 💡 **Memory 二分法**:
> - **LTM** = 持久存储的 user/task 知识（跨 session）
> - **STM** = 当前 context window 里的信息
> - 两者互补：LTM 提供知识积累，STM 保持当前上下文精简

---

However, existing research has predominantly treated LTM and STM as independent components. STM is commonly enhanced through retrieval-augmented generation (RAG) (Pan et al., 2025b), such as in MainRAG (Chang et al., 2025) and ReSum (Wu et al., 2025a), which expand usable context via external retrieval or periodic summarization. Although effective in some tasks, these methods rely heavily on predefined schedules or heuristic rules, potentially resulting in overlooked infrequent but critical details as well as unnecessary noise (Ma et al., 2025; Dong et al., 2025). In contrast, LTM management has progressed along separate lines, typically categorized into trigger-based (Kang et al., 2025; Wang and Chen, 2025; Wang et al., 2025c; Chhikara et al., 2025) and agent-based (Yan et al., 2025; Hu et al., 2025; Xu et al., 2025) paradigms. The former executes fixed memory operations at predefined moments, whereas the latter incorporates a specialized memory manager to determine what and how to store. Despite offering more flexibility, most approaches still depend on handcrafted rules or auxiliary expert models, limiting adaptability and increasing system complexity (Xiong et al., 2025).

> 💡 **现有方法的分类**:
> - **STM 增强**：RAG / 定期摘要（MainRAG, ReSum）→ 依赖预定义规则，可能漏掉稀有但关键信息
> - **LTM 管理**：
>   - Trigger-based：预定义时刻执行固定操作（如 Mem0, MemoryOS）
>   - Agent-based：专门的 memory manager 决策（如 A-Mem, Memory-R1）
>   - 问题：仍依赖手工规则或辅助专家模型

---

As a consequence, LTM and STM are typically treated as separate and loosely coupled modules. As illustrated in Figure 1, existing architectures generally follow two patterns: (a) static STM with trigger-based LTM, or (b) static STM with agent-based LTM. In both settings, the two memory systems are optimized independently and later combined in an ad hoc way, leading to fragmented memory construction and suboptimal performance in long-horizon reasoning tasks. Thus, unifying the management of LTM and STM remains a necessary yet largely unexplored challenge.

![Figure 1](../images/c2be4c3f794ec9072fb9b2dc7d78b6949ab05ffbe0338a14f4043585582111ef.jpg)
*Figure 1: Comparison between independent and unified memory management frameworks. (Left) Traditional framework with static STM and trigger-based LTM. (Middle) Independent framework with an additional Memory Manager controlling LTM in an agent-based manner, while STM remains static. (Right) The proposed AgeMem framework, where LTM and STM are jointly and intelligently managed via explicit tool-based operations.*

> 💡 **Figure 1 批读**:
> - **Left（传统）**：STM 静态不变 + LTM 用 trigger 规则管理 → 两者完全独立
> - **Middle（现有 SOTA）**：加了 Memory Manager 做 agent-based LTM，但 STM 仍然是静态的 → 半独立
> - **Right（AgeMem）**：LTM 和 STM 都由 agent 通过 tool call 统一管理 → 完全统一
> - **核心区别**：AgeMem 把 memory 管理从"外挂模块"变成了"agent 自身能力"
> - 对比 MemSkill：MemSkill 的 Controller+Executor 更像 Middle 模式的进化版，而 AgeMem 更简洁地直接用 tool

---

Nevertheless, achieving unified memory management poses three fundamental challenges. (C1) Functional heterogeneity coordination: LTM and STM serve distinct yet complementary purposes: LTM determines what to store, update, or discard, while STM governs what to retrieve, summarize, or remove from the active context (Zhang et al., 2025b). The challenge lies in designing a unified mechanism that orchestrates their interplay synergistically. (C2) Training paradigm mismatch: Existing reinforcement learning (RL) frameworks adopt markedly different training strategies for the two memory types (Ma et al., 2024). LTM-focused training often leverages session-level information available prior to interaction, whereas STM training typically injects distractors to simulate long-horizon contexts (Sun et al., 2024). Moreover, standard RL assumes continuous trajectories with stable rewards, which conflicts with the inherently fragmented and discontinuous experiences produced by memory operations (Wu et al., 2025a), making end-to-end optimization particularly challenging. (C3) Practical deployment constraints: Many agent systems rely on an auxiliary expert LLM for memory control, significantly increasing inference cost and training complexity. How to integrate unified memory management directly into an agent without dependence on external expert models remains an open problem.

> 💡 **三大挑战**:
> - **C1 功能异构协调**：LTM（存/更新/删）和 STM（检索/摘要/过滤）功能不同，如何统一编排？
> - **C2 训练范式不匹配**：LTM 训练用 session 信息，STM 训练用 distractor 注入；memory 操作产生不连续轨迹，标准 RL 难以处理
> - **C3 部署约束**：不能依赖额外的专家 LLM 做 memory 控制（成本太高）
> - 这三个挑战分别对应 AgeMem 的三个设计：统一 tool interface (C1) → 三阶段渐进 RL (C2) → 单模型端到端 (C3)

---

To address these challenges, we propose Agentic Memory (AgeMem), a unified framework that jointly manages LTM and STM, illustrated in Figure 1 (right). Unlike prior designs that treat memory as an external component, AgeMem integrates both memory types directly into the agent's decision-making process. Through a unified tool-based interface, the LLM autonomously invokes and executes memory operations for both LTM and STM. Furthermore, we design a three-stage progressive RL strategy: the model first acquires LTM storage capabilities, then learns STM context management, and finally coordinates both forms of memory under full task settings. To address the fragmented experience issue across training stages, we design a step-wise Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which transforms cross-stage dependencies into learnable signals, thereby alleviating the challenges posed by sparse and discontinuous rewards in RL. We evaluate AgeMem on five long-context, reasoning-intensive benchmarks. Comprehensive results show that AgeMem consistently outperforms strong baselines, validating the effectiveness of unified memory management.

> 💡 **AgeMem 方案概览**:
> - **统一 tool interface**：6 个 memory tool（3 LTM + 3 STM）→ 解决 C1
> - **三阶段渐进 RL**：LTM 构建 → STM 过滤 → 联合推理 → 解决 C2
> - **Step-wise GRPO**：终端奖励广播到所有 step → 解决稀疏奖励
> - **单模型端到端**：不需要额外 memory manager → 解决 C3

---

Our main contributions are as follows:

• We propose Agentic Memory (AgeMem), a unified agentic memory framework that enables LLM-based agents to autonomously decide when, what, and how to manage both long-term and short-term memory.

• We develop a three-stage progressive RL strategy equipped with a step-wise GRPO mechanism, facilitating effective end-to-end learning of unified memory management behaviors.

• We conduct comprehensive evaluations across multiple models and long-horizon benchmarks, demonstrating the robustness and effectiveness of AgeMem in complex agentic tasks.

---

## 🔖 Section 总结

### 核心洞察
1. **LTM/STM 分治是根本问题**：现有方法要么 STM 静态 + LTM trigger-based，要么 STM 静态 + LTM agent-based，都没有统一管理
2. **Memory 操作 = Tool Action** 是优雅的统一抽象：让 agent 自己决定何时调用什么 memory 操作
3. **三阶段设计很巧妙**：先学存（LTM）→ 再学过滤（STM）→ 最后学协调 → 类似课程学习
4. **Step-wise GRPO 是关键技术贡献**：memory 操作的奖励天然稀疏且不连续，广播终端奖励到所有 step 是自然的解法
