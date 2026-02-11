[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
论证"没有万能的记忆架构"，用人类学习策略的类比引出 MemEvolve 的双层优化思路，列出三大贡献。

---

Language agents and agent systems, empowered by increasingly capable foundation models (Team et al., 2025a,b) and sophisticated scaffolding (Wang et al., 2024a; LangChain, 2023), have advanced rapidly, demonstrating unprecedented performance across complex tasks such as deep research (Chen et al., 2025), scientific discovery (Bai et al., 2025; Wei et al., 2025b), and industrial report generation (Zhang et al., 2025g). A key driving force behind this success is the agent memory system (Zhang et al., 2024b; Hu et al., 2025c), which persistently captures interactions between the agent and environment, distilling them into diverse forms of knowledge and skills, and thereby enabling large language model (LLM)-based agents to evolve continuously in task solving and world exploration (Wu et al., 2025c).

> 💡 **背景**: Agent 记忆系统是 agent 持续进化的关键驱动力。注意引用了 Hu et al. 2025c 即 "Memory in the Age of AI Agents" 综述，与我们之前读的论文一脉相承。

Naturally, the choice of memory paradigm plays a decisive role in shaping an agent's capacity for on-the-fly self-evolution. Initial designs centered on raw trajectory storage and few-shot prompting (Zhong et al., 2024; Wen et al., 2024), which were later superseded by more abstracted textual artifacts such as tips, shortcuts, and reasoning templates (Ouyang et al., 2025; Zhang et al., 2025b; Ye et al., 2025; Tang et al., 2025). Recent advances have also explored structured tool interfaces (e.g., APIs (Zheng et al., 2025), MCPs (Qiu et al., 2025b,a; Zhang et al., 2025h)) and code-level repositories (Zhang et al., 2025e; Wang et al., 2025a) as memory carriers. Amid this growing diversity, an inquisitive practitioner might ask: What kind of memory architecture most effectively drives agent self-improving?

> 💡 **记忆范式的演进**:
> - **第一代**: 原始轨迹 + few-shot（MemoryBank, DILU）
> - **第二代**: 抽象文本（tips, insights, reasoning templates）
> - **第三代**: 结构化工具（API, MCP）和代码仓库
> - 核心问题：哪种最好？

We posit that no universally optimal memory architecture exists. For instance, a memory system that distill reusable APIs from past trajectories may excel in tasks such as web browsing, yet offer limited utility for mathematical and scientific reasoning. Conversely, memories predicated on self-critique, while powerful in reasoning-intensive domains (Cai et al., 2025), show diminished efficacy in coding and tool-use scenarios, as empirically discussed in (Zhang et al., 2025d). We contend that these trade-offs arise from the static nature of current memory systems. Researchers typically design a fixed memory pipeline (i.e., memory ingestion/abstraction/retrieval (Zhang et al., 2025i)) and embed it within an agent, assuming it will sustain long-term evolution through mere exposure to new experiences. Yet this overlooks a crucial reality: distinct tasks are coupled with distinct memory affordances. A memory system that cannot adapt itself to the task at hand is fundamentally misaligned with the very premise of open-ended agent evolution.

> 💡 **核心论点 — No Free Lunch for Memory**:
> - API 记忆擅长 web browsing，但对数学推理无用
> - 自我批评记忆擅长推理，但在 coding/tool-use 场景效果差
> - 根源：记忆架构是静态的，不能适应不同任务域
> - 这个 insight 与我们在多图记忆策略上的困境类似：不同类型的图（结构图、流程图、数据表）可能需要不同的记忆策略

To elucidate this dilemma, consider the analogy of human learning. Both high- and low-performing students inevitably make mistakes, yet their distinction lies in the meta-cognitive strategies they employ to learn from these errors. An underperforming student might resort to rote memorization, superficially recording an error without genuine comprehension (Zhong et al., 2024; Orhan, 2023). In contrast, a more skillful student engages in higher-order learning: they not only record errors but also distill transferable insights through reflection (Shinn et al., 2023; Zhao et al., 2024) or derive reusable schemas (Zheng et al., 2025; Qiu et al., 2025b)). Current memory systems effectively model a skillful learner. Herein lies the critical gap: the most effective human learners are not merely skillful, but adaptive. They dynamically alter their learning strategies based on the subjects, for instance, prioritizing memorization for literary analysis while abstracting solution templates for mathematics. It is precisely this transition, from a skillful to an adaptive learner (as shown in Figure 2), that we argue agent memory systems must undergo. To put it more formally:

> 💡 **学习者类比**:
> - 差生：死记硬背（raw trajectory storage）
> - 好生：提炼规律（ExpeL/AWM 式的 insight extraction）
> - 最优学习者：根据科目切换学习策略（MemEvolve）
> - 关键词是 **meta-cognitive** — 对学习过程本身的认知和调整

How can a memory system not only facilitate the agent system's evolution but also meta-evolve its own architecture to achieve superior task-domain performance gains while preserving generalizability?

> 💡 **研究问题**: 既要 task-specific adaptation，又要 generalizability。这两个目标之间存在张力，后面的实验会展示 MemEvolve 如何平衡。

To address the challenge, we introduce MemEvolve, a framework that facilitates the dual evolution of an agent's experience and its memory architecture. Conceptually, MemEvolve operates as a bilevel optimization process: the inner loop performs a first-order evolution, where the agent, guided by a fixed memory system, adapts to a continuous stream of new tasks by populating its experience base. The outer loop drives a second-order evolution, meta-learning a more effective memory architecture to accelerate future learning. This allows the agent not only to evolve, but to evolve more efficiently and intelligently over time.

> 💡 **MemEvolve 框架概览**:
> - **内层循环（一阶进化）**: 固定记忆架构，Agent 处理任务、积累经验
> - **外层循环（二阶进化）**: 根据 Agent 表现反馈，优化记忆架构本身
> - 这是一个典型的 **bilevel optimization**，类似于 MAML 中 inner loop 学习任务、outer loop 学习如何学习

However, the vast and heterogeneous design space of memory systems (e.g., knowledge graphs, skill libraries, vector databases) presents a significant challenge to controllable optimization. To render this optimization tractable, we introduce a modular design, decomposing any memory architecture into four key components: ♣ Encode (perceiving and formatting experiences), ♠ Store (committing information), ♥ Retrieve (context-aware recall), and ♠ Manage (consolidation and forgetting). MemEvolve evolves the programmatic implementations of these modules in a model-driven fashion, using feedback from the agent's performance in the inner loop. This process establishes a virtuous cycle: an improved memory architecture from the outer loop enhances the agent's learning efficiency. In turn, a more capable agent generates higher-quality trajectories, providing a more precise fitness signal for the outer loop to drive the next round of architectural evolution.

> 💡 **四组件模块化设计**:
> - ♣ **Encode**: 如何感知和格式化经验（raw trace → insights/tips/APIs）
> - ♠ **Store**: 如何存储（vector DB / graph / JSON）
> - ♥ **Retrieve**: 如何检索（semantic search / graph traversal / contrastive）
> - ♠ **Manage**: 如何维护（consolidation / forgetting / pruning）
> - 进化的单位是这四个组件的**代码实现**，而非超参数。这是一个 program synthesis / code evolution 问题。
> - **良性循环**: 好架构 → 好经验 → 更好的进化信号 → 更好的架构

To ground our framework within the diverse landscape of existing self-improving agent memories, we systematically re-implement twelve representative architectures in a unified modular design space, including ExpeL (Zhao et al., 2024), Agent Workflow Memory (Wang et al., 2024b), and Dynamic Cheatsheet (Suzgun et al., 2025). The resulting framework, denoted as EvolveLab, serves both as an empirical foundation for MemEvolve's evolutionary process and as a standardized codebase to facilitate future research on self-evolving agents. Our contributions are as follows:

> 💡 **EvolveLab 的双重价值**: (1) 为 MemEvolve 提供进化的起点和搜索空间 (2) 为社区提供统一的 benchmark 平台

- **Unified Codebase**: We introduce EvolveLab, a modular design space for self-improving agent memory systems encompassing four key components (encoding, storage, retrieval, and management), providing unified implementations and benchmark support for a wide range of prevailing agent memory systems.

- **Meta-Evolution Framework**: We propose MemEvolve, a meta-evolutionary framework that jointly evolves both agents' experiential knowledge and their underlying memory architecture, in which agent systems not only accumulate experience but also progressively refine their mechanism for learning from it.

- **Experimental Evaluation**: Extensive experiments on four challenging agentic benchmarks demonstrate that MemEvolve delivers (I) substantial performance gains, improving frameworks such as SmolAgent and Flash-Searcher by up to 17.06%; and (II) cross-domain, cross-framework and cross-LLM generalization, where memory systems evolved on TaskCraft yield 2.0–9.09% gains with unseen benchmarks and backbone models.

> 💡 **三大贡献**:
> 1. EvolveLab（工程）：12 种记忆系统的统一实现
> 2. MemEvolve（方法）：记忆架构的元进化
> 3. 实验（验证）：四个 benchmark + 三维泛化
>
> 与 MemGen/G-Memory 的关系：MemGen 是具体的记忆方法（生成式隐式记忆），G-Memory 是图结构记忆。MemEvolve 是更上层的"元方法"——自动搜索最优记忆架构，MemGen 和 G-Memory 都可以是其搜索空间中的一个点。

---

## 🔖 Section 总结

### 核心洞察
1. **No Free Lunch**: 没有万能记忆架构，不同任务需要不同记忆策略
2. **Bilevel Optimization**: 内层积累经验，外层优化架构
3. **模块化是关键**: 四组件分解使得庞大的设计空间变得可搜索
4. **对我们项目的启发**: 多图记忆也面临同样问题 — 不同图类型可能需要不同的记忆策略，MemEvolve 的思路可以借鉴
