[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
从 LLM agent 的自进化能力出发，指出 MAS 缺乏有效记忆机制是自进化的瓶颈，现有方案要么没有跨试次记忆，要么直接搬单 Agent 方案导致信息过载。

---

As Large Language Models (LLMs) continue to redefine the frontier of artificial intelligence, LLM-driven agents have exhibited unprecedented prowess in perception [2, 3, 4, 5], planning [6, 7, 8], reasoning [9, 10], and action [11, 12], which have catalyzed remarkable progress across diverse downstream domains, including code generation [13, 14], data analysis [15], embodied tasks [16] and autonomous driving [3, 17, 18]. Building upon the impressive competencies of single agents, LLM-based Multi-Agent Systems (MAS) have been demonstrated to push the boundaries of single model capacity [19, 20, 21]. Similar to collective intelligence arising from human social collaboration [22, 23, 24], MAS orchestrates multiple agents [25, 26, 27], whether through cooperation [28, 29, 30, 31] or competition [32, 33, 34], to transcend the cognitive and specialized limitations of solitary agents.

> 💡 **背景铺垫**: LLM agent 在各领域展现强大能力 → MAS 通过多 agent 协作突破单 agent 的能力上限。

**Self-Evolving Agents.** What especially characterizes LLM agents is their self-evolving capacity, i.e., the ability to continuously adapt and improve through interactions with the environment, as seen in prior works where such adaptability has led to two- to three-fold quantitative improvements [35]. The central driving force behind such self-evolving nature is memory mechanism of agents [36, 37, 38], which parallels human abilities to accumulate knowledge, process past experiences, and retrieve relevant information. Previous successful memory mechanism designs, including both inside-trial memory (i.e., context retained within solving one single query) and cross-trial memory (i.e., experience accumulated across multiple tasks) [39], have empowered agents to excel in diverse applications such as personalized chat [36, 40, 41], recommendation [42], embodied action [43, 16], and social simulation [19, 44, 45], enabling them to evolve into experiential learners that effectively leverage past experiences and world knowledge.

> 💡 **自进化的核心是记忆**: 
> - **Inside-trial memory**: 单次任务内的上下文（如 MemGPT 的滑动窗口）
> - **Cross-trial memory**: 跨任务积累的经验（如 ExpeL 的经验提取、Voyager 的技能库）
> - 记忆让 agent 从「无状态工具」变成「经验型学习者」，性能可提升 2-3 倍

![Figure 1](../images/8ec779acc0af0eea98195333dadd1dc391033f1c21c8a6d159be6eb2daf7b862.jpg)
*Figure 1: (Left) We report the token cost of several single-agent and MAS baselines on ALFWorld benchmark; (Right) The overview of G-Memory's three-tier hierarchical memory architecture, encompassing the insight graph, query graph and interaction (utterance) graph.*

> 💡 **Figure 1 批读**:
> - **左图**: MAS 的 token 消耗远超单 Agent（高达 10×），这是因为多 agent 多轮对话产生大量轨迹文本
> - **右图**: G-Memory 的三层架构概览。底层 interaction graph 存每条 utterance 及其时序关系；中层 query graph 存任务节点及其相似性连边；顶层 insight graph 存抽象的策略洞察
> - 关键设计：三层之间有明确的关联——每个 query 节点连着它的 interaction graph，每个 insight 节点连着支持它的 query 集合

**Self-Evolving MAS.** However, such self-evolving capacity remains largely absent in multi-agent systems. Most existing MAS are still constrained by manually defined workflows, such as the Standard Operating Procedures (SOP) in MetaGPT [21] and ChatDev [46], or rely on pre-defined communication topologies in MacNet [47] and AgentPrune [30]. More recent automated MASs, such as GPTSwarm [48], ADAS [49], AFlow [50], and MaAS [51] have made it to automatically optimize inter-agent topologies or prompts, which, nevertheless, ultimately yield giant and cumbersome MAS architectures, lacking the agility to self-adjust with accumulated collaboration experience.

> 💡 **MAS 自进化的缺失**: 
> - 早期 MAS（MetaGPT, ChatDev）靠手写 SOP
> - 自动化 MAS（AFlow, ADAS）能搜架构但不能随经验进化
> - 核心问题：这些方法都是 one-shot 设计，缺乏「越用越好」的能力

**Memory for MAS.** The absence of the aforementioned self-evolving capacity is, in fact, rooted in the lack of memory mechanisms specifically tailored for MAS. One may challenge this claim from two perspectives: ❶ Do existing MASs lack memory mechanisms altogether? Not entirely. Classical MAS frameworks such as MetaGPT, ChatDev, and Exchange-of-Thought [52] incorporate memory-related designs. However, these are often limited to inside-trial memory [52], while cross-trial memory, if present, remains rudimentary—typically involving the transmission of overly condensed artifacts (e.g., final solutions or execution results) [21, 46, 47], and failing to enable meaningful learning from collaborative experience. ❷ Why not directly transfer existing single-agent memory mechanisms to MAS? Unfortunately, such a transfer is far from straightforward. The inherent nature of MAS, i.e., multi-turn orchestration across multiple agents [26, 27], leads to substantially longer task-solving trajectories compared to single-agent settings (up to 10× more tokens, as demonstrated by Figure 1 (Left)). This poses a significant challenge to traditional retrieval-based memory designs [36, 37, 16], as naive feeding of the entire long-context trajectory without proper abstraction from a collaborative perspective offers little benefit.

> 💡 **两个关键质疑的回答**:
> - Q1: MAS 已有记忆？→ 只有 inside-trial 或极度压缩的 cross-trial（只存最终结果），丢失了协作过程信息
> - Q2: 搬单 Agent 记忆？→ MAS 轨迹是单 Agent 的 10×，直接塞进去 LLM 会信息过载
> - **本质矛盾**: 协作轨迹信息量大但又不能全部丢弃，需要一种「层次化压缩 + 选择性检索」的方案

Given the aforementioned challenges, a natural question arises:

*How can we design a memory mechanism capable of storing, retrieving, and managing the lengthy interaction history of multi-agent systems, such that agent teams can benefit from concise and instructive experience and insights?*

**The Present Work: G-Memory.** In response to the above question, we introduce a Graph-based Agentic Memory Mechanism for LLM-based Multi-Agent Systems, dubbed G-Memory, which manages the complex and lengthy interaction history of MAS through a three-tier hierarchical graph structure:

- **Insight Graph**, which abstracts generalizable insights from historical experience;
- **Query Graph**, which encodes meta-information of task queries and their connectivity;
- **Interaction Graph**, which stores fine-grained textual communication logs among agents.

> 💡 **三层图的直觉**:
> - **Interaction Graph**（底层）≈ 原始聊天记录，谁说了什么
> - **Query Graph**（中层）≈ 任务索引，哪些任务相似、哪些任务用到了哪些对话
> - **Insight Graph**（顶层）≈ 经验总结，从多次任务中提炼的策略
>
> 类比人类组织：底层是会议记录，中层是项目档案，顶层是管理经验/最佳实践

Figure 1 (Right) visualizes these structures, and their formal definitions are placed in Section 3. When a new query arrives, G-Memory efficiently retrieves relevant query records by leveraging the topology of the query graph, and then traverses upward (i.e., query → insight graph) to extract associated high-level insights and downward (i.e., query → interaction graph) to identify core interaction subgraphs that are most pertinent to the task at hand, thereby mitigating information overload. Based on the retrieved memory, G-Memory offers actionable guidance to the MAS, e.g., division of labor, task decomposition, and lessons from past failures. Upon the completion of a task, all three levels of the memory hierarchy are updated in an agentic manner, with newly distilled insights, enriched query records, detailed MAS trajectories, and their level of detailed associations. Through this refinement, G-Memory functions as a plug-and-play module that can be seamlessly embedded into mainstream MAS frameworks, empowering evolving inter-agent collaboration and collective intelligence.

> 💡 **工作流概述**:
> 1. 新 query 来了 → query graph 检索相似任务
> 2. 向上遍历 → 取 insight（高层策略）
> 3. 向下遍历 → 取 sparsified interaction subgraph（关键对话片段）
> 4. 根据每个 agent 的 role 定制记忆
> 5. 任务完成后 → 三层图全部更新
>
> Plug-and-play 意味着不需要改 MAS 框架本身，只在 MAS 开始运行前注入记忆。

Our contributions are summarized as follows:

- **❶ Bottleneck Identification.** We conduct a thorough review of existing multi-agent systems and identify a fundamental bottleneck in their self-evolving capabilities, which is largely attributed to the oversimplified memory architectures.
- **❷ Practical Solution.** We propose G-Memory, a hierarchical agentic memory architecture for MAS, which models complex and prolonged inter-agent collaboration through a three-tier structure comprising insight, query, and interaction graphs.
- **❸ Experimental Evaluation.** Extensive experiments across five benchmarks show that G-Memory is (I) high-performing, improving state-of-the-art MAS by up to 20.89% and 10.12% on embodied action and knowledge QA tasks, respectively; and (II) resource-friendly, maintaining comparable or even lower token usage than mainstream memory designs.

---

## 🔖 Section 总结

### 核心洞察
1. MAS 自进化的瓶颈在于**记忆机制**——现有 MAS 要么没有跨试次记忆，要么只存最终结果
2. 单 Agent 记忆无法直接迁移到 MAS——轨迹太长（10× tokens），需要层次化压缩
3. G-Memory 的核心思想：**组织记忆理论**启发的三层图结构 + 双向遍历 + role-specific 记忆注入
4. 灵感来源是 organizational memory theory（Walsh & Ungson 1991），把 MAS 类比为人类组织
