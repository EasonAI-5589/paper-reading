[← 返回 README](../README.md)

# 2 Related Works

## 📌 预览
三条线：(1) 单 Agent 记忆从 RAG 到认知启发架构的演进；(2) MAS 记忆的严重匮乏；(3) MAS 框架从手动到自动但缺乏持续进化。

---

**Single-Agent Memory.** Memory serves as a primary driving force for agents to accumulate experiences and explore the world through interactions with the environment [53, 54, 55, 56]. It plays a critical role in both task-solving and social simulation LLM agents, and this work primarily focuses on the former. Early research on agent memory was confined to simple inside-trial memory, mainly addressing limitations posed by the LLM context window in chatbot applications, including MemoryBank [36], ChatDB [40], MemoChat [41], and MemGPT [37], which typically adopt retrieval-augmented generation (RAG)-style, similarity-based chunk retrieval. Subsequent developments have progressed toward more cognitively inspired memory architectures, including (1) memory scope extended to cross-trial memory like ExpeL [43] and Synapse [57]; (2) application domains broadened to include computer control [57], embodied action [58], scientific discovery [59], coding and reasoning [60]; and (3) management techniques evolved from coarse-grained textual similarity toward more sophisticated abstraction and summarization of acquired knowledge and experiences [19], as seen in A-Mem [61], Mem0 [62] and MemInsight [63]. More discussions are in Appendix D.

> 💡 **单 Agent 记忆演进路线**:
> - **第一阶段**: Inside-trial RAG（MemoryBank, MemGPT）→ 解决上下文窗口限制
> - **第二阶段**: Cross-trial 经验学习（ExpeL, Synapse）→ 积累跨任务经验
> - **第三阶段**: 认知启发式架构（A-Mem, Mem0, MemInsight）→ 更精细的抽象和总结
>
> **对比 MemGen**: MemGen 走了一条完全不同的路——用 latent space 做隐式记忆，避免了显式文本检索的信息损失。G-Memory 则是显式文本路线的升级版，用图结构解决组织问题。

**Memory in Multi-agent System.** However, the memory mechanisms tailored for MAS remain markedly underexplored. Some representative frameworks, such as LLM-Debate [20, 33] and Mixture-of-Agent [64], omit memory components altogether. Others merely adopt simplistic inside-trial memory schemes [47, 52]. Even in frameworks that attempt cross-trial memory [46], the memory is merely compressed as the final outcome artifacts, overlooking the nuanced agent interactions. Collectively, there is a pressing need for a principled memory architecture that can capture, organize, and retrieve the inherently intricate task-solving processes unique to MAS [39].

> 💡 **MAS 记忆的现状**:
> | 框架 | 记忆类型 | 问题 |
> |------|----------|------|
> | LLM-Debate, MoA | 无记忆 | 每次从头开始 |
> | MacNet, EoT | Inside-trial | 只有当次对话 |
> | ChatDev | Cross-trial（仅存结果） | 丢失协作过程 |
>
> G-Memory 是**第一个**为 MAS 设计的、同时覆盖 inside-trial 和 cross-trial、且保留协作轨迹细节的记忆系统。

**LLM-based Multi-Agent Systems.** Our work focuses on task-solving MAS, which, unlike their single-agent counterparts, often lack the capacity for continual evolution through interaction with the environment [65, 66]. Early frameworks such as AutoGen [13], CAMEL [24], and AgentVerse [67] rely entirely on pre-defined workflows. More recent efforts [68, 69, 50, 49, 70, 31] introduce a degree of adaptivity by generating dynamic MAS in response to environmental feedback. However, such evolution is often one-shot: for example, AFlow [50] employs Monte Carlo Tree Search to construct a complex MAS tailored to a specific task domain, which yet lacks the capacity to evolve with increasing task exposure or transfer across domains [51, 71]. From this perspective, constructing MAS with genuine self-evolving capabilities remains an open and challenging research frontier.

> 💡 **MAS 框架的演进**:
> - 手动设计：AutoGen, CAMEL, AgentVerse
> - 自动搜索：GPTSwarm, AFlow, ADAS → 但是 one-shot，不能持续进化
> - G-Memory 的定位：不改框架结构，而是加「记忆模块」让框架具备持续学习能力

---

## 🔖 Section 总结

### 核心洞察
1. 单 Agent 记忆已经发展到第三代（认知启发式），但 MAS 记忆还停留在第一代（简单 RAG 或无记忆）
2. G-Memory 填补了 MAS 跨试次记忆的空白
3. 与自动 MAS 搜索（AFlow 等）互补——搜索解决「框架设计」，G-Memory 解决「经验积累」
