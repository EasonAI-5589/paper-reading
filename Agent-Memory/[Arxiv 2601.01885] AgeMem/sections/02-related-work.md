[← 返回 README](../README.md)

# 2 Background and Related Work

## 📌 预览
三条研究线：LTM 架构设计、STM 上下文管理、RL for LLMs。AgeMem 的定位：把 RL 直接整合进 memory 管理。

---

**Long-term memory (LTM).** Persistent LTM is crucial for LLM-based agents operating over extended horizons (Wang et al., 2025b; Li et al., 2025). Recent work has explored diverse architectural designs for modeling LTM. LangMem (LangChain Team, 2025) provides a modular framework that supports multiple memory types, while A-Mem (Xu et al., 2025) adopts a Zettelkasten-inspired design that links structured knowledge units to facilitate consolidation. Mem0 (Chhikara et al., 2025) proposes a scalable extract-update pipeline and extends it to a graph-based variant for structured reasoning, and Zep (Rasmussen et al., 2025) represents memory as a temporal knowledge graph to enable cross-session and time-aware reasoning. Although effective in organizing and retrieving information, these approaches largely rely on predefined memory structures or heuristic update rules. As memory grows, such designs commonly suffer from increased system complexity and lack adaptive, learning-based strategies for prioritization and forgetting. In contrast, our work aims to learn an adaptive memory policy that allows agents to dynamically decide what to store, update, or forget, depending on task demands and long-term utility.

> 💡 **LTM 相关工作**:
> - **LangMem**：模块化框架，多种 memory 类型
> - **A-Mem**：Zettelkasten 式结构化知识单元链接
> - **Mem0 / Mem0^g**：extract-update pipeline / graph-based
> - **Zep**：temporal knowledge graph
> - **共同问题**：依赖预定义结构/启发式规则，缺乏自适应学习
> - AgeMem vs 这些方法：AgeMem 不设计固定的 memory 结构，而是让 agent 通过 RL 学习自适应策略

---

**Short-term memory (STM).** STM in agentic LLMs primarily concerns context selection and retrieval (Wang et al., 2024; Jin et al., 2024). Retrieval-Augmented Generation (RAG) (Pan et al., 2025b; Salama et al., 2025; Kagaya et al., 2024) is the dominant paradigm, expanding usable context by injecting retrieved content into prompts. While effective, RAG does not fundamentally prevent context explosion in long-horizon settings and may introduce irrelevant or distracting information. To address this issue, ReSum (Wu et al., 2025a) periodically compresses interaction histories into compact reasoning states, allowing agents to operate beyond fixed context-window constraints. Yet its summarization schedule remains largely predefined, and aggressive compression risks discarding rare but crucial details. Our approach instead enables agents to learn when and how to retrieve, summarize, or filter context, achieving a more flexible balance between efficiency and information preservation.

> 💡 **STM 相关工作**:
> - **RAG**：主流范式，但无法根本防止 context 爆炸，可能引入噪声
> - **ReSum**：定期压缩交互历史 → 但摘要时机预定义，激进压缩可能丢失关键细节
> - AgeMem 的区别：agent 自己学习 when/how to retrieve/summarize/filter

---

**Reinforcement learning for LLMs.** Reinforcement learning has become an effective paradigm for improving the decision-making and reasoning capabilities of LLM-based agents (Yao et al., 2022; Jin et al., 2025; Qian et al., 2025; Chaudhari et al., 2025). Among recent advances, GRPO (Shao et al., 2024) enhances stability by optimizing policies based on the relative quality of sampled trajectories, removing the need for an explicit value function. GRPO and its variants (Gilabert et al., 2025; Wang et al., 2025a) have shown strong performance in complex reasoning tasks. However, existing RL-based systems generally treat memory as a static or external component, making them ill-suited for the discontinuous and fragmented trajectories associated with memory operations (Yan et al., 2025; Zhang et al., 2025a). In contrast, our work integrates RL directly into the memory management process, enabling unified training of both language generation and memory operations.

> 💡 **RL for LLMs 相关工作**:
> - **GRPO**（DeepSeekMath）：基于组内相对质量优化，不需要 value function
> - **现有 RL 系统的问题**：把 memory 当静态/外部组件 → 无法处理 memory 操作导致的不连续轨迹
> - AgeMem 的贡献：把 RL 直接整合进 memory 管理过程
> - 对比 Memory-R1：Memory-R1 也用 RL 训练 memory，但只管 LTM，不统一 STM

---

## 🔖 Section 总结

### 核心洞察
1. LTM 研究聚焦架构设计（Zettelkasten/KG/pipeline），但缺乏自适应学习能力
2. STM 研究以 RAG 为主，无法根本解决 context 爆炸
3. RL for LLMs 没有把 memory 当作可训练组件 → AgeMem 填补这个空白
4. AgeMem 的定位：在三条线的交汇处，用 RL 统一训练 LTM+STM 管理
