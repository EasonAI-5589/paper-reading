[← 返回 README](../README.md)

# Abstract

## 📌 预览
提出 Forms–Functions–Dynamics 三维分类框架，系统梳理 Agent Memory 研究全景，并区分其与 LLM Memory / RAG / Context Engineering 的概念边界。

---

Memory has emerged, and will continue to remain, a core capability of foundation model-based agents. It underpins long-horizon reasoning, continual adaptation, and effective interaction with complex environments. As research on agent memory rapidly expands and attracts unprecedented attention, the field has also become increasingly fragmented. Existing works that fall under the umbrella of agent memory often differ substantially in their motivations, implementations, assumptions, and evaluation protocols, while the proliferation of loosely defined memory terminologies has further obscured conceptual clarity. Traditional taxonomies such as long/short-term memory have proven insufficient to capture the diversity and dynamics of contemporary agent memory systems.

> 💡 **核心痛点**: "Agent memory" 太泛——有人说的是 KV cache 管理，有人说的是 knowledge graph，有人说的是 experience replay。传统的 long/short-term 二分法已无法覆盖这些多样性。

This survey aims to provide an up-to-date and comprehensive landscape of current agent memory research. We begin by clearly delineating the scope of agent memory and distinguishing it from related concepts such as LLM memory, retrieval augmented generation (RAG), and context engineering. We then examine agent memory through the unified lenses of forms, functions, and dynamics. From the perspective of forms, we identify three dominant realizations of agent memory, namely token-level, parametric, and latent memory. From the perspective of functions, we move beyond coarse temporal categorizations and propose a finer-grained taxonomy that distinguishes factual, experiential, and working memory. From the perspective of dynamics, we analyze how memory is formed, evolved, and retrieved over time as agents interact with their environments.

> 💡 **三维框架核心**:
> | 维度 | 问题 | 分类 |
> |------|------|------|
> | Forms | 记忆存在哪里？ | Token-level / Parametric / Latent |
> | Functions | 记忆为什么存在？ | Factual / Experiential / Working |
> | Dynamics | 记忆怎么运作？ | Formation / Evolution / Retrieval |

To support empirical research and practical development, we compile a comprehensive summary of representative benchmarks and open source memory frameworks. Beyond consolidation, we articulate a forward-looking perspective on emerging research frontiers, including automation-oriented memory design, the deep integration of reinforcement learning with memory systems, multimodal memory, shared memory for multi-agent systems, and trustworthiness issues. We hope this survey serves not only as a reference for existing work, but also as a conceptual foundation for rethinking memory as a first-class primitive in the design of future agentic intelligence.

> 💡 **定位**: "Memory as a first-class primitive"——记忆不是 agent 的附属模块，而是和 reasoning、planning 同等重要的核心原语。

---

![Figure 1](../images/59c7dcb89b84c5659faf913c40baa21d0d721fb0004a4a3bb8b6dfab62df4dc9.jpg)
*Figure 1: Overview of agent memory organized by the unified taxonomy of forms (Section 3), functions (Section 4), and dynamics (Section 5).*

> 💡 **Figure 1 批读**:
> 全文核心概览图。横轴 = Forms（Token-level → Parametric → Latent），纵轴隐含 Functions，代表性系统被映射到这个二维空间中：
> - MemGPT, Mem0 → Token-level + Factual
> - Voyager, ExpeL → Token-level + Experiential
> - Titans, MemGen → Latent + Working
> - ROME, Character-LM → Parametric + Factual

---

## 🔖 Section 总结

### 核心洞察
1. Agent Memory 领域高度碎片化，需要统一分类框架
2. Forms × Functions × Dynamics 三维分类法比传统 long/short-term 更精确
3. 本文覆盖 400+ 篇文献，定位 memory 为 agent 的 "first-class primitive"
