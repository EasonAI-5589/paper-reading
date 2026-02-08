[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
阐述 agent memory 需要新分类体系的两大动机：(1) 现有分类法滞后于 2025 年的方法论爆发，(2) 概念碎片化导致 "memory" 一词歧义严重。提出 5 个核心研究问题。

---

The past two years have witnessed the overwhelming evolution of increasingly capable large language models (LLMs) into powerful AI agents. These foundation-model-powered agents have demonstrated remarkable progress across diverse domains such as deep research, software engineering, and scientific discovery, continuously advancing the trajectory toward artificial general intelligence (AGI).

> 💡 **背景**: 2023–2025 是 LLM → Agent 的关键转型期。Agent ≠ 能对话的 LLM，而是具备 reasoning + planning + memory + tool-use 的完整系统。

Although early conceptions of "agents" were highly heterogeneous, a growing consensus has since emerged within the community: beyond a pure LLM backbone, an agent is typically equipped with capabilities such as reasoning, planning, perception, memory, and tool-use. Some of these abilities, such as reasoning and tool-use, have been largely internalized within model parameters through reinforcement learning, while some still depend heavily on external agentic scaffolds.

> 💡 **关键洞察**: Reasoning 和 tool-use 已经通过 RL（如 DeepSeek-R1）被内化到参数里了，但 **memory 仍高度依赖外部脚手架**。这是 memory 研究的独特价值——它是 agent 能力中最难被参数化的部分。

Among these agentic faculties, memory stands out as a cornerstone, explicitly enabling the transformation of static LLMs, whose parameters cannot be rapidly updated, into adaptive agents capable of continual adaptation through environmental interaction.

> 💡 **Memory 的核心价值**: 让"参数不能快速更新的静态 LLM" → "能通过环境交互持续适应的动态 Agent"。

From an application perspective, numerous domains demand agents with proactive memory management rather than ephemeral, forgetful behaviors: personalized chatbots, recommender systems, social simulations, and financial investigations all rely on the agent's ability to process, store, and manage historical information. From a developmental standpoint, one of the defining aspirations of AGI research is to endow agents with the capacity for continual evolution through environment interactions, a capability fundamentally grounded in agent memory.

---

### Agent Memory Needs A New Taxonomy

The motivation for a new taxonomy and survey is twofold:

**① Limitations of Existing Taxonomies**: While several recent surveys have provided valuable overviews of agent memory, their taxonomies were developed prior to a number of rapid methodological advances. For example, emerging directions in 2025, such as memory frameworks that distill reusable tools from past experiences, or memory-augmented test-time scaling methods, remain underrepresented in earlier classification schemes.

> 💡 **批注**: 2025 年新范式举例——把经验蒸馏成可复用工具（skill memory, Voyager）、用 memory 增强 test-time scaling（Dynamic Cheatsheet）。旧分类无法涵盖。

**② Conceptual Fragmentation**: Researchers often find that papers claiming to study "agent memory" differ drastically in implementation, objectives, and underlying assumptions. The proliferation of diverse terminologies (declarative, episodic, semantic, parametric memory, etc.) further obscures conceptual clarity.

> 💡 **碎片化实例**: MemGPT 的 memory = OS 式虚拟内存管理；Voyager 的 memory = 代码技能库；Reflexion 的 memory = 自我反思日志。都叫 "memory" 但完全不同。

---

### Key Questions

This survey aims to address the following key questions:

| # | 问题 | 对应章节 |
|---|------|---------|
| ❶ | Agent memory 如何定义？与 LLM memory / RAG / context engineering 的关系？ | §2 |
| ❷ | Agent memory 可以有哪些架构/表征形式？ | §3 |
| ❸ | 为什么 agent 需要记忆？记忆服务什么功能？ | §4 |
| ❹ | Agent memory 如何运作、适应和演化？ | §5 |
| ❺ | Agent memory 研究的前沿方向？ | §7 |

---

### Contributions

1. 提出 Forms–Functions–Dynamics 三维分类法
2. 深入讨论不同记忆形式与功能目的之间的匹配关系
3. 探讨新兴研究方向（RL 融合、多模态记忆等）
4. 汇编 30+ benchmarks 和 25+ 开源框架资源

---

## 🔖 Section 总结

### 核心洞察
1. Memory 是 agent 能力中**最难被参数化**的部分，仍高度依赖外部脚手架
2. 现有分类法（long/short-term）已过时，无法覆盖 2025 年的方法论爆发
3. 概念碎片化是当前最大挑战——不同论文的 "memory" 可能指完全不同的东西
4. 本文的野心：建立 "conceptual foundation for rethinking memory as a first-class primitive"
