[← 返回 README](../README.md)

# 8. Conclusion

## 📌 预览
全文核心论点浓缩：Memory 不是辅助存储，而是 agent 实现时间连贯性、持续适应和长程能力的本质基质。

---

This survey has examined agent memory as a foundational component of modern LLM-based agentic systems. By framing existing research through the unified lenses of forms, functions, and dynamics, we have clarified the conceptual landscape of agent memory and situated it within the broader evolution of agentic intelligence.

On the level of forms, we identify three principal realizations: token-level, parametric, and latent memory, each of which has undergone distinct and rapid advances in recent years, reflecting fundamentally different trade-offs in representation, adaptability, and integration with agent policies.

On the level of functions, we move beyond the coarse long-term versus short-term dichotomy prevalent in prior surveys, and instead propose a more fine-grained and encompassing taxonomy that distinguishes factual, experiential, and working memory according to their roles in knowledge retention, capability accumulation, and task-level reasoning.

> 💡 **核心论点**: Memory 不是辅助存储机制，而是 agent 实现以下三种能力的 essential substrate：
> - **Temporal coherence** — 跨时间的行为一致性
> - **Continual adaptation** — 持续学习和自我进化
> - **Long-horizon competence** — 长程任务的可靠执行

Together, these perspectives reveal that memory is not merely an auxiliary storage mechanism, but an essential substrate through which agents achieve temporal coherence, continual adaptation, and long-horizon competence.

Beyond organizing prior work, we have identified key challenges and emerging directions that point toward the next stage of agent memory research. In particular, the increasing integration of reinforcement learning, the rise of multimodal and multi-agent settings, and the shift from retrieval-centric to generative memory paradigms suggest a future in which memory systems become fully learnable, adaptive, and self-organizing. Such systems hold the potential to transform large language models from powerful but static generators into agents capable of sustained interaction, self-improvement, and principled reasoning over time.

> 💡 **三大趋势**:
> 1. **RL 驱动** → 可学习的记忆管理（从 heuristic 到 learned policy）
> 2. **多模态/多智能体** → 异构记忆协同（从文本到 omnimodal）
> 3. **检索→生成** → 构建式记忆重组（从 verbatim retrieval 到 constructive generation）

We hope this survey provides a coherent foundation for future research and serves as a reference for both researchers and practitioners. As agentic systems continue to mature, the design of memory will remain a central and open problem, one that is likely to play a decisive role in the development of robust, general, and enduring artificial intelligence.

---

## 🔖 全文核心 Takeaway

1. **Memory = first-class primitive**，与 reasoning/planning 同级，不是可选附件
2. **Forms × Functions × Dynamics** 三维框架是目前最完整的 agent memory 分类体系
3. **概念边界已澄清**: Agent Memory ⊃ 大部分 LLM Memory; ≈ Agentic RAG（边界模糊）; ∩ Context Engineering（working memory 处重合）
4. **RL 是统一全记忆生命周期的关键技术路径**——从 formation 到 evolution 到 retrieval
5. **生成式记忆**（而非检索式）将成为下一个主流范式
6. **可信记忆**（隐私+可解释+抗幻觉）是实际部署的前提条件
7. 开源框架生态已初步成型：**Mem0** 社区最广，**Zep** 结构化最强，**MemOS** 评测最全
