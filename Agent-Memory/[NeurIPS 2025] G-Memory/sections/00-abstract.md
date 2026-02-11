[← 返回 README](../README.md)

# Abstract

## 📌 预览
G-Memory 提出三层图记忆（insight/query/interaction）管理 MAS 的长交互历史，通过双向遍历检索多粒度经验，在 5 个 benchmark 上显著提升 MAS 性能。

---

Large language model (LLM)-powered multi-agent systems (MAS) have demonstrated cognitive and execution capabilities that far exceed those of single LLM agents, yet their capacity for self-evolution remains hampered by underdeveloped memory architectures. Upon close inspection, we are alarmed to discover that prevailing MAS memory mechanisms (1) are overly simplistic, completely disregarding the nuanced inter-agent collaboration trajectories, and (2) lack cross-trial and agent-specific customization, in stark contrast to the expressive memory developed for single agents.

> 💡 **问题定位**: MAS 记忆的两大缺陷：(1) 过于简化，忽略了 agent 间的协作轨迹细节；(2) 缺乏跨试次记忆和 agent-specific 定制。这与单 Agent 记忆（如 MemGen、ExpeL）的发展形成鲜明对比。

To bridge this gap, we introduce G-Memory, a hierarchical, agentic memory system for MAS inspired by organizational memory theory [1], which manages the lengthy MAS interaction via a three-tier graph hierarchy: insight, query, and interaction graphs. Upon receiving a new user query, G-Memory performs bi-directional memory traversal to retrieve both high-level, generalizable insights that enable the system to leverage cross-trial knowledge, and fine-grained, condensed interaction trajectories that compactly encode prior collaboration experiences.

> 💡 **核心方法**: 三层图 + 双向遍历：
> - **Insight graph**: 抽象的、可迁移的高层策略洞察
> - **Query graph**: 历史任务的元信息和关联关系
> - **Interaction graph**: 细粒度的 agent 间对话轨迹
>
> 新 query 来了 → 先在 query graph 找相似任务 → 向上遍历取 insights → 向下遍历取 condensed trajectories。

Upon task execution, the entire hierarchy evolves by assimilating new collaborative trajectories, nurturing the progressive evolution of agent teams. Extensive experiments across five benchmarks, three LLM backbones, and three popular MAS frameworks demonstrate that G-Memory improves success rates in embodied action and accuracy in knowledge QA by up to 20.89% and 10.12%, respectively, without any modifications to the original frameworks. Our codes are available at https://github.com/bingreeky/GMemory.

> 💡 **关键结果**: Embodied +20.89%, Knowledge QA +10.12%，且是 plug-and-play 无需修改原框架。代码已开源。
>
> 💡 **对我们的启发**: G-Memory 的三层图结构对多图医学记忆设计很有启发——可以类比为：insight graph ≈ 疾病诊断规则/临床经验，query graph ≈ 病例关联，interaction graph ≈ 具体诊疗对话轨迹。层次化组织是关键。

---

## 🔖 Section 总结

### 核心洞察
1. MAS 记忆问题的本质是**协作轨迹太长太复杂**，现有方法要么丢弃要么塞全部
2. 解法：层次化压缩——三层图分别存不同粒度的信息
3. 双向遍历同时获取「抽象策略」和「具体操作经验」
