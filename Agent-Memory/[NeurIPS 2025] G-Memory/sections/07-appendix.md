[← 返回 README](../README.md)

# Appendix

## 📌 预览
实验细节（数据集/Baseline/MAS 设置）、额外实验结果（Qwen-7b/14b 主表 + 学习曲线 + 额外 cost 分析）、Prompt 模板、与 A-Mem/Mem0 的讨论。

---

## A Experimental Details

### A.1 Dataset Descriptions

- **ALFWorld** [78]: 文本型 embodied 环境，家务任务，agent 通过自然语言指令导航和交互。评估指标：success rate。
- **ScienceWorld** [79]: 文本型 embodied 环境，科学实验任务。评估指标：progress rate。
- **PDDL**: 来自 AgentBoard [80] 的策略游戏数据集，agent 用 PDDL 表达式完成复杂任务。评估指标：progress rate。
- **HotpotQA** [76]: 多跳 QA 数据集。评估指标：exact match accuracy。
- **FEVER** [77]: 事实验证数据集。评估指标：exact match accuracy。

> 💡 **数据集选择评价**: 三个领域（embodied/QA/game）覆盖了 MAS 的主要应用场景。但缺少代码生成和社交模拟等场景。

### A.2 Baseline Setup

- **Voyager**: 从 Minecraft agent 改编，为 MAS 适配了 agent-specific 历史检索
- **MemoryBank**: 基于 Ebbinghaus 遗忘曲线的记忆衰减机制
- **Generative**: 基于 Generative Agents [19]，包含 raw observation + reflective memory
- **MetaGPT-M**: 仅 inside-trial memory
- **ChatDev-M**: Inside-trial + 简单 cross-trial（只存最终结果）
- **MacNet-M**: 仅保留上一轮的 final answer，丢弃所有对话轨迹

> 💡 **Baseline 对比总结**:
> | 方法 | Cross-trial | 保留轨迹 | Role-specific | 层次化 |
> |------|:---------:|:-------:|:----------:|:-----:|
> | Voyager | ✅ | ❌ | 部分 | ❌ |
> | MemoryBank | ✅ | ❌ | ❌ | ❌ |
> | Generative | ✅ | 部分 | ❌ | ✅(2层) |
> | MetaGPT-M | ❌ | ❌ | ❌ | ❌ |
> | ChatDev-M | ✅(仅结果) | ❌ | ❌ | ❌ |
> | MacNet-M | ❌ | ❌ | ❌ | ❌ |
> | **G-Memory** | **✅** | **✅** | **✅** | **✅(3层)** |

### A.3 Multi-agent System Setup

**AutoGen**: Solver + Ground Truth + Executor 三 agent 架构。

**DyLAN**: 辩论式框架 + agent importance score 做 early stopping。3 个辩论 agent + 1 个 ranker。

**MacNet**: 去中心化，无中央 agent，edge agent 在 agent 间传递指令。Random graph 拓扑，5 个 agent。

> 💡 **三个 MAS 框架覆盖了不同的协作模式**:
> - AutoGen: 流水线式（solver → ground truth → executor）
> - DyLAN: 辩论式（多 agent 辩论 + ranking）
> - MacNet: 去中心化图式（edge agent 协调）
> - G-Memory 在所有模式下都有效，证明了其通用性

---

## B Additional Experiment Results

### B.1 学习曲线

![Figure 6a](../images/753bb08f3bea1958b96c21ce50a816145b59876dda12e895d4c186125d2396fa.jpg)
*The performance trajectory of AutoGen on ALFWorld.*

![Figure 6b](../images/79209b2d36e0a786a62c4cce467909a1f8d605ef0b30ddd24099dfc61461450f.jpg)
*The performance trajectory of DyLAN on ALFWorld.*

![Figure 6c](../images/ac921f7cfde5c3de94c6286565780421f8412b04020023560449a1c06d376d31.jpg)
*The performance trajectory of MacNet on ALFWorld.*

> 💡 **学习曲线批读**: G-Memory 在所有 MAS 框架下都展现了更快的学习速度和更高的性能天花板——说明记忆确实在驱动 MAS 的渐进式自进化。

### B.2 额外 Cost 分析

![Figure 7](../images/41f2ada53be38bac9471e7675c2e801a143a98aa3512acc02fc1997fc0221c80.jpg)
*Figure 7: Cost analysis of G-Memory across additional benchmarks.*

### B.3 Case Study

#### Insight Graph 可视化

![Figure 8a](../images/f52917e70afc146460ba70cbaa422e8d116d438ee7870f49b2fb669fc665f3ea.jpg)
*Insight graph on gpt-4o-mini + MacNet + ALFWorld.*

![Figure 8b](../images/0f23a78f2fca3cfdb15e6cb6493609452f002a0a12723b7366133af37d5295dd.jpg)
*Insight graph on gpt-4o-mini + DyLAN + ALFWorld.*

> 💡 **Insight Graph 可视化批读**: 
> - 同类任务（如 clean 类、heat 类）的 insight 节点形成密集连接的子图
> - 不同任务类型之间也有跨类连接——表示可迁移的策略
> - 这证明 insight graph 确实在学习有意义的结构化知识

#### Query Graph 可视化

![Figure 9](../images/ca9e7b529735c21364d1ab09d9741c63f3fcead9412c3f02ed6f48a79593ace8.jpg)
*Figure 9: Query graph optimized from ALFWorld dataset.*

![Figure 10](../images/256aa4b72781d8588330497a91044acd7b3cac77cbea8e74afb7d00a4087e3a1.jpg)
*Figure 10: Query graph optimized from SciWorld dataset.*

![Figure 11](../images/d1c032ca0c213a52e84067bf2dbc7292c15ef802fef11a3928485ba3a71682e4.jpg)
*Figure 11: Query graph optimized from PDDL dataset.*

> 💡 **Query Graph 可视化批读**: 语义相似的 query 自动聚类，形成紧密连接的子图，稀疏的跨集群连边表示跨任务灵感。这与 G-Memory 1-hop 扩展的设计一致——1-hop 足以捕捉同类任务的关联。

---

## C Prompt Set

论文提供了完整的 prompt 模板，包括：

1. **Query Relevance Filtration**: LLM 对两个 query 的相关性打分（1-10）
2. **Graph Sparsifier**: 从成功轨迹中提取关键步骤，过滤 "Nothing happens" 等错误动作
3. **Insight Summarization**: 两种模式——(a) 对比失败和成功轨迹提取教训；(b) 从成功案例中提取共性策略
4. **Insight Merge**: 合并冗余的 insight，限制总数

> 💡 **Prompt 设计要点**:
> - Sparsifier 特别关注 "Nothing happens" 的动作——在 embodied 环境中这表示错误操作
> - Insight 生成区分成功和失败：失败时做对比分析，成功时做模式提取
> - Merge 机制控制 insight 数量增长——但论文没有详细分析 merge 的频率和效果

---

## D Discussion with Related Works

In this section, we further discuss the relationship between G-Memory and several recent agent memory frameworks.

**A-Mem [61]**: A-Mem 面向单 Agent chatbot，强调原子化记忆构建；G-Memory 面向 MAS 任务执行，强调从协作轨迹中提取可复用策略。

**Mem0 [62]**: Mem0 也用图结构，但它的图更接近知识图谱（实体+关系），而 G-Memory 的图是 agent-centric 的（轨迹+决策+协作模式）。

> 💡 **与其他记忆系统的区别总结**:
> | 维度 | A-Mem | Mem0 | G-Memory |
> |------|-------|------|----------|
> | 目标场景 | 单 Agent chatbot | 单 Agent chatbot | **MAS 任务执行** |
> | 图结构语义 | 原子记忆关联 | 知识图谱(实体-关系) | **Agent 协作轨迹** |
> | 记忆粒度 | 原子级 | 事实级 | **三层层次化** |
> | Role-specific | ❌ | ❌ | **✅** |

---

## 🔖 Section 总结

### 核心洞察
1. 三个 MAS 框架（流水线/辩论/去中心化）都从 G-Memory 获益，证明方法的通用性
2. 学习曲线显示 G-Memory 确实在驱动 MAS 的渐进式自进化（越用越好）
3. Insight/Query graph 的可视化证明 G-Memory 学到了有意义的结构化知识
4. Prompt 设计精心区分了成功/失败场景，并有 insight merge 机制控制增长
