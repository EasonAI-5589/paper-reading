[← 返回 README](../README.md)

# 2. Preliminaries: Formalizing Agents and Memory

## 📌 预览
形式化定义 LLM Agent 和 Memory System，提出 Formation/Evolution/Retrieval 三算子框架。系统区分 Agent Memory 与 LLM Memory / RAG / Context Engineering 的概念边界。

---

## 2.1 LLM-based Agent Systems

**Agents and Environment**: $N$ 个 agent 在状态空间 $S$ 中交互。每个 agent $i$ 观察 $o_t^i = O_i(s_t, h_t^i, \mathcal{Q})$，依据策略

$$a_t = \pi_i(o_t^i, m_t^i, \mathcal{Q})$$

做出动作，其中 $m_t^i$ 是 **memory-derived signal**。

> 💡 **批注**: Memory 直接出现在 agent policy 的输入中。没有 $m_t^i$，agent 就只能基于当前观察 $o_t^i$ 做决策——这就是 memoryless agent，无法利用任何历史信息。

**Action Space** 五种类型：自然语言生成、工具调用、规划动作、环境控制、通信动作。虽然语义多样，但统一通过自回归 LLM 生成。

---

## 2.2 Agent Memory Systems

Memory state $\mathcal{M}_t \in \mathbb{M}$，可以是 text buffer、key-value store、vector DB、graph 或任意混合。不预设内部结构。

**Memory Lifecycle 三算子**：

| 算子 | 公式 | 作用 |
|------|------|------|
| **Formation** $F$ | $\mathcal{M}_{t+1}^{form} = F(\mathcal{M}_t, \phi_t)$ | 从交互产物 $\phi_t$ 中选择性提取有未来价值的信息 |
| **Evolution** $E$ | $\mathcal{M}_{t+1} = E(\mathcal{M}_{t+1}^{form})$ | 合并冗余、解决冲突、丢弃低价值信息、重构索引 |
| **Retrieval** $R$ | $m_t^i = R(\mathcal{M}_t, o_t^i, \mathcal{Q})$ | 构建任务感知查询，返回相关记忆给 agent policy |

> 💡 **关键设计哲学**: **短期记忆和长期记忆不是通过架构分离实现的，而是通过 F/E/R 的时间调用模式自然涌现的**。例如：
> - 每步都做 retrieval = working memory 效果
> - 只在任务开头做一次 retrieval = long-term memory 效果
> - 每步都做 formation = episodic logging
> - 任务结束时做一次 formation = cross-trial consolidation
>
> 这是一个非常优雅的统一视角，避免了人为的架构分离。

---

## 2.3 Comparing Agent Memory with Other Key Concepts

![Figure 2](../images/688e9237c75530ba778e871307f894cf0be1bee9498671badfb9b79b4653dcae.jpg)
*Figure 2: Conceptual comparison of Agent Memory with LLM Memory, RAG, and Context Engineering.*

> 💡 **Figure 2 批读**: Venn 图展示四个概念的交集与差异。Agent Memory 的独特核心 = "persistent and self-evolving cognitive state that integrates factual knowledge and experience"。各交集区域列出了共享的技术实现（KV reuse, graph retrieval 等）。

---

### 2.3.1 Agent Memory vs. LLM Memory

Agent Memory **几乎完全包含**传统 "LLM Memory"——2023 年的 MemGPT、MemoryBank 自称 "LLM memory"，用今天的术语就是 agent memory。

**不属于 Agent Memory 的 LLM Memory**：
- 架构级修改（RWKV, Mamba, diffusion-based LMs）
- KV cache 管理和压缩
- 长上下文处理机制（attention sparsity）

> 💡 **区分标准**: 是否支持 (1) 跨任务持久化 (2) 环境驱动适应 (3) 主动记忆操作（F/E/R）。纯粹优化模型内部表示能力的 → LLM Memory；涉及外部持久化+自主管理的 → Agent Memory。

---

### 2.3.2 Agent Memory vs. RAG

| 维度 | RAG | Agent Memory |
|------|-----|-------------|
| 数据来源 | 外部静态知识库 | Agent 自身交互产生的信息 |
| 时间特性 | 单次任务调用 | 跨任务、多轮持续演化 |
| 知识库性质 | 预先存在、外部维护 | Agent 在交互中自己构建 |
| 代表 Benchmark | HotpotQA, MuSiQue | LoCoMo, LongMemEval |

**模糊地带**：
- HippoRAG/HippoRAG2 被 RAG 和 memory 社区同时引用
- **Agentic RAG** 与 agent memory 边界最模糊——两者都涉及自主决定何时/如何检索

> 💡 **实操区分**: RAG 的知识库通常预先存在且由外部维护；Agent Memory 的记忆库是 agent 在交互中自己构建、演化、管理的。当 RAG 系统开始自主更新知识库时，它就变成了 Agent Memory。

按 RAG 子类分析重叠：
- **Modular RAG**（检索流水线）→ 对应 Agent Memory 的 retrieval 阶段
- **Graph RAG**（图结构知识）→ 对应 Agent Memory 的 planar/hierarchical token-level memory
- **Agentic RAG**（自主检索决策）→ 与 Agent Memory 概念最接近，区别仅在于是否维护 internal persistent memory

---

### 2.3.3 Agent Memory vs. Context Engineering

| 维度 | Context Engineering | Agent Memory |
|------|------|------|
| 范式 | **资源管理** | **认知建模** |
| 核心问题 | 如何在有限窗口内最优排列信息？ | Agent 知道什么、经历了什么、如何演化？ |
| 时间范围 | 单次推理窗口内 | 跨会话持久化 |
| 技术焦点 | 压缩、排序、格式化、tool 调度 | 存储、合并、遗忘、检索 |

**重叠区**：管理 working memory 时两者完全重合——滚动摘要、token 修剪、重要性选择既是 context engineering 也是 working memory management。

> 💡 **精彩类比**: Context Engineering 构建"外部脚手架"（enables perception and action under resource constraints），Agent Memory 构成"内部基质"（supports learning, adaptation, and autonomy）。前者优化 agent-model 瞬时接口，后者维持跨越任何单个上下文窗口的持久认知状态。

---

## 🔖 Section 总结

### 核心洞察
1. F/E/R 三算子统一定义了 agent memory 的生命周期
2. **短期/长期记忆是 F/E/R 调用时间模式的涌现**，而非架构分离
3. Agent Memory ⊃ 大部分 LLM Memory，但不包括纯架构优化
4. Agent Memory 与 RAG 在 Agentic RAG 处边界最模糊
5. Agent Memory vs. Context Engineering = 认知建模 vs. 资源管理
