[← 返回 README](../README.md)

# 4 G-Memory

## 📌 预览
G-Memory 的完整工作流：粗粒度检索（embedding + hop expansion）→ 双向遍历（上取 insight，下取 condensed trajectory）→ role-specific 记忆注入 → 任务完成后三层图联合更新。

---

This section outlines the management workflow of G-Memory, as illustrated in Figure 2. Specifically, upon the arrival of a new query $Q$, G-Memory first conducts coarse-grained retrieval to identify pertinent trajectory records (▷ Section 4.1). It then performs bi-directional hierarchical memory traversal: upward to retrieve collective cognitive insights, and downward to distill concrete procedural trajectories (▷ Section 4.2). After the memory-augmented MAS completes the query execution, the hierarchical memory architecture is jointly updated based on environmental feedback, thereby achieving the institutionalization of group knowledge (▷ Section 4.3).

![Figure 2](../images/0ad93707f67a8003b5861cb954fa3686920f867f8069df6fe09cba95875edd3b.jpg)
*Figure 2: The overview of our proposed G-Memory.*

> 💡 **Figure 2 批读**:
> 三阶段流程清晰可见：
> 1. **Coarse-grained Retrieval** (左)：embedding 检索 + graph expansion
> 2. **Bi-directional Traversal** (中)：向上取 insight，向下取 interaction subgraph，然后为每个 agent 定制记忆
> 3. **Hierarchy Update** (右)：任务完成后更新三层图
>
> 注意中间的 role-specific memory assignment——不同角色的 agent 获取不同的记忆切片，这是 G-Memory 区别于单 Agent 记忆的关键设计。

---

## 4.1 Coarse-grained Memory Retrieval

As a plug-in designed for seamless integration into mainstream MAS, G-Memory is triggered when the MAS $\mathcal{G}$ encounters a new user query $Q$. As emphasized in organizational memory theory [1], efficient knowledge retrieval typically begins with broadly relevant schemas prior to more fine-grained access. Following this principle, G-Memory first performs a coarse-grained similarity-based retrieval over the query graph $\mathcal{G}_{\text{query}}$ to efficiently obtain a sketched set of queries $\mathcal{Q}^s$:

$$\mathcal{Q}^S = \arg\text{top-}k_{q_i \in \mathcal{Q} \text{ s.t. } |\mathcal{Q}^S|=k}\left(\frac{\mathbf{v}(Q) \cdot \mathbf{v}(q_i)}{|\mathbf{v}(Q)||\mathbf{v}(q_i)|}\right),$$

where $\mathbf{v}(\cdot)$ maps queries into fixed-length embeddings using models such as MiniLM [73].

> 💡 **第一步：Embedding 检索**: 用 MiniLM 对 query 做 embedding，找 top-k 最相似的历史 query。这是标准 RAG 操作，但只是第一步。

While Equation (4) retrieves semantically similar historical queries, the similarity may be only superficial or noisy. Therefore, G-Memory further enlarges the relevant set via hop expansion on the query graph:

$$\tilde{\mathcal{Q}}^S = \mathcal{Q}^S \cup \big\{Q_k \in \mathcal{Q} \mid \exists Q_j \in \mathcal{Q}^S, Q_k \in \mathcal{N}^+(Q_j) \cup \mathcal{N}^-(Q_j)\big\},$$

where $\tilde{\mathcal{Q}}^s$ is augmented with the 1-hop neighbors of $\mathcal{Q}^s$ on the query graph $\mathcal{G}_{\text{query}}$.

> 💡 **第二步：图扩展**: 在 query graph 上做 1-hop 扩展，把检索到的 query 的邻居也加进来。
> - 为什么？embedding 相似性可能是表面的，但 query graph 的边是基于实际协作经验建立的，更可靠
> - 消融实验（Figure 4a）显示 1-hop 最优，2-hop/3-hop 反而引入噪声
> - 这个设计与 MemGen 的区别：MemGen 用 latent space 做隐式关联，G-Memory 用显式图拓扑做关联

However, it is suboptimal to directly feed these relevant records as input akin to certain single-agent memory systems [41, 37]. On one hand, the excessive context length may overwhelm the LLM; on the other hand, agents in MAS play distinct roles and should be assigned specialized memory tailored to their functions. To address this, the next section introduces a bi-directional processing scheme in G-Memory that operates over both abstract and fine-grained memory levels.

> 💡 **为什么不能直接用？** 两个原因：(1) 上下文太长；(2) 不同角色的 agent 需要不同的记忆。这引出了双向遍历 + role-specific 分配的设计。

---

## 4.2 Bi-directional Memory Traversal

Subsequent to identifying the expanded set of relevant query nodes $\tilde{\mathcal{Q}}^s$ within $\mathcal{G}_{\text{query}}$, G-Memory executes a bi-directional memory traversal to furnish multi-granularity memory support.

**Upward Traversal (Query → Insight).** Specifically, G-Memory first performs an upward traversal ($\mathcal{G}_{\text{query}} \to \mathcal{G}_{\text{insight}}$), retrieving insight nodes that may provide high-level guidance for the current task:

$$\mathcal{Z}^S = \Pi_{\mathcal{Q} \to \mathcal{Z}}(\tilde{\mathcal{Q}}^S), \quad \Pi_{\mathcal{Q} \to \mathcal{Z}}(\mathcal{S}_q) \triangleq \left\{\iota_k \in \mathcal{Z} \mid \Omega_k \cap \mathcal{S}_q \neq \emptyset\right\},$$

where $\Pi_{\mathcal{Q} \to \mathcal{T}}$ is a query-to-insight projector that identifies all the insight nodes whose supporting query sets intersect with the input query set, and the retrieved insights $\mathcal{T}^s$ encapsulate distilled, generalized knowledge potentially relevant for orienting the MAS $\mathcal{G}$'s strategic approach to $Q$.

> 💡 **向上遍历**: 找到所有与检索到的 query 相关的 insight 节点。判断标准很简单——insight 的支持 query 集合与检索到的 query 集合有交集。
> - 这实现了从具体任务到抽象策略的映射
> - 例如：检索到 "put clean egg in microwave" → 关联到 insight "先清洗再放置"

**Downward Traversal (Query → Interaction).** Beyond generalized insights, the fine-grained textual interaction history of the MAS is equally valuable, as it reveals the underlying reasoning patterns that led to successful or failed collaborations [68, 74, 75]. To utilize these concisely, in the downward traversal ($\mathcal{G}_{\text{query}} \to \mathcal{G}_{\text{interaction}}$), G-Memory employs an LLM-facilitated graph sparsifier $S_{\text{LLM}}(\cdot, \cdot)$ to extract the core subgraph that encapsulates essential inter-agent collaboration:

$$\{\hat{\mathcal{G}}_{\text{inter}}^{Q_i}\}_{i=1}^{|M|} = \left\{\mathcal{S}_{\text{LLM}}\big(\mathcal{G}_{\text{inter}}^{(Q_j)}, Q\big) \mid q_j \in \text{arg top-}M\ \mathcal{R}_{\text{LLM}}(Q, q_k')\right\},$$

where $\mathcal{R}_{\text{LLM}}(Q, q_j)$ rates the relevancy of historical queries w.r.t. $Q$, and the sparsifier $\mathcal{S}_{\text{LLM}}(\mathcal{G}_{\text{inter}}^{(Q_j)}, Q)$ constructs a sparsified graph $\hat{\mathcal{G}}_{\text{inter}}^{(Q_j)} = (\hat{\mathcal{U}}^{(Q_j)}, \hat{\mathcal{E}}_u^{(Q_j)})$ from the original $\mathcal{G}_{\text{inter}}^{(Q_j)}$ by identifying and retaining dialogue elements. Please refer to Appendix C for their implementations.

> 💡 **向下遍历的关键：Graph Sparsifier**:
> 1. 先用 LLM 给候选 query 的相关性打分（$\mathcal{R}_{\text{LLM}}$），选 top-M 个
> 2. 对每个选中 query 的 interaction graph，用 LLM 做 sparsification——保留关键对话节点，去掉冗余
> 3. 产出：M 个 condensed interaction subgraph
>
> 这是解决 MAS 轨迹过长问题的核心——不是全部保留也不是全部丢弃，而是智能压缩。
> 
> **与 MemGen 的对比**: MemGen 用 latent representation 隐式压缩所有历史，G-Memory 用 LLM 做显式 sparsification。前者更 end-to-end，后者更可解释。

**Role-Specific Memory Assignment.** Upon completing the bi-directional traversal, we obtain both generalizable insights ($\mathcal{T}^S$) and detailed collaborative trajectories ($\{\hat{\mathcal{G}}_{\text{inter}}^{Q_i}\}_{i=1}^{|M|}$). G-Memory then proceeds to provide specialized memory support for each agent $\mathcal{C} \in \mathcal{V}$ within the MAS $\mathcal{G}$.

$$\text{Mem}_i \gets \Phi\left(\mathcal{Z}^S, \{\hat{\mathcal{G}}_{\text{inter}}^{Q_i}\}_{i=1}^{|M|}; \text{Role}_i, Q\right), \forall C_i = (\text{Base}_i, \text{Role}_i, \text{Mem}_i, \text{Plugin}_i) \in \mathcal{V},$$

where the operator $\Phi(\cdot; \cdot)$ evaluates the utility and relevance of each insight $\iota_k \in \mathcal{T}^S$ and sparsified interaction graph $\hat{\mathcal{G}}_{\text{inter}}^{(Q_j)}$ concerning the agent's specific role $\text{Role}_i$ and the task $Q$ (see Appendix C). Based on this evaluation, $\Phi$ initializes each agent's internal memory state with pertinent historical context before it participates in the subsequent reasoning epochs of the MAS.

> 💡 **Role-Specific 记忆注入**:
> - 不是所有 agent 看同样的记忆——$\Phi$ 根据 agent 的角色筛选相关的 insight 和 trajectory 片段
> - 例如：Solver agent 可能更需要「如何分解任务」的 insight，Ground Truth agent 更需要「常见错误模式」的 trajectory
> - 这是 G-Memory 相比单 Agent 记忆的关键优势——**记忆是 role-aware 的**

It is worth noting that G-Memory is invoked at the onset of solving query $Q$ in our implementation. However, practitioners may flexibly configure more fine-grained invocation strategies, such as at the beginning of each MAS dialogue round or selectively for specific agents, based on their needs.

> 💡 **灵活的调用时机**: 默认在任务开始时一次性注入，但可以改成每轮对话都查一次或只给特定 agent 查。这给了实践者很大灵活性。

---

## 4.3 Hierarchy Memory Update

After completing memory augmentation for each agent, the system $\mathcal{G}$ is executed as outlined in Section 3, yielding a final solution $a^{(T)}$ and receiving environmental feedback, including execution status $\Psi_i \in \{\text{Failed}, \text{Resolved}\}$, token usage, and other performance metrics. Subsequently, G-Memory updates its hierarchical memory architecture to incorporate this new query.

**Interaction Level Update.** At the interaction level, G-Memory traces each agent's utterances to construct the interaction graph $\mathcal{G}_{\text{inter}}^{(Q)}$, which is then stored.

**Query Level Update.** At the query level, a new query node is instantiated and added to the query graph:

$$q_{\text{new}} \gets (Q, \Psi, \mathcal{G}_{\text{inter}}^{(Q)}), \quad \mathcal{N}_{\text{conn}} \gets \mathcal{Q}^{\mathcal{R}} \cup \Big(\bigcup_{\iota_k \in \mathcal{Z}^S} \Omega_k\Big),$$
$$\mathcal{E}_{\text{new}} \gets \{(q_n, q_{\text{new}}) \mid q_n \in \mathcal{N}_{\text{conn}}\}, \quad \mathcal{G}_{\text{query}}^{\text{next}} \gets (\mathcal{Q} \cup \{q_{\text{new}}\}, \mathcal{E}_q \cup \mathcal{E}_{\text{new}}),$$

where edges are established between $q_{\text{new}}$ and (i) the set $\mathcal{Q}^{\mathcal{R}}$ containing the top-$M$ relevant historical queries identified in Equation (7), and (ii) the set of queries $\bigcup_{\iota_k \in \mathcal{T}_{\text{ret}}} \Omega_k$ that support the insights $\mathcal{T}^S$ utilized for solving $Q$.

> 💡 **Query Graph 更新规则**:
> - 新 query 与两类节点建边：(1) 检索时找到的 top-M 相关 query；(2) 检索时用到的 insight 所关联的 query
> - 这意味着 query graph 的拓扑**编码了记忆检索的历史**——谁帮助过谁、谁和谁有关联
> - 随着任务积累，query graph 会形成有意义的聚类结构（见 Appendix B 的可视化）

**Insight Level Update.** Finally, at the insight level, G-Memory integrates the learning from the completed query $Q$ into the insight graph $\mathcal{G}_{\text{insight}} = (\mathcal{T}, \mathcal{E}_i)$. First, possible new insights summarizing the experience are generated and structurally linked via a summarization function $\mathcal{I}(\cdot, \cdot)$ (see prompt in Appendix C) as follows:

$$\iota_{\text{new}} = (\mathcal{I}(\mathcal{G}_{\text{inter}}^{(Q)}, \Psi), \{q_{\text{new}}\}), \quad \mathcal{E}_{i,\text{new}} \gets \{(\iota_k, \iota_{\text{new}}, q_{\text{new}}) \mid \iota_k \in \mathcal{T}^S\}$$
$$\mathcal{G}_{\text{insight}}' \gets (\mathcal{T} \cup \{\iota_{\text{new}}\}, \mathcal{E}_i \cup \mathcal{E}_{i,\text{new}})$$

where edges are added to connect the previously utilized insights which inspires the completion of $Q$ in Equation (6).

> 💡 **Insight 生成机制**:
> - LLM 从完成的 interaction graph + 成功/失败状态中提炼新 insight
> - 新 insight 与之前检索时用到的旧 insight 建边——形成 insight 之间的因果/关联链
> - Prompt 区分成功和失败：失败时对比成功案例提取教训，成功时从模式中提取策略

Afterward, the supporting query sets ($\Omega_k$) for the utilized insights ($\mathcal{T}^s$) are updated to include $q_{\text{new}}$, reflecting their relevance to this successful (or failed) application:

$$\mathcal{T}^{\text{next}} \gets (\mathcal{T} \setminus \mathcal{T}_{\text{ret}}) \cup \{(\kappa_k, \Omega_k \cup \{q_{\text{new}}\}) \mid \iota_k = (\kappa_k, \Omega_k) \in \mathcal{T}_{\text{ret}}\} \cup \{\iota_{\text{new}}\}$$
$$\mathcal{G}_{\text{insight}}^{\text{next}} \gets (\mathcal{T}^{\text{next}}, \mathcal{E}_i \cup \mathcal{E}_{i,\text{new}}),$$

where the final node set $\mathcal{T}^{\text{next}}$ incorporates the new insight and the updated versions of the utilized insights, and the resulting graph $\mathcal{G}_{\text{insight}}^{\text{next}}$ thus encapsulates the integrated knowledge. This continuous update cycle across all hierarchical levels enables G-Memory to learn and adaptively refine its collective memory based on ongoing experience.

> 💡 **完整更新流程总结**:
> 1. **Interaction**: 直接存储完整对话图
> 2. **Query**: 新节点 + 与相关 query 建边 + 与 insight 支持的 query 建边
> 3. **Insight**: LLM 提炼新 insight + 与旧 insight 建边 + 更新旧 insight 的支持 query 集
>
> 这形成了一个**自增长的知识图谱**，随着任务积累越来越丰富。
>
> 💡 **对我们多图医学记忆设计的启发**:
> - **Insight graph** ≈ 临床经验规则/诊断模式（从多个病例中提炼）
> - **Query graph** ≈ 病例索引（哪些病例相似、哪些共享诊断模式）
> - **Interaction graph** ≈ 诊疗对话记录（医生-患者-检查结果的交互）
> - 三层图的更新机制可以直接借鉴：每处理一个新病例，更新三层结构
> - Role-specific 记忆注入也很适合医学场景：不同科室的 agent 看不同的记忆切片

---

## 🔖 Section 总结

### 核心洞察
1. **粗粒度检索 = embedding + graph hop expansion**：结合语义相似性和图拓扑，比纯 RAG 更鲁棒
2. **双向遍历的精髓**：向上取策略（抽象但可迁移），向下取轨迹（具体但可操作），两者互补
3. **Graph Sparsifier 是关键创新**：用 LLM 从冗长的 interaction graph 中提取核心子图，解决 MAS 轨迹过长的核心难题
4. **Role-specific 记忆注入**：不同角色的 agent 获取不同的记忆，这是 MAS 记忆与单 Agent 记忆的本质区别
5. **三层联动更新**：每次任务完成后三层图同时更新，形成自增长的知识库
