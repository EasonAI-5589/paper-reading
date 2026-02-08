# Summary: Memory in the Age of AI Agents

## 全文要点概括

### 1. 为什么需要这篇 Survey？
- Agent Memory 领域高度碎片化：不同论文说的 "memory" 可能指 KV cache、knowledge graph、experience replay 等完全不同的东西
- 传统 long/short-term 二分法已无法覆盖 2025 年的方法论爆发（skill memory、test-time scaling memory 等新范式）
- 需要一个统一框架来组织 400+ 篇相关文献

### 2. 核心分类框架：Forms × Functions × Dynamics

#### Forms（记忆存在哪里？）
| 形式 | 特点 | 适合场景 |
|------|------|---------|
| **Token-level** (1D/2D/3D) | 透明可编辑，即插即用 | 对话、个性化、推荐、法律合规 |
| **Parametric** (Internal/External) | 零延迟推理，强泛化 | 角色扮演、数学推理、代码 |
| **Latent** (Generate/Reuse/Transform) | 高密度，天然隐私 | 多模态、边缘部署、隐私敏感 |

#### Functions（记忆为什么存在？）
| 功能 | 核心问题 | 关键属性 |
|------|---------|---------|
| **Factual Memory** | Agent 知道什么？ | 一致性 + 连贯性 + 适应性 |
| **Experiential Memory** | Agent 如何进步？ | Case → Strategy → Skill 抽象阶梯 |
| **Working Memory** | Agent 当前在想什么？ | 被动缓冲 → 主动工作空间 |

#### Dynamics（记忆怎么运作？）
| 阶段 | 核心问题 | 关键趋势 |
|------|---------|---------|
| **Formation** | 如何提取记忆？ | 固定 prompt → RL 可训练蒸馏 |
| **Evolution** | 如何精炼记忆？ | 破坏性替换 → 软更新 → RL 自适应 |
| **Retrieval** | 如何利用记忆？ | 静态搜索 → 动态认知过程 |

### 3. 概念边界

| Agent Memory vs. | 关系 | 区分标准 |
|---|---|---|
| **LLM Memory** | Agent Memory ⊃ 大部分 LLM Memory | 纯架构优化（RWKV, Mamba）不算 Agent Memory |
| **RAG** | 在 Agentic RAG 处边界最模糊 | RAG 知识库预先存在；Agent Memory 由 agent 自建 |
| **Context Engineering** | Working memory 处完全重合 | CE = 资源管理；AM = 认知建模 |

### 4. 开源框架格局
- **Mem0**: 社区采用度最高，标准化 CRUD，graph+vector 混合
- **MemGPT (Letta)**: 开创性 OS 式架构，理论优雅但复杂
- **Zep**: Temporal KG 最适合时间推理
- **MemOS**: 评测覆盖最广（4 个 benchmark）

### 5. 八大前沿方向（按优先级）
1. 🔥🔥🔥 **RL 融合**：从 RL-free → RL-assisted → Fully RL-driven，端到端训练全记忆生命周期
2. 🔥🔥🔥 **生成式记忆**：从 "检索已有" → "按需合成"，类似人类 constructive memory
3. 🔥🔥 **自动化管理**：Memory management 成为 agentic capability（通过工具调用显式推理）
4. 🔥🔥 **可信记忆**：隐私保护 + 可解释性 + 抗幻觉，实际部署的前提条件
5. 🔥 **多智能体共享**：从孤立记忆到 agent-aware 共享记忆
6. 🔥 **多模态记忆**：视觉最成熟，音频等仍欠探索，需要 omnimodal 统一方案
7. 🔥 **世界模型记忆**：从被动缓存到主动状态维护（Dual-System: Fast/Slow）
8. 💡 **人类认知连接**：offline consolidation（"agent 睡眠"进行记忆重组）

### 6. 对 Eason 研究的启示
- **Memory 是 agent 最难被参数化的能力**，仍高度依赖外部脚手架 → 这是一个充满机会的研究方向
- **RL + Memory** 是目前最热且最有前景的交叉点（Mem-α, Memory-R1, Context-Folding, MemSearcher 等）
- **生成式记忆**（MemGen 的 latent memory trigger/weaver）代表了从检索到生成的范式跃迁
- 实际系统应该**混合使用多种记忆形式**——Token-level 保证可解释性，Latent 保证效率，Parametric 保证泛化
