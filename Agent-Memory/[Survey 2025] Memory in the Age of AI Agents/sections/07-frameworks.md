[← 返回 README](../README.md)

# 6.2 Open-Source Frameworks

## 📌 预览
25+ 开源记忆框架对比，从 agent-centric（MemGPT, Mem0, MemOS）到 infra-centric（Pinecone, Chroma）形成谱系。

---

## 框架对比总表

| Framework | Factual | Exp. | 多模态 | 架构核心 | 评测基准 |
|-----------|---------|------|--------|---------|---------|
| **MemGPT** (Letta) | ✓ | ✓ | ✗ | 层次化 S/LTM (OS 式) | LoCoMo |
| **Mem0** | ✓ | ✓ | ✗ | graph + vector | LoCoMo |
| **Memobase** | ✓ | ✓ | ✗ | structured profiles | LoCoMo |
| **MIRIX** | ✓ | ✓ | ✓ | 6 种结构化记忆类型 | LoCoMo, MemoryAgentBench |
| **MemoryOS** | ✓ | ✓ | ✗ | 层次化 S/M/LTM | LoCoMo, MemoryBank |
| **MemOS** | ✓ | ✓ | ✗ | tree memory + memcube | LoCoMo, PrefEval, LongMemEval, PersonaMem |
| **Zep** | ✓ | ✓ | ✗ | temporal knowledge graph | LongMemEval |
| **LangMem** | ✓ | ✓ | - | core API + manager | - |
| **Cognee** | ✓ | ✓ | - | knowledge graph | - |
| **SuperMemory** | ✓ | ✓ | - | vector + semantic | - |
| **Memary** | ✓ | - | - | stream + entity store | - |
| **Pinecone** | ✓ | - | ✓ | vector database | - |
| **Chroma** | ✓ | ✗ | ✓ | vector database | - |
| **Weaviate** | ✓ | - | ✓ | vector + graph | - |
| **Second Me** | ✓ | - | - | agent ego | - |

---

## 重点框架深度对比

### MemGPT (now Letta)
- **架构**: 开创性的 OS 式虚拟内存管理——main context (active) + external context (archival)
- **优势**: 理论优雅，首次将 OS 概念引入 agent memory
- **劣势**: 架构复杂，部署成本较高
- **影响**: 启发了 MemOS, MemoryOS 等后续工作

### Mem0
- **架构**: 标准化 memory CRUD 操作 + graph + vector 混合存储
- **Mem0g 变体**: 增加 graph-based 记忆组织
- **优势**: 最工程化、社区采用度最高，已有生产级部署
- **劣势**: 记忆组织相对简单
- **影响**: 事实上的行业标准

### Zep
- **架构**: 独特的 Temporal Knowledge Graph——三层图 ($\mathcal{G}_e$ Episodic + $\mathcal{G}_s$ Semantic + $\mathcal{G}_c$ Community)
- **优势**: 最适合需要时间推理的场景（bi-temporal model 支持精确时间查询）
- **劣势**: 图构建和维护成本较高
- **影响**: 代表了 structured memory 的前沿方向

### MemOS
- **架构**: tree memory + memcube，最全面的评测覆盖
- **优势**: 4 个 benchmark (LoCoMo, PrefEval, LongMemEval, PersonaMem)
- **特色**: MemScheduler 可动态在 parametric/activation/plaintext 记忆间路由

> 💡 **选型建议**:
> - 快速原型 → **Mem0**（API 简洁，社区活跃）
> - 长对话场景 → **MemGPT/Letta**（OS 式管理，无限上下文）
> - 时间敏感场景 → **Zep**（Temporal KG，精确时间推理）
> - 需要全面评测 → **MemOS**（最多 benchmark 覆盖）
> - 多模态需求 → **MIRIX**（唯一同时支持多模态+多评测的框架）

---

## 🔖 Section 总结

### 核心洞察
1. 框架从 agent-centric（MemGPT, Mem0）到 infra-centric（Pinecone, Chroma）形成谱系
2. **Mem0 是社区采用度最高的方案**，Zep 在结构化记忆方面最前沿
3. 多模态记忆框架仍然稀缺——只有 MIRIX 同时支持多模态和有报告评测
4. 大多数框架都在 LoCoMo 上评测，缺乏 embodied/multi-agent 场景覆盖
