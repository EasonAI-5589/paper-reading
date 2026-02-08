# Memory in the Age of AI Agents: A Survey

**作者**: Yuyang Hu†, Shichun Liu†, Yanwei Yue†, Guibin Zhang†◊ (Project Organizer), Boyang Liu, ... (75+ authors)  
**来源**: arXiv 2512.13564 (2025.12)  
**机构**: National University of Singapore, Renmin University of China, Fudan University, Peking University, Oxford University 等  
**链接**: [arXiv](https://arxiv.org/abs/2512.13564) | [GitHub](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

## 一句话总结
提出 **Forms–Functions–Dynamics** 三维统一分类框架，系统梳理 LLM Agent 记忆系统的载体形式（Token-level / Parametric / Latent）、功能用途（Factual / Experiential / Working）和动态生命周期（Formation / Evolution / Retrieval），并明确区分 Agent Memory 与 LLM Memory、RAG、Context Engineering 的概念边界。

## 核心贡献
1. **三维统一分类法**：从 Forms × Functions × Dynamics 三个正交维度组织现有工作，超越传统 long/short-term 二分法
2. **概念边界澄清**：首次系统区分 Agent Memory vs. LLM Memory vs. RAG vs. Context Engineering
3. **形式-功能匹配分析**：深入讨论 Token-level / Parametric / Latent 各自适合的任务场景
4. **全面资源汇编**：整理 30+ benchmarks 和 25+ 开源框架（MemGPT, Mem0, Zep 等）
5. **8 大前沿展望**：RL 融合、生成式记忆、自动化管理、多模态、多智能体共享、世界模型、可信记忆、人类认知连接

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1（全景分类图） |
| [01 - Introduction](sections/01-introduction.md) | 动机：为什么需要新分类法 + 5 个核心问题 |
| [02 - Scope](sections/02-scope.md) | Agent/Memory 形式化 + 与 LLM Memory / RAG / Context Engineering 区别 (Figure 2) |
| [03 - Memory Forms](sections/03-memory-forms.md) | Token-level (1D/2D/3D) + Parametric + Latent (Figure 3-5, Table 1-3) |
| [04 - Memory Functions](sections/04-memory-functions.md) | Factual + Experiential + Working Memory (Figure 6-7, Table 4-6) |
| [05 - Memory Dynamics](sections/05-memory-dynamics.md) | Formation + Evolution + Retrieval (Figure 8-10, Table 7) |
| [06 - Benchmarks](sections/06-benchmarks.md) | 30+ memory/lifelong/self-evolving benchmarks (Table 8) |
| [07 - Frameworks](sections/07-frameworks.md) | MemGPT / Mem0 / Zep 等 25+ 开源框架对比 (Table 9) |
| [08 - Frontiers](sections/08-frontiers.md) | 8 大前沿方向：RL 融合、生成式记忆、多模态等 (Figure 11) |
| [09 - Conclusion](sections/09-conclusion.md) | 总结 + 全文核心 takeaway |

## 关键数字

| 指标 | 数值 |
|------|------|
| 引用文献数 | 400+ |
| 分类维度 | 3（Forms × Functions × Dynamics） |
| 记忆形式 | 3 大类 9 小类 |
| 功能分类 | 3 大类（Factual / Experiential / Working） |
| 动态过程 | 3 阶段（Formation → Evolution → Retrieval） |
| Formation 操作类型 | 5 种 |
| 开源框架 | 25+ |
| Benchmark | 30+ |
| 前沿方向 | 8 个 |

## 分类框架速览

```
                        Forms (载体)
                    ┌─────────────────┐
                    │  Token-level    │ ← 外部可见、可编辑的离散单元
                    │   ├── 1D Flat   │    (MemGPT, Mem0, Voyager)
                    │   ├── 2D Planar │    (A-MEM, KGT, PREMem)
                    │   └── 3D Hier.  │    (GraphRAG, HippoRAG, Zep)
                    ├─────────────────┤
                    │  Parametric     │ ← 编码在模型参数中
                    │   ├── Internal  │    (ROME, Character-LM)
                    │   └── External  │    (K-Adapter, WISE, LoRA)
                    ├─────────────────┤
                    │  Latent         │ ← 隐藏状态/KV cache/激活值
                    │   ├── Generate  │    (Gist, Titans, MemGen)
                    │   ├── Reuse     │    (Memorizing Transformers)
                    │   └── Transform │    (SnapKV, H2O, PyramidKV)
                    └─────────────────┘

    Functions (用途)                    Dynamics (运作)
┌───────────────────┐            ┌───────────────────┐
│ Factual Memory    │            │ Formation         │
│  "Agent知道什么"   │            │  提取: 摘要/蒸馏/  │
│  → User / Env     │            │  结构化/潜态/参数化 │
├───────────────────┤    ←→     ├───────────────────┤
│ Experiential Mem  │            │ Evolution         │
│  "Agent如何进步"   │            │  精炼: 合并/更新/  │
│  → Case/Strategy/ │            │  遗忘             │
│    Skill/Hybrid   │            ├───────────────────┤
├───────────────────┤            │ Retrieval         │
│ Working Memory    │            │  利用: 时机/查询/  │
│  "Agent在想什么"   │            │  策略/后处理       │
│  → Single/Multi   │            └───────────────────┘
└───────────────────┘
```
