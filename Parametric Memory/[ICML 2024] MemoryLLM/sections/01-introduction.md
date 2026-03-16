[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
分析三类现有方法的不足（检索、模型编辑、长上下文），提出 MemoryLLM 的设计思路。

---

Despite the impressive performance LLMs demonstrate, a pivotal issue persists: How should we update the model with the latest knowledge? Previous solutions can be broadly categorized into three classes:

**(1) Retrieval-Based Methods**: These methods rely on information retrieval in a knowledge base. They can yield strong results, but face challenges when redundancy in the knowledge base presents and suffer the logistical issue of managing an ever-expanding repository of knowledge. In multi-modality scenarios, retrieval-based methods might require enormous storage space.

> 💡 **批注**: RAG 的核心问题——知识库无限增长 + 冗余信息 + 多模态存储开销。

**(2) Model Editing**: This class involves making targeted edits to the model to adapt to new facts while preserving other desired capabilities. Existing methods primarily focus on fact-based editing, typically limited to single sentences. This limitation becomes more severe when one attempts to inject new knowledge in the form of longer and more complicated contexts.

> 💡 **批注**: 模型编辑（如 ROME, MEMIT）只能改单个事实，无法处理段落级知识注入。

**(3) Long Context Methods**: Another alternative is to incorporate all knowledge into the model's context. Methods involve reducing attention complexity and modifying positional embeddings. However, as complex reasoning tasks are thirsty for massive up-to-date knowledge, the inevitable context overload becomes infeasible, as long as the context length is finite.

> 💡 **批注**: 长上下文方法的本质问题——上下文 = 临时知识库，有限窗口终究不够用。

In response to these challenges, we introduce MEMORYLLM, a model that embeds a substantial, fixed-size memory pool within its latent space, which serves as the self-updatable parameters. Specifically, we build the memory pool as hidden vectors within each layer of the transformer. At each layer, the memory pool contains memory tokens representing compressed knowledge.

> 💡 **MemoryLLM 设计要点**:
> - Memory pool 在**每一层**都有（不是只在某一层），最大化容量
> - Memory tokens 是 hidden vectors，与 Transformer 的 hidden states 同维度
> - Self-update 只更新 K 个 tokens（K << N），旧知识指数衰减
> - 近百万次更新无退化——这是个重要的工程验证

**Contributions**:
1. 在 LLM 隐空间中集成 memory pool，固定大小避免无限增长
2. 将 7B 模型扩展 1B 参数的 memory pool
3. 在模型编辑、长上下文、知识保留等多个基准上表现优异

---

## 🔖 Section 总结

### 核心洞察
1. **三类方法的共同问题**: RAG（无限增长）、模型编辑（粒度太粗）、长上下文（容量有限）→ MemoryLLM 用固定大小的 latent memory 解决
2. **$\phi$ + $\theta$ 双参数设计**: $\phi$（Transformer 权重）= 持久知识，$\theta$（memory pool）= 动态知识。这与 Titans 的 persistent memory + neural memory 思路异曲同工
3. **Self-update = Transformer 前向传播**: 不需要梯度，极其高效
