[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
两类 memory 方法（Token-level vs Latent-space），MemoryLLM 的局限，M+ 的设计思路。

---

The integration of memory modules into LLMs has gained increasing attention. Existing approaches can be broadly divided into two categories:

**(1) Token-level memory**: Memory represented as structured text (raw context, summaries, knowledge graphs). Benefits: adaptability, interpretability. Limitations: redundant, hard to resolve conflicts.

**(2) Latent-Space Memory**: Information compressed into hidden states, model parameters, or external latent space. Benefits: efficient compression, end-to-end training, closer to human memory.

> 💡 **批注**: 这个分类很清晰。M+ 属于 Latent-Space Memory 阵营，核心优势是压缩效率。

MemoryLLM creates a memory pool with 1B parameters by incorporating memory tokens into each layer. However, it faces limitations in recalling information injected beyond 20k tokens. M+ addresses this by introducing a long-term memory mechanism with a co-trained retriever.

Unlike H2O and SnapKV (which store KV caches and retrieve per query head per layer → high latency), M+ retrieves in hidden state space via co-training, only once per layer for all query heads, significantly improving efficiency. The long-term memory is stored on CPU, extending retention without increasing GPU memory.

> 💡 **M+ vs KV Cache 方法的关键区别**:
> | 方法 | 存储 | 检索粒度 | 延迟 |
> |------|------|---------|------|
> | H2O/SnapKV | KV cache | 每 head 每层 | 高 |
> | **M+** | Hidden states (CPU) | **每层一次，所有 heads** | 低 |

**Contributions**:
1. Long-term memory + co-trained retriever
2. Specialized data curriculum for long-context training
3. Significant improvement while maintaining similar GPU memory footprint

---

## 🔖 Section 总结

### 核心洞察
1. Latent-space memory 比 token-level memory 更紧凑，但保留范围有限 → M+ 用 LTM 打破这个限制
2. Co-trained retriever 的关键创新：在 hidden state 空间检索，而非 token 空间，效率更高
3. LTM 存 CPU 是个工程巧思：GPU 不增加开销，CPU 内存便宜
