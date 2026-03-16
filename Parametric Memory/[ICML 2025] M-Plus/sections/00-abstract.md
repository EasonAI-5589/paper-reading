[← 返回 README](../README.md)

# Abstract

## 📌 预览
M+ 在 MemoryLLM 基础上增加长期记忆 + co-trained retriever，将知识保留从 <20k 扩展到 160k+ tokens。

---

Equipping large language models (LLMs) with latent-space memory has attracted increasing attention as they can extend the context window of existing language models. However, retaining information from the distant past remains a challenge. For example, MemoryLLM, as a representative work with latent-space memory, compresses past information into hidden states across all layers, forming a memory pool of 1B parameters. While effective for sequence lengths up to 16k tokens, it struggles to retain knowledge beyond 20k tokens. In this work, we address this limitation by introducing M+, a memory-augmented model based on MemoryLLM that significantly enhances long-term information retention. M+ integrates a long-term memory mechanism with a co-trained retriever, dynamically retrieving relevant information during text generation. We evaluate M+ on diverse benchmarks, including long-context understanding and knowledge retention tasks. Experimental results show that M+ significantly outperforms MemoryLLM and recent strong baselines, extending knowledge retention from under 20k to over 160k tokens with similar GPU memory overhead.

> 💡 **Abstract 批读**:
> - **核心问题**: MemoryLLM 的 memory pool 固定大小 → 超过 20k tokens 的知识无法保留
> - **解决方案**: 将被丢弃的 memory tokens 存入 CPU 端 **Long-Term Memory (LTM)**，配合 **co-trained retriever** 在生成时检索
> - **关键结果**: 20k → 160k+ tokens 的知识保留，GPU 开销不变（LTM 在 CPU）
> - **与 MemoryLLM 的关系**: M+ = MemoryLLM (short-term) + LTM + Retriever

---

## 🔖 Section 总结
M+ 的核心创新：不丢弃旧 memory tokens，而是"归档"到 CPU 端 LTM。这把 MemoryLLM 的 "forget" 变成了 "archive"。
