[← 返回 README](../README.md)

# 3 Preliminary Analysis

## 3.1 Computational Overhead in LVLM Processing Pipeline

### Architecture and Processing Flow

Existing LVLMs consist of three main components: a vision encoder, a vision-language projector, and a LLM decoder. Given visual input V, the vision encoder extracts visual features → projector maps to visual token embeddings E_v → concatenated with text embeddings E_t and system prompt embeddings E_s → LLM input sequence.

During LLM prefilling: all input tokens interact via self-attention → KV cache stored. During decoding: only new tokens computed, cached KV retrieved.

### Computational Cost Analysis

```
FLOPs_encoding = FLOPs_prefilling = T × (4nd² + 2n²d + 2ndm)
FLOPs_decoding = T × (4Ld² + 2Ldm + dL(2n+L-1))
```

For LLaVA-NeXT-7B (n=3000, L=20):
- encoding : prefilling : decoding ≈ **1 : 63.6 : 0.4**

For LLaVA-NeXT-13B:
- encoding : prefilling : decoding ≈ **1 : 121.1 : 0.8**

> 💡 **批注**：Prefilling 阶段占绝对主导！这解释了为什么 Stage 1（在 embedding space 减少 token 数）的效率提升远大于 Stage 2（在 decoder 中期删除）。也说明了尽早 pruning 的重要性。

## 3.2 Intra- and Inter-Modal Redundancy

Token redundancy manifests in two **orthogonal** components:

### Intra-modal redundancy
Visual tokens exhibit significant similarity → highly similar tokens contribute little unique information. Solution: retain a **maximally diverse** set of tokens. This motivates **Stage 1: Diversity-driven Visual Token Selection**.

> 💡 **批注**："Instead of relying on complex designs for redundancy detection, we find that retaining a maximally diverse set of tokens more effectively preserves the visual representation." — 换了一个等价但更直接的视角：不做冗余检测，直接做多样性最大化选择。

### Cross-modal redundancy
Visual tokens less relevant to textual information can be safely pruned. Cross-modal attention during LLM prefilling offers a reliable signal. Key observation: **information migration** — early layers prioritize cross-modal interaction, deeper layers focus on uni-modality processing. This motivates **Stage 2: Relevance-driven Visual Token Reduction**.

> 💡 **批注**：正交性是核心论点。Diversity 只看 visual token 之间的关系（intra-modal），Relevance 看 visual-text 之间的关系（cross-modal）。两者独立操作互不干扰，这也是两阶段设计的理论基础。
