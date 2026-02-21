[← 返回 README](../README.md)

# 2 Related work

## 📌 预览
系统回顾三类 training-free visual token pruning 方法，分析各自优缺点，引出 FSR 的动机。

---

The high inference cost of modern VLMs is largely driven by the massive number of visual tokens, which dominate both attention computation and KV-cache memory. To mitigate this overhead without additional training, a growing line of work studies training-free visual token reduction. Existing methods primarily differ in the signals used to estimate token importance.

> 💡 **分类标准**: 按 importance estimation 使用的信号类型分类

---

Attention-based Pruning. Attention-based pruning estimates token importance from attention statistics, either inside the LLM decoder or within the vision encoder. On the LLM side, FastV prunes visual tokens according to cross attention scores in shallow layers Chen et al. (2024a). LLaVA-PruMerge further combines attentionbased pruning with token merging to compress redundant visual tokens while preserving spatial semantics Shang et al. (2024). SparseVLM introduces text-guided attention scoring and token recycling to reduce information loss during progressive sparsification Zhang et al. (2024b), while PyramidDrop (PDrop) applies layer-wise progressive dropping to better align pruning strength with model depth Xing et al. (2024). To enhance deployment efficiency, TopV ensures FlashAttention compatibility during prefilling Yang et al. (2025a); Dao et al. (2022) , whereas FitPrune minimizes attention-distribution divergence for budget-aware pruning Ye et al. (2025). On the vision-encoder side, FasterVLM and HiRED rank tokens using [CLS]-based attention, enabling early or region aware pruning Zhang et al. (2024a); Arif et al. (2025). SparseVILA decouples visual sparsity into query-agnostic prefill and query-aware decoding stages Khaki et al. (2025). Overall, while attention-based methods are effective and easy to deploy, their importance estimates can be biased toward salient regions, which may inadvertently limit the coverage of subtle yet critical global contextual information.

> 💡 **Attention-based 方法梳理**:
> | 方法 | 信号来源 | 特点 |
> |------|---------|------|
> | FastV | LLM shallow-layer cross-attn | 开创性工作，layer 2 后剪枝 |
> | LLaVA-PruMerge | Attention + merge | 剪枝+合并双管齐下 |
> | SparseVLM | Text-guided attn + recycling | 渐进式稀疏化 |
> | PyramidDrop | Layer-wise progressive | 逐层递增剪枝 |
> | TopV | FlashAttention-compatible | 部署导向 |
> | FitPrune | Attn distribution divergence | Budget-aware |
> | FasterVLM/HiRED | [CLS]-based (encoder side) | 早期剪枝 |
> | SparseVILA | Decoupled prefill/decode | Query-agnostic + query-aware |
>
> **共同缺陷**: 偏向 salient regions，忽略 subtle global context

---

Similarity-based Pruning. Similarity-based approaches reduce redundancy by selecting diverse visual tokens in feature space rather than relying on saliency or importance scores. These methods are motivated by the observation that attention-based criteria may not reliably reflect whether a token is redundant, and can even lead to inferior performance or incompatibility with FlashAttention. DivPrune formulates token pruning as a max–min diversity selection problem to retain a representative and diverse subset Alvar et al. (2025). DART further prunes tokens based on duplication by retaining tokens dissimilar to a small set of pivots, enabling training-free acceleration Wen et al. (2025). However, as these methods primarily concentrate on global regions, they often overlook fine-grained local details that are essential for precise reasoning.

> 💡 **Similarity-based 方法**:
> - DivPrune: max-min diversity selection
> - DART: 基于 duplication 的 pivot-based 筛选
> - **缺陷**: 偏全局，忽略 fine-grained local details
> - 与 Attention-based 形成互补但也互为盲区

---

Joint attention-similarity-based Pruning. Recent methods combine multiple cues to better trade off query-critical local evidence and complementary global context. VisionZip and VisPruner integrate attention-based importance estimation with redundancy reduction to reduce token count while maintaining coverage Yang et al. (2025b); Zhang et al. (2025a). CDPruner further incorporates instruction relevance and maximizes conditional diversity through a DPP-style formulation, encouraging the retained tokens to be both relevant and diverse under the prompt Zhang et al. (2025b). HoloV promotes holistic context retention by partition-wise allocation and connectivity aware token selection, aiming to avoid over-focusing on a few highlighted regions Zou et al. (2025). Despite their effectiveness, under a fixed and limited token budget these methods can still struggle to simultaneously preserve the most query-critical local evidence and the complementary global context needed for reliable reasoning, especially when the retained tokens become extremely sparse.

> 💡 **Joint 方法对比**:
> | 方法 | 策略 | 局限 |
> |------|------|------|
> | VisionZip | Attention + redundancy reduction | 静态混合 |
> | VisPruner | Visual cues + redundancy | 高压缩率下退化 |
> | CDPruner | DPP-style conditional diversity | 计算开销较大 |
> | HoloV | Partition-wise + connectivity-aware | Over-focus 问题仍存在 |
>
> **共同问题**: 在极端 token budget 下仍 struggle to balance

---

Prior research has investigated various token pruning strategies including attention-based, similarity-based, and joint attention-similaritybased pruning. However, effectively preserving both query-critical local evidence and complementary global context remains a formidable and persistent challenge, particularly under stringent token budgets. To address this limitation, we propose FSR, a human-inspired paradigm that dynamically balances fine-grained local detail and broad global context in accordance with the intrinsic complexity of the input.

> 💡 **引出 FSR**: 
> - 关键词 "dynamically balances" — 不是静态比例，而是根据输入复杂度动态调整
> - 这是对所有三类方法的统一回应

---

## 🔖 Section 总结

### Citation Landscape 预览
- **Attention-based**: FastV, LLaVA-PruMerge, SparseVLM, PyramidDrop, TopV, FitPrune, FasterVLM, HiRED, SparseVILA
- **Similarity-based**: DivPrune, DART
- **Joint**: VisionZip, VisPruner, CDPruner, HoloV
- FSR 属于 Joint 类但强调"动态分配"而非"静态融合"
