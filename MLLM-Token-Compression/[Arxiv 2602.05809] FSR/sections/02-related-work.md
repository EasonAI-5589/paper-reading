[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
系统梳理 training-free visual token reduction 的三大类方法：Attention-based、Similarity-based、Joint，并指出各自的局限性。

---

The high inference cost of modern VLMs is largely driven by the massive number of visual tokens, which dominate both attention computation and KV-cache memory. To mitigate this overhead without additional training, a growing line of work studies training-free visual token reduction. Existing methods primarily differ in the signals used to estimate token importance.

> 💡 开篇简明扼要地框定了 related work 的范围：**training-free** visual token reduction。这排除了需要训练的方法（如 Honeybee、MQT、Matryoshka 等）。

---

**Attention-based Pruning.** Attention-based pruning estimates token importance from attention statistics, either inside the LLM decoder or within the vision encoder. On the LLM side, FastV prunes visual tokens according to cross attention scores in shallow layers. LLaVA-PruMerge further combines attention-based pruning with token merging to compress redundant visual tokens while preserving spatial semantics. SparseVLM introduces text-guided attention scoring and token recycling to reduce information loss during progressive sparsification, while PyramidDrop (PDrop) applies layer-wise progressive dropping to better align pruning strength with model depth. To improve inference-time efficiency and deployment compatibility, TopV performs visual token pruning during prefilling stage while maintaining compatibility with FlashAttention. FitPrune derives budget aware pruning schemes by minimizing attention-distribution divergence without additional training. On the vision-encoder side, FasterVLM and HiRED rank tokens using [CLS]-based attention, enabling early or region aware pruning. Overall, while attention-based methods are effective and easy to deploy, their importance estimates can be biased toward salient regions, which may inadvertently limit the coverage of subtle yet critical global contextual information.

> 💡 **Attention-based 方法总结**:
> - **LLM 侧**: FastV（浅层 cross-attention）、PruMerge（attention + merge）、SparseVLM（text-guided + recycling）、PyramidDrop（逐层递进）、TopV（prefill 阶段 + FlashAttention 兼容）、FitPrune（attention distribution divergence 最小化）
> - **Vision Encoder 侧**: FasterVLM、HiRED（[CLS] attention）
> - **共同局限**: 偏向视觉显著区域（salient regions），可能遗漏全局上下文
>
> 注意 TopV 的 FlashAttention 兼容性是一个实际部署的重要考量。FSR 作为 vision encoder 侧的方法，天然兼容 FlashAttention。

---

**Similarity-based Pruning.** Similarity-based approaches reduce redundancy by selecting diverse visual tokens in feature space rather than relying on saliency or importance scores. These methods are motivated by the observation that attention-based criteria may not reliably reflect whether a token is redundant, and can even lead to inferior performance or incompatibility with FlashAttention. DivPrune formulates token pruning as a max–min diversity selection problem to retain a representative and diverse subset. DART further prunes tokens based on duplication by retaining tokens dissimilar to a small set of pivots, enabling training-free acceleration. However, as these methods primarily concentrate on global regions, they often overlook fine-grained local details that are essential for precise reasoning.

> 💡 **Similarity-based 方法总结**:
> - DivPrune: max-min diversity selection（最大化 token 集合的多样性）
> - DART: pivot-based duplication removal（保留与 pivot 不相似的 token）
> - **共同局限**: 偏重全局覆盖，忽略细粒度局部细节
>
> 与 attention-based 方法的偏好恰好互补：attention → local，similarity → global。FSR 的 Focus/Scan 分阶段设计正是利用了这种互补性。

---

**Joint attention-similarity-based Pruning.** Recent methods combine multiple cues to better trade off query-critical local evidence and complementary global context. VisionZip and VisPruner integrate attention-based importance estimation with redundancy reduction to reduce token count while maintaining coverage. CDPruner further incorporates instruction relevance and maximizes conditional diversity through a DPP-style formulation, encouraging the retained tokens to be both relevant and diverse under the prompt. HoloV promotes holistic context retention by partition-wise allocation and connectivity aware token selection, aiming to avoid over-focusing on a few highlighted regions. Despite their effectiveness, under a fixed and limited token budget these methods can still struggle to simultaneously preserve the most query-critical local evidence and the complementary global context needed for reliable reasoning, especially when the retained tokens become extremely sparse.

> 💡 **Joint 方法总结**:
> - VisionZip / VisPruner: attention + redundancy reduction
> - CDPruner: instruction relevance + DPP-style conditional diversity（最接近 FSR 的竞争者）
> - HoloV: partition-wise allocation + connectivity-aware selection
> - **共同局限**: 在极端压缩下仍难以同时保留 local evidence 和 global context
>
> CDPruner 用 DPP 来同时优化 relevance 和 diversity，但 DPP 是一次性选择所有 token，没有 FSR 这种"先 Focus 再 Scan"的阶段性设计。FSR 的阶段性保证了 Focus 集合的质量不被 global diversity 目标稀释。

---

Prior research has investigated various token pruning strategies including attention-based, similarity-based, and joint attention-similarity-based pruning. However, effectively preserving both query-critical local evidence and complementary global context remains a formidable and persistent challenge, particularly under stringent token budgets. To address this limitation, we propose FSR, a human-inspired paradigm that dynamically balances fine-grained local detail and broad global context in accordance with the intrinsic complexity of the input.

> 💡 **过渡段**: 自然地从 related work 的局限性引出 FSR 的定位——**human-inspired paradigm**，核心差异化在于**动态平衡**和**根据输入复杂度自适应**。
