[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
Related Work 覆盖两个主题：(1) MLLM 的基本范式及 visual token 开销问题；(2) 现有 visual token pruning/compression 方法的分类。

---

**Multimodal Large Language Models (MLLMs).** Large Language Models (LLMs)[1, 37, 3, 10, 16] have achieved remarkable success in a wide range of language understanding and generation tasks. Building on this foundation, Multimodal LLMs (MLLMs)[24, 25, 21, 50, 19, 4] have shown impressive progress in visual understanding. A prevailing paradigm in MLLMs projects visual features into a sequence of visual tokens via a vision-to-language projector, and feeds them into the LLM alongside text tokens, as exemplified by LLaVA [24, 25], Qwen-VL [4], and Mini-Gemini [21].

> 💡 **MLLM 范式**: Vision Encoder → Projector → visual tokens + text tokens → LLM。代表：LLaVA, Qwen-VL, Mini-Gemini。

However, real-world images are often high-resolution, resulting in long visual token sequences that significantly slow down inference in MLLMs [23, 30, 20, 9]. For example, LLaVA-Next [25] converts a 672×672 image into over 2,000 tokens. The situation worsens when handling multiple images or videos, further increasing the number of visual tokens. This highlights the need for effective strategies to reduce token length and accelerate vision-language inference.

> 💡 **问题量化**: LLaVA-Next 672×672 → 2000+ tokens。多图/视频场景更严重。

---

**Visual Token Pruning/Compression in MLLMs.** A number of recent studies [49, 41, 40, 7] have focused on reducing visual token redundancy in MLLMs without requiring additional model training. Most of these methods [7, 49, 41] rely on specific attention scores to rank token saliency, such as text-to-vision attention in LLMs or CLS-token attention in vision transformers. They typically retain only the top-ranked tokens using a top-k strategy, i.e., selecting tokens with the highest attention scores. For instance, FastV [7] leverages early-layer text-to-vision attention to retain salient tokens. SparseVLM [49] uses important textual words as a rater to guide token selection. VisionZip [43] applies CLS-based attention in the vision transformer for token pruning. To further increase the information density of the selected tokens, several approaches attempt to merge semantically similar tokens [43, 49, 35]. DivPrune [2] selects visual tokens by maximizing the diversity of selected tokens. In contrast, our method jointly considers both saliency and coverage, aiming to preserve semantic completeness while reducing token redundancy.

> 💡 **现有方法分类**:
> | 方法 | Saliency 来源 | 策略 |
> |------|-------------|------|
> | FastV [7] | LLM 早期层 text→vision attention | Top-k |
> | SparseVLM [49] | 重要 textual words 作为 rater | Top-k + merge |
> | VisionZip [43] | ViT CLS token attention | Top-k + merge |
> | DivPrune [2] | Diversity maximization | Diversity-based |
> | **SCOPE (ours)** | **CLS attention + coverage** | **Saliency × Coverage gain** |
> 
> 关键区别：前三者都是 saliency-only，DivPrune 考虑 diversity 但不考虑 saliency，SCOPE 是唯一联合考虑两者的。

---

## 🔖 Section 总结

### 核心洞察
1. 现有 training-free token pruning 方法几乎都依赖 attention score 做 top-k 选择
2. Token merge 是 pruning 的补充手段，可以进一步提升信息密度
3. SCOPE 的定位：填补 saliency + coverage 联合建模的空白
