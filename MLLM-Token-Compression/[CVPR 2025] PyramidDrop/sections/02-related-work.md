[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 覆盖两个方向：Token Reduction（LLM/VLM 中的 token 压缩方法）和 Large Vision Language Models（LVLM 发展及高分辨率带来的效率挑战）。

---

**Token Reduction** The large language model (LLM) realm has made several efforts in applying token reduction for inference acceleration and KV cache compression[19]. StreamLLM[47] only keeps attention sinks and the most recent tokens to reduce the size of the KV cache. FastGen[15] introduces an adaptive KV cache management approach that optimizes memory usage by adjusting retention strategies according to the specific properties of attention heads. Heavy-Hitter Oracle (H2O)[55] employs a strategy that selectively prunes key-value pairs (KVs) during generation, utilizing a scoring mechanism driven by cumulative attention to inform the removal process. ScissorHands[34] concentrates on identifying and retaining important tokens that show a consistent pattern of attention weight across previous token windows during generation. These works attempt to address the redundancy of text tokens during the inference process in LLMs. As for visual tokens, existing works [4, 21, 26, 43, 48] make explorations on Vision Language Models (VLMs) before the era of large vision-language models, focusing on token reduction for vision transformers (ViTs). A recent work, FastV [9], makes an early attempt at visual token reduction in LVLMs, which drops visual tokens at the second layer of LVLMs during inference. In contrast, our work makes a more comprehensive study of the visual redundancy in LVLMs and proposes a progressive visual token reduction solution for both training and inference of LVLMs.

> 💡 **批注**: Token Reduction 领域的相关工作可分为三类：
> 1. **LLM 文本 token 压缩**：StreamLLM、FastGen、H2O、ScissorHands — 主要压缩 KV cache
> 2. **ViT 视觉 token 压缩**：ToMe、PuMer 等 — 在 ViT 阶段做 token pruning/merging
> 3. **LVLM 视觉 token 压缩**：FastV 是先驱，但只在第 2 层做一次性丢弃
> 
> PyramidDrop 的区别：更全面地研究了 LVLM 中视觉冗余的层级特性，提出渐进式方案，且同时适用于训练和推理。

---

**Large Vision Language Models** Enabled by the open-sourcing of large language models like LLaMA[45] and Vicuna[11], LVLMs[10] have advanced the ability to understand and generate diverse content by seamlessly integrating information across multiple modalities, such as text, images, and audio. Models like LLaVA[31], InstructBLIP[12], and MiniGPT-4[57] have pushed the boundaries of this field, enabling users to interact with these intelligent systems through multimodal prompts, including images and text. Recent advances [20, 46, 53] have significantly increased the number of image tokens for high-resolution image understanding, resulting in substantial costs for training and inference in LVLMs. This underscores the critical importance of developing more efficient training and inference methods for LVLMs.

> 💡 **批注**: LVLM 的发展趋势：分辨率越来越高 → token 数越来越多 → 效率问题越来越严重。这为 PyramidDrop 这类工作提供了强烈的应用需求。

---

## 🔖 Section 总结

### 核心洞察
1. LLM 领域的 KV cache 压缩方法为视觉 token 压缩提供了思路借鉴
2. FastV 是 LVLM 视觉 token 压缩的先驱，但只在固定浅层做一次性丢弃
3. 高分辨率 LVLM 的发展使 token 压缩成为刚需
