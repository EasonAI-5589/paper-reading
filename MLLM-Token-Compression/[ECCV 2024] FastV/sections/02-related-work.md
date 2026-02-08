[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
三个方向的相关工作：(1) LVLM 架构（visual token 数量随分辨率暴增），(2) LLM 推理优化（FlashAttention、KV cache 压缩），(3) VLM token 压缩（ViT 层面的 token 减少方法）。FastV 是第一个针对 LVLM 的视觉 token 剪枝方法。

---

**Large Vision-Language Model.** To benefit from the advancement of LLM and integrate visual information into the LLM, large Vision-Language Models utilize a Visual Prompt Generator Li et al. (2023b) to transform the visual embeddings into prompts that the language model can comprehend Li et al. (2023c); Liu et al. (2023c), resulting in a significant increase in required tokens. Handling higher resolution images inevitably necessitates a quadratic increase in the number of needed tokens. For instance, LLAVA process 336x336 images into 576 tokens Liu et al. (2023b) and process images with a greater resolution of 672x672 into 2304 tokens Liu et al. (2024b). Fuyu Bavishi et al. (2023), in a similar vein, translates pixel-level images of 1080×1080 into 1296 tokens. Understanding and generating multiple images or videos also inherently demands an escalated count of tokens for vision information. Both Video-Poet Kondratyuk et al. (2023) and Unified-IO2 Lu et al. (2023) are compelled to reserve thousands of tokens within the context to facilitate the understanding and generation of multiple images or videos. Large multimodal models like Gemini Team et al. (2023) and LWM Liu et al. (2024a) highlights the significance of long context in developing a robust understanding of the world model and extending the context length to 1M to address the issue of escalating context requirements.

> 💡 **LVLM token 数量统计**:
> | 模型 | 分辨率 | Token 数 |
> |------|--------|---------|
> | LLaVA | 336×336 | 576 |
> | LLaVA (高分辨率) | 672×672 | 2304 |
> | Fuyu | 1080×1080 | 1296 |
> | Video 模型 | 多帧 | 数千 |
> 
> 分辨率翻倍 → token 数 **4 倍增长**（二次关系）。这就是 FastV 要解决的核心矛盾。

---

**Inference Optimization for LLM.** Efficient inference in LLMs is challenged by their autoregressive generation where each token prediction depends on the preceding context. Hence, considering the quadratic complexity of computation's attention during training, as the context length increases, the generation becomes progressively slower. To tackle these challenges, pioneering studies fall into two categories: methods optimizing memory consumption for attention module like FlashAttention, vLLM and RingAttention Dao et al. (2022); Dao (2023); Kwon et al. (2023); Liu et al. (2023a), which ensure no drastic shifts in the results, and methods like StreamingLLM and FastGen Xiao et al. (2023); Ge et al. (2024) that simplify computations by pruning redundant attention computation. We are interested in the second kind of methods since they are proposed inspired by the distinct attention patterns observed in LLM's inference. While these methods have boosted the inference efficiency of LLMs, they are designed for text-only language models, and whether their effectiveness can be transferred to LVLMs remain under-explored. There is previous work attempt to handle the long-context in LVLMs efficiently, like LLaMA-VID Li et al. (2023d), which utilizes cross-attention to effectively represent each video frame with two key tokens, the requirement for an additional fine-tuning stage obstructs its broad applicability for different LVLMs.

> 💡 **LLM 推理优化两大流派**:
> 1. **内存优化** (不改结果): FlashAttention, vLLM, RingAttention
> 2. **计算简化** (剪枝冗余): StreamingLLM, FastGen
> 
> FastV 属于第 2 类，但专门针对 LVLM 的 **视觉 token**，而非文本 token。
> LLaMA-VID 虽然也压缩视觉 token，但需要额外微调，不够通用。

---

**Token Reduction for VLMs.** There have been studies on improving efficiency for Vision-Language Models (VLMs) before the era of large vision-language models. A majority of them focus on token reduction for vision transformers (ViTs). Various methods, such as EViT Liang et al. (2022), SPViT Kong et al. (2022), and Pumer Cao et al. (2023), have been proposed for ViTs. More recently, PYRA Xiong et al. (2024) has enhanced the training and inference of ViTs via a specialized token merging technique. FastV is the first to explore visual token reduction for Large Vision-Language Models (LVLMs), which uses language as an interface for various vision-language tasks. FastV utilizes the signal from LLM to guide the pruning of visual tokens, a strategy not previously explored. We are the first to demonstrate the effectiveness of token reduction in video-QA and various comprehensive LVLM benchmarks. Another significant advantage of FastV over previous methods is its simplicity; it can be applied to any LVLM without requiring model retraining.

> 💡 **FastV vs 之前的 Token Reduction 方法**:
> - 之前的方法 (EViT, SPViT, PYRA) 都在 **ViT 层面**做 token 减少
> - FastV 的独特之处：
>   1. 在 **LLM decoder 层面**做剪枝
>   2. 用 **LLM 的 attention signal** 指导剪枝（而非视觉信号）
>   3. **无需重训练**，plug-and-play
>   4. 首次在 video-QA 和综合 benchmark 上验证有效性

---

## 🔖 Section 总结

### 核心洞察
1. Visual token 数量随分辨率二次增长 → 效率瓶颈
2. LLM 推理优化方法不能直接迁移到 LVLM
3. 之前的 token reduction 在 ViT 层面，FastV 首次在 LLM decoder 层面做视觉 token 剪枝
