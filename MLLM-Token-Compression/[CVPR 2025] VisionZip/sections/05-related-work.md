# 5. Related Work

> 来源: VisionZip (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: 简要定位 VisionZip 在 efficient VLM 领域的位置。

### Vision-Language Models

Building on LLMs, recent VLMs advance multimodal generation by processing extensive visual token sequences. Higher resolutions require exponentially more tokens; for example, LLaVA-NeXT processes 672×672 images into 2304 tokens. Handling videos or multiple images increases token requirements further.

> 💡 **批注**: VLM 发展趋势 = token 越来越多：
> ```
> LLaVA-1.5:    576 tokens
> LLaVA-NeXT:   2880 tokens (5×)
> Video-LLaVA:  2048 tokens (8 frames)
> 未来长视频:    可能 10,000+ tokens
> ```
> → 高效 token 压缩只会越来越重要

### Efficient VLM 方法谱系

> 💡 **批注**: 把 VisionZip 放在 efficient VLM 的方法谱系中：
>
> | 方法 | 类型 | 压缩位置 | 依赖文本 | 代表工作 |
> |------|------|----------|---------|----------|
> | Token pruning in LLM | Text-aware | LLM 中间层 | ✅ | FastV, SparseVLM, PyramidDrop |
> | KV cache compression | Text-aware | LLM KV cache | ✅ | ZipVL |
> | Vision encoder 端压缩 | Text-agnostic | Encoder 后 | ❌ | **VisionZip** |
> | Token merging | Architecture | Encoder 内部 | ❌ | ToMe |
> | Model pruning | Architecture | 全模型 | - | UPop |
>
> VisionZip 独特之处：**唯一在 vision encoder 端做 text-agnostic 压缩的方法**。

---

## 💡 Section 总结

### 与其他 Text-aware 方法的对比细节

| 方法 | 会议 | 策略 | 和 VisionZip 的关系 |
|------|------|------|-------------------|
| **FastV** | ECCV 2024 | LLM 第 2 层后按 text-visual attention 剪 token | VisionZip 全面优于，尤其 token 少时 |
| **SparseVLM** | 2024.10 | LLM 各层逐步按 attention 剪 token | VisionZip 优于 5-9% |
| **PyramidDrop** | 2024.10 | LLM 各层金字塔式逐步减少 | 类似 FastV/SparseVLM 的问题 |
| **ZipVL** | 2024.10 | KV cache 压缩 | 互补方向，可以和 VisionZip 结合 |
| **SwiftVLM** | - | 类似 FastV 的 LLM 端 pruning | Text-aware 的局限性 |
| **DivPrune** | - | Diversity-based pruning | 思路不同但也是 text-aware |
