[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 梳理了两条线：VLM 的视觉 token 增长问题，以及视觉 token 压缩的两大方向（encoder/projector 压缩 vs. decoder 阶段稀疏化）。

---

## Vision-Language Models

Recent works on vision-language models (Liu et al., 2024a; Chen et al., 2024b; Li et al., 2024c) improve multimodal comprehension and generation by processing longer visual token sequences. Moreover, the usage of higher-resolution images inevitably entails an exponential growth in the length of visual sequences. For example, LLaVA typically encodes 336×336 images into 576 tokens (Liu et al., 2024b) with up to 672×672 maximum resolution using 2880 token sequences (Liu et al., 2024a). Similarly, mini-Gemini-HD (Li et al., 2024c) converts 1536×1536 high resolution and 672×672 low resolution images into 2880 visual tokens. Moreover, comprehending videos or multiple images leads to increased token allocations for visual signals. For instance, the VideoLLaVA (Lin et al., 2024) and VideoPoet (Kondratyuk et al., 2024) use thousands of tokens to encode multiple image frames. However, large number of visual tokens results in a computational bottleneck. Further research on sparsification is urged to further unleash VLM capabilities.

> 💡 **VLM 视觉 token 增长趋势**:
> | 模型 | 分辨率 | Token 数 |
> |------|--------|---------|
> | LLaVA | 336×336 | 576 |
> | LLaVA | 672×672 | 2880 |
> | mini-Gemini-HD | 1536×1536 + 672×672 | 2880 |
> | VideoLLaVA / VideoPoet | 多帧 | 数千 |
>
> 高分辨率和视频场景使视觉 token 数量急剧增长 → 计算瓶颈。

---

## Visual Compression for VLMs

Compression of visual tokens is necessary because, on the one hand, their quantity is usually tens to hundreds of times that of language tokens. On the other hand, visual signals are inherently more sparse in information when compared to texts that have been produced by humans (Marr, 2010). Past efforts to address the above problem can be categorized into two directions. The first one centers on the compression of a vision tower or an efficient projection of vision modality. For instance, LLaMA-VID (Li et al., 2024b) exploits the Q-Former with the context token while DeCo (Yao et al., 2024) employs an adaptive pooling to downsample the visual tokens at the patch level. Methods that belong to the second direction (Ye et al., 2025; Chen et al., 2024a; Wu et al., 2024) go deeper into the text modality and sparsify visual tokens during the LLM decoding stage, but they still lack guidance from the text tokens. In this paper, SparseVLM takes note of this limitation and improves performance upon it.

> 💡 **两大压缩方向**:
>
> **方向一：Encoder/Projector 压缩**（在进入 LLM 前压缩）
> - LLaMA-VID: Q-Former + context token
> - DeCo: 自适应 pooling 下采样
> - 特点：修改模型结构，需要训练
>
> **方向二：Decoder 阶段稀疏化**（在 LLM 推理过程中裁剪）
> - FastV (Chen et al., 2024a)
> - VoCo-LLaMA (Ye et al., 2025)
> - VideoLLM-MoD (Wu et al., 2024)
> - **局限：缺乏文本引导** ← SparseVLM 的切入点
>
> SparseVLM 属于方向二，但加入了 text-aware guidance。

---

## 🔖 Section 总结

### 核心洞察
1. 视觉 token 数量是语言 token 的数十到数百倍，压缩是必要的
2. 现有 decoder 阶段稀疏化方法（FastV 等）不考虑文本信息
3. SparseVLM 的定位：decoder 阶段 + text-aware + training-free
