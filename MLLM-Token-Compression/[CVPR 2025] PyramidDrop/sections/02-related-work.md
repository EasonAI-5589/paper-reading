# 2. Related Work

> 来源: PyramidDrop (CVPR 2025)

---

## 📄 原文

### 2.1 Token Reduction

> 💡 **2.1 要点预览**: Token 压缩分两条线——LLM 文本 token 压缩（KV cache）和视觉 token 压缩（ViT/LVLM）。

**LLM 领域的 token 压缩（KV Cache）：**

The large language model (LLM) realm has made several efforts in applying token reduction for inference acceleration and KV cache compression. StreamLLM only keeps attention sinks and the most recent tokens to reduce the size of the KV cache. FastGen introduces an adaptive KV cache management approach that optimizes memory usage by adjusting retention strategies according to the specific properties of attention heads. Heavy-Hitter Oracle (H2O) employs a strategy that selectively prunes key-value pairs (KVs) during generation, utilizing a scoring mechanism driven by cumulative attention to inform the removal process. ScissorHands concentrates on identifying and retaining important tokens that show a consistent pattern of attention weight across previous token windows during generation.

> 💡 **批注**: LLM KV Cache 压缩方法谱系：
> ```
> StreamLLM  → 只保留 attention sink + 最近 token
> FastGen    → 按 attention head 特性自适应保留
> H2O        → 用累积 attention 分数来决定丢谁
> ScissorHands → 找"持续重要"的 token 保留
> ```
> 这些都是**文本 token** 的压缩，PyramidDrop 借鉴了"用 attention 评估重要性"这个思路。

**视觉 token 压缩：**

As for visual tokens, existing works make explorations on Vision Language Models (VLMs) before the era of large vision-language models, focusing on token reduction for vision transformers (ViTs). A recent work, FastV, makes an early attempt at visual token reduction in LVLMs, which drops visual tokens at the second layer of LVLMs during inference.

> 💡 **批注**: FastV 是最直接的对比对象——它在 LLM 第 2 层就砍 token。PyramidDrop 的改进是：不在某一层一次性砍，而是**多阶段渐进式砍**。

In contrast, our work makes a more comprehensive study of the visual redundancy in LVLMs and proposes a progressive visual token reduction solution for both training and inference of LVLMs.

> 💡 **2.1 小结**:
> - LLM token 压缩已有成熟方案（KV cache），PyramidDrop 借鉴了 attention-based ranking
> - 视觉 token 压缩刚起步，FastV 是先驱但太激进（第 2 层就砍）
> - PyramidDrop 的差异化：渐进式 + 训练推理都适用

---

### 2.2 Large Vision Language Models

> 💡 **2.2 要点预览**: LVLM 发展迅速，但高分辨率带来 token 爆炸问题。

Enabled by the open-sourcing of large language models like LLaMA and Vicuna, LVLMs have advanced the ability to understand and generate diverse content by seamlessly integrating information across multiple modalities, such as text, images, and audio. Models like LLaVA, InstructBLIP, and MiniGPT-4 have pushed the boundaries of this field.

Recent advances have significantly increased the number of image tokens for high-resolution image understanding, resulting in substantial costs for training and inference in LVLMs. This underscores the critical importance of developing more efficient training and inference methods for LVLMs.

> 💡 **批注**: 高分辨率趋势（mPLUG-DocOwl, Qwen2-VL, InternLM-XComposer）让 token 数量从几百飙到几千，效率问题迫在眉睫。PyramidDrop 正是为此而生。

> 💡 **2.2 小结**: LVLM 追求高分辨率 → token 数量爆炸 → 需要 token 压缩方案

---

## 💡 Section 总结

### 方法定位图
```
Token 压缩方法谱系:
├── LLM 文本 token (KV Cache)
│   ├── StreamLLM, H2O, ScissorHands...
│   └── 思路: attention-based 重要性排序
├── ViT 视觉 token (Pre-LLM era)
│   ├── ToMe, PuMer, SPViT...
│   └── 思路: 在 ViT 内部合并/剪枝
└── LVLM 视觉 token (当前热点) ⭐
    ├── FastV: 第 2 层一次性砍（太激进）
    ├── SparseVLM: 稀疏化
    ├── LLaVolta: 分阶段压缩（训练only）
    └── PyramidDrop: 渐进式金字塔丢弃 ✅
```
