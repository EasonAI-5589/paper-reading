[← 返回 README](../README.md)

# Abstract

## 📌 预览
VisionTrim 提出了一个 training-free 的 MLLM 加速框架，通过两个 plug-and-play 模块（DVTS + TGVC）在 vision encoding 和 LLM decoding 两个阶段统一压缩视觉 token。

---

Multimodal large language models (MLLMs) suffer from high computational costs due to excessive visual tokens, particularly in high-resolution and video-based scenarios. Existing token reduction methods typically focus on isolated pipeline components and often neglect textual alignment, leading to performance degradation. In this paper, we propose VisionTrim, a unified framework for training-free MLLM acceleration, integrating two effective plug-and-play modules: 1) the Dominant Vision Token Selection (DVTS) module, which preserves essential visual tokens via a global-local view, and 2) the Text-Guided Vision Complement (TGVC) module, which facilitates context-aware token merging guided by textual cues. Extensive experiments across diverse image and video multimodal benchmarks demonstrate the performance superiority of our VisionTrim, advancing practical MLLM deployment in real-world applications. The code is available at: https://github.com/hanxunyu/VisionTrim.

> 💡 **Abstract 批读**:
> - **问题**: MLLM 推理开销大，根源是视觉 token 过多，尤其是高分辨率图像和视频场景
> - **现有方法的不足**: (1) 只关注 pipeline 的某个单一阶段；(2) 忽略了文本对齐
> - **VisionTrim 方案**: 两个 plug-and-play 模块
>   - **DVTS** (Dominant Vision Token Selection): global-local 双视角选择重要 token
>   - **TGVC** (Text-Guided Vision Complement): 利用文本线索引导 token merging，补充被丢弃但与文本相关的 token
> - **关键词**: training-free, unified (两阶段都做), plug-and-play

---

## 🔖 Section 总结

### 核心洞察
1. VisionTrim 是第一个在 vision encoder 和 LLM decoder 两阶段都做 token 压缩的 training-free 方法（VScan 也是两阶段，但没有 text-guided merging）
2. 相比纯 pruning，VisionTrim 多了 "complement" 的思路——被 DVTS 丢掉的 token 不是直接扔掉，而是通过 TGVC 聚类合并后补回来
