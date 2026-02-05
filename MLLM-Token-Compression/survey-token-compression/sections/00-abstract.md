# Abstract

## 📄 原文

> Multimodal Large Language Models (MLLMs) have made significant strides in integrating vision-language perception, alignment, and reasoning.
>
> ==MLLMs 在视觉-语言感知、对齐、推理方面取得重大进展==

> However, the increasing complexity of tasks such as high-resolution image processing and long video understanding has led to an exponential rise in visual context length within MLLMs.
>
> ==核心问题：高分辨率图像 + 长视频 → 视觉 token 数量指数级增长==

> The resulting long-context token sequences impose substantial computational demands on large language models (LLMs), leading to quadratic complexity growth, heightened GPU resource consumption, and slower inference speeds.
>
> ==后果三连：O(n²) 复杂度 + GPU 内存爆炸 + 推理变慢==

> To address these challenges, token compression has emerged as a promising research direction that reduces the number of tokens processed within MLLMs while preserving essential cross-modal semantic information, thereby enhancing both training and inference efficiency.
>
> ==解决方案：Token Compression — 减少 token 数量，保留关键语义==

> This survey provides a comprehensive review of token compression techniques for MLLMs, examining the current state of research and exploring future directions.
>
> ==这篇综述干什么：全面回顾 Token Compression 技术==

> We propose a taxonomy of token compression methods based on their application modules within the MLLM system, including the vision encoder, projector, LLM backbone, and hybrid approaches.
>
> ==分类方法：按压缩位置分类 (Vision Encoder / Projector / LLM / Hybrid)==

> We analyze the strengths and limitations of widely adopted algorithms, offering practitioners a structured framework for selecting appropriate token compression strategies.
>
> ==实用价值：提供方法选择框架==

> Finally, we discuss practical applications of token compression, identify key challenges in the field, and propose potential directions for future research and development.
>
> ==结构：应用场景 + 关键挑战 + 未来方向==

> All related resources are available at https://github.com/yaolinli/MLLM-Token-Compression.

---

## 💡 Key Takeaways

1. **问题**：高分辨率图像和长视频导致 visual tokens 爆炸
2. **痛点**：O(n²) attention 复杂度 + GPU 内存 + 推理延迟
3. **方案**：Token Compression (减少 tokens，保留语义)
4. **分类**：按 MLLM 模块位置分类

---

*[返回论文目录](../README.md)*
