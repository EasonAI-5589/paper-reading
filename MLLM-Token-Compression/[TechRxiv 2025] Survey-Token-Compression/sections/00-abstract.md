[← 返回 README](../README.md)

# Abstract

## 📌 预览
这篇 survey 综述了 MLLM 中 token 压缩技术，按压缩位置（Vision Encoder / Projector / LLM / Hybrid）分类，分析各类方法的优劣，并讨论应用场景、挑战和未来方向。

---

Multimodal Large Language Models (MLLMs) have made significant strides in integrating vision-language perception, alignment, and reasoning. However, the increasing complexity of tasks such as high-resolution image processing and long video understanding has led to an exponential rise in visual context length within MLLMs. The resulting long-context token sequences impose substantial computational demands on large language models (LLMs), leading to quadratic complexity growth, heightened GPU resource consumption, and slower inference speeds.

> 💡 **核心问题**: 高分辨率图像 + 长视频 → visual token 数量爆炸 → O(n²) attention 复杂度 + GPU 内存 + 推理延迟三重痛点。

To address these challenges, token compression has emerged as a promising research direction that reduces the number of tokens processed within MLLMs while preserving essential cross-modal semantic information, thereby enhancing both training and inference efficiency.

> 💡 **解决方案**: Token Compression — 减少 token 数量同时保留关键跨模态语义信息。

This survey provides a comprehensive review of token compression techniques for MLLMs, examining the current state of research and exploring future directions. We propose a taxonomy of token compression methods based on their application modules within the MLLM system, including the vision encoder, projector, LLM backbone, and hybrid approaches. We analyze the strengths and limitations of widely adopted algorithms, offering practitioners a structured framework for selecting appropriate token compression strategies. Finally, we discuss practical applications of token compression, identify key challenges in the field, and propose potential directions for future research and development. All related resources are available at https://github.com/yaolinli/MLLM-Token-Compression.

> 💡 **Survey 结构**: 按 MLLM 模块分类（VE / Projector / LLM / Hybrid）→ 方法选择指南 → 应用场景 → 挑战与未来。

---

## 🔖 Section 总结

### 核心要点
1. **问题**: 高分辨率图像和长视频导致 visual tokens 爆炸，O(n²) 注意力复杂度不可承受
2. **方案**: Token Compression — 减少 tokens 同时保留语义
3. **分类体系**: 按压缩位置分类（Vision Encoder / Projector / LLM Backbone / Hybrid）
4. **实用价值**: 提供方法选择框架，帮助实践者选择合适的压缩策略
