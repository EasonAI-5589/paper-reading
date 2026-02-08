[← 返回 README](../README.md)

# Abstract

## 📌 预览
本综述首次系统性地梳理了多模态大语言模型（MLLM）中 token 压缩技术。按模态（图像/视频/音频）和压缩机制（变换/相似度/注意力/查询）双维度分类。

---

Multimodal large language models (MLLMs) have made remarkable strides, largely driven by their ability to process increasingly long and complex contexts, such as high-resolution images, extended video sequences, and lengthy audio input. While this ability significantly enhances MLLM capabilities, it introduces substantial computational challenges, primarily due to the quadratic complexity of self-attention mechanisms with numerous input tokens. To mitigate these bottlenecks, token compression has emerged as an auspicious and critical approach, efficiently reducing the number of tokens during both training and inference. In this paper, we present the first systematic survey and synthesis of the burgeoning field of multimodal long context token compression. Recognizing that effective compression strategies are deeply tied to the unique characteristics and redundancies of each modality, we categorize existing approaches by their primary data focus, enabling researchers to quickly access and learn methods tailored to their specific area of interest: (1) image-centric compression, which addresses spatial redundancy in visual data; (2) video-centric compression, which tackles spatio-temporal redundancy in dynamic sequences; and (3) audio-centric compression, which handles temporal and spectral redundancy in acoustic signals. Beyond this modality-driven categorization, we further dissect methods based on their underlying mechanisms, including transformation-based, similarity-based, attention-based, and query-based approaches. By providing a comprehensive and structured overview, this survey aims to consolidate current progress, identify key challenges, and inspire future research directions in this rapidly evolving domain.

> 💡 **Abstract 要点**:
> - **问题**: MLLM 处理多模态数据时，token 数量巨大，self-attention 的二次复杂度带来严重计算瓶颈
> - **方法**: Token 压缩——减少 token 数量来降低训练和推理开销
> - **分类维度 1 — 模态**: (1) 图像（空间冗余）、(2) 视频（时空冗余）、(3) 音频（时间+频谱冗余）
> - **分类维度 2 — 机制**: transformation-based、similarity-based、attention-based、query-based
> - **定位**: 首篇多模态长上下文 token 压缩的系统综述

---

## 🔖 Section 总结

### 核心洞察
1. 多模态 token（尤其视觉 token）在 MLLM 中占绝大多数序列长度，是主要计算瓶颈
2. 不同模态的冗余模式不同，需要针对性的压缩策略
3. 综述采用「模态 × 机制」的双维度 taxonomy，兼顾实用性和系统性
