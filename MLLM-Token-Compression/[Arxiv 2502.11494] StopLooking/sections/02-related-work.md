[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
简要回顾 MLLM 架构和 visual token compression 方法，重点区分 training-based 和 training-free 方法，以及 importance-based vs duplication-based 的差异。

---

**Multimodal Large Language Models** Multi-modal large language models (MLLMs) (Liu et al., 2024b; Li et al., 2023a; Zhu et al., 2023; Liu et al., 2024d) excel at image, video, and multimodal reasoning by integrating vision and text (Zhang et al., 2024a). However, visual data processing is costly due to redundancy, low information density (Liang et al., 2022; Liu et al., 2025b), and the quadratic cost of attention (Vaswani et al., 2017). For instance, models like LLaVA (Liu et al., 2023) and mini-Gemini-HD (Li et al., 2024d) encode high-resolution images into thousands of tokens, while video models like VideoLLaVA (Lin et al., 2023) and VideoPoet (Kondratyuk et al., 2023) handle even more tokens across frames. These challenges highlight the need for efficient token representations and longer context. Recent work like Gemini (Team et al., 2023) and LWM (Liu et al., 2024a) addresses this by improving token efficiency and extending context, enabling more scalable MLLMs.

> 💡 **MLLM 背景**: 视觉数据的三个成本来源——冗余性、低信息密度、attention 的二次复杂度。高分辨率和视频进一步放大问题。

---

**Visual Token Compression** Visual tokens often outnumber text tokens by tens to hundreds of times, as visual signals are more spatially redundant than information-dense text (Marr, 2010). LLaMA-VID (Li et al., 2024c) employs a Q-Former with context tokens, and DeCo (Yao et al., 2024a) uses adaptive pooling. DTMFormer (Wang et al., 2024d) improves ViTs' efficiency in medical image segmentation by merging redundant tokens during training. MADTP (Cao et al., 2024) reduces computation by aligning cross-modal features and pruning tokens. However, these require modifying components and additional training. ToMe (Bolya et al., 2023) merges tokens without training but disrupts cross-modal interactions (Xing et al., 2024). FastV (Chen et al., 2024) selects via attention scores, while SparseVLM (Zhang et al., 2024c) uses text guidance. Yet, these forgo Flash-Attention (Dao et al., 2022; Dao, 2024), neglecting token duplication. We preserve hardware acceleration (i.e., Flash-Attention) and target duplication for efficient token reduction.

> 💡 **方法分类**:
> - **Training-based**: LLaMA-VID (Q-Former), DeCo (adaptive pooling), MADTP (cross-modal alignment) — 需要额外训练
> - **Training-free, importance-based**: FastV (attention scores), SparseVLM (text-guided attention) — 不兼容 FA
> - **Training-free, merging**: ToMe — 破坏 cross-modal interaction
> - **DART 定位**: Training-free + duplication-based + FA 兼容，填补了空白

---
