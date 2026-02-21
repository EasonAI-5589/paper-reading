[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
Related Work 分两部分：(1) MLLM 的架构与挑战——视觉数据冗余、attention 二次复杂度；(2) Visual Token Compression 方法综述——从需要训练的方法（LLaMA-VID, DeCo, MADTP）到 training-free 方法（ToMe, FastV, SparseVLM），指出已有方法要么需要训练要么不兼容 FlashAttention。

---

Multimodal Large Language Models Multimodal large language models (MLLMs) (Liu et al., 2024b; Li et al., 2023a; Zhu et al., 2023; Liu et al., 2024d) excel at image, video, and multimodal reasoning by integrating vision and text (Zhang et al., 2024a). However, visual data processing is costly due to redundancy, low information density (Liang et al., 2022; Liu et al., 2025b), and the quadratic cost of attention (Vaswani et al., 2017). For instance, models like LLaVA (Liu et al., 2023) and mini-Gemini-HD (Li et al., 2024d) encode highresolution images into thousands of tokens, while video models like VideoLLaVA (Lin et al., 2023) and VideoPoet (Kondratyuk et al., 2023) handle even more tokens across frames. These challenges highlight the need for efficient token representations and longer context. Recent work like Gemini (Team et al., 2023) and LWM (Liu et al., 2024a) addresses this by improving token efficiency and extending context, enabling more scalable MLLMs.

> 💡 **批注**: 点明了 MLLM 中 vision token 冗余的三大来源：(1) 视觉信号本身的空间冗余，(2) 信息密度低（相比文本），(3) attention 的 O(N²) 复杂度。高分辨率图像（数千 token）和视频（跨帧累积）是最严重的场景。

---

Visual Token Compression Visual tokens often outnumber text tokens by tens to hundreds of times, as visual signals are more spatially redundant than information-dense text (Marr, 2010). LLaMA-VID (Li et al., 2024c) employs a Q-Former with context tokens, and DeCo (Yao et al., 2024a) uses adaptive pooling. DTMFormer (Wang et al., 2024d) improves ViTs' efficiency in medical image segmentation by merging redundant tokens during training. MADTP (Cao et al., 2024) reduces computation by aligning cross-modal features and pruning tokens. However, these require modifying components and additional training. ToMe (Bolya et al., 2023) merges tokens without training but disrupts crossmodal interactions (Xing et al., 2024). FastV (Chen et al., 2024) selects via attention scores, while SparseVLM (Zhang et al., 2024c) uses text guidance. Yet, these forgo Flash-Attention (Dao et al., 2022; Dao, 2024), neglecting token duplication. We preserve hardware acceleration (i.e., Flash-Attention) and target duplication for efficient token reduction.

> 💡 **批注**: 这段综述将已有方法分为两类：
> - **需要训练**的：LLaMA-VID（Q-Former）、DeCo（adaptive pooling）、MADTP（cross-modal alignment）→ 部署成本高
> - **Training-free**的：ToMe（token merging，但破坏跨模态交互）、FastV（attention score pruning）、SparseVLM（text-guided pruning）→ 不兼容 FlashAttention
> 
> DART 的定位：training-free + FlashAttention 兼容 + 关注 duplication。注意作者引用 Marr (2010) 的经典视觉理论来支持"视觉信号空间冗余"这一前提。
