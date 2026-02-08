[← 返回 README](../README.md)

# 2. Preliminaries

## 📌 预览
介绍 MLLM 的三组件架构（Vision Encoder + Projector + LLM）、计算复杂度分析、以及 Token Compression 的形式化定义。

---

This section lays the foundation for token compression in Multimodal Large Language Models (MLLMs). We begin with an overview of typical MLLM architectures (§2.1), followed by a formal definition of token compression techniques (§2.2).

## 2.1 Multimodal Large Language Models

The rapid advancement of artificial intelligence has witnessed a paradigm shift from unimodal models to sophisticated multimodal systems capable of understanding and reasoning across diverse data modalities. MLLMs represent a significant milestone in this evolution, combining the remarkable language understanding capabilities of Large Language Models (LLMs) [18], [163]–[166] with comprehensive visual perception abilities to create systems that can process, understand, and generate responses based on both textual and visual information.

Modern MLLMs typically adopt a three-component architecture: A vision encoder (VE) (often based on SigLIP [167] or CLIP [168]) that processes visual inputs into high-dimensional feature representations, a projector that aligns visual features with the language model's embedding space, and a powerful LLM that performs multimodal alignment, reasoning and generation. This architectural design enables end-to-end training and seamless integration of visual and textual information processing. Throughout this survey, we focus on token compression techniques designed for this mainstream three-component architecture. Alternative architectural paradigms [169], [170] that deviate from this design are beyond the scope of our discussion.

> 💡 **MLLM 三组件架构**: Vision Encoder（SigLIP/CLIP）→ Projector（对齐视觉和语言空间）→ LLM（推理生成）。这是当前主流架构，本 survey 专注于此。

Formally, let $X^v = \{I_1, I_2, \ldots, I_{n_v}\}$ with $n_v \geq 1$ denote the input image sequence or video frames, and $X^t = \{x_1, x_2, \ldots, x_{n_t}\}$ represent the textual token sequence comprising system prompts, user instructions, or dialogue history. The MLLM architecture consists of three key components:

**Vision Encoder.** The vision encoder $E_v$ transforms raw visual inputs into a sequence of dense visual token representations:

$Z^v = E_v(X^v) \in \mathbb{R}^{n_v \times d_v}$ — (Eq. 1)

where $n_v$ denotes the number of visual tokens and $d_v$ represents the feature dimension of each visual token.

**Projector.** To bridge the modality gap between visual and textual representations, a projector $P$ transforms visual features from dimension $d_v$ to the LLM's embedding space:

$H^v = P(Z^v) \in \mathbb{R}^{n_v \times d_t}$ — (Eq. 2)

where $d_t$ denotes the embedding dimension of the target language model.

**Large Language Model.** The LLM $G$ processes the concatenated sequence of projected visual tokens and embedded textual tokens:

$Y = G([H^v; E_t(X^t)])$ — (Eq. 3)

where $E_t(\cdot)$ represents the embedding layer of the LLM, $[\cdot; \cdot]$ denotes concatenation along the sequence dimension, and $Y$ is the generated output sequence.

> 💡 **形式化流程**: 图像 → VE 编码为 visual tokens $Z^v$ → Projector 映射到 LLM 空间 $H^v$ → 与文本 tokens 拼接后送入 LLM 生成输出。

### Computational Complexity

The aforementioned components in MLLMs primarily employ Transformer-based architectures [171], renowned for their powerful representation capabilities but also characterized by high computational costs for processing long input sequences. The computational complexity predominantly stems from the self-attention mechanism and feed-forward networks (FFNs) within Transformer layers.

Given a sequence of length $n$, a hidden dimension size $d$, and an intermediate dimension $m$ in the FFN, the computational cost per Transformer layer can be approximated as:

Layer FLOPs = $4nd^2 + 2n^2d + 2ndm$ — (Eq. 4)

Thus, for an $L$-layer Transformer, the total cost is:

Total FLOPs = $L \times (4nd^2 + 2n^2d + 2ndm)$ — (Eq. 5)

where $n = n_t + n_v$ is the overall sequence length (text tokens $n_t$ plus visual tokens $n_v$).

> 💡 **复杂度分析**: 关键项是 $2n^2d$（attention 的二次复杂度）。当 $n$ 增大时，这一项增长最快。在高分辨率图像/长视频场景下，$n_v$ 远大于 $n_t$，视觉 tokens 主导了计算开销。

As the sequence length $n$ increases, the quadratic complexity term $2n^2d$ in the attention mechanism grows rapidly, leading to prohibitive computational overhead. This computational bottleneck is particularly pronounced in scenarios involving: (1) high-resolution images or long videos, where $n_v$ typically dominates $n_t$ in MLLMs, and (2) multi-turn conversations or complex reasoning tasks requiring extensive contextual history.

## 2.2 Token Compression

The quadratic computational complexity in MLLMs naturally motivates the development of token compression techniques (also known as token reduction), which aim to reduce the total context length in the MLLM while preserving essential visual and textual semantics, thereby achieving computational efficiency without remarkably compromising model performance.

Formally, denote the total visual and textual token number in the MLLM as $N = n_t + n_v$, token compression aims to reduce the $N$ to a smaller $M$ to improve efficiency by selecting or aggregating original tokens, where $M < N$. The token compression process can be represented as a function $C$ that maps the original token sequence to a compressed sequence:

$H_{comp} = C(H) \in \mathbb{R}^{M \times d_t}$ — (Eq. 6)

where $H = [H^v; H^t] \in \mathbb{R}^{N \times d_t}$ is the concatenated sequence of projected visual tokens and embedded textual tokens, and $H_{comp}$ is the compressed token sequence.

> 💡 **Token Compression 形式化**: 就是一个映射函数 C: N 个 tokens → M 个 tokens（M < N），核心是在保留关键语义的前提下尽量压缩。

**Compression Ratio** is a widely-mentioned concept in token compression, defined as:

$R_{comp} = N / M$ — (Eq. 7)

where higher values (e.g., 4× or 8×) indicate greater compression levels, more compact semantic representations, and consequently larger efficiency gains.

Since the number of visual tokens typically exceeds that of textual tokens by substantial margins (e.g., by 20× [93]) in MLLMs, most existing token compression methods primarily focus on reducing $n_v$. To achieve more compact visual representations within MLLMs, two main types of redundancy can be exploited:

**(i) Intra-Visual Redundancy.** Visual content inherently contains redundant information. In images, numerous patches may represent background elements that are not crucial for understanding the primary subject matter. Similarly, in videos, consecutive frames often exhibit substantial similarity, resulting in temporal redundancy. This redundancy can be leveraged to reduce the number of visual tokens requiring processing, thereby improving computational efficiency while maintaining information quality.

**(ii) Cross-Modal Redundancy.** In multimodal tasks, particularly question-answering scenarios, textual input provides contextual guidance that can identify the most relevant visual tokens. For instance, when a question focuses on a specific object within an image, only visual tokens corresponding to that object may be necessary for accurate comprehension and response generation. By exploiting textual information, it becomes possible to selectively retain only those visual tokens that are pertinent to the specific task requirements.

> 💡 **两类冗余**:
> - **Intra-Visual（视觉内部）**: 图像中大量背景 patch 是冗余的；视频中相邻帧高度相似
> - **Cross-Modal（跨模态）**: 根据文本问题，只有与问题相关的视觉区域才是必要的
> 
> 这两类冗余对应了两大类压缩方法：purely-visual（利用 intra-visual 冗余）和 text-guided（利用 cross-modal 冗余）。

---

## 🔖 Section 总结

### 关键公式速查
| 公式 | 含义 |
|------|------|
| Eq. 1-3 | MLLM 三组件的前向传播 |
| Eq. 4-5 | Transformer 层的 FLOPs，关键项 $2n^2d$ |
| Eq. 6-7 | Token Compression 形式化定义和压缩率 |

### 核心洞察
1. Visual tokens 数量通常是文本 tokens 的 20 倍以上
2. 压缩主要针对 visual tokens（$n_v$）
3. 两类可利用的冗余：intra-visual + cross-modal
