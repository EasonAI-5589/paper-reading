# 2. Preliminaries

## 2.1 Multimodal Large Language Models

### MLLM 三组件架构

> Modern MLLMs typically adopt a **three-component architecture**:
> - A **vision encoder** (often based on SigLIP or CLIP) that processes visual inputs into high-dimensional feature representations
> - A **projector** that aligns visual features with the language model's embedding space
> - A powerful **LLM** that performs multimodal alignment, reasoning and generation
>
> ==标准 MLLM 架构：Vision Encoder → Projector → LLM==

### 形式化定义

> Let $X^v = \{I_1, I_2, ..., I_{n_v}\}$ denote the input image sequence or video frames, and $X^t = \{x_1, x_2, ..., x_{n_t}\}$ represent the textual token sequence.

**Vision Encoder:**
> $$Z^v = E_v(X^v) \in \mathbb{R}^{n_v \times d_v}$$
> where $n_v$ denotes the number of visual tokens and $d_v$ represents the feature dimension.
>
> ==Vision Encoder 输出：$n_v$ 个 visual tokens，每个 $d_v$ 维==

**Projector:**
> $$H^v = P(Z^v) \in \mathbb{R}^{n_v \times d_t}$$
> where $d_t$ denotes the embedding dimension of the target language model.
>
> ==Projector 作用：将视觉特征从 $d_v$ 维映射到 LLM 的 $d_t$ 维空间==

**Large Language Model:**
> $$Y = G([H^v; E_t(X^t)])$$
> where $E_t(\cdot)$ represents the embedding layer of the LLM, $[·;·]$ denotes concatenation.
>
> ==LLM 处理：拼接 visual tokens 和 text tokens，生成输出==

---

### 计算复杂度分析

> Given a sequence of length $n$, hidden dimension $d$, and FFN intermediate dimension $m$:
> $$\text{Layer FLOPs} = 4nd^2 + 2n^2d + 2ndm$$

> For an $L$-layer Transformer:
> $$\text{Total FLOPs} = L \times (4nd^2 + 2n^2d + 2ndm)$$
> where $n = n_t + n_v$ is the overall sequence length.
>
> ==关键：$2n^2d$ 是 attention 的二次项，序列变长时急剧增长==

> As the sequence length $n$ increases, the quadratic complexity term $2n^2d$ in the attention mechanism grows rapidly, leading to prohibitive computational overhead.
>
> ==瓶颈场景：(1) 高分辨率图像/长视频 (2) 多轮对话/复杂推理==

---

## 2.2 Token Compression

### 形式化定义

> **Token compression** aims to transform an original visual sequence into a shorter, semantically-equivalent representation while maintaining downstream task performance:
> $$\hat{Z}^v = f_\text{comp}(Z^v) \in \mathbb{R}^{M \times d}$$
> where $M < N$ and $N$ is the original sequence length.

### 压缩率 (Compression Ratio)

> $$R_\text{comp} = \frac{N}{M}$$
> Higher values (e.g., 4× or 8×) indicate greater compression levels, more compact semantic representations, and consequently larger efficiency gains.
>
> ==压缩率 = 原始长度 / 压缩后长度，4× 表示压缩到 1/4==

### 两种冗余来源

> Since the number of visual tokens typically exceeds that of textual tokens by substantial margins (e.g., by **20×**), most existing methods primarily focus on reducing $n_v$.
>
> ==视觉 tokens 数量 >> 文本 tokens（约 20 倍），所以主要压缩视觉部分==

#### (i) Intra-Visual Redundancy

> Visual content inherently contains redundant information. In images, numerous patches may represent **background elements** that are not crucial for understanding the primary subject matter. Similarly, in videos, consecutive frames often exhibit **substantial similarity**, resulting in temporal redundancy.
>
> ==视觉内部冗余：图像背景区域冗余 + 视频帧间相似==

#### (ii) Cross-Modal Redundancy

> In multimodal tasks, particularly question-answering scenarios, textual input provides **contextual guidance** that can identify the most relevant visual tokens. For instance, when a question focuses on a specific object within an image, only visual tokens corresponding to that object may be necessary for accurate comprehension.
>
> ==跨模态冗余：文本 query 可以指导只保留相关的 visual tokens==

---

## 💡 Key Takeaways

| 概念 | 说明 |
|------|------|
| MLLM 架构 | Vision Encoder → Projector → LLM |
| 计算瓶颈 | $O(n^2)$ attention 复杂度 |
| 压缩率 | $R = N/M$，值越大压缩越狠 |
| 视觉内部冗余 | 背景区域 + 帧间相似 |
| 跨模态冗余 | 与 query 无关的视觉信息 |

---

*[返回论文目录](../README.md)*
