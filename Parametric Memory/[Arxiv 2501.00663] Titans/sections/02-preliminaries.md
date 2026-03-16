[← 返回 README](../README.md)

# 2 Preliminaries

## 📌 预览
介绍 Attention、Linear Attention（kernel trick）、现代线性 RNN 的记忆视角，以及 Memory Modules 的历史。

---

$\boldsymbol{x} \in \mathbb{R}^{N \times d_{\mathrm{in}}}$ be the input. We let $M$ be a neural network (neural memory module), be the attention mask. $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ be the query, key and value of the attention mechanism. When segmenting the sequence, we use $\mathsf{S}^{(i)}$ the $i$-th segment. Through the paper, we abuse the notation and use subscripts to refer to a specific element of a matrix, vector, or segments. For example, we let $\mathsf{S}_j^{(i)}$ be the $j$-th token in the $i$-th segment. The only exception is subscripts with $t$ which we reserved to index recurrence over time, or the state of a neural network at time $t$. Given a neural network $N$ and a data sample $x$, we use $N(x)$ (resp. $N^*(x)$) to refer to the forward pass with (resp. without) weight adjustment. Also, we abuse the notation and use $N^{(k)}$ to refer to the $k$-th layer of the neural network. In the following, we first, discuss the backgrounds for attention and its efficient variants followed by a review of modern linear RNNs. Finally, we discuss a memory perspective of these architectures that motivates us to design Titans.

> 💡 **符号说明批注**: 注意 $N(x)$ vs $N^*(x)$ 的区别——前者会更新权重（训练/记忆时），后者不更新（推理/检索时）。这个区分对理解后面的 memory read/write 很关键。

---

## 2.1 Backgrounds

### Attention

Transformers (Vaswani et al. 2017) as the de facto backbone for many deep learning models are based on attention mechanism. Given input $\boldsymbol{x} \in \mathbb{R}^{N \times d_{\mathrm{in}}}$, causal attention computes output $\mathbf{y} \in \mathbb{R}^{N \times d_{\mathrm{in}}}$ based on softmax over input dependent key, value, and query matrices:

$$
\mathbf{Q} = x\mathbf{W_Q}, \quad \mathbf{K} = x\mathbf{W_K}, \quad \mathbf{V} = x\mathbf{W_V},
$$
$$
\mathbf{y}_i = \sum_{j=1}^{i} \frac{\exp(\mathbf{Q}_i^\top \mathbf{K}_j / \sqrt{d_{\mathrm{in}}}) \mathbf{V}_j}{\sum_{\ell=1}^{i} \exp(\mathbf{Q}_i^\top \mathbf{K}_\ell / \sqrt{d_{\mathrm{in}}})}
$$

where $\mathbf{W_Q}, \mathbf{W_K}$, and $\mathbf{W_V} \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{in}}}$ are learnable parameters. Despite the power and effectiveness in recall, transformers need at least $N \times d$ operators to calculate the output, resulting in larger memory consumption and lower-throughput for longer sequences.

> 💡 **批注**: 标准 causal attention 复习。关键点：$O(N^2 d)$ 复杂度，N 是序列长度。

### Efficient Attentions / Linear Attention

To improve the memory consumption and throughput of softmax attention for longer sequences, various studies focused on I/O aware implementations (Dao 2024; Dao, D. Fu, et al. 2022), sparsifying the attention matrix, approximating the softmax, or developing kernel-based (linear) attentions. In linear attention, the softmax is replaced with a kernel function $\phi(.,.)$, such that $\phi(x,y) = \phi(x)\phi(y)$:

$$
\mathbf{y}_i = \frac{\phi(Q_i)^\top \sum_{j=1}^{i} \phi(K_j) V_j}{\phi(Q_i)^\top \sum_{\ell=1}^{i} \phi(K_\ell)}
$$

When choosing the kernel as identity matrix, this can be written in recurrent format:

$$
M_t = \mathcal{M}_{t-1} + K_t^\top V_t, \quad \mathbf{y}_t = Q_t \mathcal{M}_t
$$

> 💡 **批注**: Linear attention 的本质——把 softmax 换成可分解的 kernel，就能把 $O(N^2)$ 变成 $O(N)$。但代价是记忆变成了固定大小的矩阵 $M_t$，信息通过 **累加** 压缩进去。这就是后面说的"memory overflow"问题的根源。

### Modern Linear Models and Their Memory Perspective

One can define learning as a process for acquiring effective and useful memory. Building upon this, the hidden state of RNNs can be treated as a memory unit with read and write operations:

$$
M_t = f(M_{t-1}, x_t), \quad \mathbf{y}_t = g(M_t, x_t)
$$

In this perspective, the recurrence formula of linear Transformers is equivalent to additively compress and write keys and values into a matrix-valued memory unit $M_t$. When dealing with long context data, this additive nature results in memory overflow. Two directions to address this:

1. **Adding forget mechanism**: Adaptive forgetting gates — GLA, LRU, Griffin, xLSTM, Mamba2
2. **Improving the write operation**: Delta Rule — before adding a memory, first remove its past value. Gated DeltaNet adds a forget gate on top.

> 💡 **批注**: 这段是理解 Titans 设计动机的关键。现有方法的两条路：
> - **遗忘门**（Mamba/xLSTM 路线）：可以清除不需要的记忆，但写入仍是线性累加
> - **Delta Rule**（DeltaNet 路线）：先删后写，但没有遗忘机制
> - Titans 的 Neural Memory 同时解决了两个问题：非线性写入（深层 MLP）+ 自适应遗忘（weight decay）

### Memory Modules

Memory has always been one of the core parts of neural network designs. The idea of seeing linear layers as key-value (associative) memory backs to fast weight programs (JH Schmidhuber 1992). The two learning rules of Hebbian and delta are the most popular for fast weight programs. All these models, however, are based on momentary surprise, missing the token flow in the sequences, and most lack a forgetting gate.

> 💡 **批注**: Fast Weight Programs 是 Titans 的思想源头——Schmidhuber 1992 就提出了"动态快权重作为可写记忆"。Titans 的创新在于加入了 momentum（捕捉 token flow）和 weight decay（遗忘门），以及深层非线性记忆。

---

## 🔖 Section 总结

### 关键数字速查
| 模型 | 记忆结构 | 复杂度 | 遗忘 | 非线性 |
|------|---------|--------|------|--------|
| Transformer | 增长 KV | $O(N^2)$ | ✗ | ✓ (softmax) |
| Linear Transformer | 矩阵 $M$ | $O(N)$ | ✗ | ✗ |
| Mamba2 | 矩阵 $M$ | $O(N)$ | ✓ | ✗ |
| DeltaNet | 矩阵 $M$ | $O(N)$ | ✗ | ✗ (delta rule) |
| Gated DeltaNet | 矩阵 $M$ | $O(N)$ | ✓ | ✗ |
| **Titans (LMM)** | **深层 MLP** | $O(N)$ | ✓ | ✓ |

### 核心洞察
1. 所有序列模型可用「记忆结构 + 读写操作」统一描述
2. 线性模型的累加写入导致 memory overflow，遗忘门和 delta rule 是两种缓解方式
3. Titans 的根本区别：**深层非线性记忆**，从在线线性回归升级为在线非线性回归
