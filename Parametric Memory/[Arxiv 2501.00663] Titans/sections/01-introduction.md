[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
从 Transformer 的局限性出发，引入记忆视角分析现有架构，提出 5 个核心问题（Q1-Q5），然后概述 Neural Memory + Titans 的贡献。

---

"The true art of memory is the art of attention!"

— Samuel Johnson, 1787

Transformers, pure attention-based architectures (Vaswani et al. 2017), have been firmly established as state-of-the-art models in sequence modeling, mainly due to their in-context learning and ability to learn at scale (Kaplan blocks (Bietti et al. 2024), where they learn to store key-value associations and retrieve them by computing pairwise similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer is exclusively conditioned on the direct dependencies of tokens in the current context window. This accurate modeling of dependencies, however, comes with quadratic time and memory complexity in terms of the context length. In complex real-world tasks (e.g., language modeling (N. F. Liu et al. 2024), video understanding (C.-Y. Wu et al. 2019), long-term time series forecasting (H. Zhou et al. 2021)), the context window can become extremely large, making the applicability of Transformers challenging in these downstream tasks.

> 💡 **批注**: Transformer 的核心矛盾——精确建模依赖关系（softmax attention）vs. 二次复杂度。context window 越长，计算成本越高，这是所有后续工作的出发点。

To overcome the scalability issue of Transformers, recent studies aim to design different variants of linear Transformers (Kacham, Mirrokni, and P. Zhong 2024; Katharopoulos et al. 2020; S. Yang, B. Wang, Shen, et al. 2024), where softmax is replaced by a kernel function in the attention (see §2.1 for details), resulting in a significant drop in memory consumption. Despite efficiency and the ability to scale to longer context, linear Transformers do not show competitive performance compared to Transformers as the kernel trick makes the model a linear recurrent network, in which the data is compressed into a matrix-valued states (Katharopoulos et al. 2020). This, however, brings a contradictory fact about linear recurrent (or linear Transformers) models: On one hand, we use these linear models to enhance scalability and efficiency (linear vs. quadratic complexity), whose advantages is appeared for very long context; On the other hand, a very long context cannot be properly compressed in a small vector-valued or matrix-valued states (S. Wang 2024).

> 💡 **批注**: 指出 Linear Transformer 的**根本悖论**：你用线性模型就是为了处理长序列，但长序列恰恰是线性模型压缩不了的。这个 argument 很有力，直接动摇了 Mamba/GLA 等方法的理论基础。

Furthermore, beyond efficiency, most existing architectures–ranging from Hopfield Networks (Hopfield 1982) to LSTMs (Jürgen Schmidhuber and Hochreiter 1997) and Transformers (Vaswani et al. 2017)–face challenges when dealing with generalization, length extrapolation, and/or reasoning (Anil et al. 2022; Qin, Y. Zhong, and Deng 2024), all of which are inseparable parts of many hard real-world tasks. Although these architectures draw inspiration from the human brain, each of which are missing: (1) a crucial component for learning process—such as short-term memory, long-term memory, meta-memory, attending to current context, etc. (Cowan 2008); (2) how these components are interconnected systems that can operate independently; and/or (3) the ability to actively learn from data and memorize the abstraction of past history. We argue that in an effective learning paradigm, similar to human brain, there are distinct yet interconnected modules, each of which is responsible for a component crucial to the learning process.

> 💡 **批注**: 关键 insight——现有架构都只实现了人脑记忆系统的一部分。Hopfield = 联想记忆，LSTM = 门控记忆，Transformer = 短期精确记忆。没有一个模型同时拥有：(1) 完整的记忆组件，(2) 组件间的互联，(3) 主动学习和抽象的能力。

---

## Memory Perspective

> 💡 **Memory Perspective 要点预览**: 用记忆视角重新审视所有序列模型——RNN 是向量记忆 + 压缩写入，Transformer 是增长记忆 + 无压缩写入。这引出 5 个核心设计问题。

Memory is a fundamental mental process and is an inseparable component of human learning (Terry 2017). Without a properly functioning memory system, humans and animals would be restricted to basic reflexes and stereotyped behaviors. Accordingly, memory has been the inspiration for many seminal research in machine learning literature; e.g., Hopfield Networks (Hopfield 1982), LSTMs (Jürgen Schmidhuber and Hochreiter 1997), and Transformers (Vaswani et al. 2017).

Taking inspiration from the common definitions of memory and learning in neuropsychology literature (Okano, Hirano, and Balaban 2000), most existing architectures consider memory as a neural update caused by an input, and define learning as a process for acquiring effective and useful memory, given an objective. In this perspective, Recurrent Neural Networks (RNNs) (Williams and Zipser 1989) can be defined as models with a vector-valued memory module $M$ (also called hidden state) with two main steps: Given a new input $x_t$ at time $t$, the model (1) updates the memory using a function $f(\mathscr{M}_{t-1}, x_t)$ (with compression); and (2) retrieves the corresponding memory of input using a function $g(\boldsymbol{M}_t, \boldsymbol{x}_t)$ (see §2.1 for details). Similarly, Transformers can be seen as architectures with a growing memory and two similar steps. That is, the pair of key and value matrices acts as the model's memory, and the model: (1) updates the memory by appending the key and value to the memory (without compression), and (2) retrieves query vectors' corresponding memory by finding the similarity of query and key vectors, which is then used to weight the value vectors for the output.

> 💡 **批注**: 非常优雅的统一框架：
> | 模型 | 记忆结构 | 写入方式 | 读取方式 |
> |------|---------|---------|---------|
> | RNN | 向量 $M$ | 压缩写入 $f(M, x)$ | $g(M, x)$ |
> | Transformer | 增长的 KV 矩阵 | 追加（无压缩） | softmax 相似度 |
> | Linear Transformer | 固定矩阵 $M$ | 累加压缩 | 线性投影 |

This perspective, can help us better understand existing paradigms, their critical differences, and design more effective architectures. For example, the main difference between Transformers (Vaswani et al. 2017) and linear Transformers (Katharopoulos et al. 2020) is the memory structure as well as the memory updating step, in which linear Transformers compress the historical data into a fixed-size matrix-valued memory while Transformers keep all historical data (within the context length) without any compression. While both linear Transformers and linear RNNs (including state space models) compress the information in memory update step, the critical difference lies in the structure of the memory, where linear RNNs (vs. linear Transformers) use a vector-valued memory (vs. matrix-valued memory). Therefore, this perspective motivates us to ask: (Q1) What constitute a good structure for the memory? (Q2) What is a proper memory update mechanism? and (Q3) What is a good memory retrieval process?

Revisiting our understanding of human memory, it is neither a unitary process nor it serves a single function (Cowan 2008). In fact, memory is a confederation of systems–e.g., short-term, working, and long-term memory–each serving a different function with different neural structures, and each capable of operating independently (Willingham 1997). This fact motivates us to ask: (Q4) How to design an efficient architecture that incorporates different interconnected memory modules. Finally, storing a memory is a neural process that requires to encode and store the abstraction of the past. It can be over-simplification to assume a single vector or a matrix, whose parameters are encoding the data in a linear manner, are enough for storing long-term history. (Q5) Is a deep memory module needed to effectively store/remember long past?

> 💡 **5 个核心问题总结**:
> - **Q1**: 记忆结构该是什么样的？（向量/矩阵/深层网络）
> - **Q2**: 怎么更新记忆？（压缩/追加/选择性遗忘）
> - **Q3**: 怎么检索记忆？
> - **Q4**: 如何设计多记忆系统的架构？（短期+长期+持久）
> - **Q5**: 是否需要深层记忆？（线性 vs 非线性）
>
> 这 5 个问题就是全文的研究路线图。

---

## Contributions and Roadmap

In this paper, we aim to answer the above five questions by designing a long-term neural memory module, that can efficiently and effectively learn to memorize at test time. Building upon its design, we discuss how it can be incorporated into an architecture.

**Neural Memory (§3)**. We present a (deep) neural long-term memory that (as a meta in-context model) learns how to memorize/store the data into its parameters at test time. Inspired by human long-term memory system (Mandler 2014), we design this memory module so an event that violates the expectations (being surprising) is more memorable. To this end, we measure the surprise of an input with the gradient of the neural network with respect to the input in associative memory loss (see §3.1 for details). To better handle the limited memory, we present a decaying mechanism that consider the proportion of memory size and the amount of data surprise, resulting in better memory management. We show that this decay mechanism is in fact the generalization of forgetting mechanism in modern recurrent models (Dao and Gu 2024; Gu and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2024). Interestingly, we find that this mechanism is equivalent to optimizing a meta neural network with mini-batch gradient descent, momentum, and weight decay. Building upon tensorizing mini-batch gradient descent to use more matmul operations (Yu Sun et al. 2024), we present a fast and parallelizable algorithm to train our deep neural long-term memory.

> 💡 **Neural Memory 贡献解读**:
> - 核心 idea: 用 **梯度** 衡量 surprise（梯度大 = 输入与过去差异大 = 更值得记住）
> - 加入 **momentum**（过去的 surprise 也有影响）和 **weight decay**（遗忘机制）
> - 巧妙之处：这些恰好等价于标准的 SGD + momentum + weight decay，所以可以复用优化器的并行化技巧

**Titans Architectures (§4)**. After designing the long-term neural memory, an important remaining question is how to effectively and efficiently incorporate memory into a deep learning architecture. We present Titans, a family of deep models that consists of three hyper-heads: (1) Core: this module consists of the short-term memory, and is responsible for the main flow of processing the data (we use attention with limited window size); (2) Long-term Memory: this branch is our neural long-term memory module that is responsible to store/remember long past; (3) Persistent Memory: this is a set of learnable but date-independent parameters that encodes the knowledge about a task. Finally, as a proof of concept, we present three variants of Titans, in which we incorporate memory as: (i) a context, (ii) a layer, and (iii) a gated branch.

> 💡 **Titans 三组件**:
> | 组件 | 角色 | 参数状态 |
> |------|------|---------|
> | Core (Attention) | 短期记忆，处理当前窗口 | 固定（推理时不更新） |
> | Long-term Memory | 长期记忆，存储历史抽象 | **测试时仍在学习** |
> | Persistent Memory | 任务知识，数据无关参数 | 固定 |

**Experimental Results (§5)**. We perform experimental evaluations on language modeling, commonsense reasoning, recallintensive, needle in haystack, time series forecasting, and DNA modeling tasks. We observe that our Titan architecture outperforms all modern recurrent models as well as their hybrid variants (combining with sliding-window attention) across a comprehensive set of benchmarks. Furthermore, Titans outperforms Transformers with the same context window, and show competitive performance with Transformers that use the entire context. This results are achieved while, contrary to Transformers, Titans scale to larger than 2M context window size.

---

## 🔖 Section 总结

### 核心洞察
1. **记忆视角** 是全文最重要的 framework：把所有序列模型统一为「记忆结构 + 读写操作」
2. Linear model 的悖论：需要长序列才能体现优势，但长序列恰恰压缩不了
3. Titans 的三层记忆系统直接对应人脑：短期（attention）、长期（neural memory）、元记忆（persistent memory）
4. Neural memory 的测试时学习特性是关键差异——其他模型在推理时参数冻结
