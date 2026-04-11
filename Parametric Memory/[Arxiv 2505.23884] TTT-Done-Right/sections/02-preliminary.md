[← 返回 README](../README.md)

# 2. Preliminary

## 📌 预览

本节建立 LaCT 的理论基础，分为两个部分：2.1 节回顾 TTT 的数学形式（update 操作和 apply 操作），2.2 节通过计算访存比的定量分析揭示为何小 chunk size 导致 GPU 效率低下，从而为大 chunk 策略提供理论依据。本节还包含 LaCT block 的整体架构图（Figure 2）。

---

# 2.1 Test-Time Training

Consider a one-dimensional sequence of $N$ tokens $\mathbf { x } = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } ]$ , where each token $x _ { i } \in \mathbb { R } ^ { d }$ Following attention formulation, each input tokens $x _ { i }$ is projected into query $( q _ { i } )$ , key $( k _ { i } )$ , and value $( v _ { i } )$ vectors. For clarity, we assume all these vectors $q _ { i } , \dot { k } _ { i } , \dot { v } _ { i } \in \mathbb { R } ^ { d }$ .

Test-Time Training (TTT) [2] introduces a neural network with rapidly adaptable weights—called fast weights [3]—that are updated during both training and inference to dynamically store context information. This contrasts with the slow weights (i.e., model parameters) that are frozen during inference. Formally, TTT defines fast weights in the form of a neural network: $f _ { W } ( \cdot ) : \mathbb { R } ^ { d } \to \mathbb { R } ^ { \breve { d } }$ parameterized by the fast weights $W$ , and it involves two primary operations:

$$
W \gets W - \eta \nabla _ { W } \mathcal { L } \big ( f _ { W } ( k ) , v \big )
$$

where $\mathcal L ( \cdot , \cdot )$ is a loss function between the transformed key $f _ { W } ( k )$ and the value $v$ , commonly Mean Squared Error, designed to encourage the network to associate keys with corresponding values. $\eta$ is the learning rate. Intuitively, this learning objective is to encode the KV cache into a neural memory with fixed state size as accurate as possible [4].

> 💡 **批注：TTT Update 操作的直觉理解**
>
> 这个公式的本质是让 fast weight 网络 $f_W$ 学会将 key $k$ 映射到 value $v$。可以把 $f_W$ 想象成一个"神经网络形式的 KV 存储"：
> - **训练阶段**：每看到一对 (k, v)，就用梯度下降更新 $W$，使网络"记住" key → value 的映射关系
> - **推理阶段（apply）**：用当前的 query $q$ 查询网络，得到输出 $o = f_W(q)$
>
> 与 softmax attention 相比，TTT 不需要存储所有历史 KV 对（O(N) 存储），而是将它们压缩进固定大小的 $W$（O(1) 存储），但代价是压缩可能有损。

# Apply operation:

$$
o = f _ { W } ( q ) ,
$$

where the updated fast weights $W$ are used to compute the output vector $o$ given the query $q$ . The per-token TTT layer iteratively perform the update and apply operations on each token $x _ { i }$ in sequence.

> 💡 **批注：Update 与 Apply 的解耦**
>
> TTT 的 update（写入记忆）和 apply（读取记忆）是两个解耦的操作，这是 LaCT 灵活性的来源：
> - **Per-token TTT**：每个 token 先 update 再 apply，形成严格的因果依赖链
> - **LaCT**：在一个大 chunk 内，可以先 update 所有 tokens（批量写入）再 apply（批量读取），也可以先 apply 再 update，形成不同的"注意力掩码等价模式"（详见 Figure 3）
>
> 这种解耦使得 LaCT 可以用一套框架表达多种数据依赖结构，无需为每种模态设计专用架构。

---

# 2.2 Challenges in Efficient Implementation

Frequent online update of fast weights is inefficient due to memory bandwidth limitations. Consequently, previous works [11, 12, 13, 14, 15] often employ customized kernels that keep fast weights in SRAM across updates to reduce memory load. However, this strategy typically requires fast weights to evolve mostly independently within SMs to reduce communications, which is not valid for large nonlinear states (e.g., the nonlinear SwiGLU fast weight in Sect. 3.1 and the Muon update in Sec. 3.2). Moreover, developing such kernel code is cumbersome, with far longer development cycles than native PyTorch code, hindering rapid research exploration.

> 💡 **批注：自定义 Kernel 的局限性**
>
> 现有 TTT 实现（如 Mamba 的 selective scan）通过将 fast weights 保持在 SRAM（芯片内高速缓存）中跨 token 更新来避免反复读写 HBM（高带宽内存）。但这要求每个 SM（流多处理器）独立演化自己的 fast weights，这对于：
> - **大型非线性网络**（如 SwiGLU MLP）：中间激活太大，无法完全放入 SRAM
> - **Muon 等复杂优化器**：需要跨 SM 通信（all-reduce），破坏了独立演化的假设
>
> 因此，自定义 kernel 方案从根本上限制了 fast weight 的网络架构和优化器的选择空间。

---

On the other hand, a PyTorch-based implementation, while simpler, is typically bounded by memory speed. As an illustration, consider a PyTorch implementation of simple MLP fast weight, the core of which is a matrix multiplication between fast weight (e.g., $h \times h$ matrix) and the mini-batch input $b \times h$ where $\mathbf { b }$ is the chunk size). The ideal compute-to-memory ratio is:

$$
r = { \frac { 2 h ^ { 2 } b } { 2 h ^ { 2 } + 4 h b } } = { \frac { h / 2 } { 1 + { \frac { h } { 2 b } } } } = { \frac { b } { 1 + { \frac { 2 b } { h } } } } \leq \operatorname* { m i n } ( h / 2 , b ) .
$$

Here, $2 h ^ { 2 } b$ is the FLOPs to for matrix multiplication, the denominator $2 h ^ { 2 } + 4 h b$ is the memory workload for two input matrices and the output in BF16 (2 bytes). Small fast weight size (e.g., $h = 6 4$ ) or small chunk size (e.g., $b = 1 6$ ) will bound the ratio $r$ far below the theoretical peak (e.g., 290 FLOPs per byte on H100), making the operation memory-bound and limiting compute usage.

> 💡 **批注：计算访存比分析——大 chunk 为何有效**
>
> 这是全文最重要的定量分析之一。公式的含义：
>
> $$r \leq \min(h/2, b)$$
>
> - **$h/2$ 限制**：fast weight 矩阵维度 $h$ 太小时，即使 batch 再大也没用。例如 $h=64$ 时，$r \leq 32$，远低于 H100 的 290 FLOPs/byte 峰值。
> - **$b$ 限制**：chunk size $b$ 太小时，batch 维度成为瓶颈。例如 $b=16$ 时，$r \leq 16$，利用率不到 6%。
>
> **LaCT 的解法**：同时增大 $h$（大非线性 fast weight）和 $b$（大 chunk），使 $r$ 接近 GPU 理论峰值。
>
> | 参数设置 | 计算访存比上界 | H100 利用率估计 |
> |---------|-------------|--------------|
> | h=64, b=16 (现有方法) | min(32, 16) = 16 | < 6% |
> | h=1024, b=2048 (LaCT) | min(512, 2048) = 512 | ~70% |

---

In light of this, we advocate for using large chunk sizes (from 2048 to 1M). This allows us to achieve higher throughput (Fig. 1a) leading to better performance in less training wall-clock time(Fig. 1d). Our design also allows the state size to be scaled up efficiently(Fig. 1b), leading to significant results improvement with such scaling (Fig 1c, Fig. 7a). Our architecture achieves a state-to-parameter size ratio $\geq 4 0 \%$ , which is an order of magnitude larger than previous methods' ratio of $0 . 1 \%$ to $5 \%$ . Detailed pseudocode is provided in Appendix 1.

Parallelism over the sequence length dimension, in addition to the batch and head dimensions, is crucial to achieve high occupancy when handling long sequences (where the batch size is often small). Linear Attention variants like Mamba [12], Gated Linear Attention [13] and DeltaNet [15] enable such parallelism by utilizing the associative property of linear recurrence. Attention [1, 16] can be parallelized along the sequence length dimension using online softmax [17], a key improvement in FlashAttention-2 [16] over FlashAttention-1 [18]. For test-time training with non-linear updates, sequence dimension parallelism can only be implemented within online chunks, further motivating the use of extremely large chunk sizes. When implementing large-chunk TTT with PyTorch, this sequence dimension parallelism within a device across multiple thread blocks is automatically handled by PyTorch and low-level compilers. An example of such sequence parallelism across multiple devices is provided in Section 3.4, with pseudocode in Appendix 3.

> 💡 **批注：序列维度并行性——大 chunk 的额外优势**
>
> 这段揭示了大 chunk 的另一个深层次好处：**序列维度并行**。
>
> - **线性 attention**（Mamba/GLA/DeltaNet）：利用线性递推的结合律，可以将序列并行拆分（类似前缀和并行）
> - **Softmax attention**（FlashAttention-2）：利用 online softmax 在序列维度上并行
> - **非线性 TTT（per-token）**：每个 token 的 fast weight 依赖前一个，无法并行 → 序列必须串行处理
> - **非线性 TTT（LaCT 大 chunk）**：chunk 内部可以并行，chunk 间串行 → chunk 越大，并行度越高
>
> 这是为什么非线性 fast weight + 大 chunk 的组合特别有效：弥补了非线性导致无法使用线性递推结合律的劣势。

---

![Figure 2](../images/9df32abf87dc1c95f8dc5a4fadf307885353253bec974b2f01fcb23185a3b376.jpg)
*Figure 2: The basic diagram for a LaCT block. The large-chunk TTT layer updates the fast weight $W$ to store historical context information, while the window attention handles the locality and internal structures within the chunk. The solid line denotes the information flow over model depth and the dashed line denotes the information flow over time (i.e., the fast weight $W$ passing through chunks). Various instantiations in Sec. 4 use different chunk sizes and window attention types according to the specific data structure. Additionally, window attention and large-chunk TTT layers can be combined within the same layer by sharing the QKV and summing their outputs; this in-layer mixing is used in our language modeling and video generation experiments (see Appendix 2 for such pseudocode).*

> 💡 **Figure 2 批读**：
> - **整体架构**：LaCT block = Window Attention（局部结构）+ Large-Chunk TTT（长程记忆）+ Feed-Forward（通道混合），每个子层都有残差连接。
> - **信息流向**：实线表示深度方向（layer-by-layer），虚线表示时间方向（fast weight $W$ 在 chunk 间传递）。这个双流设计是 LaCT 区别于标准 Transformer 的关键：模型既有深度方向的特征变换，也有时间方向的记忆传递。
> - **灵活性**：不同任务使用不同的 chunk size 和 window attention 类型（双向/因果），展示了 LaCT 框架的模块化设计。
> - **In-layer Mixing**：在语言模型和视频生成任务中，Window Attention 和 TTT 层共享 QKV 并将输出相加（类似 GAU），节省了一次 QKV 投影的计算开销。
> - **核心权衡**：TTT 层用固定大小的 $W$ 压缩长程上下文（$O(1)$ 存储，有损），Window Attention 用精确的局部注意力处理 chunk 内结构（$O(L_w^2)$ 计算，无损）。两者互补，共同覆盖长程和短程依赖。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| H100 理论峰值计算访存比 | 290 FLOPs/byte |
| 小 h=64, b=16 时计算访存比上界 | 16 FLOPs/byte（< 6% 利用率）|
| LaCT 状态与参数比 | ≥ 40% |
| 现有方法状态与参数比 | 0.1% ~ 5% |

### 核心洞察
1. **计算访存比是核心约束**：$r \leq \min(h/2, b)$，同时增大 fast weight 维度 $h$ 和 chunk size $b$ 是提升 GPU 利用率的必要条件。
2. **自定义 kernel 的死胡同**：将 fast weight 保持在 SRAM 的策略虽然减少了内存访问，但限制了 fast weight 只能是小型线性网络，且无法使用需要跨 SM 通信的 Muon 等优化器。
3. **非线性 fast weight 的序列并行**：非线性更新无法使用线性递推结合律，因此只能在 chunk 内部并行。大 chunk = 高并行度，这在数学上是必然的。
4. **双流架构设计**：fast weight 沿时间维度传递（长程记忆），特征沿深度维度传递（逐层变换），二者正交互补。
