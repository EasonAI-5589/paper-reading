[← 返回 README](../README.md)

# 3. Method

## 📌 预览
Method 是全文核心，包含四个子部分：(1) VLM 注意力机制回顾；(2) 文本引导的视觉稀疏化（重要性评估 + rater 选择 + 自适应裁剪比例）；(3) Token Recycling（聚类重构）；(4) 计算复杂度理论分析。

---

In this section, we present our SparseVLM for efficient VLM inference. We first review the attention mechanism in VLMs and then introduce the detailed strategies for our visual sparsification including visual significance estimation, relevant text token selection, and sparsification level adaptation. We further propose token recycling to reduce information loss and provide a theoretical analysis of computation savings. The pipeline is shown in Figure 2.

> 💡 **Section 路线图**: Preliminary → 重要性评估 → Rater 选择 → 自适应比例 → Token Recycling → 理论分析

---

![Figure 2](../images/760be4367466ee10afe7387909f5183ba8903703b759b14caaf1f13adf98b7ed.jpg)
*Figure 2. The architecture of SparseVLM. In stage (a), text raters are pre-selected before entering the sparsification LLM. In stage (b), adaptive sparsification is performed on LLM layers, involving computing redundancy and the recycling of reconstructed tokens.*

> 💡 **Figure 2 批读**:
> - **Stage (a)**: 在进入 LLM 前，先用视觉-文本 embedding 的交叉注意力筛选 text raters
> - **Stage (b)**: 在 LLM 每一层中，用自注意力矩阵计算视觉 token 重要性 → rank-based 自适应裁剪 → token recycling
> - 整个流程是 **逐层递进** 的，每层都可能裁剪不同数量的 token

---

## 3.1. Preliminary: Attention in VLM Decoders

> 💡 **3.1 要点预览**: 回顾 VLM decoder 中的 causal self-attention，引出注意力矩阵 A 的定义。

VLM decoders typically rely on the causal self-attention from the original transformer architecture (Vaswani et al., 2017) for token interactions. Without loss of generality, we describe the single-head attention below. Formally, the self-attention matrix with logits $A \in \mathbb{R}^{L \times L}$, where $L$ denotes the length of a sequence with all kinds of tokens e.g. text and visual, is computed by

$$A = \text{Attention}(Q, K) = \text{Softmax}\left(\frac{QK^T}{\sqrt{D}}\right),$$

where the scalar $D$ represents the matrix dimension, and the $Q \in \mathbb{R}^{L \times D}$ and $K \in \mathbb{R}^{L \times D}$ are the query and key matrices, respectively. The keys and queries in a self-attention layer are computed in parallel by using multi-layer perceptrons to transform the input hidden states $H$ into a common space, where aligned interactions between modalities occur.

> 💡 **批注**: 标准的 causal self-attention。关键点是注意力矩阵 $A$ 中已经包含了跨模态交互信息——文本 query 对视觉 key 的注意力权重天然反映了视觉 token 的重要性。SparseVLM 的核心思路就是复用这个已有的 $A$，而不是额外训练一个评分网络。

Often, the matrix $A$ cannot be directly accessed due to FlashAttention-type (Dao et al., 2022) optimizations. Therefore, we develop an approach to extract $A$ while maintaining compatibility with the FlashAttention when applying our sparsification. Please refer to the Appendix B.

> 💡 **实现细节**: FlashAttention 为了节省显存不会显式存储注意力矩阵。SparseVLM 设计了 dual-flash attention 操作来兼容（见 Appendix B）。

---

## 3.2. Sparsification Guidance from Text to Vision

> 💡 **3.2 要点预览**: 这是方法的核心部分，包含三个子模块：
> 1. 视觉 token 重要性评估（用文本-视觉注意力打分）
> 2. Text rater 选择（筛选与视觉相关的文本 token）
> 3. 自适应稀疏化比例（用矩阵 rank 决定每层裁剪量）

---

### Estimation of Visual Token Significance

For a multimodal model, we aim to estimate an impact of deleting a single token from one modality to other modalities. In the VLM case, we need to quantify how relevant a visual token is to text tokens in order to determine whether it can be pruned. Therefore, we naturally reuse the self-attention logits from VLM's transformer layers as a reference since they already contain language-to-vision query results.

In particular, we take the interaction between the query-dimensional part of the language modality and the key-dimensional part of the vision modality as the basis for sparsification priority matrix $P \in \mathbb{R}^{L_t \times L_v}$, where $L_t$ and $L_v$ are the lengths of text and visual tokens, defined by

$$P = A_{i,j}, \text{ and } (i,j) \in \{\mathbb{L}, \mathbb{I}\},$$

where $\mathbb{L}$ and $\mathbb{I}$ denote the language instruction and image token sets, respectively.

> 💡 **批注**: 从完整注意力矩阵 $A$ 中提取文本→视觉的子矩阵 $P$。$P$ 的每个元素 $P_{i,j}$ 表示第 $i$ 个文本 token 对第 $j$ 个视觉 token 的注意力权重。

Next, we obtain a vector $\tilde{p}$ that estimates the significance of all visual tokens w.r.t. the text dimension as

$$\tilde{p} = [\tilde{p}_1, \tilde{p}_2, \dots \tilde{p}_{L_v}] = \frac{1}{L_t} \sum_{i=1}^{L_t} P_i,$$

where we use $\tilde{p}$ as an indicator for sparsification and a larger value in $\tilde{p}$ means higher significance of the corresponding visual token. Calculation of (3) costs $L_t \times L_v$ FLOPs only while the access to already computed $A$ is considered as free, which highlights low complexity of the SparseVLM.

> 💡 **批注**: 对 $P$ 按文本维度取平均，得到每个视觉 token 的重要性分数 $\tilde{p}$。计算量极小（$L_t \times L_v$ FLOPs），因为注意力矩阵已经算好了。

---

### Relevant Text Token Selection

It is not appropriate to use all text tokens as a reference for visual sparsification. Figure 3 shows four representative cases where we compute the correlation between the prompt and the image. Case 3 highlights Tylenol, Advil, ibuprofen, while sticker, fridge in case 4 are significant, where a large proportion of question tokens in light red include little visual relevance. Therefore, it is unreasonable to make insignificant text tokens to rate visual tokens, and we need to select relevant text tokens (i.e., "raters") for guidance.

> 💡 **批注**: 关键洞察 — 问题中的介词、代词等功能词与视觉内容无关，不应该参与视觉 token 评分。只有与图像内容相关的实义词（如 "Tylenol", "fridge"）才有资格做 "rater"。

---

![Figure 3](../images/07af15341c5e9df178eefcc5c589c1ad90facbd024dd1b198d6d024e17e5d065.jpg)
*Figure 3. Sample prompts from four representative multimodal benchmarks. The darker the word, the greater its relationship to the image and the more valuable it is for reference. We see that some words are irrelevant to the vision domain (e.g., prepositions and pronouns) and should not be considered for visual sparsification. It is best viewed in color.*

> 💡 **Figure 3 批读**:
> - 深色词 = 与图像高度相关的词（如名词、专有名词）
> - 浅色词 = 视觉不相关的词（介词、代词等）
> - Case 3 的 "Tylenol, Advil, ibuprofen" 和 Case 4 的 "sticker, fridge" 是有效 rater
> - 说明不加筛选地用所有文本 token 会引入噪声

---

Specifically, for an input image $x_v$, the vision embedding tokens $H_v$ can be computed as

$$H_v = WZ_v,$$

where $Z_v$ is the visual feature provided by visual encoder $Z_v = g(x_v)$, and $W$ is the projection matrix to convert $Z_v$ into vision embedding tokens $H_v$. For the language instruction $x_q$, it is transformed into text embedding tokens $H_q$ through the tokenizer. The above tokens both have the same dimensionality as the word embedding space. Then, we start to recognize which characters in the prompt are visually relevant and assign them the role of raters, which can be formulated as

$$s = \{i | r_i \geq m\}, i \in \{1, 2, ..., L_t\},$$

$$r = \frac{1}{L_v} \sum_{j=1}^{L_v} \left(\text{Softmax}(H_v H_q^T)\right)_j,$$

where $m = \text{mean}(r)$ and only candidates that exceed the $m$ threshold become raters. The strategy $s$ contains the indices of selected raters from the candidate list of $L_t$ tokens. The (6) costs $L_t \times L_v \times 2D$ FLOPs that is only computed once before the decoder layer processing.

> 💡 **Text Rater 选择流程**:
> 1. 用视觉 embedding $H_v$ 和文本 embedding $H_q$ 计算交叉注意力
> 2. 对每个文本 token，计算其与所有视觉 token 的平均相关度 $r_i$
> 3. 取均值 $m = \text{mean}(r)$ 作为阈值，超过均值的文本 token 成为 rater
> 4. 只计算一次，在进入 decoder 前完成（$L_t \times L_v \times 2D$ FLOPs）
>
> **注意**: 这个选择是在 embedding 空间完成的，不需要 decoder 的注意力矩阵。

---

### Sparsification Level Adaptation

Having obtained the token significance, we further propose a rank-based strategy to adaptively determine the level of vision sparsification at each decoder layer. Considering that a full-rank matrix implies that all its rows or columns are linearly independent, we use the rank of $P$ to demonstrate the redundancy of the visual tokens. We argue that the difference between the dimension and rank of $P$ reflects its redundancy and utilize a scaling factor $\lambda$ to determine the number of deletions as

$$N = \lambda \times (L_v - \text{rank}(P)).$$

We then remove $N$ visual tokens with the smallest values in $P$. Notably, if the result of $N$ in a decoder layer is 0, we skip the layer without sparsification. This stage requires $L_t \times L_v \times \min(L_t, L_v)$ FLOPs for rank computation.

> 💡 **Rank-based 自适应裁剪**:
> - **直觉**: 矩阵 rank 越低 → 行/列之间线性相关性越高 → 视觉 token 越冗余
> - **公式**: 裁剪数 $N = \lambda \times (L_v - \text{rank}(P))$
>   - $L_v - \text{rank}(P)$ 表示冗余维度数
>   - $\lambda$ 是缩放因子
> - **逐层自适应**: 每一层的 $P$ 不同 → rank 不同 → 裁剪量不同
> - 如果 $N=0$，跳过该层，不做裁剪
> - **计算成本**: SVD 分解，$L_t \times L_v \times \min(L_t, L_v)$ FLOPs

---

## 3.3. Visual Token Recycling

> 💡 **3.3 要点预览**: 被裁剪的 token 不直接丢弃，而是通过密度峰值聚类算法分组，然后加和重构为更紧凑的 token。

We progressively sparsify visual tokens in each layer in the decoder, which results in more discarded tokens at later stages. Despite being less significant, the pruned visual tokens with relatively large values in $P$ still contain certain information. To efficiently preserve more visual details with fewer tokens, we propose a token recycling strategy to aggregate and reconstruct tokens to be pruned.

---

### Token Aggregation

We first recycle the pruned visual tokens $\bar{h}_v$ with the top-$\tau$ (%) highest values in $P$ from the deleted pool. Then, we group $\bar{h}_v$ tokens with $k$-nearest neighbor density peak aggregation algorithm (Rodriguez, 2014) for adaptive token aggregation.

In particular, we first compute the local density $\rho_i$ of the ith token of total $\tau \times N$ recycled tokens according to its $k$-nearest neighbors $\mathcal{K}(\bar{h}_v^i)$ as

$$\rho_i = \exp\left(-\frac{1}{k} \sum_{\bar{h}_v^j \in \mathcal{K}(\bar{h}_v^i)} \|\bar{h}_v^i - \bar{h}_v^j\|_2^2\right).$$

Then, we compute the minimum distance between the recycled token $\bar{h}_v^i$ and any other token with higher density (denoted as the distance indicator $\delta_i$) that is defined by

$$\delta_i = \begin{cases} \min \|\bar{h}_v^i - \bar{h}_v^j\|_2, & \text{if } \exists j \text{ s.t. } \rho_j > \rho_i, \\ \max \|\bar{h}_v^i - \bar{h}_v^j\|_2, & \text{otherwise.} \end{cases}$$

We use $\rho_i \times \delta_i$ to indicate the score of each token, where the tokens with higher scores are likely to be cluster centers. Other tokens are then assigned to the nearest cluster center via cosine similarity. The FLOPs cost in this stage is $L_r \times (3L_r - 1) \times 2D + L_r$, where $L_r = \tau \times N$ is the length of recycled tokens, $C = \theta \times L_r$ is the number of cluster centers, and $\tau$ and $\theta$ are hyperparameters.

> 💡 **Token 聚类流程** (Density Peak Clustering):
> 1. 从被裁剪的 token 中回收 top-$\tau$% 的（重要性不那么低的）
> 2. 计算每个 token 的**局部密度** $\rho_i$（基于 KNN 距离）
> 3. 计算每个 token 的**距离指标** $\delta_i$（到更高密度 token 的最小距离）
> 4. $\rho_i \times \delta_i$ 高的 token 成为聚类中心
> 5. 其余 token 按余弦相似度分配到最近的聚类中心
>
> 这是 Rodriguez (2014) 的 density peak clustering 算法的应用。

---

### Token Reconstruction

Having performed token aggregation, the recycled tokens with similar semantics are classified into the same group. Then, the tokens $\mathbb{T} \in \mathbb{R}^{N_k \times D}$ in the $k$th group are reconstructed into a new compressed token $\mathcal{T}_k \in \mathbb{R}^{1 \times D}$ via the element-wise sum operation as

$$\mathcal{T}_k = \sum_{i=1}^{N_k} \mathbb{T}[i], k \in \{1, 2, \ldots, C\},$$

where $N_k$ is the token number of the kth group and the operation costs $D \times (L_r - C)$ FLOPs.

> 💡 **Token 重构**: 每个聚类内的 token 直接求和，得到一个压缩 token。简单高效，计算量只有 $D \times (L_r - C)$ FLOPs。最终 $L_r$ 个回收 token 被压缩为 $C$ 个重构 token，重新加入序列。

---

## 3.4. Theoretical Analysis of Computational Complexity

We consider the computation of multi-head attention and feed-forward network (FFN) modules in the FLOPs estimation. Assuming $N$ is the number of pruned tokens, $D$ is the hidden state size, which is the same as the intermediate size in FFN, the FLOPs for one Transformer layer can be reduced by $6(N-C)D^2 + 2(N-C)^2D$. Besides, our partial step introduces minimal computation with the details provided in Appendix C. Thus, we estimate the FLOPs savings as the reduction part minus the additional overhead:

$$\underbrace{\sum_i 6(N_i - C_i)D^2 + 2(N_i - C_i)^2 D}_{\text{reduction part}} - \underbrace{2L_t L_v D - \sum_i L_t^i L_v^i(1+\min(L_t^i, L_v^i)) - (6{L_r^i}^2 + 2L_r^i)D - L_r^i}_{\text{overhead part}}$$

$$\approx -2L_t L_v D + \sum_i DN_i(6D + 2N_i) - {L_t^i}^2 L_v^i,$$

where $i \in \{1, 2, \ldots, \Omega\}$ and $\Omega$ is the number of total layers, and $x = \tau \times \theta$ is a very small decimal that can be ignored.

> 💡 **理论分析**:
> - **节省**: 每层减少 $6(N-C)D^2 + 2(N-C)^2 D$ FLOPs（attention + FFN）
> - **开销**: text rater 计算 + rank 计算 + 聚类，但这些是轻量操作
> - **净节省**: reduction - overhead ≈ 正值，当裁剪足够多 token 时
> - $x = \tau \times \theta$ 很小可忽略，说明回收的额外开销可忽略不计

---

## 🔖 Section 总结

### SparseVLM 完整流程
```
输入图像 + 问题
    ↓
[Stage a: Text Rater 选择]
    H_v × H_q^T → 交叉注意力 → 选出视觉相关文本 token
    ↓
[Stage b: 逐层自适应稀疏化]  (对每个 decoder layer)
    1. 提取 P = A[text, vision]（文本→视觉注意力子矩阵）
    2. 用 raters 计算视觉 token 重要性 p̃
    3. rank(P) → 决定裁剪数量 N
    4. 裁剪重要性最低的 N 个视觉 token
    5. 回收 top-τ% → 密度峰值聚类 → 求和重构 → 重新加入
    ↓
输出（更少的视觉 token，更快的推理）
```

### 核心洞察
1. **复用注意力矩阵**而非额外训练评分网络 → training-free
2. **文本 rater 筛选**避免无关词干扰视觉评分
3. **Rank-based 自适应**让每层裁剪量自动匹配冗余程度
4. **Token recycling**通过聚类压缩减少信息损失
