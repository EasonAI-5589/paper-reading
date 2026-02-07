# 3. Method

> 来源: SparseVLM (ICML 2025)

---

## 📄 原文

> 💡 **Section 概览**: 四步走 — (1) Attention 基础 (2) Text-guided 剪枝 (3) Token recycling (4) 计算量分析

---

### 3.1 Preliminary: Attention in VLM Decoders

标准 causal self-attention:

$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{D}}\right)$$

其中 $Q, K \in \mathbb{R}^{L \times D}$，$L$ 是所有 token（文本+视觉）的总长度。

> 💡 **批注**: SparseVLM 的核心思路是复用这个已经计算好的 $A$ 矩阵。由于 FlashAttention 不暴露 $A$，他们开发了兼容方案（Appendix B）。

---

### 3.2 Sparsification Guidance from Text to Vision

这是本文方法论的**核心**，分三步：

#### Step 1: Estimation of Visual Token Significance

从 self-attention 矩阵 $A$ 中提取 text→vision 的交互部分：

$$P = A_{i,j}, \quad (i,j) \in \{\mathbb{L}, \mathbb{I}\}$$

其中 $\mathbb{L}$ 是文本 token 集合，$\mathbb{I}$ 是图像 token 集合。

对所有文本 token 取平均，得到每个 visual token 的重要性评分：

$$\tilde{p} = \frac{1}{L_t}\sum_{i=1}^{L_t} P_i$$

> 💡 **批注**: 这一步很直观 — 如果一个 visual token 被很多 text token "关注"（attention weight 高），它就重要。计算量只有 $L_t \times L_v$ FLOPs，几乎免费。

#### Step 2: Relevant Text Token Selection (Text Raters)

**关键创新**: 不是所有文本 token 都适合当"评审"！介词、代词等与图像无关的词会引入噪声。

筛选方法：
1. 计算 vision embedding 和 text embedding 的相似度：$r = \frac{1}{L_v}\sum_j \text{Softmax}(H_v H_q^T)_j$
2. 选择相似度 > 均值的文本 token 作为 "rater"

![Figure 3](../images/07af15341c5e9df178eefcc5c589c1ad90facbd024dd1b198d6d024e17e5d065.jpg)
*Figure 3: 四个 benchmark 的 prompt 示例。颜色越深 = 与图像关系越大。浅色词（介词、代词）不应参与视觉 token 评估。*

> 💡 **Figure 3 批读**:
> ```
> Case 3 (医药问题): 
>   重要词 ✅: Tylenol, Advil, ibuprofen
>   无关词 ❌: the, in, is, of
>
> Case 4 (冰箱贴纸):
>   重要词 ✅: sticker, fridge
>   无关词 ❌: what, does, say
> ```
> **核心洞察**: 如果让 "the", "is" 这些词也参与评估 visual token，会稀释真正有用的信号。

#### Step 3: Sparsification Level Adaptation (Rank-based)

用 attention 矩阵 $P$ 的秩（rank）来衡量视觉 token 的冗余度：

$$N = \lambda \times (L_v - \text{rank}(P))$$

> 💡 **批注**:
> ```
> 直觉理解:
> ├── 如果 P 满秩 → visual tokens 线性独立 → 冗余度低 → 少剪
> └── 如果 P 低秩 → visual tokens 高度相关 → 冗余度高 → 多剪
> ```
> 这比 FastV 的固定比例剪枝更灵活。$\lambda$ 是缩放因子，如果某层 $N=0$ 就跳过不剪。

![Figure 2](../images/760be4367466ee10afe7387909f5183ba8903703b759b14caaf1f13adf98b7ed.jpg)
*Figure 2: SparseVLM 架构。Stage (a) 在进入 LLM 前预选 text raters。Stage (b) 在 LLM 各层中执行自适应稀疏化 + token recycling。*

> 💡 **Figure 2 批读**:
> ```
> 整体流程:
> Stage (a) - 预处理 (LLM 外部):
>   Input text → 筛选 text raters (与图像相关的词)
>
> Stage (b) - 逐层稀疏化 (LLM 内部):
>   每一层:
>   ├── 1. 提取 text raters → vision 的 attention
>   ├── 2. 计算 rank → 决定剪多少
>   ├── 3. 剪掉不重要的 visual tokens
>   └── 4. 回收部分被剪 token → 聚类压缩
> ```

---

### 3.3 Visual Token Recycling

被剪掉的 token 不全丢弃，而是**回收 + 重建**：

#### Token Aggregation
1. 从被剪 token 中回收 top-τ% 重要的
2. 用 k-nearest neighbor 密度峰值聚类算法分组
3. 用 $\rho_i \times \delta_i$ 指标选聚类中心（$\rho$ = 局部密度，$\delta$ = 到更高密度点的最近距离）

#### Token Reconstruction
每个聚类内的 token element-wise 求和，压缩成一个 token：

$$\mathcal{T}_k = \sum_{i=1}^{N_k} \mathbb{T}[i]$$

> 💡 **批注**:
> ```
> Recycling 流程:
> 被剪 100 个 token
>   → 回收 top-τ% (如 top-50%) = 50 个
>   → 聚类成 θ×50 个组 (如 10 个组)
>   → 每组压缩成 1 个 token
>   → 最终: 丢 100 个，回来 10 个
>
> 效果: 减少了 90 个 token 的计算量，但保留了关键信息
> ```
> 这是比 FastV 优的另一个关键点 — FastV 直接丢弃，SparseVLM 回收。

---

### 3.4 Theoretical Analysis of Computational Complexity

FLOPs 节省估算：

$$\text{Savings} \approx -2L_tL_vD + \sum_i DN_i(6D + 2N_i) - L_t^{i^2}L_v^i$$

> 💡 **批注**: 主要开销是 text rater 选择（一次性）和 rank 计算（每层）。但节省的是整个 FFN 和 attention 的计算。当 $N_i$ 足够大时，节省远大于开销。

---

## 💡 Section 总结

### 方法对比表
| 组件 | FastV | SparseVLM |
|------|-------|-----------|
| 剪枝依据 | attention score | text-guided attention |
| Text 引导 | ❌ | ✅ (text rater selection) |
| 自适应比例 | ❌ 固定 | ✅ rank-based |
| 被剪 token 处理 | 直接丢弃 | 回收 + 聚类压缩 |
| 需要训练 | 不需要 | 不需要 |
| 额外参数 | 无 | 无 |

### 核心洞察
1. **Text rater 选择是精髓** — 不是所有文本都有资格评价视觉 token
2. **Rank-based 自适应** — 比固定比例更合理，不同图像冗余度不同
3. **Token recycling 是保底** — 激进剪枝时减少信息损失
