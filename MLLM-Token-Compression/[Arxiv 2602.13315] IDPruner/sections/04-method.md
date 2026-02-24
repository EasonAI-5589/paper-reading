[← 返回 README](../README.md)

# 4. Harmonizing Importance and Diversity via MMR

## 📌 预览

Method 分三个子模块：(1) Token 重要性估计（借用 VisionSelector 的 DiffTopK 输出 + min-max 归一化）；(2) 语义冗余量化（余弦相似度）；(3) IDPruner 算法（MMR-based 迭代选择 + 高效更新策略）。

---

## 4.1 Token Importance Estimation

The analysis in Section 3 has demonstrated that the MMR mechanism effectively harmonizes diversity and importance. However, applying this framework in practice requires a computable importance metric.

To this end, we adopt the importance estimation mechanism of VisionSelector (Zhu et al., 2025a), which currently represents the state-of-the-art among importance-based pruning approaches. Specifically, this method employs a trainable estimation module coupled with a differentiable selection mechanism, DiffTopK, to learn token importance through end-to-end training. To maintain consistency with the training phase, we utilize the output of the DiffTopK mechanism as our raw importance scores, denoted as **w**.

> 💡 **关键设计选择：为什么用 VisionSelector？**
> VisionSelector 是 importance-based 方法中当时的 SOTA。IDPruner 的思路是：在最强的重要性估计器基础上，再加 MMR 多样性平衡，理论上应该进一步提升性能。这也是消融实验的逻辑起点（见 Appendix B，固定 VisionSelector 作为 importance scorer，比较 DPP/Naive Hybrid/MMR 三种组合策略）。
>
> ⚠️ **重要警告（对 STAR-Pro 的启示）**: 因为 IDPruner 使用了 VisionSelector，**IDPruner 不是 training-free 方法**。VisionSelector 需要端到端训练一个可学习模块（DiffTopK）。这意味着 IDPruner 在实际部署时需要训练代价，对新架构的泛化需要重新训练或微调。这是 training-free 方法（如 STAR-Pro）相比 IDPruner 的重要优势。

However, since MMR involves a direct subtraction between importance and similarity, both metrics must have comparable scales to prevent one from dominating the selection process. We therefore apply min-max normalization to the raw importance vector **w** to define the normalized importance metric:

Imp(v_i) = (w_i - min(**w**)) / (max(**w**) - min(**w**) + ε)

where ε is a small constant for numerical stability. This procedure maps importance scores to the interval [0, 1], ensuring they are commensurate with the similarity constraint.

> 💡 **归一化的必要性**: MMR 公式是 λ·Imp(v_i) - (1-λ)·Sim(v_i, v_j)，两项直接相减。如果 Imp 的量纲（比如 logit 值，范围可能是 -10 到 +10）和 Sim（余弦相似度，范围 [0,1]）不一致，λ 的物理含义就没了，实际上某一项会主导选择。min-max 归一化把两者都映射到 [0,1]，使 λ 真正控制权衡比例。

## 4.2 Quantifying Collective Redundancy

In addition to importance, the MMR framework requires a metric to quantify semantic redundancy. In the latent feature space of MLLMs, tokens representing similar visual concepts tend to cluster together. Thus, we define the pairwise similarity between a candidate token v_i and a reference token v_j using cosine similarity:

Sim(v_i, v_j) = (v_i^T v_j) / (||v_i|| · ||v_j||)

where ||·|| denotes the Euclidean norm. This metric enables the algorithm to identify tokens that are semantically similar to those already selected.

> 💡 **余弦相似度的选择**: 余弦相似度只看角度，不看模长，对于 token feature 来说很合适——不同 patch 的 feature 模长可能因图像局部亮度等因素不同，但语义相似性更多体现在方向上。Appendix E 验证了视觉 token 特征空间中所有 pairwise 角度都在 [0°, 90°] 范围内（峰值约 74°），保证余弦相似度恒为非负，从而 MMR 惩罚项始终是惩罚而非奖励。

## 4.3 IDPruner: An MMR-based Selection Strategy

Building upon the normalized importance and semantic similarity metrics defined above, we formally present the Importance and Diversity Pruner (IDPruner). This method harmonizes the two conflicting objectives within the MMR framework to iteratively construct the optimal subset. At each step t, IDPruner selects the token v* from the remaining candidates V\S_{t-1} by maximizing the following objective:

v* = arg max_{v_i ∈ V\S_{t-1}} [λ · Imp(v_i) - (1 - λ) · m_i]

where m_i = max_{v_j ∈ S_{t-1}} Sim(v_i, v_j) represents the maximum similarity between the candidate v_i and any token in the currently selected set, and λ ∈ [0, 1] is the hyperparameter balancing importance and diversity.

> 💡 **算法直觉**: 每一步都选「重要且与已选 token 不重复」的 token。m_i 是候选 token 与任意已选 token 的最大余弦相似度——用 max 而不是 mean，确保候选 token 与所有已选 token 都保持差异，而不仅仅是平均差异。这避免了「与部分已选 token 非常相似，只是与其他少数已选 token 距离远」的情况。

**Algorithm 1 IDPruner**

```
Require: Tokens V, Raw Importance Scores w, Budget K, Hyperparameter λ
Ensure: Pruned subset S

1: Imp ← (w - min w) / (max w - min w + ε)
2: S ← ∅, m ← fill(N, -1.0)
3: for t = 1 to K do
4:   if t = 1 then
5:     v* ← arg max_{v_i ∈ V} Imp(v_i)
6:   else
7:     v* ← arg max_{v_i ∉ S} [λ Imp(v_i) - (1 - λ) m_i]
8:   end if
9:   S ← S ∪ {v*}
10:  m ← max(m, Sim(V, v*))
11: end for
12: return S
```

> 💡 **算法细节**:
> - **第一步特殊处理（t=1）**: 第一个 token 直接选最重要的（因为 S 为空，无法计算相似度惩罚）
> - **高效更新（第 10 行）**: 每次只需更新 m 向量（与新选 token 的相似度 vs. 之前最大相似度取 max），无需重算所有 pair
> - **m 初始化为 -1.0**: 确保第一次更新时 max 操作正确工作

To minimize computational overhead, we adopt an efficient updating strategy. Instead of recomputing the similarity scores for all pairs at every step, we maintain a vector **m** ∈ R^N that tracks the maximum similarity for each candidate. After selecting v*, we simply update this vector: m_i ← max(m_i, Sim(v_i, v*)). This implementation reduces the computational complexity from O(K^2 N) to O(KN), rendering the overhead negligible relative to the model's forward pass. The complete procedure is summarized in Algorithm 1.

> 💡 **复杂度分析（重要！）**:
> - **朴素实现**: 每步 t 都需要对所有候选 token 重算与 S 中所有已选 token 的最大相似度 → O(K × K × N) = O(K^2 N)
> - **高效实现**: 维护 m 向量（每个候选 token 当前的最大相似度），每次只需计算新选 token 与所有候选的相似度并更新 m → O(K × N) = O(KN)
> - **实际开销**: 相比于模型前向传播（O(N^2)，attention 复杂度），O(KN) 可忽略不计。
>
> 💡 **与 DPP 对比（效率）**: DPP 需要计算 kernel 矩阵的行列式，复杂度更高（O(N^3) 或 O(K^2 N)），而且难以高效近似。MMR 的 O(KN) 复杂度是其相比 DPP 的重要工程优势。
>
> 💡 **对 STAR-Pro 方法的启示**: 如果 STAR-Pro 也涉及迭代选择，类似的高效 m 向量更新技巧同样可以应用，将复杂度从 O(K^2 N) 降到 O(KN)。

## 🔖 Section 总结

Method 章节非常简洁，分三步：(1) 借用 VisionSelector 输出作为重要性 + min-max 归一化；(2) 余弦相似度量化冗余；(3) MMR 迭代选择 + O(KN) 高效更新。整个算法实现简单，计算开销小，理论有保障。唯一的"外部依赖"是 VisionSelector（需要训练），这是 IDPruner 不是 training-free 方法的原因。
