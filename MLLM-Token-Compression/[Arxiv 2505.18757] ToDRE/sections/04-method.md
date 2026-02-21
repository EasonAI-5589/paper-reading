[← 返回 README](../README.md)

# 4 Visual Token Pruning with Token Diversity and Task Relevance

ToDRE is a two-stage, training-free, plug-and-play visual token compression framework:
- Stage 1: similarity-guided greedy search in LLM embedding space → maximally diverse subset
- Stage 2: adaptive task-relevance-based pruning within LLM decoder

## 4.1 Diversity-Driven Token Selection

Adopts a **greedy max-sum diversification algorithm** (from Gollapudi & Sharma, VLDB 2009) with two steps:

### Step 1: Pivot Token Selection

Use [CLS] attention from the **last layer of the vision encoder** as importance indicator:

```
a_[CLS] = Softmax(q_[CLS] · K_v^T / √d)
```

Pivot selection strategy depends on input type:
- **Image with AnyRes**: Select highest [CLS]-attention token from **global thumbnail** only
- **Image without AnyRes**: Select from all visual tokens
- **Video**: For each frame find highest [CLS]-attention token, then select the best among frames

> 💡 **批注**：Pivot 选择使用 global thumbnail 的 [CLS] attention 而非 local crops，因为 thumbnail 捕获最全局的信息。对于没有 [CLS] token 的 encoder（如某些 ViT 变体），random selection 也接近原始性能——说明算法对 pivot 选择不太敏感，鲁棒性好。

### Step 2: Greedy Max-Sum Diversification

从 pivot 开始，迭代选择与当前已选集合**累积相似度最小**的 token：

```
c^(t) = argmin_{v ∈ V \ C^(t-1)} [Σ_{c ∈ C^(t-1)} s(x_v, x_c)]
```

其中 `s(x_v, x_c) = cos(x_v, x_c)` 为余弦相似度。

等价地，这是在最大化 sum of distances（d = 1 - s）。

**增量更新**：每选一个新 token c^(t)，更新累积相似度：
```
∀v ∈ V \ C^(t): S_v^(t) = S_v^(t-1) + s(x_v, x_{c^(t)})
```

重复直到选够 k 个 token（如 k=288，约 10%）。

> 💡 **批注**：算法复杂度分析：
> - 初始化：O(n) 计算 pivot 与所有 token 的相似度
> - 每次迭代：O(n) 更新累积相似度 + O(n) 找最小值 = O(n)
> - 总计：O(kn) 其中 k << n
> - 实际上可以用矩阵运算加速：s ← s + X · X_c^T 是一个 matrix-vector product
>
> 这比 attention-based 方法的 overhead 小得多（不需要跑 LLM forward pass），而且在 embedding space 操作，可以直接用 normalized features 的点积。

> 💡 **批注**：与 DivPrune 的关键区别：DivPrune 也用 diversity，但 ToDRE 的 max-sum diversification 有明确的优化目标（最大化被选 token 间的距离之和），而不仅仅是启发式的 diversity metric。另外 ToDRE 额外有 Stage 2。

### Algorithm 1 Pseudocode

```
Input: V ∈ R^{n×d}, α ∈ R^n (CLS attention), k (# tokens to retain)
Output: C (indices of selected tokens)

# Pivot selection
p ← argmax(α)

# Greedy diversification
X ← row_normalize(V)          # L2 normalize for cosine similarity
C ← {p}
s ← X · X_p^T                 # cumulative similarity vector
s_p ← +∞                      # mask pivot

for i = 1 to k-1:
    c ← argmin(s)             # pick least similar to current set
    C ← C ∪ {c}
    s ← s + X · X_c^T         # update cumulative similarity
    s_c ← +∞                  # mask selected
return C
```

> 💡 **批注**：实现非常简洁。核心操作就是反复做 matrix-vector product 和 argmin。GPU 上可以高效并行。值得注意的是 L2 normalize 之后，cosine similarity 就等于点积，避免了重复计算范数。

## 4.2 Relevance-Driven Token Compression

Stage 2: 在 LLM decoder 的后半段找到一个合适的层，**一次性删除全部 visual tokens**。

### Layer Selection

基于经验观察（deeper layers 的 cross-modal attention 已衰减），只在 **fractional depth 7L/8** 处检查。

在选定层 ℓ 计算两个 cross-modal attention ratios：

```
α_{t→v}^(ℓ) = Σ_{i∈T} Σ_{j∈V} A_ij^(ℓ) / Σ_{i∈T} Σ_{j∈S∪V∪T} A_ij^(ℓ)
α_{v→t}^(ℓ) = Σ_{i∈V} Σ_{j∈T} A_ij^(ℓ) / Σ_{i∈V} Σ_{j∈S∪V∪T} A_ij^(ℓ)
```

### Pruning Criterion

All visual tokens are removed at layer ℓ if and only if **both** α_{t→v} and α_{v→t} < threshold τ.

> 💡 **批注**：关键设计选择：
> 1. **只检查 7L/8 处**而非逐层扫描——因为 deeper layers 的 attention ratio 趋于稳定，逐层检查是浪费
> 2. **双向检查**（t→v 和 v→t）确保 visual info 确实已完全迁移
> 3. **全部删除**而非渐进删除——因为 Stage 1 已经大幅减少了 token 数量，剩余 token 数量少，没必要分批删
> 4. 与 VTW 对比：VTW 用 post-hoc KL divergence 选层（需要额外 forward pass），ToDRE 直接用 attention ratio（forward pass 中顺手计算），更高效

> 💡 **批注**：Stage 2 对性能的影响是正向的（Table 5: 95.8% → 96.0%）！原因是删除了 task-irrelevant visual tokens 后减少了它们对 text reasoning 的干扰。这与 video 实验中 pruning 后性能反超 baseline 的现象一致。
