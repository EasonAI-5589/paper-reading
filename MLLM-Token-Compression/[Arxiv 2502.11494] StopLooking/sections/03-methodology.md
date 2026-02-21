# 3. Methodology

## 3.1 Preliminary

MLLM 标准架构：Visual Encoder → Modality Projector (MLP) → LLM
- 图像 I 编码为 visual tokens e_v
- 与 text tokens e_t 拼接后送入 LLM
- LLM 自回归生成输出 y_i = f(I, p_t, y_0, ..., y_{i-1})

## 3.2 Beyond Token Importance: Questioning the Status Quo

### 现有 paradigm

在 transformer 中，attention map = softmax(QK^T/√d_k)，现有方法提取 attention map 后计算每个 visual token 的平均 attention score：

φ_attn(x_i) = (1/N) Σ Attention(x_i, x_j)

然后按 importance score 和预定 reduction ratio 保留最重要的 tokens：R = {x_i | φ_attn(x_i) ≥ τ}

### 核心问题分析

**静态 vs 动态交互的矛盾**：

Importance-based 方法隐含独立性假设——score s_j 不随子集 X' 变化：
X_pruned = argmax_{X'⊆X, |X'|=k} Σ s_j

但实际上，移除 x_q 后 x_p 的 importance 应该更新：
s_p' = F(x_p | X' \ {x_q}) > s_p

这导致估计偏差 Δ = s_p' - s_p，可量化为：
E_{X'⊂X} [Σ (F(x_i|X') - F(x_i|X))]

**Position bias**：Attention scores 偏向序列末端的 token，导致保留的 token 集中在图像的右下区域。FastV 甚至比 vanilla model 产生更多 hallucination。

**FlashAttention 不兼容**：需要关闭 FA 才能获取 attention scores，实际加速大打折扣。

## 3.3 Token Duplication: Rethinking Reduction

受 transformer 中 over-smoothing 现象启发（token 趋向均匀化），提出以 duplication 为核心的 token reduction。

### Definition 1: Pivot Tokens
P = {p_1, ..., p_k} ⊆ X, where k << n

### Definition 2: ε-Duplicate Score
dup(p_i, x_j) = cos(p_i, x_j) = p_i^T x_j / (||p_i|| ||x_j||)

Two tokens are ε-duplicates if dup(p_i, x_j) > ε

### Retention Set
对每个 pivot p_i: R_i = {x_j | dup(p_i, x_j) ≤ ε}
最终保留集: R = P ∪ (∪ R_i)

ε 根据 reduction ratio 动态确定。

### Pivot Selection 策略

可以用 attention scores、K-norm、V-norm、甚至 random 来选 pivot——**结果差异仅 1.2%**，说明 duplication-based reduction 对 pivot 选择不敏感。这进一步证明 "removing duplicates" 比 "finding important tokens" 更关键。

> 💡 **这是方法论上最精彩的部分**。DART 的核心创新是 shift the paradigm：从 "哪些 token 重要" 到 "哪些 token 冗余"。这看起来是一个简单的视角转换，但效果差异巨大。

> 💡 Pivot token 选择的鲁棒性是一个强有力的 ablation 结论。Max K-norm 和 min K-norm 选出的 pivot token 保留的 token 集合 overlap < 50%，但性能相当。这说明存在多个 "valid" token 子集，而非唯一的 "critical token set"——挑战了 importance-based 方法的基本假设。

> 💡 从 ToMe 的角度看，DART 可以理解为 "半个 ToMe"：ToMe 合并相似 token，DART 删除与 pivot 相似的 token。区别是 DART 不做合并（保留原始表示），且在 LLM 内部做而非 ViT 阶段。

## 3.4 Theoretical Analysis

### Assumption 1 (Transformer Property)
- (A1) Lipschitz continuity under Hausdorff distance: ||f(X₁) - f(X₂)|| ≤ K · d_H(X₁, X₂)
- (A2) Bounded embedding: ||x|| ≤ B

### Lemma 1 (Bounded Distance)
被裁掉的 token 到最近 pivot 的距离有界：min_{p∈P} |p - x_j| ≤ √(2(1-ε))·B

### Lemma 2 (Bounded Approximation Error)
Hausdorff distance: d_H(X, R) ≤ √(2(1-ε))·B

### Theorem 1 (Performance Guarantee)
||f(X) - f(R)|| ≤ K·√(2(1-ε))·B

> 💡 理论分析给出了 performance bound，但实际意义有限：K (Lipschitz constant) 对深度 transformer 可能非常大，bound 可能很松。不过作为定性分析——ε 越大（越严格的 duplication 标准），保留的信息越多——是合理的。

> 💡 这个理论框架的一个好处是它说明了 DART 的 information preservation 性质：被裁掉的 token 都有近似的 "代表" 在保留集中（pivot token）。这比 importance-based 方法的理论基础更强——后者没有这种 coverage guarantee。
