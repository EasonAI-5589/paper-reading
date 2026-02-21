[← 返回 README](../README.md)

# 3 Methodology

## 📌 预览
方法论分四部分：MLLM 架构 preliminary → 质疑 importance-based 范式 → 提出 duplication-based DART → 理论分析证明 bounded error。

---

## 3.1 Preliminary

**Architecture of MLLM.** The architecture of Multimodal Large Language Models (MLLMs) typically comprises three core components: a visual encoder, a modality projector, and a language model (LLM). Given an image I, the visual encoder and a subsequent learnable MLP are used to encode I into a set of visual tokens eᵥ. These visual tokens eᵥ are then concatenated with text tokens eₜ encoded from the text prompt pₜ, forming the input for the LLM. The LLM decodes the output tokens y sequentially, which can be formulated as: yᵢ = f(I, pₜ, y₀, y₁, · · · , yᵢ₋₁).

> 💡 **标准 MLLM 三件套**: Visual Encoder + Projector + LLM。Vision tokens 和 text tokens concat 后送入 LLM 做 autoregressive decoding。这里 visual tokens 的数量远超 text tokens 是问题根源。

---

## 3.2 Beyond Token Importance: Questioning the Status Quo

Given the computational burden associated with the length of visual tokens in MLLMs, numerous studies have embraced a paradigm that utilizes attention scores to evaluate the significance of visual tokens, thereby facilitating token reduction. Specifically, in transformer-based MLLMs, each layer performs attention computation as illustrated below:

Attention(Q, K, V) = softmax(Q · K⊤ / √dₖ) · V, (1)

where dₖ is the dimension of K. The result of Softmax(Q · K⊤/√dₖ) is a square matrix known as the attention map. Existing methods extract the corresponding attention maps from one or multiple layers and compute the average attention score for each visual token based on these attention maps:

ϕ_attn(xᵢ) = (1/N) Σⱼ Attention(xᵢ, xⱼ), (2)

where Attention(xᵢ, xⱼ) denotes the attention score between token xᵢ and token xⱼ, ϕ_attn(xᵢ) is regarded as the importance score of the token xᵢ, N represents the number of visual tokens. Finally, based on the importance score of each token and the predefined reduction ratio, the most important visual tokens are selectively retained:

R = {xᵢ | (ϕ_attn(xᵢ) ≥ τ)}, (3)

where R represents the set of retained visual tokens, and τ is a threshold determined by the predefined reduction ratio.

> 💡 **现有范式形式化**: 公式 (1)-(3) 完整描述了 importance-based 方法的流程：attention map → 平均 attention score → 阈值筛选。这个范式被 FastV、SparseVLM 等方法共享。

---

**Problems:** Although this paradigm has demonstrated initial success in enhancing the efficiency of MLLMs, it is accompanied by several inherent limitations that are challenging to overcome.

One key limitation is disregarding the dynamic nature of token importance during pruning. For a token sequence {x₁, . . . , xₙ}, importance-based methods compute static token importance via a scoring function sᵢ = F(xᵢ|X), where X is the full token set. The strategy retains Top-k tokens:

X_pruned = arg max_{X'⊆X, |X'|=k} Σ_{xⱼ∈X'} sⱼ (4)

This implies an independence assumption: the score sⱼ remains unchanged for any subset X' ⊂ X, ignoring dynamic token interactions. For example, if two similar tokens xₚ, xᵧ have sₚ ≈ sᵧ, removing xᵧ should recalibrate sₚ as:

s'ₚ = F(xₚ|X' \ {xᵧ}) > sₚ, (5)

which leads to a bias in importance estimation Δ = s'ₚ − sₚ. This contradiction between static scoring and dynamic interaction can be quantified as:

E_{X'⊂X}[Σ_{xᵢ∈X'} |F(xᵢ|X') − F(xᵢ|X)|] (6)

> 💡 **形式化分析静态 scoring 的问题**: 公式 (4) 揭示了 importance-based 方法隐含的独立性假设——每个 token 的 score 不随其他 token 的去留而变化。公式 (6) 量化了这个假设带来的 bias。这在 information theory 中等价于：用 marginal MI 代替 conditional MI 来做 feature selection。

---

Additionally, Figure 1 visualizes the results of token reduction, revealing that selecting visual tokens based on attention scores introduces a noticeable bias toward tokens in the lower-right region of the image, those appearing later in the visual token sequence. However, this region is not always the most significant in every image. Further, we present the outputs of various methods. Notably, FastV generates more hallucinations than the vanilla model, while DART effectively reduces them. We attribute this to the inherent bias of attention-based methods, which tend to retain tokens concentrated in specific regions, often neglecting the broader context of the image. In contrast, DART removes highly duplication tokens and preserves a more balanced distribution across the image, enabling more accurate and consistent outputs.

> 💡 **可视化证据**: FastV 保留的 token 集中在图像右下角（position bias），导致 hallucination 比 vanilla 更严重。DART 保留的 token 空间分布更均匀。这是 duplication-based 方法的天然优势——重复 token 往往在空间上相邻（因为相邻 patch 视觉相似），所以去重后自然保留空间分散的 token。

---

Furthermore, methods relying on attention scores for token importance are incompatible with Flash Attention, compromising speed, and sometimes even underperforming random token reduction in effectiveness (See Fig. 2).

> 💡 一句话总结三个问题：不兼容 FA + 不如 random。简洁有力。

---

## 3.3 Token Duplication: Rethinking Reduction

Given the numerous drawbacks associated with the paradigm of using attention scores to evaluate token importance for token reduction, what additional factors should we consider beyond token importance in the process of token reduction? Inspired by the intuitive ideas mentioned in §1 and the phenomenon of tokens in transformers tending toward uniformity (i.e., over-smoothing) (Nguyen et al., 2023; Gong et al., 2021), we propose that token duplication should be a critical focus.

> 💡 **理论动机**: Over-smoothing 是 deep transformer 的已知现象——层数越深，token 表示越趋同。这意味着深层中必然存在大量重复 token，duplication-based 方法自然有丰富的可去除目标。

---

Due to the prohibitively high computational cost of directly measuring duplication among all tokens, we adopt a paradigm that involves selecting a minimal number of pivot tokens.

**Definition 1 (Pivot Tokens).** Let P = {p₁, p₂, . . . , pₖ} ⊆ X denote the pivot tokens, where k ≪ n and n is the total length of the tokens X = {x₁, x₂, . . . , xₙ}. The pivot tokens P are a subset of X, selected for their representativeness of the entire set.

> 💡 **Pivot tokens 设计**: k ≪ n（实际 ≤8 个），避免 O(n²) 的全量相似度计算。Pivot 的选择不太敏感（§5.2 验证了 random pivot 也 work），这说明方法的关键在 duplication 度量本身，而非 pivot 选择策略。

---

**Definition 2 (ε-duplicate Score).** The token duplication score between a pivot token pᵢ and a visual token xⱼ is defined as:

dup(pᵢ, xⱼ) = pᵢ⊤xⱼ / (‖pᵢ‖‖xⱼ‖), (7)

where ‖·‖ denotes the Euclidean norm. Two tokens pᵢ, xⱼ are ε-duplicates if

dup(pᵢ, xⱼ) > ε. (8)

With the ε-duplicate score, for each pivot pᵢ, the associated retained token set is defined as:

Rᵢ = {xⱼ | dup(pᵢ, xⱼ) ≤ ε} (9)

The final retained set is:

R = P ∪ (⋃_{pᵢ∈P} Rᵢ) (10)

where ε is the threshold dynamically determined for each pivot pᵢ based on reduction ratio. This ensures that only tokens that are sufficiently different from the pivot tokens are kept.

> 💡 **DART 核心算法**: 用 cosine similarity 度量 duplication → 保留与所有 pivot 都不太相似的 token。ε 不是手动设定的，而是根据目标 reduction ratio 动态确定的。计算复杂度 O(k·n)，k≤8 时近乎 O(n)。

---

Our method is orthogonal to the paradigm of using attention scores to measure token importance, meaning it is compatible with existing approaches. Specifically, we can leverage attention scores to select pivot tokens, and subsequently incorporate token duplication into the process.

However, this still does not fully achieve compatibility with Flash Attention. Therefore, we explored alternative strategies for selecting pivot tokens, such as using K-norm, V-norm, or even random selection. Surprisingly, all these strategies achieve competitive performance across multiple benchmarks. This indicates that our token reduction paradigm based on token duplication is not highly sensitive to the choice of pivot tokens. Moreover, it suggests that removing duplicate tokens may be more critical than identifying "important tokens", highlighting token duplication as a more significant factor in token reduction. Detailed discussion on pivot token selection is provided in §5.2.

> 💡 **Pivot 选择不敏感**: K-norm、V-norm、random 都 work，最差的 random 也比 importance-based 方法好 2.1%。这是论文最有力的论据之一：**duplication removal 本身比 pivot selection 策略重要得多**。这暗示 vision token 中存在大量冗余是一个 intrinsic property，几乎任何合理的去重方式都能利用它。

---

## 3.4 Theoretical Analysis

**Assumption 1 (Transformer Property).** For transformer property, we assume the following:

(A1). (Lipschitz continuity under Hausdorff distance). The model f is Lipschitz continuous with respect to the Hausdorff distance between token sets. Formally, there exists K > 0 such that for any two token sets X₁, X₂ ⊆ ℝᵈ:

‖f(X₁) − f(X₂)‖ ≤ K · d_H(X₁, X₂),

where d_H(X₁, X₂) ≜ max{sup_{x₁∈X₁} inf_{x₂∈X₂} ‖x₁ − x₂‖, sup_{x₂∈X₂} inf_{x₁∈X₁} ‖x₁ − x₂‖}.

(A2). (Bounded embedding). All tokens have bounded Euclidean norms: ‖x‖ ≤ B, ∀x ∈ X, where B > 0 is a constant.

> 💡 **假设**: A1 是一个较强的假设——transformer 对 token set 的 Hausdorff 距离是 Lipschitz 的。实际 transformer 有 softmax 非线性，Lipschitz 常数 K 可能很大。但作为理论框架这是合理的。A2 bounded embedding 在实践中通常成立（LayerNorm 后 token norm 有上界）。

---

**Lemma 1 (Bounded Distance).** min_{pᵢ∈P} |pᵢ − xⱼ| ≤ (2(1−ε))^{1/2} B, ∀xⱼ ∈ X \ R.

**Proof.** Using A2 and Definition 2, we obtain:

min_{pᵢ∈P} |pᵢ − xⱼ|² = min_{pᵢ∈P}(|pᵢ|² + |xⱼ|² − 2pᵢ⊤xⱼ) ≤ min_{pᵢ∈P}(B² + B² − 2ε · B · B) ≤ 2(1−ε)B²

Therefore, the duplication distance bound is given by: min_{pᵢ∈P} |pᵢ − xⱼ|² ≤ (2(1−ε))^{1/2} B

> 💡 **Lemma 1**: 被删除的 token 到最近 pivot 的距离有上界，由 ε 和 B 控制。ε 越大（去重越严格），上界越小，说明只删掉非常接近 pivot 的 token。

---

**Lemma 2 (Bounded Approximation Error).** Under Assumption 1, the Hausdorff distance between original and retained tokens satisfies:

d_H(X, R) ≤ √(2(1−ε)) · B.

**Proof.** For any x ∈ X:
• If x ∈ R, then inf_{r∈R} ‖x − r‖ = 0
• If x ∉ R, by definition and Lemma 1 there exists pᵢ ∈ P ⊆ R with ‖x − pᵢ‖ ≤ √(2(1−ε)) · B

Thus: sup_{x∈X} inf_{r∈R} ‖x − r‖ ≤ √(2(1−ε)) · B.

Since R ⊆ X, Hausdorff distance simplifies to: d_H(X, R) = sup_{x∈X} inf_{r∈R} ‖x − r‖ ≤ √(2(1−ε)) · B.

---

**Theorem 1 (Performance Guarantee).** Under Assumptions 1, the output difference between original and pruned token sets is bounded by:

‖f(X) − f(R)‖ ≤ K√(2(1−ε)) · B.

**Proof.** Direct application of Lipschitz continuity (A1) with Lemma 2: ‖f(X) − f(R)‖ ≤ K · d_H(X, R) ≤ K√(2(1−ε)) · B.

This provides a theoretical guarantee that DART preserves model output within a controllable bound, thereby supporting the trustworthiness and robustness of our method.

> 💡 **理论保证**: Output error ≤ K√(2(1−ε))B。这给出了一个 ε → 1 时 error → 0 的 graceful degradation。但注意：(1) K 可能很大且难以估计，所以 bound 可能很 loose；(2) 这个 bound 对 importance-based 方法同样可以建立类似结论，所以理论分析更多是 "DART 不会太差" 的保证，而非 "DART 优于 importance-based" 的证明。真正的优势还是靠实验验证。

---
