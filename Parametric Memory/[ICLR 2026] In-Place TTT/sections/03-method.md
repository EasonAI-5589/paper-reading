[← 返回 README](../README.md)

# 3. In-Place Test-Time Training

## 📌 预览
本节提出 In-Place TTT 框架。核心思路：不替换 attention、不加新模块，而是**复用已有的 MLP block 的 $W_{down}$ 作为 fast weights**，配合 chunk-wise 更新实现高效推理时适应（3.1）；设计 LM-Aligned 目标函数，用 Conv1D 构造包含未来 token 信息的 target，与 NTP 目标对齐（3.2）；理论证明 LM-Aligned target 能提升正确 token 的 logit 而 reconstruction target 做不到（3.3）；最后给出基于 prefix sum 的 Context Parallelism 实现细节（3.4）。

---

## 3.1 Overall Framework

### Repurposing MLP Blocks for In-Place Adaptation

Previous TTT research has largely positioned it as a potential solution to replace the attention mechanism. However, these prior studies were typically conducted at moderate scales, a regime vastly different from that of modern, billion-parameter LLMs. Consequently, replacing the core attention mechanism—whose learned properties are critical to an LLM's capabilities—is a high-risk architectural modification. Moreover, introducing any new, randomly-initialized layer also creates a conflict with the billions of trained parameters of LLMs, necessitating costly and often impractical retraining to resolve this imbalance.

> 💡 **批注**: 这段话直接否定了以往 TTT 工作（包括 TTT-Linear、Titans 等）"替换 attention"的路线。两个关键理由：(1) attention 的学到的属性对 LLM 能力至关重要，替换风险太高；(2) 新增随机初始化层与已有数十亿参数的 LLM 存在冲突，需要重新训练。这意味着 TTT 要想在大模型时代落地，**必须走"无创"路线**。

Our core insight is to sidestep these challenges entirely. Instead of replacing or adding components, we repurpose a ubiquitous module–the Multi-Layer Perceptron (MLP) block–to also serve as the fast weights. Recalling the TTT formulations in Section 2, there exist no constraints on the choice of fast weights, i.e., any parameters can serve as fast weights updated via the TTT mechanism. In particular, the MLP blocks in Transformers can also be viewed as a form of key-value memory (Geva et al., 2020), functioning as a "slow weights" for the vast, general knowledge acquired during pre-training. It is therefore a natural extension to leverage this same component to also function as the adaptive "fast weights", dynamically internalizing transient, in-context information at inference time.

> 💡 **批注**: 这是全文最核心的设计洞察。Geva et al. (2020) 证明了 **MLP 层 ≈ key-value memory**——$W_{up}$ 的行是 key pattern，$W_{down}$ 的列是对应的 value。既然 MLP 本身就是记忆，那让它在推理时**继续记忆新信息**就是自然的延伸。这避免了引入任何新架构，直接复用已有参数。

Formally, we adapt the widely used gated MLP architecture (Grattafiori et al., 2024; Yang et al., 2025). Given the hidden representation H, the gated MLP computes its output representation $O = (\phi(HW_{gate}^T) \odot (HW_{up}^T)) W_{down}^T$. In our framework, we treat the input projections $W_{up}$ and $W_{gate}$ as frozen slow weights, while repurposing the final projection matrix, $W_{down}$, as the adaptable fast weights. By exclusively updating $W_{down}$ in-place, we preserve the model's architectural integrity, transforming TTT from a disruptive restructuring into a lightweight, "drop-in" enhancement for LLMs.

> 💡 **批注**: 为什么只更新 $W_{down}$ 而不是 $W_{up}$ 或 $W_{gate}$？
> - **结构角度**: 在 gated MLP 中，$W_{up}$ 和 $W_{gate}$ 负责将输入投影到高维空间并计算门控，它们定义了 MLP 的"特征提取"功能；而 $W_{down}$ 负责从高维空间投影回输出空间，相当于 key-value memory 中的 "value 存储"。更新 value 存储 = 写入新记忆。
> - **稳定性角度**: 冻结 $W_{up}, W_{gate}$ 意味着 MLP 的输入表征空间不变，只改变输出映射，最大限度保持模型稳定性。
> - **计算角度**: 只更新一个矩阵，梯度计算和存储开销最小。

### Efficient Adaptation with Chunk-Wise Updates

Beyond architectural compatibility, our in-place design also unlocks significant computational efficiencies. Conventional TTT methods, by aiming to replace the attention mechanism, were bound to inefficient per-token updates to enforce strict causality and perform fine-grained token mixing. Chunk-wise updating approaches have been explored by recent works to achieve acceleration. Our framework also follows to sidestep the trade-off entirely. Since we only adapt the MLP blocks and leave the attention layers intact, we are liberated from the per-token constraint, enabling a far more efficient chunk-wise update strategy which further bypasses the small chunk constraints (our ablation study results (Section 4.3) also verify that our framework is naturally well-suited for chunk-wise—and specifically large chunk-wise—updates, achieving optimal performance with chunk sizes of 512 to 1024.

> 💡 **批注**: 这里揭示了 In-Place TTT 能用大 chunk 的根本原因：**MLP 是 attention 的补充，而非替代**。以往 TTT 方法要替换 attention，所以 fast weights 必须承担 token mixing 的职责，需要 per-token 更新来维持因果性。而 In-Place TTT 保留了 attention 来做 token mixing，MLP 的 fast weights 只负责存储信息，不需要细粒度的因果约束。因此可以用 512-1024 的大 chunk，计算效率大幅提升。

The process operates as follows. Given the intermediate activations $Z = \phi(HW_{gate}^T) \odot (HW_{up}^T) \in \mathbb{R}^{n \times d_{ff}}$ and corresponding value targets and outputs $V, O \in \mathbb{R}^{n \times d_{model}}$, we partition them into k non-overlapping chunks of size C. Let $W_{down}^{(i)}$ be the fast weights state before processing chunk i and $W_{down}^{(0)} = W_{down}$. For each chunk $i \in [k]$, we perform two sequential operations:

1. **Apply Operation:** The current state of the fast weights $W_{down}^{(i)}$ are used to process chunk $Z_{[i]}$, i.e., $O_{[i]} = Z_{[i]}(W_{down}^{(i)})^T$.
2. **Update Operation:** The fast weight $W_{down}^{(i)}$ are updated using $Z_{[i]}$ as keys and $V_{[i]}$ as values, which is performed via one gradient descent step with a loss function $\mathcal{L}$ and a learning rate $\eta$: $W_{down}^{(i+1)} = W_{down}^{(i)} - \eta \nabla_W \mathcal{L}(Z_{[i]}(W_{down}^{(i)})^T, V_{[i]})$.

> 💡 **批注**: 注意这里的顺序是 **Apply-then-Update**，与标准 TTT（如 Titans）的 Update-then-Apply 不同。
> - **标准 TTT**: 先用当前 token 更新 fast weights，再用更新后的 weights 计算输出 → 当前 token 的信息会影响自身的输出（类似自回归中 token 能"看到自己"）
> - **In-Place TTT**: 先用当前状态的 fast weights 计算输出，再更新 → chunk $i$ 的输出只反映 chunk $0, \ldots, i-1$ 写入的信息，当前 chunk 的信息要等下一个 chunk 才能被读取
>
> 这种设计更符合因果性：**你只能从已经记忆过的信息中检索，不能从正在写入的信息中获益**。同时避免了 update step 中的梯度影响当前输出的计算图，简化了反向传播。

---

## 3.2 LM-Aligned Objective

Prior TTT approaches typically use a reconstruction target, e.g., $\mathcal{L}(f_W(k), v)$ where both k and v are linear projection outputs of the same input token x, which encourages the model to simply memorize the current token's representation. We argue this is suboptimal for language modeling tasks. Instead, we propose to align the objective with the Next-Token Prediction (NTP) goal governing LLMs.

> 💡 **批注**: 这是对 Titans 等先前工作的直接批评。Titans 的 associative memory loss $\|\mathcal{M}(k_t) - v_t\|^2$ 中 $k_t$ 和 $v_t$ 都来自同一个 token $x_t$，本质是让 fast weights 学会**自编码**（输入 = 输出）。但 LLM 的核心目标是 NTP（输入当前 token → 预测下一个 token），两个目标方向不一致。

To achieve this, we specify the target v to include future token information. Formally, we derive our target $\bar{V} = \text{Conv1D}(X_0) W_{target}$, where $X_0 \in \mathbb{R}^{n \times d_{model}}$ denotes the token embedding, Conv1D(·) is the 1D Convolution operator and $W_{target} \in \mathbb{R}^{d_{model} \times d_{model}}$ is a trainable projection matrix. Under this formulation, the amount of future token information can be controlled in our target $\bar{V}$, e.g., the Next-Token target can be achieved by parameterizing $W_{target}$ as an identity transformation and assigning Conv1D(·)'s kernel weights to be 1 for the next token and 0 for others.

> 💡 **批注**: Conv1D + $W_{target}$ 的设计非常巧妙：
> - **Conv1D** 的作用是**沿序列维度混合 token embedding**。通过设定卷积核权重，可以精确控制 target 包含哪些位置的信息。例如 kernel = $[0, 1]$（当前位为 0，下一位为 1）就得到纯 Next-Token target。
> - **$W_{target}$** 是可学习的投影矩阵，让模型自己学如何组合未来信息。
> - 这个设计的灵活性在于：kernel 也可以是可学习的，让模型自适应决定"看多远的未来"。
> - 注意 target 来自 **token embedding $X_0$**（最底层的表征），而 key $Z$ 来自中间层的 MLP 激活。这种跨层设计让 fast weights 学习"从中间表征预测未来 token"的映射。

With this aligned target, we use the widely used similarity measure to instantiate our loss function for simplicity, i.e., $\mathcal{L}(\cdot, \cdot) = -\langle \cdot, \cdot \rangle_F$. Under this loss function, the gradient with respect to the fast weights in our chunk-wise mechanism can be directly derived:

$$W_{down}^{(i)} = W_{down}^{(i-1)} + \eta \bar{V}_{[i]}^T Z_{[i]} \quad \text{...(Eq. 1)}$$

> 💡 **批注**: 使用内积损失（而非 MSE）的好处：梯度形式极其简洁——$\Delta W = \eta \bar{V}^T Z$，就是一个外积累加。
> - 这与 linear attention 的更新规则 $S \leftarrow S + v \cdot k^T$ 形式完全一致
> - 不需要存储或计算 Hessian，不需要迭代优化，一步到位
> - 计算复杂度 = 一次矩阵乘法 $O(C \cdot d_{ff} \cdot d_{model})$
>
> 对比 Titans 的 MSE loss：梯度是 $\nabla = 2(Wk - v)k^T$，需要先做一次前向计算 $Wk$，然后再计算外积，多了一步。

---

## 3.3 Theoretical Analysis

The theoretical analysis uses the canonical induction head setting: a key-value pair $(k^*, v^*)$ appears at position $t^*$, and at query position $n > t^*$, the key $k^*$ reappears; the model must predict $v^*$.

**Theorem 1** (Logit-wise Effect of LM-Aligned Target v.s. Reconstruction Target): Under the specified setup and assumptions, for a learning rate $\lambda_{lr} > 0$, the expected change in logits $\Delta \ell_n$ after one update step using the LM-Aligned target satisfies:

- (Correct logit increases) $\mathbb{E}[\Delta \ell_n[v^*]] \geq \lambda_{lr} \cdot c_{norm}^2 \cdot c_{align}$
- (Other logits almost unchanged) $|\mathbb{E}[\Delta \ell_n[w]]| \leq \lambda_{lr} \cdot \epsilon \cdot c_{align}, \quad \forall w \neq v^*$

In contrast, for the reconstruction target, the expected change in logits is negligible for the correct token.

> 💡 **批注**: Theorem 1 的核心含义用一句话概括：**Reconstruction target 更新 fast weights 后，对 next token 的 logit 没有帮助；LM-Aligned target 则能显著提升正确 next token 的 logit。**
>
> 直觉理解：
> - **Reconstruction target**: fast weights 学的是 "key $k^*$ → value $v^* = f(x_{t^*})$"，即"看到 $x$ 输出 $x$ 的表征"。当 $k^*$ 再次出现时，它只能恢复 $x_{t^*}$ 的自身表征，但这对预测 $x_{t^*+1}$（下一个 token）没有直接帮助。
> - **LM-Aligned target**: fast weights 学的是 "key $k^*$ → target $\bar{v}^* = \text{embedding}(x_{t^*+1})$"，即"看到 $x$ 输出下一个 token 的信息"。当 $k^*$ 再次出现时，它直接输出了预测所需的信息。
>
> 数学上，correct logit 的增量有 $c_{norm}^2 \cdot c_{align}$ 的下界（正值），而错误 logit 的变化被 $\epsilon$（近似为 0）约束。这保证了 LM-Aligned target 的更新是"精准的"——只增强正确答案，不扰动其他选项。

---

## 3.4 Implementation Details

The framework is fully compatible with Context Parallelism (CP). The associative nature of the update rule (Eq. 1) makes In-Place TTT amenable to a context-parallel implementation:

(i) compute intermediate activations $Z_{[i]}$ and fast weight update $\Delta W_{down}^{(i)}$ in parallel;

(ii) a single prefix sum (CUMSUM) aggregates the deltas associatively;

(iii) the effective fast weights for each chunk are computed in parallel.

> 💡 **批注**: Context Parallelism 的可行性源于 Eq. 1 的**结合律**（associativity）：
> $$W_{down}^{(i)} = W_{down}^{(0)} + \eta \sum_{j=1}^{i} \bar{V}_{[j]}^T Z_{[j]}$$
> 每个 chunk 的 $\Delta W = \eta \bar{V}_{[i]}^T Z_{[i]}$ 可以独立计算（embarrassingly parallel），然后通过 prefix sum 累加得到每个 chunk 对应的 $W_{down}^{(i)}$。Prefix sum 是经典的并行原语，复杂度 $O(\log k)$（$k$ 为 chunk 数），在 GPU 上有成熟的高效实现。
>
> 对比 Titans：其更新包含 momentum 项 $S_t = \eta_t S_{t-1} - \theta_t \nabla \ell$，虽然也可以用 parallel scan，但需要额外处理数据相关的 $\eta_t$，实现更复杂。In-Place TTT 的更新规则是纯加法，最简单的结合操作。

Causality is maintained via causal padding on the 1D convolution. Fast weights are reset at document boundaries.

> 💡 **批注**: 两个重要的工程细节：
> - **Causal padding**: Conv1D 的 target 包含未来 token 信息，但这是训练时的 ground truth target，不违反因果性。关键是 Conv1D 使用 causal padding（只向右看），确保第 $t$ 个 target 只使用位置 $\leq t+k$ 的 embedding（$k$ 为卷积核大小）。
> - **Document boundaries**: 在多文档输入（如长上下文）中，fast weights 在文档边界重置为 $W_{down}^{(0)}$，防止跨文档的信息泄露。这也意味着 fast weights 的作用范围是**单文档内**。

---

## 🔖 Section 总结

### 关键设计速查

| 设计维度 | In-Place TTT 方案 | 对比先前 TTT（如 Titans） |
|---------|-------------------|------------------------|
| 架构修改 | 无——复用已有 MLP $W_{down}$ | 替换 attention 或新增模块 |
| Fast weights | $W_{down} \in \mathbb{R}^{d_{model} \times d_{ff}}$ | 独立的记忆模块参数 |
| 更新顺序 | Apply-then-Update | Update-then-Apply |
| 更新粒度 | Chunk-wise, C=512~1024 | Per-token 或小 chunk |
| 目标函数 | LM-Aligned（Conv1D + $W_{target}$，含未来 token 信息） | Reconstruction（自编码，$k,v$ 来自同一 token） |
| 损失函数 | 内积 $-\langle \cdot, \cdot \rangle_F$ | MSE $\|Wk - v\|^2$ |
| 更新规则 | $W^{(i)} = W^{(i-1)} + \eta \bar{V}^T Z$（纯加法） | SGD + momentum + weight decay |
| 并行方式 | Prefix sum（最简结合律） | Parallel scan（需处理 momentum） |
| 理论保证 | 正确 logit 增大，其余不变 | 无直接 NTP 相关保证 |

### 核心洞察

1. **MLP = key-value memory（Geva et al. 2020）**: 这一认知是整个框架的基石。既然 MLP 预训练时就在做"记忆"，那推理时继续记忆新信息是最自然的选择，无需修改架构。
2. **TTT 目标必须与 LM 目标对齐**: Reconstruction target 让 fast weights 学"自编码"，这与 NTP 目标正交。Theorem 1 从理论上证明了这种 misalignment 会导致 fast weights 更新对 next-token 预测没有帮助。
3. **保留 attention 解放了 chunk 约束**: 因为 token mixing 由 attention 负责，MLP 的 fast weights 不需要维护 per-token 因果性，可以用大 chunk 更新，计算效率提升数十倍。
4. **简洁的更新规则带来并行化红利**: 内积损失 → 纯加法更新 → 天然满足结合律 → prefix sum 即可实现 Context Parallelism，无需复杂的 parallel scan。
