# 3. Method

> 来源: SwiftVLM

---

## 📄 原文

> 💡 **Section 概览**: 方法分四部分——(1) VLM attention 基础；(2) 用动态规划选最优剪枝层；(3) Bypass 架构细节；(4) 表示对齐的理论分析；(5) FLOPs 计算。

---

### 3.1 Preliminary: Attention in VLMs

> 💡 **3.1 要点预览**: 交代 VLM 中 attention 的基本公式，以及如何用 T-V attention 衡量 visual token 的重要性。

Let $L$ denote the total number of tokens. The query and key matrices are obtained via linear projections:

$$\mathbf{Q} = \mathbf{h}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{h}\mathbf{W}_K$$

A single-head attention matrix:

$$A = \text{Softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{d}}\right)$$

VLMs adopt causal attention, under which each token is restricted to attending only to preceding tokens. As a result, the last text token attends to all input tokens. In practice, we extract its attention scores as the cross-modal component to evaluate the importance of visual tokens.

> 💡 **批注**: 为什么用"最后一个 text token"的 attention？因为 causal attention 下，最后一个 token 能看到前面所有 token（包括所有 visual token）。它对各 visual token 的 attention 分数就反映了"这个 visual token 对回答问题有多重要"。

> 💡 **3.1 小结**:
> - 用最后一个 text token 对所有 visual token 的 attention 分数作为重要性指标
> - 保留位置信息（positional encoding 不变）

---

### 3.2 Pruning Layer Selection

> 💡 **3.2 要点预览**: 核心问题——在哪几层做剪枝？不是拍脑袋，而是通过实验分析 + 动态规划来选。

For a model with $L$ layers, we first record the top $V\%$ visual tokens selected by T–V attention at each layer using the vanilla model. Then we re-evaluate the model by retaining only the layer-specific top $V\%$ visual tokens from the third layer onward, producing a layer-wise performance profile.

> 💡 **批注**: 实验方法很巧妙——在每一层都试一下"只保留这一层选出的 top 20% token"，看最终性能怎样。这就得到了每层的"选 token 能力"打分。

We formulate the pruning-layer selection problem as a dynamic programming task, enforcing a monotonic increase in selection capability across the chosen pruning layers.

**性能序列**: $\{x_i\}_{i=1}^L$，其中 $x_i$ 是第 $i$ 层的选择能力分数。

**约束**: 选出的剪枝层性能要单调递增：$x_{i_1} < x_{i_2} < \cdots < x_{i_K}$

**目标函数**: 最大化所有层的加权性能总和：

$$P(s) = \frac{\sum_{k=0}^{K} x_{i_k}(i_{k+1} - i_k)}{L - 2}$$

> 💡 **大白话解释**:
> ```
> 想象你是工厂质检员，有 30 个检查站。
> 你只能在其中 3 个站设置检查点。
> 每个站的"检出率"不同，而且不是越靠后越好。
> 
> 目标: 选 3 个站，让整体检出率最高。
> 约束: 后面的站检出率要比前面的高（单调递增）。
> 
> 动态规划: 从后往前算，每个位置考虑
>   "如果在这里设检查点，后面最优怎么选？"
> ```

**状态转移**:

$$x_j(L - j) \geq x_i(L - i) - x_{i_{m-1}}(j - i)$$

> 💡 **批注**: 这个 DP 的意思是——比较在层 $i$ vs 层 $j$ 设置最后一个剪枝点的收益。考虑了"剪枝点影响后续所有层"这个传播效应。

**实验结果**: 在 LLaVA-1.5-7B 上，对 6 个数据集的 1000 个样本做分析，最终选出的剪枝层是 **3, 11, 15**。

> 💡 **3.2 小结**:
> - 各层选 token 能力非单调，中间层最强
> - 用动态规划选最优剪枝层组合
> - LLaVA-1.5-7B: 层 3, 11, 15
> - 关键: 这个选层过程只需在少量样本上做一次，不影响测试时间

---

### 3.3 Architecture

> 💡 **3.3 要点预览**: SwiftVLM 的完整架构——在层 x 做第一次剪枝 + merge，在层 y 做 bypass 对齐 + 第二次剪枝。

![Figure 5](../images/b3198d4e364c33dea49653d377741b70e4ded0b56553b1344fcdd5d32ba78371.jpg)
*Figure 5: SwiftVLM 架构总览。(a) 层 x 后：不重要的 token 分组走旁路，合并版继续参与计算。(b) 层 y 前：通过 token alignment 恢复旁路 token，重新评估重要性。*

> 💡 **Figure 5 批读**:
> ```
> 完整流程:
> 
> 输入 → [层 1-x] → 第 1 次剪枝
>                    ├── Top tokens → 继续正常推理 → [层 x+1 ~ y-1]
>                    ├── Bottom tokens → 保存到旁路 (bypass)
>                    └── Bottom tokens → 按相似度 merge → 也参与推理
>                                                          ↓
>                                              [层 y] 第 2 次剪枝
>                    ├── 旁路 token + offset 对齐 → 重新打分
>                    ├── Top tokens → 继续推理 → [层 y+1 ~ L]
>                    └── Bottom tokens → 丢弃
> ```

**第 1 次剪枝 (层 x)**:
1. 用 T-V attention 给 visual token 排序
2. Top tokens 直接进入层 x+1
3. Bottom tokens 按余弦相似度分组：$s_{i,j} = \frac{(\mathbf{h}_i^x)^\top \mathbf{h}_j^x}{|\mathbf{h}_i^x||\mathbf{h}_j^x|}$
4. 每组取平均得到合并 token：$\tilde{\mathbf{h}}_g^x = \frac{1}{|\mathcal{G}_g|}\sum_{i \in \mathcal{G}_g} \mathbf{h}_i^x$
5. 合并 token 参与后续推理，原始 bottom tokens 存入旁路

> 💡 **批注**: 合并 token 有两个作用：(1) 代替 bottom tokens 参与层 x+1 到 y-1 的推理，维持信息完整性；(2) 后面用它的变化量来估算旁路 token 该怎么更新。

**第 2 次剪枝 (层 y) —— Bypass + Token Alignment**:

1. 计算合并 token 从层 x 到层 y-1 的变化量（offset）：$\Delta\mathbf{h}_{gm} = \tilde{\mathbf{h}}_{gm}^{y-1} - \tilde{\mathbf{h}}_{gm}^x$
2. 用这个 offset 更新旁路 token：$\hat{\mathbf{h}}_i^{y-1} = \mathbf{h}_i^x + \Delta\mathbf{h}_{gm}$
3. 用更新后的 token 和层 y 的 $W_K^y$ 构建 key，用最后一个 text token 的 $W_Q^y$ 构建 query
4. 重新计算 T-V attention，选出 Top tokens 继续推理

> 💡 **大白话**:
> ```
> 旁路 token 在层 x 就"冻住"了，到层 y 时跟其他 token 不在一个表示空间。
> 怎么对齐？用合并 token 当"代理人"。
> 
> 合并 token 从层 x 走到层 y，积累了 offset。
> 把这个 offset 加到旁路 token 上 → 近似对齐。
> 
> 类似于: 你请假了 10 天，同事帮你记了笔记。
>         你回来后看同事的笔记 (offset) 就能跟上进度。
> ```

> 💡 **3.3 小结**:
> - 两次剪枝，每次独立决策
> - Bypass = 保留 + 合并代理 + offset 对齐 + 重新评估
> - 只有第 2 次剪枝的 bottom tokens 被真正丢弃

---

### 3.4 Representation Alignment Analysis

> 💡 **3.4 要点预览**: 理论分析——为什么用合并 token 的 offset 能近似原始 token 的变化？

Transformer layers adopt a residual formulation:

$$\mathbf{h}^\ell = \mathbf{h}^{\ell-1} + \mathcal{F}^\ell(\mathbf{h}^{\ell-1})$$

For a visual token $i$ in group $\mathcal{G}_g$, its hidden state evolves from layer $x+1$ to $y-1$ as:

$$\mathbf{h}_i^{y-1} = \mathbf{h}_i^x + \sum_{\ell=x+1}^{y-1}\mathcal{F}^\ell(\mathbf{h}_i^{\ell-1})$$

Taking the group average:

$$\tilde{\mathbf{h}}_g^{y-1} = \tilde{\mathbf{h}}_g^x + \sum_{\ell=x+1}^{y-1}\frac{1}{|\mathcal{G}_g|}\sum_{i \in \mathcal{G}_g}\mathcal{F}^\ell(\mathbf{h}_i^{\ell-1})$$

> 💡 **批注**: 核心假设——同一组内的 token（语义相似）在每层的变化方向也相似。所以组平均的变化量 ≈ 组内每个 token 的变化量。这在 4.4 节通过 t-SNE 可视化得到了实验验证。

> 💡 **3.4 小结**:
> - 理论基础: residual connection + 语义相似 token 变化方向相似
> - 实验验证: t-SNE 可视化显示 offset 近似非常好

---

### 3.5 FLOPs Computation

> 💡 **3.5 要点预览**: 计算 SwiftVLM 的计算量，分析 bypass 额外开销。

标准 FLOPs:
$$C_n = 4nd^2 + 2n^2d + 3ndm$$
$$F = K \times C_n + (T-K) \times C_{\hat{n}}$$

Bypass 额外开销:
$$F_o = 2RZd + Rd + 2n_vd + 2d^2 + 2(1-r)n_vd^2$$

> 💡 **批注**: Bypass 的额外开销主要来自：(1) merge 操作 $2RZd$；(2) offset 对齐 $Rd$；(3) 重新计算 T-V attention $2n_vd^2$。相比省下的主体计算量，这些开销很小。

> 💡 **3.5 小结**:
> - Bypass 的额外开销远小于它省下的计算量
> - 主要省在：剪枝后层数 × 减少的 token 数

---

## 💡 Section 总结

### SwiftVLM 方法速查
| 组件 | 作用 | 关键公式 |
|------|------|----------|
| Layer Selection | 找最优剪枝层 | DP 最大化 $P(s)$ |
| 1st Pruning | T-V attention 排序 + merge | 余弦相似度分组 |
| Bypass | 保留 bottom tokens | 原始状态走旁路 |
| Token Alignment | 对齐到当前表示空间 | $\hat{\mathbf{h}}_i = \mathbf{h}_i^x + \Delta\mathbf{h}_{gm}$ |
| 2nd Pruning | 重新评估重要性 | 再次 T-V attention |

### 设计亮点
1. **两次独立决策** → 避免一次性不可逆的错误
2. **Merge 当代理人** → 用少量 token 追踪表示空间变化
3. **Offset 对齐** → 简单有效的近似方法
4. **DP 选层** → 数据驱动，不靠拍脑袋
