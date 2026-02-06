# 4. FastV Method

---

## 4.1 Dynamically Prune Vision Tokens

### 核心思想

> Given that image tokens contribute minimally to output generation in deeper layers due to diminished attention, why not consider **removing them at these stages**?
>
> ==既然 visual tokens 在深层贡献极小，为什么不直接删掉？==

### 方法设计

```
Input tokens → [Layer 1] → ... → [Layer K] → 🔪 Rank & Filter → [Layer K+1] → ... → Output
                                                    ↑
                                         保留 Top (1-R%) tokens
```

**三个关键组件：**

| 组件 | 说明 | 默认值 |
|------|------|--------|
| **Ranking function f_ϕ** | 重要性排序函数 | Attention score |
| **Filtering layer K** | 在第 K 层之后剪枝 | K=2 |
| **Filtering ratio R%** | 剪枝比例 | R=50% |

### 重要性评估标准

> We simply compute the **average attention-score one token received from all other tokens** as the criteria ϕ_attn in our experiment.
>
> ==用 token 收到的平均 attention score 作为重要性标准==

$$\phi_{attn}(t_i) = \frac{1}{n} \sum_{j=1}^{n} \alpha_{j \rightarrow i}$$

### Plug-and-Play 特性

> FastV is **plug-and-play** to different token-based LVLMs for various vision language tasks **without the need of training** the model.
>
> ==无需训练，即插即用！==

---

## 4.2 Computing Cost Estimation

### FLOPs 计算

**单层 Transformer FLOPs:**
$$\text{FLOPs} = 4nd^2 + 2n^2d + 2ndm$$

其中 n=token数, d=hidden size, m=FFN intermediate size

**FastV 的 FLOPs 减少比例:**

$$\text{Reduction} = 1 - \frac{K \times F(n) + (T-K) \times F(\hat{n})}{T \times F(n)}$$

其中 $\hat{n} = (1-R\%) \times n$

### 参数影响

| K | R | FLOPs 减少 |
|---|---|------------|
| 2 | 50% | 45% |
| 2 | 75% | 67% |
| 2 | 90% | 80% |
| 5 | 50% | 43% |

> ==K 越小、R 越大，减少越多==

---

## 4.3 Comparison: Training With Less Visual Tokens

> An alternative method to reduce visual tokens is directly **training with less visual tokens** (e.g., pooling on vision encoder output).
>
> ==对比方法：训练时就用更少的 visual tokens（如 pooling）==

**FastV vs Training-time Pooling:**

| 维度 | FastV | Training-time Pooling |
|------|-------|----------------------|
| 需要训练？ | ❌ 不需要 | ✅ 需要重新训练 |
| 灵活性 | 推理时可调 K, R | 固定 |
| 效果 | 更好 | 略差（见实验） |

---

## 💡 Key Takeaways

1. **方法极简**：在第 K 层后按 attention score 剪枝 R% tokens
2. **默认配置**：K=2, R=50% → 45% FLOPs 减少
3. **Plug-and-play**：无需训练，适用于各种 LVLM
4. **优于训练时压缩**：推理时动态剪枝效果更好

---

## 与 STAR-Pro 的对比

| 维度 | FastV | STAR-Pro |
|------|-------|----------|
| 剪枝阶段 | 单阶段 (Decoder) | 两阶段 (VE + Decoder) |
| 重要性评估 | 固定：attention score | 演化：similarity → reasoning |
| 问题 | 早期层剪太多会损失信息 | 解决 guidance inconsistency |

**STAR-Pro 可以指出 FastV 的局限：**
- FastV 依赖 attention score，但 attention 有 positional bias
- 单阶段剪枝无法适应 guidance 的演化

---

*[返回论文目录](../README.md)*
