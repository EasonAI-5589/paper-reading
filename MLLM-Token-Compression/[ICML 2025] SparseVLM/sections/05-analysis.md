# 5. Analysis

> 来源: SparseVLM (ICML 2025)

---

## 📄 原文

> 💡 **Section 概览**: 四个消融实验 — text rater 选择、token recycling、计算效率、可视化

---

### 5.1 Relevant Text Token Selection

在 64 tokens 设置下，对比三种 rater 策略：

| 策略 | TextVQA | POPE |
|------|---------|------|
| All tokens (baseline) | - | - |
| Only text tokens | - | 下降 |
| Text raters (ours) | +0.8% vs baseline | +2.7% vs only-text |

> 💡 **批注**: POPE benchmark 对 text rater 选择特别敏感（+2.7%），说明 POPE 的问题 prompt 中无关词比较多（如 "Is there a..." 中的 is/there/a 都是噪声）。

---

### 5.2 Recycling of Pruned Tokens

#### Table 4: Token Recycling 消融

| Benchmark | 64 tokens | 96 | 128 | 192 | Avg. |
|-----------|-----------|-----|-----|-----|------|
| GQA | 52.2 → **53.8** | 55.2→56.4 | 58.1→58.4 | 59.4→59.5 | +0.8 |
| POPE | 72.8 → **77.5** | 77.5→81.9 | 83.7→85.0 | 85.2→85.3 | +2.6 |

> 💡 **批读**:
> ```
> Token Recycling 效果 (POPE):
>   192 tokens: +0.1% (剪得少，回收价值有限)
>    64 tokens: +4.7% (剪得多，回收价值巨大!)
>
> 规律: 剪得越多 → recycling 越重要
> ```
> **核心洞察**: Recycling 是 SparseVLM 在极端压缩下仍能保持性能的关键原因。

---

### 5.3 Computational Efficiency

在 LLaVA-7B, A100-80GB 上测试：

| 指标 | Vanilla | SparseVLM (128 tokens) | 降低 |
|------|---------|----------------------|------|
| CUDA time | 57.82ms | 33.28ms | **-42.5%** |
| FLOPs | 4.62T | 1.72T | **-62.8%** |
| KV Cache | 302.4MB | 100.8MB | **-67%** |
| 准确率 | 100% | 96.7% | -3.3% |

> 💡 **批注**: KV cache 降 67% 对部署很关键 — 意味着同样的 GPU 显存可以跑更多并发请求。

---

### 5.4 Qualitative Visualization

![Figure 6](../images/9f87e73c9d476f8bbd7ce90956c7c2d2e3fabc9048eea25c720e9c16365378ee.jpg)
*Figure 6: SparseVLM 在不同 VQA prompt 上的可视化。从左到右，视觉表示越来越稀疏。*

> 💡 **Figure 6 批读**:
> ```
> 可视化展示逐层剪枝过程:
> Layer 0 (开始) → Layer N (结束)
>   ├── ROI 逐渐精炼
>   ├── 无关区域被移除
>   └── 保留的 token 集中在与问题相关的区域
>
> 例: 问"What color is the car?"
>   → 最终只保留汽车区域的 token
> ```

---

## 💡 Section 总结

### 核心洞察
1. **Text rater 选择**: POPE 上提升 2.7%，说明 prompt 中噪声词的影响显著
2. **Token recycling**: 极端压缩时是救命稻草（POPE 64 tokens: +4.7%）
3. **效率**: FLOPs 降 63%，latency 降 43%，KV cache 降 67%
4. **可视化**: 证实方法确实在保留与问题相关的视觉区域
