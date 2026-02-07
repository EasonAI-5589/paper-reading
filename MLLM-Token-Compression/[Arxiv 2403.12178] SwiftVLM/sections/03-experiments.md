# 4. Experiments

> 来源: SwiftVLM (Arxiv 2403.12178)

---

## 📄 原文

> 💡 **Section 概览**: 主结果 + 效率 + 消融 + bypass 分析 + 泛化

---

### 4.1 Overall Performance (Table 1)

任务分两类：**定位 (Localization)** 和 **非定位 (Non-localization)**

> 💡 **Table 1 批读 (192 tokens)**:
> ```
> Localization (RefCOCO 系列):
>   SwiftVLM:   86.9% ⭐⭐⭐
>   FEATHER:    66.9%
>   FastV:      40.3%
>   PDrop:      27.6%
>   SparseVLM:  10.9% (崩了)
>
> Non-localization:
>   SwiftVLM:   99.0% ⭐
>   SparseVLM:  98.0%
>   VisionZip:  97.5%
>   FEATHER:    96.5%
>   FastV:      95.9%
> ```
>
> **关键发现**:
> 1. 定位任务差异巨大 — SwiftVLM 遥遥领先
> 2. SparseVLM 在定位上崩溃 (10.9%) — 因为剪枝后不保留位置信息
> 3. 非定位任务差异很小 — 所有方法都还行

> 💡 **128 tokens 时**:
> ```
> Localization:
>   SwiftVLM:   69.8%
>   FEATHER:    50.8%
>   FastV:      17.7%
>   SparseVLM:   6.0%
>   PDrop:       3.6% (几乎失效)
> ```
> PDrop 在 128 tokens 的定位任务上几乎完全失效！说明它的渐进式剪枝在细粒度任务上不够好。

---

### 4.2 Efficiency (Table 2)

| Method | Prefill Time (192 tokens) | Speedup |
|--------|--------------------------|---------|
| Vanilla | 67.3ms | 1× |
| FastV | 34.7ms | 1.92× (最快) |
| SparseVLM | 40.7ms | 1.65× |
| SwiftVLM | 37.6ms | 1.79× |

> 💡 **批注**: SwiftVLM 比 SparseVLM 快（因为只用 last text token 做 query），但比 FastV 慢一点（多了 bypass 的对齐操作）。效率和性能的 tradeoff 非常好。

---

### 4.3 Ablation (Table 3)

| 组件 | RefCOCO (192) | RefCOCO (128) |
|------|---------------|---------------|
| Baseline (PDrop) | 42.6 | 23.2 |
| + Layer Selection | 64.5 (+21.9) | 42.8 (+19.6) |
| + Token Merging | 63.7 (-0.8) | 51.9 (+9.1) |
| + Bypass | **66.6** (+2.9) | **55.2** (+3.3) |

> 💡 **批读**:
> ```
> Layer Selection 贡献最大 (RefCOCO +21.9)
>   → 选对层很关键！
>
> Token Merging:
>   192 tokens: 略降 (token 够多，merging 反而压缩信息)
>   128 tokens: 大幅提升 (+9.1) (token 不够，merging 保留信息)
>
> Bypass: 稳定提升 (+2.9 / +3.3)
>   → 不管 token 多少都有帮助
> ```

---

### 4.4 Why Bypass Works?

t-SNE 可视化显示 merged token 的偏移量 (Δh_gm) 和 vanilla 模型中同组 token 的平均偏移量 (Δh_g) 几乎完全重叠。

> 💡 **批注**: 这为 token alignment 提供了实证支持 — 用 merged token 的变化量来近似原始 token 的变化是合理的。

---

### 4.5 Why Bypass > Drop?

对比 Layer 15 选出的 token 与 vanilla 模型的重叠率：
- Bypass 的重叠率显著高于 Drop
- RefCOCO 上差异更大（定位任务更需要保留完整信息）

---

### 4.6 Generalization (Table 4, LLaVA-NeXT-7B)

| Method | RefCOCO | VQA^Text | GQA | MMB | Rel. Acc |
|--------|---------|----------|-----|-----|----------|
| **Retain 33.3%** |
| FastV | 40.5 | 58.7 | 59.0 | 48.3 | 75.1% |
| FEATHER | 68.8 | 62.6 | 62.5 | 67.5 | 92.8% |
| SwiftVLM | **80.7** | **64.1** | **63.6** | **68.0** | **98.0%** |

> 💡 **批注**: 在 LLaVA-NeXT 上 SwiftVLM 也保持了绝对优势。33% tokens 时 98% 性能，非常强。

---

## 💡 Section 总结

### 关键数字速查
| 场景 | SwiftVLM | 最佳对手 | 差距 |
|------|----------|----------|------|
| Localization 192 tokens | 86.9% | 66.9% (FEATHER) | +20.0% |
| Localization 128 tokens | 69.8% | 50.8% (FEATHER) | +19.0% |
| Non-loc 192 tokens | 99.0% | 98.0% (SparseVLM) | +1.0% |
| LLaVA-NeXT 33.3% | 98.0% | 92.8% (FEATHER) | +5.2% |

### 核心洞察
1. **定位任务是 SwiftVLM 的杀手级应用** — 其他方法都崩了
2. **Layer selection 贡献最大** — 选对层比换策略更重要
3. **Bypass 在所有设置下稳定提升** — 是一个通用的改进方向
