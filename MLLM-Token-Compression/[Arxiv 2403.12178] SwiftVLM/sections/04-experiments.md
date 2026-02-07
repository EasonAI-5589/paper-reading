# 4. Experiments

> 来源: SwiftVLM

---

## 📄 原文

> 💡 **Section 概览**: 实验包括五大块——(1) 主实验对比；(2) 效率分析；(3) 消融实验；(4) bypass 为什么有效的分析；(5) 泛化性验证。

---

### 4.1 Overall Performance

> 💡 **4.1 要点预览**: 在 LLaVA-1.5-7B 上，9 个 benchmark，两种 token 预算下的全面对比。SwiftVLM 在定位任务上碾压式领先。

**数据集分类**:
- **Localization** (细粒度): RefCOCO, RefCOCO+, RefCOCOg
- **Non-localization** (粗粒度): TextVQA, GQA, SQA, MME, MMB, POPE

**Table 1: 192 tokens (保留 33.3%) 主要结果**

| 方法 | Loc. Avg. | Non-loc. Avg. | FLOPs |
|------|-----------|---------------|-------|
| Vanilla (上界) | 100% | 100% | 4.29T |
| FastV | 40.3% | 95.9% | 1.71T |
| VisionZip | 8.9% | 97.5% | 1.71T |
| PDrop | 27.6% | 93.8% | 1.72T |
| SparseVLM | 10.9% | 98.0% | 1.72T |
| FEATHER | 66.9% | 96.5% | 1.82T |
| **SwiftVLM** | **86.9%** | **99.0%** | 1.75T |

> 💡 **Table 1 批读 (192 tokens)**:
> ```
> 定位任务排行:
> ├── SwiftVLM:  86.9% ⭐ 碾压
> ├── FEATHER:   66.9%
> ├── FastV:     40.3%
> ├── PDrop:     27.6%
> ├── SparseVLM: 10.9%
> └── VisionZip:  8.9% (text-agnostic 在定位上几乎不能用)
>
> 非定位任务排行:
> ├── SwiftVLM: 99.0% ⭐ (几乎无损!)
> ├── SparseVLM: 98.0%
> ├── VisionZip: 97.5%
> ├── FEATHER: 96.5%
> ├── FastV: 95.9%
> └── PDrop: 93.8%
> ```
> **关键发现**:
> 1. 非定位任务上大家都差不多（95%+），说明 visual token 冗余确实很多
> 2. 定位任务才是真正的试金石——SwiftVLM 比第二名 FEATHER 高 20%
> 3. VisionZip/SparseVLM 定位只有 ~10%，说明不保留位置信息是致命的

**Table 1: 128 tokens (保留 22.2%) 主要结果**

| 方法 | Loc. Avg. | Non-loc. Avg. | FLOPs |
|------|-----------|---------------|-------|
| FastV | 17.7% | 91.3% | 1.29T |
| VisionZip | 5.8% | 96.1% | 1.29T |
| PDrop | 3.6% | 91.8% | 1.28T |
| SparseVLM | 6.0% | 95.8% | 1.30T |
| FEATHER | 50.8% | 95.0% | 1.44T |
| **SwiftVLM** | **69.8%** | **96.7%** | 1.31T |

> 💡 **Table 1 批读 (128 tokens)**:
> ```
> 更激进的剪枝下 (保留 22.2%):
> ├── SwiftVLM: 69.8% loc / 96.7% non-loc ⭐
> ├── FEATHER:  50.8% / 95.0% (FLOPs 更高!)
> └── 其他方法: 定位基本崩了 (<18%)
>
> 注意 FEATHER 的 FLOPs 是 1.44T，SwiftVLM 只有 1.31T
> → SwiftVLM 更快且更准
> ```

---

**可视化结果**:

![Figure 6a](../images/9a44f2658cd9e352c30d9cdd8bb272701389146af400c4f0e7860e2935f89121.jpg)
*Figure 6(a): 192 tokens 下各方法保留的 visual token 可视化*

![Figure 6b](../images/9c931e1ee5ab072308c2889ef1e639828b10fffc4802872483d76c335e21bc3c.jpg)
*Figure 6(b): 128 tokens 下各方法保留的 visual token 可视化*

> 💡 **Figure 6 批读**:
> ```
> RefCOCO 示例 (定位 "small white car"):
>   FEATHER/PDrop: 丢了车所在区域的 token → 定位失败
>   SwiftVLM: 保留了车的 token → 定位成功
>
> TextVQA 示例 (读招牌文字):
>   类似情况，SwiftVLM 保留了关键文字区域
> ```

> 💡 **4.1 小结**:
> - 非定位任务: 所有方法都还行，SwiftVLM 最好 (99%)
> - 定位任务: SwiftVLM 碾压 (比 FEATHER 高 20%+)
> - 越激进的剪枝，SwiftVLM 的优势越大

---

### 4.2 Efficiency Study

> 💡 **4.2 要点预览**: 实际延迟对比，不只看 FLOPs。

**Table 2: LLaVA-1.5-7B 效率对比**

| Tokens | Method | Total Time | Δ | Prefill Time | Δ |
|--------|--------|-----------|---|-------------|---|
| 576 | Vanilla | 850.7s | - | 67.3ms | - |
| 192 | FastV | 551.8s | 1.54× | 34.7ms | 1.92× |
| 192 | SparseVLM | 612.3s | 1.39× | 40.7ms | 1.65× |
| 192 | **SwiftVLM** | 573.8s | **1.48×** | 37.6ms | **1.79×** |
| 128 | FastV | 539.4s | 1.58× | 32.8ms | 2.05× |
| 128 | SparseVLM | 583.9s | 1.46× | 37.5ms | 1.79× |
| 128 | **SwiftVLM** | 546.2s | **1.56×** | 33.0ms | **2.04×** |

> 💡 **Table 2 批读**:
> ```
> 128-token 设置下延迟排行:
> ├── FastV:     1.58× (最快，但准确率最差)
> ├── SwiftVLM:  1.56× ⭐ (几乎一样快，准确率最好)
> └── SparseVLM: 1.46× (最慢)
>
> 为什么 SwiftVLM 比 SparseVLM 快?
> → SparseVLM 要算所有 text token 的 attention
> → SwiftVLM 只算最后一个 text token 的 attention
> ```

> 💡 **4.2 小结**:
> - SwiftVLM 速度接近 FastV（最简单的方法），但准确率远超
> - 128 token 下达到 2.04× prefill 加速

---

### 4.3 Ablation Study

> 💡 **4.3 要点预览**: 逐步加入各组件，看各自贡献。

**Table 3: 消融实验**

| Tokens | Method | RefCOCO | TextVQA |
|--------|--------|---------|---------|
| 192 | Baseline (PDrop) | 42.6 | 43.2 |
| 192 | + Layer Selection | 64.5 | 45.3 |
| 192 | + Merge | 63.7 | 44.8 |
| 192 | + Merge + Bypass | **66.6** | **45.3** |
| 128 | Baseline | 23.2 | 41.2 |
| 128 | + Layer Selection | 42.8 | 40.1 |
| 128 | + Merge | 51.9 | 40.7 |
| 128 | + Merge + Bypass | **55.2** | **41.8** |

> 💡 **Table 3 批读**:
> ```
> 192-token:
> ├── Layer Selection: +22 RefCOCO ⭐ (最大收益!)
> ├── + Merge: -0.8 (反而略降，因为 token 够用不需要 merge)
> └── + Bypass: +2.9 (恢复并超过)
>
> 128-token:
> ├── Layer Selection: +19.6 RefCOCO
> ├── + Merge: +9.1 (token 不够用时 merge 有帮助!)
> └── + Bypass: +3.3
> ```
> **关键发现**:
> 1. Layer Selection 贡献最大——选对层比什么都重要
> 2. Merge 在 token 充裕时反而有害（压缩了不该压的），token 紧张时有益
> 3. Bypass 稳定提升，尤其在激进剪枝下

> 💡 **4.3 小结**:
> - Layer Selection >> Bypass > Merge (贡献排序)
> - Merge 的效果取决于 token 预算
> - Bypass 在所有设置下稳定正向

---

### 4.4 Why Bypass Works?

> 💡 **4.4 要点预览**: 通过 t-SNE 可视化验证 offset 对齐的有效性。

![Figure 7a](../images/1ac16612ab760f5bf360afc2735315eafe1aa5476cf6e21112da1c4c1ce5e12b.jpg)
*Figure 7(a): 细粒度分组下的 token hidden-state 变化 t-SNE 可视化*

![Figure 7b](../images/9f10ba18bd8da69a2c22b6fe2e150d5b085f5778ba12da1e6693a0b9784b13e1.jpg)
*Figure 7(b): 粗粒度分组下的 t-SNE 可视化*

> 💡 **Figure 7 批读**:
> ```
> 图中三种标记:
> • = 单个 token 的变化 (vanilla model)
> × = 组内平均变化 (vanilla model)
> ★ = 合并 token 的变化 (SwiftVLM)
>
> 关键观察:
> (a) 细粒度分组: ★ 和 × 几乎完全重叠
>     → 合并 token 的 offset 完美近似了组内平均变化
> (b) 粗粒度分组 (n=18): 仍然很接近
>     → 合并 token 只占 <5%，依然能追踪变化
> ```
> **结论**: Offset 对齐的理论假设在实验中得到了验证。

---

### 4.5 Why Is Bypass Better Than Drop?

> 💡 **4.5 要点预览**: 对比 bypass 和 drop 在层 15 选出的 token 与 vanilla 模型的重叠率。

![Figure 8a](../images/e2c89e96290627e1b9d99d2183f453ccbfc4f1361620656995b98a74bff14f6400eec05127ee31d.jpg)
![Figure 8b](../images/503fa0c5759459a601c8584ac247232e191b98a74bff14f6400eec05127ee31d.jpg)
*Figure 8: Bypass vs Drop 的 token 选择重叠率对比（TextVQA 和 RefCOCO）*

> 💡 **Figure 8 批读**:
> ```
> 比较: 各方法在层 15 选出的 token 与 vanilla 的重叠率
>
> TextVQA:
>   Bypass > Drop (重叠率更高)
>
> RefCOCO:
>   Bypass >> Drop (差距更大!)
>   → 与定位任务上更大的性能差距一致
> ```
> **结论**: Bypass 让深层能"看到"更多跟 vanilla 一致的关键 token，因为它没把这些 token 在浅层就丢掉。

> 💡 **4.5 小结**:
> - Bypass 的 token 选择行为更接近 vanilla model
> - 在细粒度任务上优势更明显

---

### 4.6 Generalization

> 💡 **4.6 要点预览**: 在 LLaVA-NeXT-7B 上验证泛化性。

**Table 4: LLaVA-NeXT-7B 结果**

| Method | RefCOCO | TextVQA | GQA | MMB | Rel. Acc |
|--------|---------|---------|-----|-----|----------|
| Vanilla | 85.3 | 65.5 | 63.9 | 67.9 | 100% |
| FastV (33.3%) | 40.5 | 58.7 | 59.0 | 48.3 | 75.1% |
| FEATHER (33.3%) | 68.8 | 62.6 | 62.5 | 67.5 | 92.8% |
| **SwiftVLM (33.3%)** | **80.7** | **64.1** | **63.6** | **68.0** | **98.0%** |
| FastV (22.2%) | 26.1 | 52.6 | 56.9 | 46.0 | 66.9% |
| FEATHER (22.2%) | 53.1 | 60.9 | 61.9 | 66.5 | 87.5% |
| **SwiftVLM (22.2%)** | **79.6** | **62.4** | **63.5** | **67.7** | **97.1%** |

> 💡 **Table 4 批读**:
> ```
> 保留 33.3% tokens:
> ├── SwiftVLM: 98.0% 相对准确率 ⭐
> ├── FEATHER:  92.8%
> └── FastV:    75.1%
>
> 保留 22.2% tokens:
> ├── SwiftVLM: 97.1% ⭐ (几乎无损!)
> ├── FEATHER:  87.5%
> └── FastV:    66.9%
>
> RefCOCO 上 SwiftVLM (79.6) vs FEATHER (53.1)
> → 差距 26.5，比 LLaVA-1.5 上更大
> ```
> **关键发现**: SwiftVLM 在更新的模型上泛化良好，优势甚至更大。

> 💡 **4.6 小结**:
> - 在 LLaVA-NeXT 上同样大幅领先
> - 保留 22.2% tokens 仍达 97.1% 相对准确率

---

## 💡 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 最佳定位保持率 (192 tok) | 86.9% |
| 最佳非定位保持率 (192 tok) | 99.0% |
| Prefill 加速 (128 tok) | 2.04× |
| LLaVA-NeXT 相对准确率 (22.2%) | 97.1% |

### 核心结论
1. SwiftVLM 在定位任务上碾压所有对手（20%+ 领先）
2. 非定位任务几乎无损（99%）
3. 速度接近最简单的 FastV，但准确率远超
4. Layer Selection 是最重要的组件
5. 泛化性好，在 LLaVA-NeXT 上优势更大
