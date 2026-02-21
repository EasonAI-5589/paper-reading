[← 返回 README](../README.md)

# 4. Experiments

**Experimental Setup**: LLaVA-1.5-7B, LLaVA-NeXT-7B. 10 VQA benchmarks + 3 VG benchmarks (RefCOCO series). NVIDIA A100-40G GPUs.

## 4.1 Main Results

### Table 5: VQA Performance (LLaVA-1.5-7B)

| Method | Source | GQA | MMB | MMMU | MME | VQAv2 | VQAtext | POPE | SQA | SEED | MMVet | avg |
|--------|--------|-----|-----|------|-----|-------|---------|------|-----|------|-------|-----|
| Vanilla | CVPR'24 | 61.9 | 64.7 | 36.3 | 1862 | 78.5 | 58.2 | 85.9 | 69.5 | 58.6 | 31.1 | 100% |
| **192 tokens (↓66.7%)** |
| VisionZip | CVPR'25 | 59.3 | 63.0 | 36.6 | 1782 | 76.8 | 57.3 | 85.3 | 68.9 | 56.4 | 31.7 | 98.26% |
| **Nüwa** | - | **60.9** | **64.3** | 35.5 | **1834** | 75.9 | **57.4** | **86.4** | 68.2 | **59.7** | 30.5 | **98.80%** |
| **128 tokens (↓77.8%)** |
| VisionZip | CVPR'25 | 57.6 | 62.0 | **37.9** | 1761 | 75.6 | 56.8 | 83.2 | 68.9 | 54.9 | **32.6** | 97.63% |
| **Nüwa** | - | **60.2** | **63.4** | 35.8 | **1828** | 75.1 | **57.0** | **85.5** | 67.8 | **58.7** | 29.8 | **97.87%** |
| **64 tokens (↓88.9%)** |
| VisionZip | CVPR'25 | 55.1 | 60.1 | 36.2 | 1690 | 72.4 | 55.5 | 77.0 | 69.0 | 52.2 | **31.7** | 93.99% |
| **Nüwa** | - | **58.3** | **62.0** | **36.4** | **1706** | **72.8** | 54.9 | **83.0** | 67.5 | **56.44** | 28.2 | **94.91%** |

> 💡 **VQA 结果分析**:
> - Nüwa 在所有 token budget 下都略优于 VisionZip（0.5-1% 优势）
> - GQA 和 POPE 上优势最明显——这两个 benchmark 涉及空间推理
> - MMVet 上 Nüwa 反而略差于 VisionZip（28.2 vs 31.7），可能因为 MMVet 更侧重 OCR/推理
> - 整体 VQA 差距不大，核心优势在 VG

### Table 6: Visual Grounding Performance (RefCOCO Series)

| Method | Source | RefCOCO-test | RefCOCO-val | RefCOCO+-testA | RefCOCO+-testB | RefCOCO+-val | RefCOCOg-test | RefCOCOg-val | avg |
|--------|--------|-------------|-------------|----------------|----------------|-------------|---------------|-------------|-----|
| Vanilla | CVPR'24 | 58.30 | 56.42 | 59.43 | 38.88 | 46.32 | 48.50 | 48.82 | 100% |
| **192 tokens** |
| FEATHER* | ICCV'25 | 27.7 | - | 24.7 | - | - | 27.2 | - | 48.38% |
| **Nüwa** | - | **47.91** | **46.12** | **43.18** | **31.86** | **37.68** | **37.64** | **37.90** | **79.29%** |
| **128 tokens** |
| FastV | ECCV'24 | 10.34 | 10.13 | 8.53 | 9.83 | 8.16 | 8.87 | 9.10 | 18.55% |
| SparseVLM | ICML'25 | 6.27 | 6.17 | 5.79 | 4.22 | 9.85 | 6.35 | 6.47 | 12.84% |
| VisionZip | CVPR'25 | 4.49 | 4.11 | 4.06 | 4.86 | 3.88 | 3.50 | 3.48 | 8.1% |
| **Nüwa** | - | **45.09** | **43.69** | **42.63** | **28.98** | **35.32** | **36.59** | **36.00** | **75.20%** |
| **64 tokens** |
| FastV | ECCV'24 | 2.73 | 2.01 | 1.17 | 1.02 | 2.41 | 2.19 | 2.01 | 3.81% |
| SparseVLM | ICML'25 | 1.04 | 1.01 | 0.96 | 1.28 | 0.96 | 0.61 | 0.66 | 1.88% |
| VisionZip | CVPR'25 | 4.04 | 3.81 | 3.73 | 3.86 | 3.50 | 3.38 | 3.21 | 7.28% |
| **Nüwa** | - | **29.43** | **28.60** | **28.22** | **17.47** | **22.22** | **21.81** | **21.42** | **47.19%** |

> 💡 **VG 结果是本文最亮眼的数据**:
> - 64 tokens: Nüwa 47.19% vs VisionZip 7.28% — **6.5× 提升**
> - 128 tokens: Nüwa 75.20% vs VisionZip 8.1% — **9.3× 提升**
> - 192 tokens: Nüwa 79.29% vs FEATHER 48.38% — **1.6× 提升**
>
> **但要注意**: 47.19% 的保留率意味着绝对性能仍然只有 vanilla 的一半（RefCOCO-test 29.43 vs 58.30）。在实际应用中这个性能可能仍不够用。
>
> **与 STAR-Pro 的比较**: 论文没有直接对比 STAR-Pro（可能同期工作），但 STAR-Pro 的 indicator inconsistency 分析和 Nüwa 的 spatial integrity 分析是互补的。

### Table 4: Efficiency Analysis

| Method | Avg Token | main (TFLOPs) | metric (MFLOPs) | Prefill-Time (ms) |
|--------|-----------|---------------|-----------------|-------------------|
| Vanilla | 576 | 5.9730 | 0 | 124 |
| FastV | 64 | 0.8341 | 4.72 | 92 ↓26% |
| SparseVLM | 64 | 0.8141 | 5.51 | 104 ↓16% |
| VisionZip | 64 | 0.6461 | 8.91 | 45 ↓63% |
| **Nüwa** | 64 | 0.6476 | 17.56 | 46 ↓62% |

> 💡 **效率评价**:
> - Nüwa 和 VisionZip 几乎相同的 prefill 时间（46 vs 45 ms），因为 Stage 1 只需一次 attention 计算
> - metric 开销 17.56 MFLOPs vs main 0.6476 TFLOPs，可忽略不计
> - 兼容 FlashAttention，实际部署友好
> - 比 FastV/SparseVLM 快是因为 encoder-side pruning 在 LLM 前就减少了 token 数

## 4.2 Ablation Study

### Spatial Proximity Threshold

Performance peaks at τ = 26% of maximum distance (dist280). Smaller values restrict aggregation, larger values introduce noise.

### Key Components (Table 8)

| Region | Pillar | Random S2 | GQA | MMB | RefCOCO-test | RefCOCO+-testA |
|--------|--------|-----------|-----|-----|-------------|----------------|
| ✘ | ✘ | ✘ | 58.84 | 58.18 | 6.83 | 6.54 |
| ✘ | ✔ | ✘ | 59.62 | 62.98 | 6.35 | 6.12 |
| ✔ | ✘ | ✘ | 57.94 | 56.68 | 43.50 | 39.85 |
| **✔** | **✔** | **✘** | **60.18** | **63.40** | **45.09** | **42.63** |
| ✔ | ✔ | ✔ | 59.03 | 62.14 | 44.30 | 41.20 |

> 💡 **消融分析关键发现**:
> 1. **Region Partitioning 对 VG 至关重要**: 没有 region → VG ≈ 6%，有 region → VG ≈ 43%。但对 VQA 影响极小
> 2. **Pillar Token 对 VQA 有益**: 从 58.18 → 62.98 (MMB)。对 VG 贡献较小（43.50 → 45.09）
> 3. **Random Stage 2 反而有害**: 加入 random S2 后所有指标都略降。说明 text-guided 优于 random
> 4. **Region + Random 组合最差**: 因为 region 引入的 task-irrelevant token 被 random 保留了
>
> **核心结论**: Stage 1 的 region partitioning（即空间均匀性）是 VG 性能的决定因素，其他组件是锦上添花。
