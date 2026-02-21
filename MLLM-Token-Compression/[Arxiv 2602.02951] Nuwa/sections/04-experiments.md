[← 返回 README](../README.md)

# 4 Experiment

## 📌 预览
实验验证 Nüwa 在 VQA（10 benchmarks）和 VG（3 benchmarks）上的 SOTA 性能，包括效率分析和消融实验。

---

**Experimental Setup:**

To validate the generality and effectiveness of our method, we conduct experiments on multiple VLMs and diverse benchmarks for image understanding and visual grounding tasks. The evaluated models are LLaVA-1.5, LLaVA-NeXT. We use 10 VQA benchmarks (e.g., GQA, TextVQA) and 3 VG benchmarks (RefCOCO, etc.). All experiments are run on NVIDIA A100-40G GPUs. Detailed configurations are in the Appendix B.2.

> 💡 **批注**: 实验配置：LLaVA-1.5 + LLaVA-NeXT，10 VQA + 3 VG benchmarks，A100-40G。

---

## 4.1 Main Result

Performance on VQA Tasks: We apply Nüwa during the inference stage of LLaVA-1.5-7B. More results in Table 5 demonstrate that Nüwa achieves optimal performance across nearly all benchmarks, with average performance further improving upon existing SOTA methods. On more VLM models with different scales, such as LLaVA-NeXT-7B, Nüwa consistently demonstrates performance gains, establishing its strong generalizability. Results can be found in Appendix 10.

> 💡 **批注**: VQA 上 Nüwa 在三个 token 配置（192/128/64）下都取得了最好或接近最好的平均性能。

---

**Table 5: VQA performance comparison On LLaVA-1.5 7B.**

| Method | GQA | MMB | MMMU | MME | VQAv2 | VQAtext | POPE | SQA | SEED | MMVet | avg |
|--------|-----|-----|------|-----|-------|---------|------|-----|------|-------|-----|
| Vanilla | 61.9 | 64.7 | 36.3 | 1862 | 78.5 | 58.2 | 85.9 | 69.5 | 58.6 | 31.1 | 100% |
| **Avg Token 192 ↓ 66.7%** |
| FastV | 52.7 | 61.2 | 34.3 | 1612 | 67.1 | 52.5 | 64.8 | 67.3 | 57.1 | 27.7 | 89.53% |
| PDrop | 57.1 | 63.2 | 34.1 | 1766 | 74.9 | 56.1 | 82.3 | 70.2 | 54.7 | 30.5 | 95.87% |
| SparseVLM | 57.6 | 62.5 | 33.8 | 1721 | 75.6 | 56.1 | 83.6 | 69.1 | 55.8 | 31.5 | 96.11% |
| VisionZip | 59.3 | 63.0 | 36.6 | 1782 | 76.8 | 57.3 | 85.3 | 68.9 | 56.4 | 31.7 | 98.26% |
| **Nüwa** | **60.9** | **64.3** | 35.5 | **1834** | 75.9 | **57.4** | **86.4** | 68.2 | **59.7** | 30.5 | **98.80%** |
| **Avg Token 128 ↓ 77.8%** |
| FastV | 49.6 | 56.1 | 34.9 | 1490 | 61.8 | 50.6 | 59.6 | 60.2 | 55.9 | 28.1 | 85.04% |
| PDrop | 56.0 | 61.1 | 34.2 | 1664 | 73.5 | 55.1 | 82.3 | 69.9 | 53.3 | 30.8 | 94.32% |
| SparseVLM | 56.0 | 60.0 | 33.8 | 1696 | 73.8 | 54.9 | 80.5 | 67.1 | 53.4 | 30.0 | 93.36% |
| VisionZip | 57.6 | 62.0 | 37.9 | 1761 | 75.6 | 56.8 | 83.2 | 68.9 | 54.9 | 32.6 | 97.63% |
| PruMerge | 57.8 | 59.6 | 36.2 | 1712 | 74.7 | 54.3 | 81.5 | 67.6 | - | 30.4 | 95.06% |
| **Nüwa** | **60.2** | **63.4** | 35.8 | **1828** | 75.1 | **57.0** | **85.5** | 67.8 | **58.7** | 29.8 | **97.87%** |
| **Avg Token 64 ↓ 88.9%** |
| FastV | 46.1 | 48.0 | 34.0 | 1256 | 55.0 | 47.8 | 59.6 | 51.1 | 51.9 | 25.8 | 79.36% |
| PDrop | 41.9 | 33.3 | 26.5 | 1092 | 57.3 | 45.9 | 55.9 | 69.2 | 40.0 | 24.9 | 71.56% |
| SparseVLM | 53.8 | 60.1 | 35.44 | 1589 | 68.2 | 53.4 | 77.5 | 69.8 | 51.1 | 24.9 | 89.93% |
| VisionZip | 55.1 | 60.1 | 36.2 | 1690 | 72.4 | 55.5 | 77.0 | 69.0 | 52.2 | 31.7 | 93.99% |
| PruMerge | 55.4 | 59.6 | 35.8 | 1616 | 71.3 | 52.0 | 75.7 | 69.5 | - | 28.0 | 91.71% |
| **Nüwa** | **58.3** | **62.0** | **36.4** | **1706** | **72.8** | 54.9 | **83.0** | 67.5 | **56.44** | 28.2 | **94.91%** |

> 💡 **批注**: VQA 性能亮点：
> - **64 tokens**：Nüwa 94.91% vs VisionZip 93.99%（+0.92%），vs SparseVLM 89.93%（+5%）
> - **128 tokens**：Nüwa 97.87% vs VisionZip 97.63%（持平）
> - **192 tokens**：Nüwa 98.80% vs VisionZip 98.26%（+0.54%）
> - Nüwa 在 POPE（85.5/83.0）和 SEED（58.7/56.44）上优势明显
> - 在极端压缩（64 tokens = 88.9% reduction）下优势更显著

---

Performance on Visual Grounding Tasks: Visual grounding tasks are highly sensitive to spatial information in tokens, constituting a critical evaluation dimension for compression methods. On RefCOCO series visual grounding benchmarks, as shown in Table 6, our method substantially outperforms alternative approaches, achieving approximately 35% performance improvement over previous methods under 64 average tokens configuration. When retaining 192 tokens, our method maintains 79% of the original model's performance.

---

**Table 6: Performance comparison on the RefCOCO series benchmark On LLaVA-1.5 7B.**

| Method | Refcoco-test | Refcoco-val | Refcoco+-testA | Refcoco+-testB | Refcoco+-val | Refcocog-test | Refcocog-val | avg |
|--------|-------------|-------------|----------------|----------------|-------------|---------------|-------------|-----|
| Vanilla | 58.30 | 56.42 | 59.43 | 38.88 | 46.32 | 48.50 | 48.82 | 100% |
| **Avg Tokens 192 ↓ 66.7%** |
| FEATHER* | 27.7 | - | 24.7 | - | - | 27.2 | - | 48.38% |
| **Nüwa** | **47.91** | **46.12** | **43.18** | **31.86** | **37.68** | **37.64** | **37.90** | **79.29%** |
| **Avg Tokens 128 ↓ 77.8%** |
| FastV | 10.34 | 10.13 | 8.53 | 9.83 | 8.16 | 8.87 | 9.10 | 18.55% |
| SparseVLM | 6.27 | 6.17 | 5.79 | 4.22 | 9.85 | 6.35 | 6.47 | 12.84% |
| VisionZip | 4.49 | 4.11 | 4.06 | 4.86 | 3.88 | 3.50 | 3.48 | 8.1% |
| **Nüwa** | **45.09** | **43.69** | **42.63** | **28.98** | **35.32** | **36.59** | **36.00** | **75.20%** |
| **Avg Tokens 64 ↓ 88.9%** |
| FastV | 2.73 | 2.01 | 1.17 | 1.02 | 2.41 | 2.19 | 2.01 | 3.81% |
| SparseVLM | 1.04 | 1.01 | 0.96 | 1.28 | 0.96 | 0.61 | 0.66 | 1.88% |
| VisionZip | 4.04 | 3.81 | 3.73 | 3.86 | 3.50 | 3.38 | 3.21 | 7.28% |
| **Nüwa** | **29.43** | **28.60** | **28.22** | **17.47** | **22.22** | **21.81** | **21.42** | **47.19%** |

> 💡 **批注**: **这是全文最核心的实验结果**：
> - **64 tokens**：Nüwa 47.19% vs VisionZip 7.28% vs SparseVLM 1.88% — **碾压级别的差距**
> - **128 tokens**：Nüwa 75.20% vs FastV 18.55% — **4倍以上的性能保留率**
> - **192 tokens**：Nüwa 79.29% vs FEATHER 48.38%
> - 这直接证明了空间参考系保持的重要性——Nüwa 通过 grid partitioning + RPME 式的 PE 保持，实现了远超其他方法的 VG 性能

---

**Table 4: Comparison of Model Efficiency.**

| Method | Avg Token | main (TFLOPs) | metric (MFLOPs) | Prefill-Time (ms) |
|--------|-----------|---------------|-----------------|-------------------|
| Vanilla | 576 | 5.9730 | 0 | 124 |
| FastV | 64 | 0.8341 | 4.7185 | 92 ↓ 26% |
| SparseVLM | 64 | 0.8141 | 5.5050 | 104 ↓ 16% |
| VisionZip | 64 | 0.6461 | 8.9128 | 45 ↓ 63% |
| Nüwa | 64 | 0.6476 | 17.5636 | 46 ↓ 62% |

> 💡 **批注**: 效率分析：
> - Nüwa 的 main TFLOPs（0.6476）与 VisionZip（0.6461）几乎相同
> - metric 计算开销（17.56 MFLOPs）是 VisionZip 的 2 倍，但相对于 main 计算（0.6476 TFLOPs）可以忽略
> - **Prefill 时间**：46ms vs Vanilla 124ms，**减少 62%**，与 VisionZip 持平
> - FastV 和 SparseVLM 的 prefill 时间反而更高（92ms, 104ms），因为它们在 LLM 内部做 pruning，前面的层仍需处理全部 token

---

## 4.2 Ablation Study

**Ablation on Spatial Proximity Threshold**

To enable aggregation based on spatial proximity, we define local neighborhoods via a distance threshold $\tau$. Empirical evaluation (Table 8) shows that performance peaks at $\tau = 26\%$ of the maximum distance. Smaller values restrict aggregation scope, leading to suboptimal results, while larger values incorporate noise from distant regions, also degrading performance. These results confirm the effectiveness of localized aggregation in preserving spatial integrity.

> 💡 **批注**: 距离阈值的 sweet spot 在最大距离的 26%，即约 $0.26 \times \sqrt{24^2 + 24^2} \approx 8.8$ 个 token 的距离。太小（18 pixels）不够聚合，太大（>500）引入噪声。

---

**Table 7: Ablation Study on cohesion distance.**

| Config | GQA | MMB | MME | Refcoco-test | Refcoco+-testA | Refcoco+-testB | Refcocog-test |
|--------|-----|-----|-----|-------------|----------------|----------------|---------------|
| dist18 | 0.5784 | 60.29 | 1695 | 0.2783 | 0.2818 | 0.1655 | 0.2018 |
| dist148 | 0.5853 | 61.68 | 1705 | 0.2922 | 0.2936 | 0.1730 | 0.2189 |
| **dist280** | **0.5833** | **62.03** | **1707** | **0.2943** | **0.2822** | **0.1747** | **0.2181** |
| dist412 | 0.5826 | 62.11 | 1711 | 0.2879 | 0.2705 | 0.1698 | 0.2135 |
| dist544 | 0.5811 | 62.03 | 1703 | 0.2834 | 0.2637 | 0.1651 | 0.2100 |

> 💡 **批注**: 距离增大后 VQA（MMB, MME）先升后平，VG 先升后降。最优点 dist280 平衡了两者。

---

**Ablation on Key Components**

Experimental results in Table 8 show that region partitioning is essential for grounding tasks, as it implements a more precise RPME strategy, but has negligible effects on VQA tasks. The L2-norm criterion positively enhances baseline token selection across all tasks, consistent with our analysis in Sec. 3.1.3. For two-stage pruning, gains over random pruning remain modest. Notably, combining random pruning with region partitioning substantially degrades performance, as the partitioning introduces potentially task-irrelevant tokens that random selection may retain.

---

**Table 8: Ablation Study on each design.**

| region | pillar token | random S2 | GQA | MMB | MME | Refcoco-test | Refcoco+-testA | Refcoco+-testB | Refcocog-test |
|--------|-------------|-----------|-----|-----|-----|-------------|----------------|----------------|---------------|
| ✘ | ✘ | ✘ | 58.84 | 58.18 | 1791 | 6.83 | 6.54 | 4.50 | 5.58 |
| ✘ | ✘ | ✔ | 57.07 | 56.43 | 1736 | 6.72 | 6.48 | 4.38 | 5.25 |
| ✘ | ✔ | ✘ | 59.62 | 62.98 | 1807 | 6.35 | 6.12 | 4.65 | 5.60 |
| ✔ | ✘ | ✘ | 57.94 | 56.68 | 1742 | 43.50 | 39.85 | 26.10 | 34.20 |
| ✔ | ✘ | ✔ | 57.35 | 56.10 | 1724 | 43.17 | 38.94 | 25.58 | 33.74 |
| **✔** | **✔** | **✘** | **60.18** | **63.40** | **1828** | **45.09** | **42.63** | **28.98** | **36.59** |
| ✔ | ✔ | ✔ | 59.03 | 62.14 | 1791 | 44.30 | 41.20 | 27.50 | 35.80 |

> 💡 **批注**: **消融实验的关键发现**：
> 1. **Region Partitioning 是 VG 性能的决定性因素**：无 region 时 Refcoco-test 仅 6.83%，有 region 后跳到 43.50%（+36.67%！）
> 2. **Pillar Token** 对 VQA 有帮助（GQA: 57.94→60.18, MMB: 56.68→63.40），对 VG 也有小幅提升
> 3. **Random S2 反而有害**：text-guided > random，但差距不大（45.09 vs 44.30）
> 4. **没有 region 时，pillar token 对 VG 几乎无帮助**（6.83→6.35），说明 **空间保持是前提**
> 5. Region + Random S2 组合反而下降，因为 region 引入的 task-irrelevant tokens 被 random 保留了
