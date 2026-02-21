[← 返回 README](../README.md)

# 2. Dissecting the Visual Processing Pipeline: From Semantic Flow to Spatial Integrity

In this section, we first perform a systematic analysis (Sec 2.1) of existing pruning methods to address two key questions. We then examine the visual information processing pipeline (Sec 2.2) in VLMs through two analytical experiments, tracing the progression from global attention mechanisms to local processing paradigms. Finally, position reconstruction experiments (Sec 2.3) uncover the root causes of performance degradation in grounding tasks, thereby providing insights for the design of pruning methods.

> 💡 这一整个 Section 是论文最有价值的部分。三个 Finding 层层递进，形成了完整的分析链条。

## 2.1 Evaluating Competitive Advantages: Simple Baselines versus Advanced Pruning Methods

### Experimental Setup

We conduct a comprehensive evaluation across 12 datasets, covering a broad spectrum of capabilities including image grounding, fine-grained understanding, and complex reasoning. To facilitate a systematic comparison, we categorize mainstream visual token pruning methods into three distinct families based on their architectural placement and operation stage:

- **Vision Encoder-Side Pruning**: VisionZip, PruMerge — 在 vision encoder 输出端减少冗余
- **LLM Single-Layer Pruning**: FastV — 在 LLM 特定层一次性固定比例 pruning
- **LLM Multi-Layer Pruning**: PyramidDrop, SparseVLM — 在 LLM 连续层动态 pruning

Benchmarked against two simple baselines: **random sampling** and **average pooling**.

### Table 1: VQA Performance (LLaVA-1.5-7B)

| Method | Source | GQA | MMB | MMMU | MME | VQAv2 | VQAtext | POPE | SQA | MMVet | Avg (%) |
|--------|--------|-----|-----|------|-----|-------|---------|------|-----|-------|---------|
| Vanilla | CVPR'24 | 61.9 | 64.7 | 36.3 | 1862 | 78.5 | 58.2 | 85.9 | 69.5 | 31.1 | 100.0 |
| FastV | ECCV'24 | 46.1 | 48.0 | 34.0 | 1255 | 55.0 | 47.8 | 59.6 | 68.7 | 23.3 | 78.3 |
| Random (single-layer) | – | 51.2 | 41.8 | 34.1 | 1351 | 65.4 | 44.9 | 61.1 | 66.8 | 16.9 | 77.3 |
| Pooling (single-layer) | – | 52.2 | 48.7 | 34.0 | 1380 | 69.1 | 45.3 | 67.8 | 67.9 | 16.3 | 80.3 |
| SparseVLM | ICML'25 | 53.8 | 60.1 | 35.4 | 1589 | 68.2 | 53.4 | 77.5 | 69.8 | 24.9 | 90.2 |
| Random (multi-layer) | – | 51.5 | 46.0 | 34.1 | 1342 | 67.1 | 46.7 | 71.8 | 68.1 | 23.1 | 82.5 |
| VisionZip | CVPR'25 | 55.1 | 60.1 | 36.2 | 1690 | 72.4 | 55.5 | 77.0 | 69.0 | 31.7 | 94.5 |
| Random (encoder-side) | – | 54.3 | 51.1 | 34.0 | 1410 | 66.2 | 46.5 | 68.2 | 65.5 | 21.1 | 82.4 |
| Pooling (encoder-side) | – | 51.5 | 44.4 | 32.1 | 1151 | 68.1 | 42.9 | 68.0 | 64.7 | 18.7 | 77.2 |

> 💡 **关键观察**: VisionZip (94.5%) 和 SparseVLM (90.2%) 确实显著优于 random baseline (77-82%)，但 Pooling baseline 在 single-layer 设置下达到 80.3%，和 FastV (78.3%) 相当。这说明 **encoder-side 方法在 VQA 上确实有优势**，但 LLM-side 方法优势不明显。

### Table 2: Visual Grounding Performance (RefCOCO Series)

| Avg Tokens | Method | RefCOCO-test | RefCOCO+-testA | RefCOCO+-testB | RefCOCOg-test |
|------------|--------|-------------|----------------|----------------|---------------|
| 576 | LLaVA | 58.30 | 59.43 | 38.88 | 48.50 |
| 128 | FastV | 10.34 | 8.53 | 9.83 | 8.87 |
| 128 | SparseVLM | 6.27 | 5.79 | 4.22 | 6.35 |
| 128 | VisionZip | 4.49 | 4.06 | 4.86 | 3.50 |
| 128 | Pooling | 23.01 | 24.37 | 15.04 | 19.69 |
| 64 | FastV | 2.73 | 1.17 | 1.02 | 2.19 |
| 64 | SparseVLM | 1.04 | 0.96 | 1.28 | 0.61 |
| 64 | VisionZip | 4.04 | 3.73 | 3.86 | 3.38 |
| 64 | Pooling | 12.01 | 12.20 | 7.55 | 11.40 |

> 💡 **这张表是全文最震撼的数据**:
> - 所有高级 pruning 方法在 VG 上几乎完全失效（128 tokens: FastV 10%, SparseVLM 6%, VisionZip 4%）
> - **Average Pooling 远好于所有高级方法**（128 tokens: 23% vs 4-10%）
> - 说明位置信息比语义选择更重要——Pooling 隐式保留了网格拓扑结构
> - VisionZip 表现最差是因为 PERC 策略完全丢失了位置信息

**Finding 1**: Advanced pruning methods provide limited benefits over simple baselines on VQA tasks, whereas all methods suffer systematic degradation on grounding tasks, with average pooling achieving the best performance.

## 2.2 Unveiling Task-Dependent Visual Processing Pipeline

Building on the task-dependent performance degradation observed in Sec. 2.1, we conduct two analytical experiments: visualizing attention flows from the final token to vision tokens during decoding, and applying gradient-weighted attribution methods to trace critical visual information pathways across tasks.

> 💡 Figure 3 展示了 attention flow 和 gradient-weighted attention flow 的差异。关键发现是 VG 任务在 LLM 中间层对 visual tokens 的依赖显著高于 VQA 任务。

### Visual Attention Entropy (VAE) and Object-Centric Cohesion (OCC)

Two fine-grained metrics:

**VAE** — Shannon entropy of visual self-attention, measures global vs. local focus:

$$H(v_i) = -\sum_{j=1}^{i-1} p(v_j|v_i) \log_2 p(v_j|v_i), \quad \text{VAE} = \frac{1}{N-1}\sum_{i=2}^{N} H(v_i)$$

**OCC** — IoU between ground-truth object tokens and top-k similar tokens to object center:

$$\text{OCC}(\mathcal{O}) = \frac{|V_k^{\text{model}} \cap V_{\mathcal{O}}|}{|V_k^{\text{model}} \cup V_{\mathcal{O}}|}$$

> 💡 VAE 和 OCC 是很好的分析工具：
> - ViT 中间层 VAE 下降 + OCC 上升 → 从全局到 object-centric 的转变
> - LLM 中间层 OCC 峰值 → object-level 表示在此阶段形成
> - 这解释了为什么 VG 任务在中间层需要更多 visual information

**Finding 2**: Visual processing in VLMs unfolds through a multi-stage pipeline, progressing from global semantic integration to fine-grained object-centric focus, with task-specific reliance on vision tokens. Grounding tasks require heightened visual integration during middle stages for spatial reasoning.

## 2.3 Spatial Integrity: Reconstructing the Global Reference Frame

### 2.3.1 A Taxonomy of Position Embedding Strategies

Three paradigms abstracted from existing pruning methods:

- **PERC (Position Embedding Range Compression)**: 压缩 PE 到小范围，丢失全局参考系。如 VisionZip。
- **PESP (Position Embedding Sparse Preservation)**: 保留原始 PE 但形成稀疏子集，空间不连续。如 FastV。
- **RPME (Relative Position Mapping Extension)**: 保留相对空间距离并线性映射扩展 PE 至原始全范围，恢复空间完整性。

> 💡 **RPME 是本文最重要的分析贡献**。三种 PE 策略的分类非常清晰：
> - PERC: 64 个 token 的 PE 就是 [0, 63]，但原本应该覆盖 [0, 575]
> - PESP: 保留原始编号如 [3, 17, 42, ...]，是稀疏的
> - RPME: 线性映射让 64 个 token 均匀覆盖 [0, 575]
>
> 这个分析直接导出了为什么 Pooling 效果好——它隐式实现了类似 RPME 的效果。

### Table 3: Position Reconstruction Experiment

| Method | RefCOCO-test | RefCOCO+-testA | RefCOCOg-test | GQA | MMB |
|--------|-------------|----------------|---------------|-----|-----|
| **64 tokens** |
| VisionZip-fix | 11.57 (+7.53) | 9.27 (+5.54) | 8.19 (+4.81) | 55.6 (+0.5) | 61.8 (+1.7) |
| FastV-fix | 4.52 (+1.79) | 3.84 (+2.67) | 4.17 (+1.98) | 46.2 (+0.1) | 47.8 (-0.2) |
| Pooling | 12.01 | 12.20 | 11.40 | – | – |
| **128 tokens** |
| VisionZip-fix | 21.39 (+16.90) | 19.96 (+15.90) | 15.69 (+12.19) | 58.5 (+0.9) | 63.4 (+1.4) |
| FastV-fix | 13.41 (+3.07) | 11.69 (+3.16) | 12.02 (+3.15) | 51.3 (+0.8) | 57.7 (+1.6) |
| Pooling | 23.01 | 24.37 | 19.69 | – | – |

> 💡 **RPME 的效果验证**:
> - VisionZip + RPME: 64 tokens 下 VG 从 4.04 → 11.57 (+186%)，128 tokens 下从 4.49 → 21.39 (+376%)
> - FastV + RPME: 提升较小，因为 PESP 至少保留了部分位置信息
> - **RPME 对 VQA 几乎无影响**——说明 VQA 不依赖精确空间信息
> - 但即使加了 RPME，VisionZip-fix (11.57) 仍不如 Pooling (12.01)，说明空间完整性不是唯一因素，聚合方式也很重要

**Finding 3**: The degradation of VLMs on grounding tasks is principally driven by the loss of Global Spatial Reference Frame within token pruning strategies, which can be restored by preserving global position embedding.
