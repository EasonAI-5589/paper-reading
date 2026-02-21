[← 返回 README](../README.md)

# 2 Dissecting the Visual Processing Pipeline: From Semantic Flow to Spatial Integrity

## 📌 预览
本节是全文最精彩的分析部分，包含三个递进的实验：(1) 简单基线 vs 高级 pruning 方法的对比；(2) VLM 视觉处理流水线的 task-dependent 特性分析；(3) 位置编码重建实验证明空间参考系的关键作用。

---

In this section, we first perform a systematic analysis (Sec 2.1) of existing pruning methods to address two key questions. We then examine the visual information processing pipeline (Sec 2.2) in VLMs through two analytical experiments, tracing the progression from global attention mechanisms to local processing paradigms. Finally, position reconstruction experiments (Sec 2.3) uncover the root causes of performance degradation in grounding tasks, thereby providing insights for the design of pruning methods.

> 💡 **批注**: 本节的三段式分析结构非常工整：Finding 1（方法对比）→ Finding 2（流水线分析）→ Finding 3（根因定位）。每个小节都有明确的 Finding 总结，这种写法值得学习。

---

## 2.1 Evaluating Competitive Advantages: Simple Baselines versus Advanced Pruning Methods

Recent research (Wen et al., 2025; Endo et al., 2024) has questioned the effectiveness of existing visual token pruning methods. To investigate two key aspects — (1) Generalization: whether advanced methods consistently outperform simple baselines, and (2) Robustness: whether performance remains stable across tasks with diverse requirements, we conduct a comprehensive cross-task evaluation.

> 💡 **批注**: 两个核心问题设置得很好：泛化性（是否稳定优于简单基线）和鲁棒性（跨任务稳定性）。

---

**Experimental Setup**

We conduct a comprehensive evaluation across 12 datasets, covering a broad spectrum of capabilities including image grounding, fine-grained understanding, and complex reasoning. To facilitate a systematic comparison, we categorize mainstream visual token pruning methods into three distinct families based on their architectural placement and operation stage: Vision Encoder-Side Pruning, which focuses on reducing redundancy within or at the output of the vision encoder to save memory early on (e.g., VisionZip (Yang et al., 2024), PruMerge (Shang et al., 2024)); LLM Single-Layer Pruning, which applies a one-time, fixed-ratio pruning operation at specific layers within the LLM (e.g., FastV (Chen et al., 2024)); and LLM Multi-Layer Pruning, which dynamically identifies and removes non-essential vision tokens across consecutive LLM layers (e.g., PyramidDrop (Xing et al., 2024), SparseVLM (Zhang et al., 2025b)). To ensure a fair and rigorous assessment, we benchmark these sophisticated methods against two simple yet effective baselines, random sampling and average pooling, to determine the value of complex pruning designs.

> 💡 **批注**: 方法分类非常清晰：
> | 类别 | 位置 | 代表方法 |
> |------|------|----------|
> | Vision Encoder-Side | 视觉编码器输出 | VisionZip, PruMerge |
> | LLM Single-Layer | LLM 单层 | FastV |
> | LLM Multi-Layer | LLM 多层 | PyramidDrop, SparseVLM |
> 
> 用 random sampling 和 average pooling 作为基线，这个对比设计非常有说服力。

---

**Table 1: Performance comparison of various vision token pruning methods on LLAVA1.5-7B.**

| Method | GQA | MMB | MMMU | MME | VQAv2 | VQAtext | POPE | SQA | MMVet | Avg (%) |
|--------|-----|-----|------|-----|-------|---------|------|-----|-------|---------|
| Vanilla | 61.9 | 64.7 | 36.3 | 1862 | 78.5 | 58.2 | 85.9 | 69.5 | 31.1 | 100.0 |
| FastV | 46.1 | 48.0 | 34.0 | 1255 | 55.0 | 47.8 | 59.6 | 68.7 | 23.3 | 78.3 |
| Random (single) | 51.2 | 41.8 | 34.1 | 1351 | 65.4 | 44.9 | 61.1 | 66.8 | 16.9 | 77.3 |
| Pooling (single) | 52.2 | 48.7 | 34.0 | 1380 | 69.1 | 45.3 | 67.8 | 67.9 | 16.3 | 80.3 |
| PDrop | 41.9 | 33.3 | 26.5 | 1092 | 57.3 | 45.9 | 55.9 | 69.2 | 24.9 | 72.0 |
| SparseVLM | 53.8 | 60.1 | 35.4 | 1589 | 68.2 | 53.4 | 77.5 | 69.8 | 24.9 | 90.2 |
| Random (multi) | 51.5 | 46.0 | 34.1 | 1342 | 67.1 | 46.7 | 71.8 | 68.1 | 23.1 | 82.5 |
| VisionZip | 55.1 | 60.1 | 36.2 | 1690 | 72.4 | 55.5 | 77.0 | 69.0 | 31.7 | 94.5 |
| PruMerge+ | 55.4 | 59.6 | 35.8 | 1616 | 71.3 | 52.0 | 75.7 | 69.5 | 28.0 | 91.7 |
| Random (encoder) | 54.3 | 51.1 | 34.0 | 1410 | 66.2 | 46.5 | 68.2 | 65.5 | 21.1 | 82.4 |
| Pooling (encoder) | 51.5 | 44.4 | 32.1 | 1151 | 68.1 | 42.9 | 68.0 | 64.7 | 18.7 | 77.2 |

> 💡 **批注**: 关键发现：
> - **FastV 不如 Pooling 基线**（78.3% vs 80.3%）！
> - **SparseVLM**（90.2%）是多层 pruning 中最好的，但 Random 基线也有 82.5%
> - **VisionZip**（94.5%）在 encoder-side 方法中最好
> - 总体上，简单基线（random, pooling）表现并不差，尤其是 pooling

---

**Table 2: Performance comparison on RefCOCO series datasets.**

| Avg Tokens | Method | Refcoco-test | Refcoco+-testA | Refcoco+-testB | Refcocog-test |
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

> 💡 **批注**: **这张表是全文最震撼的数据**：
> - VisionZip 在 VQA 上保留 94.5%，但在 RefCOCO 上只剩 **4.49/58.30 ≈ 7.7%**！
> - SparseVLM 更惨：**6.27/58.30 ≈ 10.8%**
> - **Pooling 显著优于所有精心设计的方法**（128 tokens: 23.01 vs 最好的 10.34）
> - 这说明 pooling 隐式保留了空间结构，而其他方法完全破坏了它

---

Our results reveal key patterns across task types. On general-purpose VQA benchmarks (Table 1), simple baselines achieve competitive performance, often matching advanced pruning methods. In contrast, on object-centric grounding tasks (Table 2.1), all methods show systematic performance degradation, regardless of design complexity. Notably, average pooling yields the best results among pruning approaches, likely because it partially preserves spatial structural features.

> 💡 **批注**: 总结精炼。关键洞察：pooling 之所以表现最好，是因为它在粗网格上聚合特征，隐式地维持了全局拓扑结构。

---

**Finding 1:** Advanced pruning methods provide limited benefits over simple baselines on VQA tasks, whereas all methods suffer systematic degradation on grounding tasks, with average pooling achieving the best performance.

> 💡 **批注**: Finding 1 直接挑战了现有 pruning 方法的核心价值——在 VQA 上它们并不比 random/pooling 好多少，在 VG 上则全面崩溃。

---

## 2.2 Unveiling Task-Dependent Visual Processing Pipeline

Building on the task-dependent performance degradation observed in Sec. 2.1, prior explainability studies on LLMs and VLMs (Selvaraju et al., 2016; Ding et al., 2017; Zhang et al., 2024; Yin et al., 2025) have not sufficiently explored how visual processing adapts to shifts in task focus, such as from VQA to VG. To address this, we conduct two analytical experiments: visualizing attention flows from the final token to vision tokens during decoding, and applying gradient-weighted attribution methods to trace critical visual information pathways across tasks. Additionally, we evaluate the model's object-centric perception at different stages using two fine-grained metrics.

> 💡 **批注**: 分析方法很扎实：attention flow 可视化 + gradient-weighted attribution + 两个新指标（VAE 和 OCC）。

---

Figure 3 depicts task-dependent characteristics of visual processing in VLMs. Panels (a) and (b) at the first row show that attention flows exhibit distinct early and mid-stage phases. However, gradient-weighted analysis (Second row) reveals a pronounced task-dependent divergence in the mid-stage, underscoring the model's sensitivity to task requirements during visual integration — with VG tasks showing greater reliance on vision tokens. Panel (c) highlights a task-independent aspect: early multimodal interactions, suggesting universal visual processing in initial stages. Panel (d) illustrates task-varying differences in text information handling. Further experiments on attention blocking (Appendix B.5) indicate that, in VG tasks, textual cues extract critical visual details, resulting in unique last-to-text attention patterns.

> 💡 **批注**: Figure 3 的分析揭示了 VLM 视觉处理的关键特性：
> - **Early layers**：task-independent，全局多模态交互
> - **Mid layers**：**task-dependent** 分歧出现，VG 任务对 vision tokens 的依赖显著增加
> - 这解释了为什么 mid-layer pruning（如 FastV 在 layer 2 后 pruning）对 VG 特别有害

---

**Visual Attention Entropy And Object-Centric Cohesion:**

Attention flows offer insights into the model's information processing dynamics. To further quantify the multi-stage visual processing pipeline identified in the prior analysis and its task-dependent characteristics, we introduce two fine-grained metrics: Visual Attention Entropy (VAE) and Object-Centric Cohesion (OCC). VAE measures the distribution of information in the visual self-attention mechanism by computing the average Shannon entropy across visual tokens (Eq. (1)). High VAE values indicate diffuse, global attention patterns, whereas low values reflect concentrated, local focus. Complementing this, OCC assesses object-level feature cohesion by calculating the Intersection over Union (IoU) between ground-truth object tokens and the top-k tokens most similar to the object's center token (Eq. (2)). Higher OCC scores denote stronger localization of features to relevant objects, capturing fine-grained processing.

$$H(v_i) = -\sum_{j=1}^{i-1} p(v_j|v_i) \log_2 p(v_j|v_i), \quad \text{VAE} = \frac{1}{N-1} \sum_{i=2}^{N} H(v_i) \tag{1}$$

$$\text{OCC}(\mathcal{O}) = \frac{|V_k^{\text{model}} \cap V_{\mathcal{O}}|}{|V_k^{\text{model}} \cup V_{\mathcal{O}}|} \tag{2}$$

> 💡 **批注**: 两个新指标设计得很好：
> - **VAE**（Visual Attention Entropy）：衡量注意力分布的分散程度。高 = 全局关注，低 = 局部聚焦
> - **OCC**（Object-Centric Cohesion）：衡量模型特征是否聚焦在目标物体上。用 ground-truth object tokens 和 top-k similar tokens 的 IoU
> 
> 这两个指标互补：VAE 看整体模式，OCC 看物体级别的聚焦度。

---

As shown in Figure 4, the VAE of the ViT encoder exhibits a decreasing trend in the middle stage, indicating a gradual shift from global context integration to fine-grained feature extraction. In contrast, the VAE of the LLM decoder fluctuates after a sharp initial increase, suggesting a more complex process of reorganizing visual features and integrating them into the textual semantic space. The OCC scores provide a clearer explanation — they peak in the middle stage of both ViT and LLM, signifying the formation of object-level representations. This phenomenon also effectively explains the earlier observation: why grounding tasks demand such high levels of visual information at this stage.

> 💡 **批注**: 关键发现：
> - ViT 中段：VAE 下降（全局→局部），OCC 上升（object-level 表示形成）
> - LLM 中段：OCC 也在中间层达到峰值
> - 这意味着 **ViT 和 LLM 的中间层都是物体感知的关键阶段**，pruning 在此处最具破坏性

---

**Finding 2:** Visual processing in VLMs unfolds through a multi-stage pipeline, progressing from global semantic integration to fine-grained object-centric focus, with task-specific reliance on vision tokens. Grounding tasks require heightened visual integration during middle stages for spatial reasoning, in contrast to the reduced demands in image understanding tasks.

> 💡 **批注**: Finding 2 明确了 VLM 的视觉处理流水线是 **global → fine-grained** 的多阶段过程，且 VG 任务对中间层的视觉信息需求远高于 VQA。

---

## 2.3 Spatial Integrity: Reconstructing the Global Reference Frame

Building on the mid-stage visual integration demands in Sec. 2.2, where pruning disrupts task-specific vision reliance, we hypothesize that spatial integrity — via the Global Spatial Reference Frame from position embeddings — is essential for spatial perception, as pooling methods' superior grounding performance indicates. To validate this, we design experiments restoring integrity through modified position embedding strategies.

> 💡 **批注**: 从 Finding 2（中间层很重要）到这里的假设：**空间完整性 = position embedding 的全局参考系**。Pooling 效果好正是因为它隐式保留了这个参考系。

---

### 2.3.1 A Taxonomy of Position Embedding Strategies

To rigorously test our hypothesis, we first deconstruct the implicit position embedding (PE) handling strategies within existing pruning methods, abstracting them into three distinct paradigms:

**Position Embedding Range Compression (PERC):** Compresses the PE of pruned tokens into a tiny range, missing the global reference frame, like VisionZip.

**Position Embedding Sparse Preservation (PESP):** Retains the original PE for each pruned token, forming a sparse subset within an incomplete spatial frame, like FastV.

**Relative Position Mapping Extension (RPME):** Preserves the relative spatial distance of the pruned tokens and extends their PE via linear mapping, to span the entire original range and retain the spatial integrity.

> 💡 **批注**: 这个 PE 策略分类是本文最重要的理论贡献之一：
> - **PERC**（VisionZip）：把 64 个 token 的 PE 压到 [0, 63]，丢失全局参考
> - **PESP**（FastV）：保留原始 PE 但稀疏化，如 [3, 17, 42, ...]，空间不连续
> - **RPME**（提出的修复策略）：线性映射保持相对距离并覆盖完整范围
> 
> 这个分析直接揭示了为什么 VisionZip 在 VG 上比 FastV 更差。

---

**Experiment Setup**

We select two representative methods, VisionZip (PERC) and FastV (PESP), replacing their PE strategy with RPME, and then evaluate the performance of these "fixed" models on visual grounding benchmarks.

---

**Table 3: Position Reconstruction Experiment on Refcoco series and VQA Benchmarks.**

| Method | Refcoco-test | Refcoco+-testA | Refcoco+-testB | Refcocog-test | GQA | MMB | VQAv2 | MME |
|--------|-------------|----------------|----------------|---------------|-----|-----|-------|-----|
| Vanilla | 58.30 | 59.43 | 38.88 | 48.50 | 61.9 | 64.7 | 78.5 | 1862 |
| **Average 64 Tokens** |
| VisionZip-fix | 11.57 (+7.53) | 9.27 (+5.54) | 7.57 (+3.71) | 8.19 (+4.81) | 55.6 (+0.5) | 61.8 (+1.7) | 70.6 (-1.8) | 1700 (+10) |
| FastV-fix | 4.52 (+1.79) | 3.84 (+2.67) | 2.75 (+1.73) | 4.17 (+1.98) | 46.2 (+0.1) | 47.8 (-0.2) | 54.1 (-0.9) | 1247 (-8) |
| Pooling | 12.01 | 12.20 | 7.55 | 11.40 | – | – | – | – |
| **Average 128 Tokens** |
| VisionZip-fix | 21.39 (+16.90) | 19.96 (+15.90) | 13.45 (+8.59) | 15.69 (+12.19) | 58.5 (+0.9) | 63.4 (+1.4) | 74.3 (-1.3) | 1751 (-10) |
| FastV-fix | 13.41 (+3.07) | 11.69 (+3.16) | 12.29 (+2.46) | 12.02 (+3.15) | 51.3 (+0.8) | 57.7 (+1.6) | 60.3 (-1.5) | 1494 (+4) |
| Pooling | 23.01 | 24.37 | 15.04 | 19.69 | – | – | – | – |

> 💡 **批注**: **验证实验非常有说服力**：
> - RPME 为 VisionZip 带来了 **+7.53（64 tokens）和 +16.90（128 tokens）** 的 RefCOCO 提升
> - VisionZip 收益 >> FastV 收益，因为 PERC 的信息损失更严重
> - Token 数量越多，RPME 收益越大（128 > 64），因为更多 token 需要更完整的空间框架
> - VQA 任务上基本无负面影响（±1-2%），说明 RPME 是"免费"的改进
> - 但即使加了 RPME，仍然不如 Pooling，说明**仅修复 PE 不够，还需要保留空间均匀覆盖**

---

Results in Table 3 show that RPME yields notable improvements across benchmarks: VisionZip achieves gains of 5.6% and 13.4% in two settings, while FastV sees more modest increases of 1.8% and 3.2%. These differences confirm our analysis: PERC in VisionZip eliminates positional information, whereas PESP in FastV preserves absolute coordinates but disrupts spatial continuity. Gains grow with larger token budgets, underscoring the increasing importance of complete spatial frameworks for richer visual organization. Pooling methods outperform others consistently by aggregating features on coarse grids that implicitly maintain global topology, reinforcing that reconstructing continuous spatial coordinates is vital for grounding tasks. This strategy has a negligible impact on image understanding and reasoning benchmarks, indicating broad applicability.

> 💡 **批注**: 总结到位。RPME 验证了假设但也暴露了不足：仅修复 PE 不够，还需要**空间均匀覆盖**（pooling 的隐式优势）。这为 Nüwa 的 grid partitioning 设计提供了直接动机。

---

**Finding 3:** The degradation of VLMs on grounding tasks is principally driven by the loss of Global Spatial Reference Frame within token pruning strategies, which can be restored by preserving global position embedding.

> 💡 **批注**: Finding 3 完成了因果链：token pruning → PE 破坏 → 全局空间参考系丢失 → VG 性能崩溃。这是全文的核心理论贡献。
