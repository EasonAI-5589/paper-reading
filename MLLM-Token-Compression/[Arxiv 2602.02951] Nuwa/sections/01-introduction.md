[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 回顾了现有 token pruning 方法的三类范式，指出它们在 VG 任务上的系统性退化，提出三个核心问题，并概述 Nüwa 的两阶段框架及贡献。

---

VLMs (Liu et al., 2024a; Bai et al., 2025; Zhu et al., 2025) exhibit strong multimodal capabilities through pre-training on massive image-text pairs. However, the large number of vision tokens generated during inference leads to substantial computational overhead and reduced throughput. Recent visual token pruning methods aim to accelerate inference while preserving model performance. These include approaches based on visual-semantic similarity (Yang et al., 2024; Li and Shin, 2024; Jeddi et al., 2025), textual semantic filtering (Chen et al., 2024; Xing et al., 2024; Endo et al., 2024), and multi-stage pruning (Zhang et al., 2025a; Liu et al., 2025b), which improve inference throughput. Additionally, token merging (Bolya et al., 2023), a technique within token pruning, increases token sparsity in VLMs during inference. Fundamentally, reducing the number of tokens enhances sparsity and thereby improves VLM's inference throughput.

> 💡 **批注**: 开篇梳理了三类 token pruning 方法：
> 1. **Visual-semantic similarity**（VisionZip, PruMerge）— 基于视觉语义相似度
> 2. **Textual semantic filtering**（FastV, PyramidDrop）— 基于文本语义的 attention 筛选
> 3. **Multi-stage pruning** — 多层级动态裁剪
> 
> 还提到了 token merging（如 ToMe）作为 pruning 的子类。

---

Nevertheless, recent studies (Wen et al., 2025; Endo et al., 2024) have questioned the effectiveness of existing pruning methods (Chen et al., 2024; Zhang et al., 2025b). In particular, random pruning and pooling-based merging can achieve competitive performance, yet these methods exhibit substantial degradation on visual grounding (VG) tasks compared with visual question and answering (VQA) tasks (Long et al., 2025; Shao et al., 2025a). To assess whether these issues are widespread, we systematically categorize existing pruning methods and compare them with simpler baselines across multiple datasets. Our experiments confirm that these limitations persist. These findings raise fundamental questions: ❶ Why do existing pruning methods exhibit significant task-dependent degradation? ❷ How is vision information encoded and utilized within the VLM's processing pipeline? ❸ How to Mend grounding performance gaps in VLM's token pruning setting?

> 💡 **批注**: 这里引用了两篇质疑现有 pruning 有效性的论文（Wen et al., 2025; Endo/FEATHER 2024），指出一个关键现象：**random pruning 和 pooling 居然能和精心设计的方法打平**。更严重的是，所有方法在 VG 任务上都大幅退化。三个问题构成了全文的分析框架。

---

Through systematic experimental analysis, we uncover that VLMs employ a multi-stage visual processing pipeline that progresses from global to fine-grained integration, with task-specific requirements. In particular, grounding tasks depend on preserving global spatial reference frames, which are constructed from token position information and can be disrupted by token pruning. Informed by these insights, we introduce Nüwa, as shown in Figure 2, a two-stage spatial-aware token pruning framework, patching up the torn spatial integrity. The first stage operates in the visual semantic space to reduce token redundancy while maintaining spatial topology. It employs a Boids-inspired algorithm (Reynolds, 1998) with three operations: (1) Separation: partitioning the token map into localized regions; (2) Alignment: selecting representative tokens based on their alignment with the global context and information density; and (3) Aggregation: merging features of neighboring tokens around representatives using semantic similarity. The second stage performs text-guided refinement in the intermediate layers of the LLM after multimodal feature alignment, using textual semantics to guide further pruning.

> 💡 **批注**: Nüwa 的核心发现：VLM 有一个**多阶段视觉处理流水线**（global → fine-grained），grounding 任务依赖于**全局空间参考系**的完整性。
> 
> 方法设计借鉴了 **Boids 算法**（Reynolds, 1998）——经典的鸟群模拟算法，三个规则完美映射到 token pruning：
> - **Separation** → Grid Partitioning（保持空间均匀覆盖）
> - **Alignment** → Salience Identification（选择高显著性的 benchmark token）
> - **Aggregation** → Spatial Proximity Merging（基于空间邻近性聚合特征）

---

Nüwa demonstrates significant improvements, as shown in Figure 1, on VG benchmarks (7%→47%, 18%→75%) across multiple pruning configurations in LLaVA-1.5, alongside enhancements in VQA benchmarks, including image reasoning and understanding performance (94%→95%), and validates its effectiveness across additional models.

> 💡 **批注**: 性能提升非常显著：VG 从 7% 到 47%（64 tokens），从 18% 到 75%（128 tokens）。VQA 也有小幅提升（94%→95%）。注意这些数字是 performance retention rate，不是绝对精度。

---

Our contributions are as follows:

- 1. Task-specific Analysis: We systematically examine VLM's processing pipelines and show that current pruning methods fail on grounding tasks by overlooking task-specific requirements and disrupting spatial structure. Position reconstruction experiments confirm that spatial perception arises from the integrity of the global reference frame.

- 2. Nüwa Framework: We propose a two-stage spatial-aware pruning framework that retains global spatial anchors through separation and adaptive region aggregation, thereby preserving both spatial and semantic integrity. It further leverages textual information in the LLM for multimodal alignment-based pruning.

- 3. Performance Validation: Our approach yields superior results across 13 datasets and multiple VLMs, establishing new SOTA on VQA (95% performance retention) and VG (47.2% performance retention) tasks while achieving 89% reduction in TFLOPs and 62% reduction in prefill time with a 88.9% tokens reduction.

> 💡 **批注**: 三大贡献结构清晰：
> 1. **分析贡献**：揭示了 task-dependent 退化的根因（空间参考系被破坏）
> 2. **方法贡献**：两阶段空间感知 pruning 框架
> 3. **实验贡献**：13 个数据集、多个 VLM、SOTA 结果
> 
> 88.9% token reduction + 89% TFLOPs reduction + 62% prefill time reduction 是很好的效率指标。
