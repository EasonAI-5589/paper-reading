[← 返回 README](../README.md)

# 1. Introduction

VLMs (Liu et al., 2024a; Bai et al., 2025; Zhu et al., 2025) exhibit strong multimodal capabilities through pre-training on massive image-text pairs. However, the large number of vision tokens generated during inference leads to substantial computational overhead and reduced throughput. Recent visual token pruning methods aim to accelerate inference while preserving model performance. These include approaches based on visual-semantic similarity (Yang et al., 2024; Li and Shin, 2024; Jeddi et al., 2025), textual semantic filtering (Chen et al., 2024; Xing et al., 2024; Endo et al., 2024), and multi-stage pruning (Zhang et al., 2025a; Liu et al., 2025b), which improve inference throughput. Additionally, token merging (Bolya et al., 2023), a technique within token pruning, increases token sparsity in VLMs during inference. Fundamentally, reducing the number of tokens enhances sparsity and thereby improves VLM's inference throughput.

> 💡 对现有方法的分类：(1) visual-semantic similarity (VisionZip, PruMerge) (2) textual semantic filtering (FastV, PyramidDrop) (3) multi-stage pruning。这个分类和我们 survey 的一致。

Nevertheless, recent studies (Wen et al., 2025; Endo et al., 2024) have questioned the effectiveness of existing pruning methods (Chen et al., 2024; Zhang et al., 2025b). In particular, random pruning and pooling-based merging can achieve competitive performance, yet these methods exhibit substantial degradation on visual grounding (VG) tasks compared with visual question and answering (VQA) tasks (Long et al., 2025; Shao et al., 2025a). To assess whether these issues are widespread, we systematically categorize existing pruning methods and compare them with simpler baselines across multiple datasets. Our experiments confirm that these limitations persist. These findings raise fundamental questions: ❶ Why do existing pruning methods exhibit significant task-dependent degradation? ❷ How is vision information encoded and utilized within the VLM's processing pipeline? ❸ How to Mend grounding performance gaps in VLM's token pruning setting?

> 💡 **三个关键问题**非常清晰地驱动了后续分析。注意 Shao et al., 2025a 就是 STAR-Pro，说明这两篇工作在同一时间段独立发现了 VG 退化问题。

Through systematic experimental analysis, we uncover that VLMs employ a multi-stage visual processing pipeline that progresses from global to fine-grained integration, with task-specific requirements. In particular, grounding tasks depend on preserving global spatial reference frames, which are constructed from token position information and can be disrupted by token pruning. Informed by these insights, we introduce Nüwa, a two-stage spatial-aware token pruning framework, patching up the torn spatial integrity. The first stage operates in the visual semantic space to reduce token redundancy while maintaining spatial topology. It employs a Boids-inspired algorithm (Reynolds, 1998) with three operations: (1) Separation: partitioning the token map into localized regions; (2) Alignment: selecting representative tokens based on their alignment with the global context and information density; and (3) Aggregation: merging features of neighboring tokens around representatives using semantic similarity. The second stage performs text-guided refinement in the intermediate layers of the LLM after multimodal feature alignment, using textual semantics to guide further pruning.

> 💡 **Boids 算法类比**是论文的一大亮点。Reynolds 1987/1998 的经典群体智能三规则：
> - Separation（分离）→ 网格分区避免重叠
> - Alignment（对齐）→ 选代表 token（CLS attention × L2-norm）
> - Cohesion（聚合）→ 语义 + 空间邻近度加权合并
>
> 这个类比虽然不严格（原 Boids 是动态行为，这里是静态操作），但为方法设计提供了直觉框架。

Nüwa demonstrates significant improvements on VG benchmarks (7%→47%, 18%→75%) across multiple pruning configurations in LLaVA-1.5, alongside enhancements in VQA benchmarks, including image reasoning and understanding performance (94%→95%), and validates its effectiveness across additional models.

Our contributions are as follows:

1. **Task-specific Analysis**: We systematically examine VLM's processing pipelines and show that current pruning methods fail on grounding tasks by overlooking task-specific requirements and disrupting spatial structure. Position reconstruction experiments confirm that spatial perception arises from the integrity of the global reference frame.

2. **Nüwa Framework**: We propose a two-stage spatial-aware pruning framework that retains global spatial anchors through separation and adaptive region aggregation, thereby preserving both spatial and semantic integrity. It further leverages textual information in the LLM for multimodal alignment-based pruning.

3. **Performance Validation**: Our approach yields superior results across 13 datasets and multiple VLMs, establishing new SOTA on VQA (95% performance retention) and VG (47.2% performance retention) tasks while achieving 89% reduction in TFLOPs and 62% reduction in prefill time with a 88.9% tokens reduction.

> 💡 **贡献评价**:
> - 贡献 1（分析）最有价值——RPME 实验直接证明了位置嵌入策略是 VG 退化的根因
> - 贡献 2（方法）设计合理但不算革命性——本质是 grid partition + weighted merging + text-guided pruning
> - 贡献 3（实验）VG 的大幅提升确实醒目，但要注意 47% 的绝对性能仍然不高（RefCOCO-test 29.43 vs vanilla 58.30）
