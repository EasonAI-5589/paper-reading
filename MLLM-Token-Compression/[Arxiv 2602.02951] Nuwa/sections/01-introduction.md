[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 梳理 token pruning 方法的三大类别，指出它们在 VG 任务上的系统性退化，提出三个核心问题，并概述 Nüwa 的两阶段方案（Boids 启发 + text-guided）。贡献：(1) 任务依赖的流水线分析；(2) 空间感知剪枝框架；(3) 13 个数据集上的 SOTA。

---

VLMs (Liu et al., 2024a; Bai et al., 2025; Zhu et al., 2025) exhibit strong multimodal capabilities through pre-training on massive image-text pairs. However, the large number of vision tokens generated during inference leads to substantial computational overhead and reduced throughput. Recent visual token pruning methods aim to accelerate inference while preserving model performance. These include approaches based on visual-semantic similarity (Yang et al., 2024; Li & Shin, 2024; Jeddi et al., 2025), textual semantic filtering (Chen et al., 2024; Xing et al., 2024; Endo et al., 2024), and multi-stage pruning (Zhang et al., 2025a; Liu et al., 2025b), which improve inference throughput. Additionally, token merging (Bolya et al., 2023), a technique within token pruning, increases token sparsity in VLMs during inference. Fundamentally, reducing the number of tokens enhances sparsity and thereby improves VLM's inference throughput.

> 💡 **批注**: 开篇将现有 pruning 方法分为三类：(1) visual-semantic similarity (VisionZip, PruMerge)；(2) textual semantic filtering (FastV, PyramidDrop)；(3) multi-stage pruning。这个分类框架贯穿全文。

---

![](../images/538f0f9c5a3106638b933c1739a85b22a1e13a21bd8a19f7369e4cbecdf6a4ce.jpg)
*Figure 2: The left panel contrasts our Nuwa framework with prior token pruning methods. (a) Pruning at the vision encoder stage; (b) Text-guided pruning within the LLM; (c) Our two-stage approach: initial spatial-aware pruning via local aggregation that preserves global anchors in the vision encoder, followed by text-guided refinement in the LLM.*

> 💡 **Figure 2 批读**: 三列对比清晰展示了 Nüwa 与前作的结构差异。(a) encoder 侧方法（VisionZip）只做全局语义选择，丢失空间；(b) LLM 侧方法（FastV）依赖 attention score，同样丢失空间；(c) Nüwa 两阶段：先在 encoder 侧做 spatial-aware 聚合（保留全局坐标），再在 LLM 中做 text-guided 细筛。

---

Nevertheless, recent studies (Wen et al., 2025; Endo et al., 2024) have questioned the effectiveness of existing pruning methods (Chen et al., 2024; Zhang et al., 2025b). In particular, random pruning and pooling-based merging can achieve competitive performance, yet these methods exhibit substantial degradation on visual grounding (VG) tasks compared with visual question and answering (VQA) tasks (Long et al., 2025; Shao et al., 2025a). To assess whether these issues are widespread, we systematically categorize existing pruning methods and compare them with simpler baselines across multiple datasets. Our experiments confirm that these limitations persist. These findings raise fundamental questions: $\bullet$ Why do existing pruning methods exhibit significant task-dependent degradation? ❷ How is vision information encoded and utilized within the VLM's processing pipeline? $\cdot$ How to Mend grounding performance gaps in VLM's token pruning setting?

> 💡 **批注**: 三个递进问题构成了全文的逻辑骨架：Q1 → Sec 2.1 (benchmark baseline 对比)；Q2 → Sec 2.2 (attention flow 分析)；Q3 → Sec 2.3 (位置重建) + Sec 3 (方法)。

---

Through systematic experimental analysis, we uncover that VLMs employ a multi-stage visual processing pipeline that progresses from global to fine-grained integration, with task-specific requirements. In particular, grounding tasks depend on preserving global spatial reference frames, which are constructed from token position information and can be disrupted by token pruning. Informed by these insights, we introduce Nuwa ¨ , as shown in Figure 2, a two-stage spatial-aware token pruning framework, patching up the torn spatial integrity. The first stage operates in the visual semantic space to reduce token redundancy while maintaining spatial topology. It employs a Boids-inspired algorithm (Reynolds, 1998) with three operations: (1) Separation: partitioning the token map into localized regions; (2) Alignment: selecting representative tokens based on their alignment with the global context and information density; and (3) Aggregation: merging features of neighboring tokens around representatives using semantic similarity. The second stage performs text-guided refinement in the intermediate layers of the LLM after multimodal feature alignment, using textual semantics to guide further pruning.

> 💡 **批注**: Boids 算法（1987 年 Craig Reynolds 提出的群体智能模型）的三条规则——Separation、Alignment、Cohesion——被巧妙映射到 token pruning 的三步操作。这是本文最独特的设计灵感来源。

---

Nuwa ¨ demonstrates significant improvements, as shown in Figure 1, on VG benchmarks $( 7 \%  4 7 \%$ , $18 \%  7 5 \%$ ) across multiple pruning configurations in LLaVA-1.5, alongside enhancements in VQA benchmarks, including image reasoning and understanding performance $( 9 4 \% \to 9 5 \%$ ), and validates its effectiveness across additional models.

> 💡 **批注**: 两组数字对应两个 token 预算：64 tokens (88.9% 压缩) 下 VG 7%→47%；128 tokens (77.8% 压缩) 下 VG 18%→75%。VG 的提升幅度远大于 VQA，说明 Nüwa 确实针对性修复了空间信息。

---

Our contributions are as follows:

1. Task-specific Analysis: We systematically examine VLM's processing pipelines and show that current pruning methods fail on grounding tasks by overlooking task-specific requirements and disrupting spatial structure. Position reconstruction experiments confirm that spatial perception arises from the integrity of the global reference frame.

2. Nuwa Framework: ¨ We propose a two-stage spatial-aware pruning framework that retains global spatial anchors through separation and adaptive region aggregation, thereby preserving both spatial and semantic integrity. It further leverages textual information in the LLM for multimodal alignment-based pruning.

3. Performance Validation: Our approach yields superior results across 13 datasets and multiple VLMs, establishing new SOTA on VQA $9 5 \%$ performance retention) and VG $4 7 . 2 \%$ performance retention) tasks while achieving $89 \%$ reduction in TFLOPs and $62 \%$ reduction in prefill time with a $8 8 . 9 \%$ tokens reduction.

> 💡 **批注**: 三条贡献层层递进：分析问题 → 提出方案 → 验证效果。第三条的效率数据（89% TFLOPs↓、62% prefill↓）说明 Nüwa 不仅修复了 VG，效率也与前作持平。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| Token 压缩比 (最激进) | 88.9% (576→64) |
| VQA 性能保持 | 94% → 95% |
| VG 性能保持 (64 tokens) | 7% → 47% |
| VG 性能保持 (128 tokens) | 18% → 75% |
| TFLOPs 减少 | 89% |
| Prefill 时间减少 | 62% |

### 核心洞察
1. 现有 pruning 方法在 VQA 上接近 random baseline，在 VG 上全面崩溃
2. 根因：token pruning 破坏了全局空间参考系 (Global Spatial Reference Frame)
3. Nüwa = Boids-inspired spatial aggregation (Stage 1) + text-guided pruning (Stage 2)
