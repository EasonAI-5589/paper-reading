[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
从 VLM 推理瓶颈出发，梳理现有 token pruning 三大类方法的不足，引出 FSR 的人类视觉认知灵感和三阶段设计。

---

With the rapid progress of large language models (LLMs), vision–language models (VLMs) have advanced substantially in multimodal perception and reasoning. A typical VLM encodes an image into a sequence of visual tokens, concatenates them with text tokens, and performs autoregressive decoding with an LLM. To preserve fine details, modern VLMs increasingly adopt high-resolution encoders and tiling strategies, which often produce massive visual tokens. Since Transformer attention scales quadratically with sequence length, these tokens greatly increase latency and memory, becoming a key bottleneck for deployment.

> 💡 **背景**: 标准问题陈述。高分辨率 VLM（如 LLaVA-NeXT 2880 tokens、Qwen2.5-VL 动态分辨率）的 visual token 数量是推理瓶颈的核心来源。

A practical remedy is training-free visual token pruning, which reduces visual tokens under a fixed budget. Existing methods can be categorized by the signals they exploit:
(i) **Attention-based pruning** selects tokens with high cross-attention or [CLS]-based attention, and thus tends to favor locally salient regions;
(ii) **Similarity-based pruning** relies on inter-token similarity to encourage token diversity, and therefore tends to retain tokens that provide global scene coverage;
(iii) **Joint attention–similarity-based pruning** combine both cues, but still struggle to balance local evidence and global context under high reduction ratios.

> 💡 **方法分类**: 这个三分法很清晰：
> - Attention-based (FastV, PruMerge, SparseVLM, PyramidDrop) → **偏 local**
> - Similarity-based (DivPrune, DART) → **偏 global**
> - Joint (VisionZip, VisPruner, CDPruner, HoloV) → **试图平衡但在极端压缩下仍不够**
> 
> 与 STAR-Pro 对比：STAR-Pro 的 attention-guided merge 属于第一类但加了 merge，FSR 明确把 local/global 分为两阶段处理。

Importantly, the desired allocation between local and global tokens is **task-dependent**. Tasks involving multiple objects, relations, or reasoning typically require collecting multiple local cues across different regions, while fine-grained recognition often depends on a small set of concentrated evidence.

> 💡 **关键洞察**: Task-dependent 动态分配是 FSR 的核心卖点。Figure 1 很直观地展示了简单存在性问题（Focus=9, Scan=23）vs. 推理密集型问题（Focus=15, Scan=17）的 budget 动态变化。这个动态性来自 Eq.4 的 cumulative density threshold ρ。

Studies of human perception in visual question answering tasks show that humans selectively focus on task relevant regions, expand attention to scan the global context, and integrate peripheral cues via ensemble coding for a holistic representation. Inspired by this cognitive process, we propose the **Focus-Scan-Refine (FSR)** pruning framework:

(i) **Focus**: dual-pathway scoring mechanism fusing visual saliency with instruction relevance; top tokens kept until cumulative information density threshold ρ is met.
(ii) **Scan**: conditioned on the focused set, select complementary tokens most different from focused evidence and diverse among themselves.
(iii) **Refine**: merge nearby informative tokens into scan anchors via similarity-based assignment and score-weighted aggregation, keeping token budget unchanged.

> 💡 **认知科学映射**:
> - Focus → 选择性注意（selective attention）
> - Scan → 外周视觉扫描（peripheral scanning）
> - Refine → 集成编码（ensemble coding）
> 
> 这个映射在叙事上很有说服力，但实际算法设计与认知科学的对应关系比较松散。

### Main Contributions:
1. Human-inspired, training-free pruning framework with dynamic local/global budget allocation
2. Comprehensive pipeline: dual-pathway scoring + conditional sampling + aggregation refinement
3. Consistent SOTA across multiple VLM backbones and benchmarks

> 💡 **贡献评估**: 
> - 贡献 1 是 narrative contribution（human-inspired framing）
> - 贡献 2 是真正的技术贡献：三个具体算法组件
> - 贡献 3 是实验验证，覆盖面确实广（LLaVA-1.5/NeXT 7B/13B, Qwen2.5-VL, LLaVA-Video）

---

## 🔖 Section 总结

### 关键信息
- 现有方法的根本问题：无法在极端压缩下同时保 local 和 global
- FSR 的创新：将 pruning 显式分为 Focus（local）+ Scan（global）+ Refine（enrichment）三阶段
- 动态分配机制通过 cumulative density threshold 实现
- Figure 1 & 2 是很好的 motivation visualization
