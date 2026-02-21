[← 返回 README](../README.md)

# Abstract

## 📌 预览
FSR 是一个受人类视觉感知启发的三阶段 training-free visual token pruning 框架：Focus（局部证据）→ Scan（全局上下文）→ Refine（聚合精炼），在多个 VLM 和 benchmark 上实现 SOTA 的精度-效率权衡。

---

Vision-language models (VLMs) often generate massive visual tokens that greatly increase inference latency and memory footprint; while training-free token pruning offers a practical remedy, existing methods still struggle to balance local evidence and global context under aggressive compression. We propose Focus-Scan-Refine (FSR), a human-inspired, plug-and-play pruning framework that mimics how humans answer visual questions: focus on key evidence, then scan globally if needed, and refine the scanned context by aggregating relevant details. FSR first focuses on key evidence by combining visual importance with instruction relevance, avoiding the bias toward visually salient but query-irrelevant regions. It then scans for complementary context conditioned on the focused set, selecting tokens that are most different from the focused evidence. Finally, FSR refines the scanned context by aggregating nearby informative tokens into the scan anchors via similarity-based assignment and score-weighted merging, without increasing the token budget. Extensive experiments across multiple VLM backbones and vision-language benchmarks show that FSR consistently improves the accuracy-efficiency trade-off over existing state-of-the-art pruning methods. The source codes can be found at https://github.com/ILOT-code/FSR

> 💡 **Abstract 批注**:
> - **核心问题**: 现有 training-free pruning 在激进压缩下难以平衡 local evidence 和 global context
> - **方法**: 三阶段 Focus-Scan-Refine，灵感来自人类视觉认知（选择性注意 → 外周扫描 → 集成编码）
> - **Focus**: 双通道打分（visual saliency + instruction relevance），避免只选视觉显著但与 query 无关的 token
> - **Scan**: 条件互补采样，选与 Focus 集合最不同的 token
> - **Refine**: 将丢弃 token 信息聚合到 Scan anchor，不增加 budget
> - **定位**: Training-free, plug-and-play，无需修改模型权重
> - **与同类对比**: 相比 CDPruner（DPP-style joint pruning）、HoloV（partition-wise）、VisPruner 等，FSR 在极端压缩比下优势更明显

**Keywords:** Vision–Language Models, Human-Inspired Visual Processing, Visual Token Pruning, Efficient Multimodal Inference
