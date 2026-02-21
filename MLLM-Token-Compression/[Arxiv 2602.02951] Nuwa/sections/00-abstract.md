[← 返回 README](../README.md)

# Abstract

## 📌 预览
Nüwa 是一个两阶段 token pruning 框架，专门解决现有方法在 visual grounding 任务上的严重退化问题。核心洞察：pruning 破坏了全局空间参考系（Global Spatial Reference Frame），导致空间定位能力丧失。

---

Vision token pruning has proven to be an effective acceleration technique for the efficient Vision Language Model (VLM). However, existing pruning methods demonstrate excellent performance preservation in visual question answering (VQA) and suffer substantial degradation on visual grounding (VG) tasks. Our analysis of the VLM's processing pipeline reveals that strategies utilizing global semantic similarity and attention scores lose the global spatial reference frame, which is derived from the interactions of tokens' positional information. Motivated by these findings, we propose Nüwa, a two-stage token pruning framework that enables efficient feature aggregation while maintaining spatial integrity. In the first stage, after the vision encoder, we apply three operations, namely separation, alignment, and aggregation, which are inspired by swarm intelligence algorithms to retain information-rich global spatial anchors. In the second stage, within the LLM, we perform text-guided pruning to retain task-relevant visual tokens. Extensive experiments demonstrate that Nüwa achieves SOTA performance on multiple VQA benchmarks (from 94% to 95%) and yields substantial improvements on visual grounding tasks (from 7% to 47%).

> 💡 **批注**: 这篇论文的核心贡献在于发现了一个被忽视的问题：**token pruning 不仅是语义信息的损失，更是空间结构的破坏**。现有方法（FastV、VisionZip、SparseVLM）在 VQA 上表现尚可，但在 VG 上几乎崩溃（性能保留仅 1.88%~7.28%），而 Nüwa 将其提升到 47.2%。方法灵感来自群体智能算法（Boids），这个 analogy 很有趣：separation-alignment-aggregation 对应鸟群的分离-对齐-聚合行为。

---

Code: https://github.com/Man-PaperRejected/Nuwa
