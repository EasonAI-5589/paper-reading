[← 返回 README](../README.md)

# Abstract

## 📌 预览
现有 VLM token pruning 在 VQA 上还行，在 VG 上崩了。Nüwa 找到根因（空间参考系被破坏）并提出两阶段修复方案。

---

Vision token pruning has proven to be an effective acceleration technique for the efficient Vision Language Model (VLM). However, existing pruning methods demonstrate excellent performance preservation in visual question answering (VQA) and suffer substantial degradation on visual grounding (VG) tasks. Our analysis of the VLM's processing pipeline reveals that strategies utilizing global semantic similarity and attention scores lose the global spatial reference frame, which is derived from the interactions of tokens' positional information. Motivated by these findings, we propose Nüwa, a two-stage token pruning framework that enables efficient feature aggregation while maintaining spatial integrity. In the first stage, after the vision encoder, we apply three operations, namely separation, alignment, and aggregation, which are inspired by swarm intelligence algorithms to retain information-rich global spatial anchors. In the second stage, within the LLM, we perform text-guided pruning to retain task-relevant visual tokens. Extensive experiments demonstrate that Nüwa achieves SOTA performance on multiple VQA benchmarks (from 94% to 95%) and yields substantial improvements on visual grounding tasks (from 7% to 47%).

> 💡 **Abstract 批读**:
> - **核心发现**: pruning 破坏 global spatial reference frame → VG 崩溃
> - **方法**: 两阶段——Stage 1 在 vision encoder 后做 spatial-aware 聚合，Stage 2 在 LLM 内做 text-guided pruning
> - **灵感来源**: Boids 群体智能算法（separation, alignment, cohesion → 对应 separation, alignment, aggregation）
> - **关键数字**: VQA 94%→95%, VG 7%→47%
> - **对比意义**: 首次将 VG benchmark 作为 token pruning 的核心评估维度
