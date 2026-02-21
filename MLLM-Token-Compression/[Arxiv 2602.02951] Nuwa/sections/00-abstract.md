[← 返回 README](../README.md)

# Abstract

## 📌 预览
Nüwa 发现现有 token pruning 方法在 VQA 上表现良好但在 visual grounding (VG) 上严重退化，根因是破坏了全局空间参考系。提出两阶段框架：Stage 1 在 vision encoder 中通过 Boids 启发的分离-对齐-聚合保留空间锚点；Stage 2 在 LLM 中做 text-guided pruning。VQA 保持率从 94%→95%，VG 从 7%→47%。

---

Vision token pruning has proven to be an effective acceleration technique for the efficient Vision Language Model (VLM). However, existing pruning methods demonstrate excellent performance preservation in visual question answering (VQA) and suffer substantial degradation on visual grounding (VG) tasks. Our analysis of the VLM's processing pipeline reveals that strategies utilizing global semantic similarity and attention scores lose the global spatial reference frame, which is derived from the interactions of tokens' positional information. Motivated by these findings, we propose Nuwa, a two-stage token pruning framework ¨ that enables efficient feature aggregation while maintaining spatial integrity. In the first stage, after the vision encoder, we apply three operations, namely separation, alignment, and aggregation, which are inspired by swarm intelligence algorithms to retain information-rich global spatial anchors. In the second stage, within the LLM, we perform text-guided pruning to retain task-relevant visual tokens. Extensive experiments demonstrate that Nuwa achieves SOTA performance on multiple ¨ VQA benchmarks (from $94 \%$ to $9 5 \%$ ) and yields substantial improvements on visual grounding tasks (from $7 \%$ to $47 \%$ ).

> 💡 **批注**: Abstract 清晰地定义了问题（pruning 破坏 VG）、根因（全局空间参考系丢失）和方案（两阶段空间感知 pruning）。关键数字：88.9% token 压缩下 VQA 95% 保持 + VG 47% 保持。与 FastV 等前作最大的区别在于**首次系统分析了 pruning 对空间任务的影响**并提出修复方案。

---

Code: https://github.com/Man-PaperRejected/Nuwa
