# Stop Looking for "Important Tokens" in Multimodal Language Models: Duplication Matters More

> **DART (Duplication-Aware Reduction of Tokens)**
> arXiv: 2502.11494 | Feb 2025
> Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, Linfeng Zhang
> Shanghai Jiao Tong University, Shanghai AI Laboratory, Sun Yat-sen University, Peking University
> Code: https://github.com/ZichenWen1/DART

## 一句话总结

**重要性指标不靠谱**——基于 attention score 的 token importance pruning 甚至不如随机剪枝；改用 **token duplication**（cosine similarity to pivot tokens）来决定裁哪些 token，效果远超 SOTA。

## 核心贡献

1. **揭示 importance-based pruning 的四大缺陷**：忽略 token 间交互、与 FlashAttention 不兼容、position bias、精度甚至不如 random
2. **提出 DART**：基于 token duplication 的 training-free token reduction，选少量 pivot tokens（≤2%），按 cosine similarity 去除重复 token
3. **理论保证**：通过 Hausdorff distance bound 证明 pruning 后输出误差有界
4. **极端压缩下仍然强劲**：88.9% reduction 下 LLaVA-1.5-7B 保留 93.7% 性能，beat second-best 2.2%
5. **兼容 FlashAttention**：overhead < 0.08s，1.99× total / 2.99× prefill speedup

## 方法概述

```
Input Tokens → Select Pivot Tokens (≤2%, by K-norm/random/etc.)
            → Compute ε-Duplicate Score (cosine sim to pivots)
            → Retain tokens with LOW duplication to pivots
            → Continue LLM inference with reduced tokens
```

- **Pivot Selection**: K-norm, V-norm, attention score, or even random — all work comparably
- **ε-Duplicate Score**: `dup(p_i, x_j) = cos(p_i, x_j)` — tokens with high similarity to pivots are redundant
- **Pruning Point**: After layer 2, with 8 pivot tokens (default)

## 关键实验结果

| Model | Tokens Retained | Avg. Performance |
|---|---|---|
| LLaVA-1.5-7B | 192 (↓66.7%) | 98.8% |
| LLaVA-1.5-7B | 128 (↓77.8%) | 98.0% |
| LLaVA-1.5-7B | 64 (↓88.9%) | **93.7%** |
| LLaVA-Next-7B | 320 (↓88.9%) | **93.9%** |
| Qwen2-VL-7B | ↓88.9% | 87.5% |
| MiniCPM-V2.6 | ↓88.9% | 76.1% |

## 局限性

- Pivot token 选择对不同模型可能有不同最优策略
- 在 OCR-heavy 任务上（如 MiniCPM-V2.6）极端压缩下性能下降明显
- 理论分析依赖 Lipschitz 连续假设，实际 transformer 不一定严格满足

## 与其他工作的关系

- **vs FastV/SparseVLM**: 这些基于 attention importance 的方法在极端压缩下甚至不如 random pruning
- **vs ToMe**: Token merging 在 ViT 阶段做，会破坏 cross-modal interaction
- **vs PyramidDrop/PDrop**: Progressive pruning 但仍依赖 importance 指标
- **与 HiDivDrop 的关系**: HiDivDrop 关注 layer-wise progressive pruning 的 schedule 设计，DART 关注 what criteria to prune — 互补

---

## Citation Landscape

**被引 55 次** (Semantic Scholar, as of 2026-02)

### 主要引用方向

| Category | Representative Papers |
|---|---|
| **Importance + Diversity 结合** | IDPruner, D2Pruner, DivPrune, ToDRE |
| **Video Token Compression** | TimeChat-Online, QuickVideo, VideoScan, FlexSelect |
| **VLM Acceleration** | EfficientVLA, FastDriveVLA, BlindSight, ERGO |
| **Benchmark & Analysis** | "Are We Using the Right Benchmark", "All You Need Are Random Visual Tokens?" |
| **Downstream Applications** | Prune2Drive (autonomous driving), OmniDocLayout |

### 主要参考文献

| Paper | Venue | Relation |
|---|---|---|
| FastV | ECCV 2024 | Attention-based importance pruning baseline |
| SparseVLM | ICML 2025 | Text-guided cross-modal attention pruning |
| ToMe | ICLR 2023 | Training-free token merging in ViT |
| PyramidDrop | arXiv 2024 | Progressive visual redundancy reduction |
| FlashAttention | NeurIPS 2022 | Efficient attention, DART is compatible |
