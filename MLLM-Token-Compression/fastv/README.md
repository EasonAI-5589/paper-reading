# 📄 FastV: An Image is Worth 1/2 Tokens After Layer 2

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference |
| **作者** | Liang Chen, Haozhe Zhao, Tianyu Liu, Shuang Bai 等 (北大 + 阿里) |
| **发布** | 2024.03 (arXiv), ECCV 2024 |
| **类型** | Method |
| **arXiv** | [2403.06764](https://arxiv.org/abs/2403.06764) |
| **GitHub** | [pkunlp-icler/FastV](https://github.com/pkunlp-icler/FastV) |

## 一句话总结

首次发现 **Vision tokens 在 LLM 深层 attention 极度稀疏**（仅为 system prompt 的 1/472），提出在第 2 层后剪枝 50% visual tokens，实现 45% FLOPs 减少且几乎无性能损失。

## 论文结构

| # | 章节 | 笔记 | 状态 |
|---|------|------|------|
| 0 | Abstract | [📝](./sections/00-abstract.md) | ✅ |
| 1 | Introduction | [📝](./sections/01-introduction.md) | 🚧 |
| 2 | Related Work | - | - |
| 3 | Inefficient Visual Attention | [📝](./sections/03-inefficient-attention.md) | ✅ ⭐ |
| 4 | FastV Method | [📝](./sections/04-method.md) | ✅ ⭐ |
| 5 | Experiments | [📝](./sections/05-experiments.md) | 🚧 |
| 6 | Conclusion | - | - |

## 🔥 核心发现

### Inefficient Visual Attention

| 层深度 | Image Tokens Attention Efficiency | 对比 System Prompt |
|--------|-----------------------------------|-------------------|
| Shallow (Layer 1-2) | 相对平衡 | 1:2 |
| Deep (Layer 3+) | **极低** | **1:472** |

> "Image tokens receive only 0.21% of the attention score compared to system prompts in deep layers"

### 原因解释

```
浅层: 信息从 visual tokens → 聚合到 "anchor tokens" (通常是 system prompt)
深层: 模型主要 attend to anchor tokens，不再需要原始 visual tokens
```

## FastV 方法

```
Input → [Layer 1] → [Layer 2] → 🔪 Prune R% tokens → [Layer 3+] → Output
                                    ↑
                        Rank by attention score
```

**参数：**
- **K** = 剪枝层 (默认 2)
- **R** = 剪枝比例 (默认 50%)

**FLOPs 减少公式：**
```
Reduction = 1 - [K×FLOPs(n) + (T-K)×FLOPs(n̂)] / [T×FLOPs(n)]
where n̂ = (1-R%) × n
```

## 实验结果

| 配置 | FLOPs | Nocaps | Flickr30k | A-OKVQA | MMMU | Avg |
|------|-------|--------|-----------|---------|------|-----|
| Baseline | 100% | 99.8 | 67.9 | 76.7 | 34.8 | 69.8 |
| K=2, R=50% | **55%** | 99.7 | 67.5 | 77.0 | 34.4 | **69.7** |
| K=2, R=75% | 33% | 94.6 | 63.6 | 75.5 | 34.8 | 67.1 |
| K=2, R=90% | 20% | 72.1 | 43.7 | 70.1 | 35.0 | 55.2 |

> **K=2, R=50% 是最佳配置**：45% FLOPs 减少，性能几乎无损

## 与 STAR-Pro 的关系

| 维度 | FastV | STAR-Pro |
|------|-------|----------|
| 压缩位置 | LLM Decoder (单阶段) | VE + LLM (两阶段) |
| 重要性评估 | Attention score | Stage 1: similarity+diversity; Stage 2: text raters |
| 训练 | Training-free | Training-free |
| 核心发现 | Attention 稀疏 | Attention inconsistency (不一致性) |

**STAR-Pro 可以引用 FastV 支持的观点：**
1. Vision tokens 在深层 attention 确实稀疏
2. 单阶段剪枝（仅在 decoder）的局限性

---

*笔记由 3号机 📚 整理*
*阅读日期：2026-02-06*
