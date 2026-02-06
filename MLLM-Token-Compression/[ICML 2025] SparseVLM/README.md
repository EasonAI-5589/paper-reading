# SparseVLM: Visual Token Sparsification for Efficient VLM Inference

**会议**: ICML 2025 (Poster)  
**arXiv**: [2410.04417](https://arxiv.org/abs/2410.04417)  
**GitHub**: https://github.com/Gumpest/SparseVLMs

---

## 一句话总结

**Training-free** 的视觉 token 稀疏化方法，用 **文本引导** 来决定哪些视觉 token 重要，配合 **token recycling** 减少信息损失。

---

## 核心贡献

1. **Text-guided Sparsification**: 首个用文本 token 指导视觉 token 剪枝的 training-free 方法
2. **Rater Selection**: 选择与视觉强相关的文本 token 作为"评委"，过滤无关词（介词、代词等）
3. **Rank-based Adaptation**: 用 attention matrix 的 rank 自适应决定每层剪枝比例
4. **Token Recycling**: 被剪掉的 token 不直接丢弃，而是聚类压缩成更紧凑的表示

---

## 方法概述

### 1. 文本评委选择 (Text Rater Selection)

问题：不是所有文本 token 都与图像相关（如 "What is the..."）

解决：计算文本-视觉 cross-attention，选择超过均值的文本 token 作为 raters

$$\mathbf{r} = \frac{1}{L_v} \sum_{j=1}^{L_v} \text{Softmax}(H_v H_q^T)_j$$

### 2. 视觉 Token 重要性估计

从 self-attention matrix 中提取文本→视觉的 attention 作为重要性分数：

$$\tilde{p} = \frac{1}{L_t} \sum_{i=1}^{L_t} P_i$$

### 3. 自适应剪枝比例

用 attention matrix 的 rank 反映冗余度：

$$N = \lambda \times (L_v - \text{rank}(P))$$

### 4. Token Recycling

被剪掉但重要性较高的 token → k-NN 密度峰值聚类 → 重建为更少的 token

---

## 关键实验结果

### Image Understanding (LLaVA-7B)

| Tokens | Method | Avg Acc | FLOPs (T) | Latency |
|--------|--------|---------|-----------|---------|
| 576 | Vanilla | 100% | 4.62 | 57.82ms |
| 192 | FastV | 87.9% | 2.11 | 34.87ms |
| 192 | **SparseVLM** | **99.1%** | 2.14 | 36.50ms |
| 128 | FastV | 82.4% | 1.70 | 30.70ms |
| 128 | **SparseVLM** | **96.7%** | 1.72 | 33.28ms |
| 64 | FastV | 72.0% | 1.29 | 27.30ms |
| 64 | **SparseVLM** | **89.3%** | 1.30 | 29.89ms |

**核心数据**: 
- 192 tokens (↓66.7%): 只掉 0.9% 准确率
- 比 FastV 好 **11.2-17.3%**

### Video Understanding (VideoLLaVA)

- 2048 → 194 tokens (90.5% 剪枝)
- SparseVLM: 95.0% avg accuracy
- FastV: 80.3% avg accuracy
- **比 FastV 好 14.7%**

---

## 与其他方法的对比

| 方法 | Training | Text-guided | Recycling | 特点 |
|------|----------|-------------|-----------|------|
| FastV | ✗ | ✗ | ✗ | 固定从第2层开始剪 |
| ToMe | ✗ | ✗ | ✗ | 直接合并相似 token |
| PyramidDrop | ✗ | ✗ | ✗ | 金字塔式渐进剪枝 |
| VoCo-LLaMA | ✓ | ✗ | ✗ | 需要训练 |
| **SparseVLM** | ✗ | ✓ | ✓ | 文本引导 + 回收机制 |

---

## 亮点发现

1. **文本引导很关键**: POPE 上用文本引导比不用好 2.7%
2. **Token recycling 在高压缩率时效果更明显**: 64 tokens 时 POPE 提升 17.7%
3. **兼容 FlashAttention**: 设计了 dual-flash attention 操作

---

## 局限性

- Rank 计算有一定开销
- 需要访问 attention matrix（与某些优化冲突）
- 视频任务只测了简单的 QA benchmark

---

## 我的评价

**优点**:
- 思路清晰：文本引导是 VLM 场景下很自然的设计
- 实验充分：LLaVA / Mini-Gemini / Qwen2-VL / VideoLLaVA 全测了
- Training-free 实用性强

**可改进**:
- 与 STAR (reasoning-guided) 的思路是否可以结合？
- 在更复杂的视频理解任务上的表现？

---

## 相关论文

- FastV (ECCV 2024) - 层内固定剪枝
- PyramidDrop (CVPR 2025) - 金字塔式剪枝
- VoCo-LLaMA (CVPR 2025) - 需要训练的压缩
- ToMe (ICLR 2023) - Token merging

---

*3号机整理 @ 2026-02-06*
