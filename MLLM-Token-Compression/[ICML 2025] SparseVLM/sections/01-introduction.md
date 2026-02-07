# 1. Introduction

> 来源: SparseVLM (ICML 2025)

---

## 📄 原文

> 💡 **Section 概览**: 三段式论证 — (1) VLM 视觉 token 太多太贵 (2) 现有方法的局限 (3) SparseVLM 的方案

---

### 背景：VLM 的视觉 token 开销

VLMs 把图像编码成 visual tokens 送入 LLM decoder。高分辨率图像导致 token 爆炸：
- 672×672 图像 → **2304 个 visual tokens**，占上下文长度一半以上
- 图像信息天然比文本稀疏 (Marr, 2010)

> 💡 **批注**: 这是 token compression 领域的共识起点。576 tokens (336×336) 已经很多了，高分辨率更是灾难。

---

### 现有方法的问题

两类方法：
1. **修改 vision encoder/projector** — Q-Former (BLIP-2), 自适应 pooling (DeCo) 等
2. **LLM 解码阶段剪枝** — FastV, VoCo-LLaMA 等

> 💡 **关键批判**: 现有解码阶段的方法 **忽略了语言 token 的引导**，与多模态范式矛盾。不同问题应该关注图像的不同区域（前景 vs 背景），但它们一刀切地剪枝。

![Figure 1](../images/20340f3afcca8ba9339a442121159f10c42a1c2ca730a7e4389f478f5b2ec8d6.jpg)
*Figure 1: 视觉 token 稀疏化方法对比。(a) 原图 (b) SparseVLM 根据问题引导选择相关 patch (c) 之前方法如 VoCoLLaMA 的 text-agnostic 方式*

> 💡 **Figure 1 批读**:
> ```
> 传统方法 (c): 不管问什么，都用同样的方式剪枝
> SparseVLM (b): 问什么就保留什么
>   ├── 问"图中有什么动物?" → 保留动物区域
>   └── 问"背景是什么?" → 保留背景区域
> ```
> 这是本文最核心的 motivation。

---

### SparseVLM 方案概述

1. **Text rater selection**: 先筛选出与视觉信号强相关的文本 token 作为 "rater"
2. **Visual token 评估**: 用 self-attention 矩阵衡量 visual token 对 rater 的贡献
3. **自适应剪枝**: 用 attention 矩阵的 rank 决定每层剪枝比例
4. **Token recycling**: 被剪的 token 聚类压缩成紧凑表示

> 💡 **批注**: 整个流程完全复用 decoder 已有的 self-attention 矩阵，**零训练成本**。这是对 FastV 的本质改进 — FastV 只看 attention 大小，SparseVLM 还考虑了"谁在看"。

---

### 核心贡献

1. 首个 **training-free + text-aware** 的高效 VLM 推理方法
2. 提出 text rater 选择 + visual token 评估 + recycling 机制
3. 在多个 VLM 上一致超越 SOTA

---

## 💡 Section 总结

### 核心洞察
1. **Text-aware 是关键**: 同一张图，不同问题应保留不同 token — 这是 SparseVLM 相比 FastV 的根本区别
2. **Training-free 实用性强**: 可以直接 plug-and-play，不需要额外训练数据
3. **Recycling 减少信息损失**: 被剪的 token 不是完全丢弃，而是压缩保留
