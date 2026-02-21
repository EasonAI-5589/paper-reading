[← 返回 README](../README.md)

# Abstract

## 📌 预览
HiDivDrop 挑战了现有 MLLM token pruning 的两个根本误区（浅层关键论 + 固定剪枝调度），提出 Late Injection + Concave Pyramid Pruning + Early Exit 三板斧，在 ~90% 压缩率下保持原始性能并加速训练 1.72×。

---

The computational cost of Multimodal Large Language Models (MLLMs), driven by the quadratic complexity of processing vision tokens, remains a significant barrier to their widespread adoption. While progressive vision token pruning is a promising solution, we find that its full potential has been unrealized due to two key limitations: it misinterprets the role of shallow layers as being crucial for fusion and employs overly rigid, non-adaptive pruning schedules. To address these flaws, we introduce HiDivDrop, a framework that tailors token pruning to the true hierarchical function of MLLM layers. HiDivDrop incorporates two key innovations: (1) a Late Injection strategy that bypasses passive shallow layers, introducing visual tokens directly where active fusion begins; and (2) a Concave Pyramid Pruning scheme with an Early Exit mechanism that dynamically adjusts the pruning rate throughout the middle and deep layers. This process is optimized via an inter-layer similarity measure and a differentiable top- $k$ operator. Extensive experiments show that HiDivDrop compresses ${ \sim } 9 0 \%$ visual tokens while matching the original performance and accelerating training by $1 . 7 2 \times$ . Our work not only sets a new state-of-the-art for efficient MLLM training and inference but also provides valuable insights into the hierarchical nature of multimodal fusion.

> 💡 **Abstract 批读**:
> - **问题**: Vision token 的二次复杂度是 MLLM 普及的主要障碍
> - **诊断**: 现有 progressive pruning 被两个误区拖累——(1) 误认为浅层对融合至关重要；(2) 剪枝调度过于僵硬
> - **方法**: HiDivDrop — 根据层的**真实功能**定制剪枝策略
>   - Late Injection：跳过被动浅层，直接在融合开始处注入视觉 token
>   - Concave Pyramid Pruning + Early Exit：中间层加速剪、深层全丢
>   - 可微 top-k + ILVAS 度量端到端优化
> - **效果**: ~90% 压缩率，性能持平，训练加速 1.72×
> - **定位**: 不仅是 SOTA 效率方法，更提供了关于多模态融合分层本质的新见解

---

## 🔖 Section 总结

### 核心洞察
1. 浅层是"传声筒"而非"融合器" → 可以完全跳过
2. 中间层融合虽关键但高度稀疏 → 可以激进剪枝
3. 深层是语言主导推理 → 视觉 token 可以提前退出
4. 端到端可微的 token 选择优于硬性 top-k
