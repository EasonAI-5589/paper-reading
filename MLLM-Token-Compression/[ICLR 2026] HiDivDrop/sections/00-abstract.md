[← 返回 README](../README.md)

# Abstract

## 📌 预览
HiDivDrop提出了一种针对MLLM层级特性量身定制的vision token pruning框架，核心思想是：浅层不做融合（跳过）、中层积极剪枝（concave pyramid）、深层完全丢弃（early exit）。

---

The computational cost of Multimodal Large Language Models (MLLMs), driven by the quadratic complexity of processing vision tokens, remains a significant barrier to their widespread adoption. While progressive vision token pruning is a promising solution, we find that its full potential has been unrealized due to two key limitations: it misinterprets the role of shallow layers as being crucial for fusion and employs overly rigid, non-adaptive pruning schedules. To address these flaws, we introduce HiDivDrop, a framework that tailors token pruning to the true hierarchical function of MLLM layers. HiDivDrop incorporates two key innovations: (1) a Late Injection strategy that bypasses passive shallow layers, introducing visual tokens directly where active fusion begins; and (2) a Concave Pyramid Pruning scheme with an Early Exit mechanism that dynamically adjusts the pruning rate throughout the middle and deep layers. This process is optimized via an inter-layer similarity measure and a differentiable top-k operator. Extensive experiments show that HiDivDrop compresses ∼90% visual tokens while matching the original performance and accelerating training by 1.72×. Our work not only sets a new state-of-the-art for efficient MLLM training and inference but also provides valuable insights into the hierarchical nature of multimodal fusion.

> 💡 **Abstract 批读**:
> - **核心问题**: 现有progressive pruning方法有两个根本性误解——(1) 误认为浅层对多模态融合至关重要；(2) 采用过于僵化的pruning schedule
> - **两个创新**: Late Injection（跳过浅层）+ Concave Pyramid Pruning with Early Exit（中层积极剪、深层全丢）
> - **关键数字**: 压缩~90% visual tokens，性能几乎无损，训练加速1.72×
> - **与STAR-Pro的关键区别**: HiDivDrop关注的是**WHERE**——在哪些层做什么（层级功能划分），而STAR-Pro关注的是**WHAT**——用什么indicator来判断token重要性。两者是互补的视角。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| Visual token压缩率 | ~90% (576→64) |
| 训练加速 | 1.72× |
| 性能保持 | 98.3% (88.9% pruning) |

### 核心洞察
1. 浅层是"传播者"而非"融合器"——vision tokens在浅层几乎不变化
2. Pruning schedule应该与层级功能对齐，而非一刀切
3. 可微Top-K比Hard Top-K更优，尤其在高压缩率下
