# Abstract

> 来源: Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs

---

> 💡 **论文一句话总结**: 提出 CDPruner，用 DPP（行列式点过程）最大化视觉 token 的**条件多样性**（同时考虑 token 间的相似度和与指令的相关性），实现 training-free 的视觉 token 剪枝，在多个 MLLM 上达到 SOTA。

---

## 📄 原文

In multimodal large language models (MLLMs), the length of input visual tokens is often significantly greater than that of their textual counterparts, leading to a high inference cost. Many works aim to address this issue by removing redundant visual tokens. However, current approaches either rely on attention-based pruning, which retains numerous duplicate tokens, or use similarity-based pruning, overlooking the instruction relevance, consequently causing suboptimal performance.

> 💡 **批注**: 现有方法的两大问题：
> 1. **Attention-based**（如 FastV）：只看重要性，不管重复 → 保留很多重复 token
> 2. **Similarity-based**（如 DivPrune）：只看多样性，不管用户问了什么 → 不能根据问题动态调整

In this paper, we go beyond attention or similarity by proposing a novel visual token pruning method named CDPruner, which maximizes the conditional diversity of retained tokens. We first define the conditional similarity between visual tokens conditioned on the instruction, and then reformulate the token pruning problem with determinantal point process (DPP) to maximize the conditional diversity of the selected subset.

> 💡 **核心思路（大白话）**:
> ```
> 传统方法：
> - Attention 派：哪个 token 重要就留哪个（但重要的可能长得一样）
> - Similarity 派：尽量留不一样的 token（但可能留了跟问题无关的）
>
> CDPruner：两个都要！
> - 既要留的 token 之间尽量不同（多样性）
> - 又要跟用户的问题相关（条件）
> - 用 DPP 这个数学工具来同时优化这两个目标
> ```

The proposed CDPruner is training-free and model-agnostic, allowing easy application to various MLLMs. Extensive experiments across diverse MLLMs show that CDPruner establishes new state-of-the-art on various vision-language benchmarks. By maximizing conditional diversity through DPP, the selected subset better represents the input images while closely adhering to user instructions, thereby preserving strong performance even with high reduction ratios.

When applied to LLaVA, CDPruner reduces FLOPs by 95% and CUDA latency by 78%, while maintaining 94% of the original accuracy.

> 💡 **关键数字**:
> | 指标 | 数值 |
> |------|------|
> | FLOPs 降低 | 95% |
> | CUDA 延迟降低 | 78% |
> | 性能保持 | 94% |
> | 额外训练 | 无（training-free） |

Our code is available at https://github.com/Theia-4869/CDPruner.

---

## 💡 Abstract 总结

### 核心贡献
1. **Conditional Diversity** = Feature Similarity + Instruction Relevance
2. **DPP 建模** = 数学上优雅地同时优化多样性和相关性
3. **Training-free & Model-agnostic** = 即插即用，适用于各种 MLLM

### 与已有方法的定位
```
方法谱系:
├── Attention-based: FastV, PyramidDrop, SparseVLM
│   └── 问题: 保留重复 token
├── Similarity-based: DART, DivPrune
│   └── 问题: 忽略用户指令
└── CDPruner (本文): Conditional Diversity via DPP ⭐
    └── 同时解决上述两个问题
```
