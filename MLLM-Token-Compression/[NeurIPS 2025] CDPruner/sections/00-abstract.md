[← 返回 README](../README.md)

# Abstract

## 📌 预览
CDPruner 提出用 **条件多样性（Conditional Diversity）** 来做视觉 token 剪枝，核心思路是通过 DPP（行列式点过程）同时考虑 token 间的特征相似性和与指令的相关性，从而选出既多样又相关的 token 子集。

---

In multimodal large language models (MLLMs), the length of input visual tokens is often significantly greater than that of their textual counterparts, leading to a high inference cost. Many works aim to address this issue by removing redundant visual tokens. However, current approaches either rely on attention-based pruning, which retains numerous duplicate tokens, or use similarity-based pruning, overlooking the instruction relevance, consequently causing suboptimal performance.

> 💡 **问题定位**: 现有 token pruning 两大类方法各有缺陷：
> - **Attention-based**: 保留高注意力 token → 大量重复
> - **Similarity-based**: 基于相似度去重 → 忽略用户指令

In this paper, we go beyond attention or similarity by proposing a novel visual token pruning method named CDPruner, which maximizes the conditional diversity of retained tokens. We first define the conditional similarity between visual tokens conditioned on the instruction, and then reformulate the token pruning problem with determinantal point process (DPP) to maximize the conditional diversity of the selected subset.

> 💡 **核心方法**: CDPruner = Conditional Diversity Pruner
> - 定义"条件相似度"：在指令条件下计算视觉 token 间相似度
> - 用 DPP（行列式点过程）建模子集多样性，选择条件多样性最大的子集
> - 关键创新：**把指令相关性作为条件融入 DPP 的 kernel matrix**

The proposed CDPruner is training-free and model-agnostic, allowing easy application to various MLLMs. Extensive experiments across diverse MLLMs show that CDPruner establishes new state-of-the-art on various vision-language benchmarks. By maximizing conditional diversity through DPP, the selected subset better represents the input images while closely adhering to user instructions, thereby preserving strong performance even with high reduction ratios.

> 💡 **优势**: Training-free + Model-agnostic，即插即用

When applied to LLaVA, CDPruner reduces FLOPs by 95% and CUDA latency by 78%, while maintaining 94% of the original accuracy. Our code is available at https://github.com/Theia-4869/CDPruner.

> 💡 **关键数字**: LLaVA 上 FLOPs ↓95%, 延迟 ↓78%, 保留 94% 性能

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| FLOPs 减少 | 95% |
| CUDA 延迟减少 | 78% |
| 性能保留 | 94% |

### 核心洞察
1. 现有方法要么只看 attention（重复多），要么只看 similarity（忽略指令）
2. CDPruner 通过"条件多样性"统一了两者：diversity + instruction relevance
3. 用 DPP 数学框架优雅地解决子集选择问题
