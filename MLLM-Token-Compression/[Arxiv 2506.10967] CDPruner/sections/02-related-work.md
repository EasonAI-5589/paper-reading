# 2 Related Work

> 来源: Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs

---

> 💡 **Section 概览**: 三个方向：MLLM 的发展与视觉 token 膨胀问题、视觉 token 减少的三类方法、DPP 的背景。

---

## 2.1 Multimodal Large Language Models

> 💡 **2.1 要点预览**: MLLM 的视觉 token 数量随分辨率和帧数爆炸式增长。

These models typically encode visual inputs as tokens to fully leverage the capabilities of LLMs. However, the sparsity of visual signals results in a significantly larger number of visual tokens compared to their textual counterparts. For example, LLaVA-1.5 converts a 336×336 image into 576 tokens, while its high-resolution variant, LLaVA-NeXT, generates 2,880 tokens from an image with twice the resolution. In video understanding scenarios, LongVA transforms 2,000 frames into over 200K visual tokens, and LongVILA can even handle up to 6,000 frames and produce an ultra-long input sequence of over 1M visual tokens.

> 💡 **2.1 小结**: 视觉 token 数量问题严重，高分辨率和视频场景尤甚，高效推理势在必行。

---

## 2.2 Visual Token Reduction

> 💡 **2.2 要点预览**: 三类方法各有优劣，CDPruner 要统一它们的优点。

Reducing the number of input visual tokens is an effective way for MLLM inference acceleration. Some works attempt to compress visual tokens via vision-text pre-fusion, but these approaches require architectural modifications and additional training. Other works adopt a training-free approach by removing redundant visual tokens during inference, known as token pruning. These methods can be broadly categorized into three groups:

**第一类：Attention-based（在 LLM 内部剪枝）**
- FastV, PyramidDrop, SparseVLM 等
- 用 text-visual attention 评估 token 重要性
- 缺点：attention shift 导致剪枝不准；不兼容 FlashAttention

**第二类：Vision-based（在 LLM 之前剪枝）**
- LLaVA-Prumerge, VisionZip 等
- 依赖 visual encoder 的特征
- 缺点：依赖特定视觉编码器架构；不考虑用户指令

**第三类：Similarity-based（基于特征相似度剪枝）**
- DART, DivPrune 等
- 直接根据 token 间的特征相似度去重
- 缺点：同样不考虑用户指令

> 💡 **方法分类对比**:
> ```
> Token Pruning 方法谱系
> ├── 需要训练: Pre-fusion 方法 (改架构，成本高)
> └── Training-free:
>     ├── Attention-based: FastV, PyramidDrop, SparseVLM
>     │   ├── ✅ 考虑指令相关性（通过 attention）
>     │   ├── ❌ attention shift 问题
>     │   └── ❌ 不兼容 FlashAttention
>     ├── Vision-based: LLaVA-Prumerge, VisionZip
>     │   ├── ✅ 不需要 attention score
>     │   ├── ❌ 依赖特定视觉编码器
>     │   └── ❌ 不考虑指令
>     └── Similarity-based: DART, DivPrune
>         ├── ✅ 保证多样性
>         ├── ✅ Model-agnostic
>         └── ❌ 不考虑指令
> ```

---

## 2.3 Determinantal Point Process

> 💡 **2.3 要点预览**: DPP 的物理起源和在多样性建模中的应用。

DPP was first introduced to describe the distribution of fermion systems in thermal equilibrium, where no two fermions can occupy the same quantum state, resulting in an "anti-bunching" effect that can be interpreted as diversity. Later, DPPs have been widely adopted in list-wise diversity modeling across various domains. Unlike MaxMin Diversity Problem (MMDP), which also aims to maximize diversity, DPP emphasizes global diversity and typically yields more balanced and representative subset selections.

> 💡 **DPP vs MMDP（大白话）**:
> ```
> MMDP（DivPrune 用的）:
>   目标：最大化被选 token 之间的最小距离
>   问题：过度关注极端情况，可能忽略整体分布
>   类比：选人时只保证最不像的两个人尽可能不同
>
> DPP（CDPruner 用的）:
>   目标：最大化被选子集的"体积"（行列式）
>   优点：考虑全局多样性，选出更均衡的子集
>   类比：选人时保证整个团队的技能覆盖面最大
> ```

Traditional DPP focuses solely on feature similarity among samples. In this work, we extend this formulation by incorporating instruction relevance as a condition, enabling a unified consideration for superior visual token pruning performance in MLLMs.

> 💡 **2.3 小结**: CDPruner 的创新点：在 DPP 的基础上加入了**条件**（instruction relevance），从"纯多样性"变成"条件多样性"。

---

## 💡 Section 总结

### 关键洞察
1. 现有三类方法各有盲区，没有一个能同时做到"多样+相关+通用"
2. DPP 比 MMDP 更适合全局多样性建模
3. CDPruner 的核心创新：将 DPP 从无条件扩展到有条件（conditioned on instruction）
