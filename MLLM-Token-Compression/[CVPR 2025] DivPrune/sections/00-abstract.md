# Abstract

> 来源: DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models (Arxiv 2503.02175)

---

## 📄 原文

> 💡 **Abstract 概览**: 这篇论文提出了一种全新的视觉 token 剪枝方法 DivPrune，核心创新在于把 token 剪枝问题建模为 **最大-最小多样性问题 (MMDP)**，通过最大化保留 token 之间的多样性来减少冗余，而不是像以往方法那样依赖注意力分数。

Large Multimodal Models (LMMs) have emerged as powerful models capable of understanding various data modalities, including text, images, and videos. LMMs encode both text and visual data into tokens that are then combined and processed by an integrated Large Language Model (LLM). Including visual tokens substantially increases the total token count, often by thousands. The increased input length for LLM significantly raises the complexity of inference, resulting in high latency in LMMs.

> 💡 **批注**: 问题背景——视觉 token 数量庞大（往往上千个），导致 LLM 推理复杂度飙升。因为 Transformer 的计算量和序列长度呈二次方关系，所以减少 token 数量是提速的直接手段。

To address this issue, token pruning methods, which remove part of the visual tokens, are proposed. The existing token pruning methods either require extensive calibration and fine-tuning or rely on suboptimal importance metrics which results in increased redundancy among the retained tokens.

> 💡 **批注**: 现有方法的两大痛点：
> 1. **需要额外训练/校准**（费时费力，换个模型要重来）
> 2. **用注意力分数选 token 不够好**（选出来的 token 彼此相似，冗余高）

In this paper, we first formulate token pruning as Max-Min Diversity Problem (MMDP) where the goal is to select a subset such that the diversity among the selected tokens is maximized. Then, we solve the MMDP to obtain the selected subset and prune the rest. The proposed method, DivPrune, reduces redundancy and achieves the highest diversity of the selected tokens.

> 💡 **批注**: 核心思想用大白话说就是：
> ```
> 传统方法：挑"最重要"的 token → 但重要的 token 可能长得都差不多
> DivPrune：挑"最不像"的 token → 保证选出来的 token 尽量覆盖所有信息
> ```
> MMDP 的目标：在选出来的子集中，任意两个 token 之间的最小距离要尽可能大。就像在一块地上撒点，让点尽量分散开。

By ensuring high diversity, the selected tokens better represent the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments with various LMMs show that DivPrune achieves state-of-the-art accuracy over 16 image- and video-language datasets. Additionally, DivPrune reduces both the end-to-end latency and GPU memory usage for the tested models.

> 💡 **批注**: 关键卖点总结：
> - **无需训练、无需校准数据**，即插即用
> - 在 **16 个数据集**上 SOTA
> - 极端压缩比（≥80% 剪枝）下优势特别明显
> - 同时降低延迟和显存

---

## 💡 Section 总结

### 核心贡献一句话
把 token pruning 从"挑重要的"转变为"挑多样的"，用 MMDP 数学框架保证最优多样性。

### 关键信息速查
| 项目 | 内容 |
|------|------|
| 方法名 | DivPrune |
| 核心问题 | 视觉 token 冗余导致 LMM 推理慢 |
| 核心方案 | Max-Min Diversity Problem (MMDP) |
| 优势 | 无训练、即插即用、高压缩比下性能好 |
| 评测规模 | 16 个数据集（11 图像 + 5 视频） |
