[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
Related Work 涵盖三个方面：MLLM 架构及其 token 冗余问题、Visual token reduction 方法分类、DPP 的背景知识。

---

## Multimodal large language models

The remarkable achievements of large language models (LLMs) [Touvron et al., 2023a,b, Jiang et al., 2023, Bai et al., 2023, Yang et al., 2024a, Cai et al., 2024b] have lead to a growing trend of extending their powerful reasoning capabilities to other modalities, eventually forming multimodal large language models (MLLMs) [Liu et al., 2023, Li et al., 2024a, Wang et al., 2024, Bai et al., 2025, Chen et al., 2024c, Zhu et al., 2025]. These models typically encode visual inputs as tokens to fully leverage the capabilities of LLMs. However, the sparsity of visual signals results in a significantly larger number of visual tokens compared to their textual counterparts. For example, LLaVA-1.5 [Liu et al., 2024a] converts a 336×336 image into 576 tokens, while its high-resolution variant, LLaVA-NeXT [Liu et al., 2024b], generates 2,880 tokens from an image with twice the resolution. In video understanding scenarios, LongVA [Zhang et al., 2024a] transforms 2,000 frames into over 200K visual tokens, and LongVILA [Chen et al., 2024b] can even handle up to 6,000 frames and produce an ultra-long input sequence of over 1M visual tokens, leading to enormous computational overhead. Therefore, achieving more efficient inference for MLLMs is becoming increasingly critical.

> 💡 **MLLM token 数量一览**:
> | 模型 | 输入 | Token 数 |
> |------|------|---------|
> | LLaVA-1.5 | 336×336 图片 | 576 |
> | LLaVA-NeXT | 672×672 图片 | 2,880 |
> | LongVA | 2000 帧视频 | 200K+ |
> | LongVILA | 6000 帧视频 | 1M+ |
>
> 视觉信号的稀疏性导致需要大量 token 才能表征图像信息。

---

## Visual token reduction

Reducing the number of input visual tokens is an effective way for MLLM inference acceleration. Some works attempt to compress visual tokens via vision-text pre-fusion [Li et al., 2024d, Hu et al., 2024b, Cai et al., 2024a, Zhang et al., 2025], but these approaches require architectural modifications and additional training, thereby increasing computational costs. Other works adopt a training-free approach by removing redundant visual tokens during inference, known as token pruning. These methods can be broadly categorized into two groups.

> 💡 **Token reduction 两大路线**:
> - **Pre-fusion**：在 projector 阶段压缩（需要改架构 + 训练）
> - **Token pruning**：推理时去掉冗余 token（training-free，本文关注的方向）

The first group leverages text-visual attentions within the language model to assess the importance of visual tokens [Chen et al., 2024a, Ye et al., 2025, Xing et al., 2024, Zhang et al., 2024c, Liu et al., 2024c]. However, as pointed out by Zhang et al. [2024b] and Wen et al. [2025a], such methods suffer from attention shift, which compromises pruning accuracy. Moreover, the reliance on attention scores makes them incompatible with efficient attention implementations like FlashAttention [Dao et al., 2022]. The second group avoids these issues by pruning before the language model [Shang et al., 2024, Yang et al., 2024b, Song et al., 2024]. Nonetheless, these methods rely on specific visual encoder architectures and thus cannot be applied across different MLLMs. The third group directly prunes tokens based on feature similarity among visual tokens [Wen et al., 2025b, Alvar et al., 2025, Jeddi et al., 2025]. However, like the second group, they fail to consider the relevance between visual tokens and user instructions during pruning, leading to suboptimal performance.

> 💡 **三组 Pruning 方法对比**:
> | 类别 | 代表方法 | 优点 | 缺点 |
> |------|----------|------|------|
> | Text-visual attention | FastV, PyramidDrop, SparseVLM | 考虑指令相关性 | Attention shift + 不兼容 FlashAttention |
> | Vision encoder-based | PruMerge, VisionZip, TRIM | 不需要 LLM attention | 依赖特定视觉编码器 |
> | Similarity-based | DART, DivPrune | 不需要 attention | 忽略指令相关性 |
>
> CDPruner 的定位：结合 similarity-based 的优势（不需要 attention）+ 引入指令相关性

---

## Determinantal point process

Determinantal Point Process (DPP) was first introduced to describe the distribution of fermion systems in thermal equilibrium [Macchi, 1975], where no two fermions can occupy the same quantum state, resulting in an "anti-bunching" effect that can be interpreted as diversity. Later, DPPs have been widely adopted in list-wise diversity modeling across various domains [Chen et al., 2018, Celis et al., 2018, Li et al., 2024c, Sun et al., 2025]. Unlike MaxMin Diversity Problem (MMDP) [Porumbel et al., 2011], which also aims to maximize diversity, DPP emphasizes global diversity and typically yields more balanced and representative subset selections [Kulesza et al., 2012]. Traditional DPP focuses solely on feature similarity among samples. In this work, we extend this formulation by incorporating instruction relevance as a condition, enabling a unified consideration for superior visual token pruning performance in MLLMs.

> 💡 **DPP vs MMDP**:
> - **MMDP**（DivPrune 用的）: 最大化最小 pairwise 距离 → 关注极端情况
> - **DPP**: 最大化子集 kernel 矩阵的行列式 → 关注全局多样性，子集更均衡
> - 本文创新：传统 DPP 只考虑 feature similarity，CDPruner 加入 instruction relevance 作为条件

---

## 🔖 Section 总结

### 核心洞察
1. Token pruning 的三个流派各有局限，CDPruner 填补了"多样性 + 指令相关性"的空白
2. DPP 比 MMDP 更适合 token pruning，因为它关注全局多样性而非局部极端
3. 不依赖 attention scores 是工程上的重要优势（FlashAttention 兼容性）
