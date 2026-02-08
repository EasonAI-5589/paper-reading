[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 从 MLLM 视觉 token 过多的问题出发，分析现有 attention-based 和 similarity-based 方法的缺陷，提出 CDPruner 的动机和三点贡献。

---

Benefiting from the remarkable success of large language models (LMMs) [Touvron et al., 2023a,b, Jiang et al., 2023, Bai et al., 2023, Yang et al., 2024a, Cai et al., 2024b], multimodal large language models (MLLMs) [Liu et al., 2023, 2024a, Li et al., 2024a, Wang et al., 2024, Chen et al., 2024d,c] have extended their powerful reasoning capabilities to more modalities, such as images or videos. To fully leverage the strengths of LMMs, MLLMs typically encode visual inputs into a form that language models can understand, known as tokens. Within the input sequence, the length of visual tokens often numbers in the hundreds, exceeding their textual counterparts by tens of times. And in video streams [Zhang et al., 2023, Lin et al., 2023, Zhang et al., 2024d] or high-resolution [Liu et al., 2024b, Luo et al., 2024, Guo et al., 2024] scenarios, this number can grow even larger. Since attention-based models [Vaswani et al., 2017] exhibit computational complexity that scales quadratically with token length, an excessive number of visual tokens makes the use of MLLMs costly and impractical for low-latency or resource-constrained applications. [Team et al., 2024, Hu et al., 2024a].

> 💡 **背景**: MLLM 中视觉 token 数量远超文本 token（如 LLaVA 576 个 visual tokens），在视频/高分辨率场景下更严重。Attention 的 O(n²) 复杂度让推理成本很高。

---

![Figure 1](../images/e3a730b99c0e85a3ae6c37a31f837f9aa43f02b172a345ae1fddde86b2415295.jpg)
*Figure 1: Comparison of different token pruning methods. Attention-based methods retain numerous duplicate tokens, failing to achieve effective visual token compression. Similarity-based methods neglect user instructions, always pruning the same tokens and paying insufficient attention to regions most relevant to the question. Our CDPruner considers the conditional diversity of the selected subset, dynamically adjusting pruning according to the user instructions and retaining maximal visual information. In this example, CDPruner successfully preserves tokens related to crucial details, such as the "ICHIRAN" logo on the bowl and chopsticks, the chili pepper on the ramen, and the anti-slip design on the spoon handle, while both alternative methods failed.*

> 💡 **Figure 1 批读**:
> - **Attention-based**（左）: 保留的 token 高度集中在局部区域，大量重复
> - **Similarity-based**（中）: 均匀分散但忽略问题，跟指令无关的区域也保留
> - **CDPruner**（右）: 根据指令动态调整，保留与问题相关且多样的 token
> - 例子很直观：问"描述这碗拉面"时，CDPruner 精准保留了 ICHIRAN logo、辣椒、勺子防滑纹等关键细节

---

Abundant efforts have been made to reduce the inference cost of MLLMs by pruning visual tokens, and existing methods can be roughly divided into two categories. The first is to identify visual tokens with high attention scores as important and discard those deemed less critical [Chen et al., 2024a, Xing et al., 2024, Zhang et al., 2024c]. The second is to remove redundant parts based on feature similarity between visual tokens [Wen et al., 2025b, Alvar et al., 2025, Jeddi et al., 2025]. As illustrated in Figure 1, both approaches suffer from inherent weaknesses, leading to suboptimal performance after pruning. Attention-based methods only consider the importance of visual tokens, resulting in a large number of duplicate tokens being retained, while similarity-based methods neglect user instructions, failing to achieve dynamic pruning in alignment with the current question.

> 💡 **两类方法的根本缺陷**:
> - Attention-based: 只看"重要性"，不考虑"冗余" → token 重复严重
> - Similarity-based: 只看"多样性"，不考虑"相关性" → 静态剪枝，与指令无关

---

To address these issues, we propose CDPruner, a plug-and-play method for MLLM inference acceleration by maximizing the conditional diversity of the selected subset. Conditional diversity simultaneously considers feature similarity and instruction relevance, maintaining considerable performance at high reduction ratios without the need for additional training. Specifically, we first calculate pairwise similarity between visual tokens conditioned on their relevance to the input instruction. To obtain the retained tokens, we reformulate the token pruning problem with determinantal point process (DPP), which is widely used for modeling list-wise diversity based on pairwise similarity [Kulesza et al., 2012, Chen et al., 2018, Celis et al., 2018, Li et al., 2024c, Sun et al., 2025].

> 💡 **CDPruner 核心思路**:
> - "条件多样性" = 特征多样性 × 指令相关性
> - DPP 是一种经典的多样性建模工具，原本用于量子物理中费米子的排斥效应
> - 这里创新地把指令相关性作为条件引入 DPP

As a simple yet effective solution, CDPruner offers several practical advantages. First, in contrast to attention-based methods [Chen et al., 2024a, Zhang et al., 2024c], CDPruner does not require access to attention scores, which ensures its complete compatibility with efficient attention acceleration implementations [Dao et al., 2022]. Second, CDPruner does not depend on a specific visual encoder or language model, and can be readily implemented across any token-based MLLM [Li et al., 2024a, Bai et al., 2025, Zhu et al., 2025]. Extensive experiments across various MLLMs demonstrate the effectiveness and efficiency of CDPruner. When applied to LLaVA-NeXT-7B, it reduces FLOPs by 95%, CUDA latency by 78%, and GPU memory by 17%, while maintaining 94% of the original performance in a training-free manner.

> 💡 **实用优势**:
> 1. 不需要 attention scores → 兼容 FlashAttention
> 2. 不依赖特定 visual encoder → model-agnostic
> 3. Training-free → 即插即用

---

In summary, the contributions of our work are three-fold:

1. We introduce CDPruner, a plug-and-play and model-agnostic solution for visual token pruning that maximizes conditional diversity.

2. We reformulate the token pruning problem with determinantal point process, which facilitates dynamic pruning by jointly considering feature similarity and instruction relevance.

3. We conduct extensive experiments on various vision-language benchmarks, demonstrating that CDPruner consistently achieves state-of-the-art across different reduction ratios.

> 💡 **三点贡献总结**:
> 1. 提出 CDPruner：plug-and-play + model-agnostic
> 2. 用 DPP 重新建模 token pruning：多样性 + 指令相关性
> 3. 在多个 benchmark 上 SOTA

---

## 🔖 Section 总结

### 核心洞察
1. 现有方法的根本矛盾：importance vs. diversity vs. relevance，CDPruner 用"条件多样性"统一
2. DPP 是关键数学工具——自然地建模"反聚集"效应
3. 不需要 attention scores 是重要的工程优势（兼容 FlashAttention）
