# 1. Introduction

## 📄 原文

> Large Vision-Language Models (LVLMs) have become a hit in both computer vision and natural language processing studies. From describing the given picture to navigating the internet, using smartphones and making decisions in the real world, large language models with vision abilities are reshaping how we interact with AI systems.
>
> ==LVLMs 正在改变人机交互方式：图片描述、网页导航、手机操作、现实决策==

> Currently, a majority of popular LVLMs rely on **sequential visual representation**, where images are transformed into **hundreds or thousands of tokens** when feeding them to LLM along with language prompts.
>
> ==主流 LVLMs 用序列化视觉表示：图像变成数百到数千个 tokens==

> As LVLMs leverage the advanced emergent capabilities inherent in their language components, they concurrently face a **surge in computational complexity**, correlating with cost increments.
>
> ==问题：计算复杂度激增，成本上升==

---

## 两个未被充分探索的问题

> Two critical areas remain under-explored:
> 1. How do language models process and interpret images?
> 2. While the efficient training and inference of LLMs have attracted considerable attention, these dimensions within LVLMs are yet to be thoroughly examined.
>
> ==1) LLM 如何处理图像？2) LVLM 的高效训练和推理尚未被充分研究==

---

## 核心发现

> In this paper, we uncover the fact that current LVLMs actually apply an **inefficient way** while processing image information.
>
> ==核心发现：当前 LVLMs 处理图像的方式是低效的！==

> Specifically, the image tokens receive **strikingly lower attention scores** compared to their textual counterparts within the token-based LVLMs like LLaVA.
>
> ==Image tokens 收到的 attention score 远低于 text tokens==

> In the image captioning tasks, we observed that within the deep layers (after layer 2) of LLaVA 1.5, image tokens garner an average attention score that amounts to **only 0.21%** of the score attributed to system prompts. In contrast, this figure reaches **50%** in the initial two layers.
>
> ==量化数据：深层只有 0.21%，浅层是 50%==

---

## 解释

> We assume a plausible explanation is that the high redundancy in visual signals leads to the **aggregation of image-related, instruction-specific features onto certain "anchor" tokens** through the self-attention mechanism in the shallow layers.
>
> ==解释：视觉信号冗余 → 浅层 self-attention 将信息聚合到 "anchor tokens"==

> In deep layers, attentions are focused on those anchor tokens, leading to significantly reduced attention on the image tokens themselves.
>
> ==深层只 attend anchor tokens，不再需要 image tokens==

---

## FastV 方法

> The phenomena inspires to propose FastV, a dynamic image tokens pruning method. Given that image tokens contribute minimally to output generation in deeper layers due to diminished attention, **why not consider removing them at these stages**?
>
> ==启发：既然深层 image tokens 贡献小，为什么不删掉？==

> FastV implements an image token pruning strategy at one specific layer of LLM. Prior to this layer, computations proceed as usual. Beyond this selected layer, image tokens are re-evaluated based on their **average received attention scores**. Tokens falling below a predefined attention score threshold are then selectively discarded.
>
> ==FastV：在特定层按 attention score 阈值剪枝==

---

## 与其他方法的区别

> Compared to other attention-based methods for accelerating inference, such as sparse attention, FastV's most notable distinction lies in its **direct elimination of tokens**. This approach not only bypasses the computational demand of the self-attention module but also the **Feed-Forward Network (FFN) module** in deeper layers.
>
> ==FastV 直接删除 tokens，不仅省 attention 还省 FFN==

---

## 实验亮点

> Our experiment on LLaVA-1.5-13B model shows that we can filter out **50% image tokens after layer 2** without sacrificing the average performance.
>
> ==LLaVA-1.5-13B 在第 2 层后删 50% image tokens，性能无损==

> Our latency test experiment on A-OKVQA showed that **LLaVA-13B model with FastV could achieve a lower latency than LLaVA-7B model** while maintaining superior performance.
>
> ==FastV + LLaVA-13B 延迟比 LLaVA-7B 更低，性能更好！==

---

## 贡献总结

> 1. **Identify and analyze** the inefficient visual attention phenomena in prevailing LVLMs.
> 2. **Propose FastV**, a plug-and-play method to significantly reduce inference cost without sacrificing performance.
> 3. **Validate** the effectiveness of FastV on a wide range of vision-language tasks across different LVLMs.
>
> ==三大贡献：发现现象 → 提出方法 → 验证有效==

---

*[返回论文目录](../README.md)*
