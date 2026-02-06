# 2. Related Work

## 📄 原文

### Large Vision-Language Model

> Large Vision-Language Models utilize a **Visual Prompt Generator** to transform the visual embeddings into prompts that the language model can comprehend, resulting in a **significant increase in required tokens**.
>
> ==VPG 将视觉嵌入转为 prompt → token 数量大增==

> Handling higher resolution images inevitably necessitates a **quadratic increase** in the number of needed tokens.
>
> ==高分辨率 → token 数量二次增长==

**Token 数量对比：**

| 模型 | 分辨率 | Token 数 |
|------|--------|----------|
| LLaVA | 336×336 | 576 |
| LLaVA | 672×672 | 2304 |
| Fuyu | 1080×1080 | 1296 |
| Video-Poet, Unified-IO2 | 多图/视频 | 数千 |
| Gemini, LWM | 长上下文 | 1M |

---

### Inference Optimization for LLM

> Efficient inference in LLMs is challenged by their **autoregressive generation** where each token prediction depends on the preceding context.
>
> ==自回归生成：每个 token 依赖前面所有上下文==

**两类优化方法：**

| 类型 | 代表 | 思路 |
|------|------|------|
| Memory 优化 | FlashAttention, vLLM, RingAttention | 优化 attention 内存，结果不变 |
| Computation 简化 | StreamingLLM, FastGen | 剪枝冗余 attention 计算 |

> We are interested in the second kind of methods since they are proposed inspired by the **distinct attention patterns** observed in LLM's inference.
>
> ==FastV 属于第二类：利用 attention pattern 剪枝==

> While these methods have boosted the inference efficiency of LLMs, they are designed for **text-only** language models, and whether their effectiveness can be transferred to LVLMs remain **under-explored**.
>
> ==现有方法针对纯文本 LLM，LVLM 场景未被充分探索==

---

### Token Reduction for VLMs

> There have been studies on improving efficiency for Vision-Language Models (VLMs) before the era of large vision-language models. A majority of them focus on **token reduction for vision transformers (ViTs)**.
>
> ==早期工作主要关注 ViT 的 token reduction==

**相关工作：**
- EViT, SPViT, Pumer — ViT token 减少
- PYRA — ViT token merging
- LLaMA-VID — cross-attention 压缩视频帧为 2 tokens（但需要微调）

> FastV is the **first to explore visual token reduction for Large Vision-Language Models (LVLMs)**, which uses language as an interface for various vision-language tasks.
>
> ==FastV 是首个探索 LVLM visual token reduction 的工作！==

> FastV utilizes the signal from LLM to guide the pruning of visual tokens, a strategy **not previously explored**.
>
> ==创新点：用 LLM 的信号（attention score）指导视觉 token 剪枝==

> Another significant advantage of FastV over previous methods is its **simplicity**; it can be applied to any LVLM **without requiring model retraining**.
>
> ==FastV 的优势：简单 + 无需重新训练==

---

## 💡 Key Takeaways

1. **Token 数量问题**：高分辨率/视频 → token 爆炸
2. **现有优化**：主要针对纯文本 LLM
3. **FastV 的定位**：首个 LVLM visual token reduction，用 LLM attention 信号指导

---

*[返回论文目录](../README.md)*
