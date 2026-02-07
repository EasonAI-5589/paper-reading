# 1. Introduction

> 来源: DivPrune (Arxiv 2503.02175)

---

## 📄 原文

> 💡 **Section 概览**: Introduction 从 LMM 的背景出发，分析了现有 token pruning 方法的不足，引出 DivPrune 的核心动机——用多样性代替重要性来选 token。

Following the success of Large Language Models (LLMs) in language understanding [1, 6, 43], Large Multimodal Models (LMMs) [21, 24, 25, 55] have emerged to handle diverse data types like images and video, by leveraging the foundational capabilities of LLMs. Typically, LMMs encode text and visual modalities into tokens, also known as embeddings. These tokens are then combined and processed by an integrated LLM. The inclusion of visual tokens significantly increases the total number of tokens, often adding thousands to the combined set. Since the running time and memory requirements scale quadratically with input size [7, 8, 17, 41], the addition of visual tokens can substantially raise the running time for LMMs.

> 💡 **批注**: 计算量和序列长度的二次方关系是核心痛点。LLaVA 1.5 有 576 个视觉 token，LLaVA 1.6 更是 3-5 倍，再加上文本 token，总量轻松破千。

![Figure 1](../images/5d4ca29a6241cf8a1e634fde44bd1e735c6d126d1529b06b734ee2a996ef2350.jpg)
*Figure 1: 不同 token pruning 方法在 LLaVA 1.5-7B 上的对比。DivPrune 在各个 TFLOP 压缩比下显著优于所有 baseline，尤其在高压缩比（≤25% TFLOP）下优势更大。*

> 💡 **Figure 1 批读**:
> ```
> 性能排行（TFLOP ratio ≤ 25%）:
> ├── DivPrune ⭐ 显著领先
> ├── FitPrune（需要校准数据）
> ├── FastV
> └── VTW
>
> 关键观察:
> - 其他方法在 TFLOP ≤ 10% 时性能断崖式下跌
> - DivPrune 下降更平缓（graceful degradation）
> - 高 TFLOP ratio 时各方法趋同
> ```

---

### 现有方法的问题

Previous research [4, 38, 50] has demonstrated that there is a high degree of redundancy in the visual information processed by LMMs. As a result, visual token pruning has emerged as a promising solution to address the computational complexity challenges faced by LMMs. Specifically, previous research has demonstrated that reducing the number of visual tokens by 50% [4] to 95% [38] can significantly enhance the inference speed of LMMs.

While promising, token pruning methods have certain shortcomings. For example, the works in [3, 19, 23, 50] require calibration or finetuning for each model which is costly and time-consuming to implement. FastV [4] and PruMerge [38] use attention scores to identify less important tokens for pruning. However, it is shown that using attention scores is not optimal, as some important tokens are overlooked [23]. Additionally, attention-based pruning tends to retain tokens that are similar to each other, leading to redundancy.

> 💡 **批注**: 这里点出了 attention-based pruning 的根本缺陷：
> ```
> 问题: 注意力分数高的 token 往往聚集在相似区域
>       ↓
> 结果: 选出来的 token 冗余度高
>       ↓
> 后果: 高压缩比下信息丢失严重，性能暴跌
> ```
> 这正是 DivPrune 要解决的核心问题。

---

### DivPrune 的解决方案

To address the above-mentioned issues, we formulate token pruning as a Max-Min Diversity Problem (MMDP) [37]. In an MMDP, the objective is to select a subset of elements such that the diversity among them is maximized. We apply this concept to token pruning, which we call DivPrune, aiming to maximize the diversity of the selected tokens by increasing the minimum distance between them. By ensuring high diversity, DivPrune captures a broader range of visual tokens, making it inherently more robust compared to attention-based methods that focus only on token importance scores.

> 💡 **批注**: 和已有方法的核心对比：
> | 方法类型 | 选择标准 | 问题 |
> |---------|---------|------|
> | FastV/PruMerge | 注意力分数（重要性） | token 间冗余高 |
> | FitPrune/VTW | 校准数据优化 | 需要额外数据和计算 |
> | M³ | 微调嵌套表示 | 需要大量训练 |
> | **DivPrune** | **最大化多样性** | **无需训练，覆盖更全** |

---

### 实际优势

DivPrune also offers practical advantages that make it a highly useful solution in real-world scenarios. DivPrune is a plug-and-play solution that can be used without requiring offline optimization with a calibration set, or fine-tuning of the model. DivPrune is applicable to LMMs with any LLM architecture and vision encoder. Additionally, DivPrune is compatible with inference optimization techniques, such as KV caching, resulting in practical speedup in real-world applications.

> 💡 **批注**: 实用性很强——不挑模型、不挑架构、兼容 KV cache，真正的即插即用。

---

### 贡献总结

In summary, our major contributions are as follows:

• We introduce DivPrune, a token pruning method based on MMDP that maximizes diversity among visual tokens, effectively reducing redundancy and ensuring a highly representative subset.
• DivPrune is a training-free, calibration-data-free, plug-and-play solution that can be seamlessly integrated with off-the-shelf LMMs.
• We conduct evaluations using 16 datasets on image- and video-language models with image and video understanding tasks. DivPrune achieves state-of-the-art performance, with noticeable gains under extreme pruning (i.e., ratio ≥ 80%).
• DivPrune reduces GPU memory usage and inference latency while maintaining comparable accuracy compared to the original model across most datasets.

> 💡 **批注**: 四大贡献清晰明了。对我们 STAR-Pro 的启示：DivPrune 证明了在不微调的情况下，合理的 token 选择策略可以大幅减少 token 而不损性能，这为我们设计 token 压缩模块提供了参考。

---

## 💡 Section 总结

### 核心洞察
1. **注意力 ≠ 最佳选择标准**：attention score 高的 token 容易聚集，造成冗余
2. **多样性 > 重要性**：保证选出的 token 尽量不同，比选"最重要"的更有效
3. **即插即用是重要优势**：不需要校准/微调，降低了实际部署门槛
