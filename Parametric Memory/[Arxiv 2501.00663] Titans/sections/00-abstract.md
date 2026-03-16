[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文提出神经长期记忆模块 + Titans 架构族，核心思路：attention = 短期记忆，neural memory = 长期记忆，两者互补。

---

Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps an attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of a fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines.

> 💡 **Abstract 批读**:
> - **核心矛盾**: RNN 压缩到固定大小 → 信息丢失；Attention 全量关注 → 二次复杂度。两者各有优劣。
> - **解决方案**: 设计一个 **neural long-term memory module**，学习记忆历史上下文，帮助 attention 只关注当前窗口但仍能利用长期信息。
> - **记忆视角**: Attention = 短期记忆（精确但有限），Neural Memory = 长期记忆（可压缩但持久）。这个类比来自认知科学。
> - **关键卖点**: 可并行训练 + 快速推理 + 扩展到 2M+ 上下文 + 多任务 SOTA。

---

## 🔖 Section 总结

### 核心洞察
1. Titans 的哲学：不同记忆系统各司其职（短期/长期/持久），类似人脑的 confederation of systems
2. Neural memory 是 meta model，在测试时仍在学习（参数在更新），这与传统模型冻结参数的做法根本不同
