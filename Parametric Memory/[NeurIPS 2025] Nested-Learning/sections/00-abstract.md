[← 返回 README](../README.md)

# Abstract

## 📌 预览
Nested Learning: 将深度学习模型和优化器统一为嵌套的多层优化问题。

---

Over the last decades, developing more powerful neural architectures and simultaneously designing optimization algorithms to effectively train them have been the core of research efforts to enhance the capability of machine learning models. Despite the recent progresses, particularly in developing Language Models (LMs), there are fundamental challenges and unanswered questions about how such models can continually learn/memorize, self-improve, and find effective solutions. In this paper, we present a new learning paradigm, called Nested Learning (NL), that coherently represents a machine learning model with a set of nested, multi-level, and/or parallel optimization problems, each of which with its own "context flow". Through the lenses of NL, existing deep learning methods learns from data through compressing their own context flow, and in-context learning naturally emerges in large models. NL suggests a philosophy to design more expressive learning algorithms with more "levels", resulting in higher-order in-context learning and potentially unlocking effective continual learning capabilities.

In addition to its neuro-scientific motivation, we advocate for NL by presenting three core contributions: (1) **Expressive Optimizers**: We show that known gradient-based optimizers, such as Adam, SGD with Momentum, etc., are in fact associative memory modules that aim to compress the gradients' information. Building on this insight, we present other "more expressive" optimizers with deep memory and/or more powerful learning rules; (2) **Self-Modifying Learning Module**: A sequence model that learns how to modify itself by learning its own update algorithm; and (3) **Continuum Memory System**: A new formulation that generalizes the traditional "long-term/short-term memory" viewpoint. Combining our self-modifying sequence model with the continuum memory system, we present a continual learning module, called **Hope**.

> 💡 **Abstract 批读**:
> - **核心洞察**: 架构和优化器不是两个独立的东西，而是同一个嵌套优化系统的不同层级
> - **三大贡献**:
>   1. 优化器 = 联想记忆（Adam 是最优的！）
>   2. Self-modifying module = 能学习自身更新规则的模型
>   3. CMS = 连续频谱的记忆系统（不再是二分法的长期/短期）
> - **与 Titans 的关系**: Nested Learning 是 Titans 的**理论升级版**。Titans 提出了 neural memory，NL 给出了统一理论框架，解释了为什么 Titans 有效。
> - **Hope 架构**: Titans 的精神续作，加入了 CMS 和 self-modification

---

## 🔖 Section 总结
这篇论文是 Titans 作者的理论深化之作。如果说 Titans 是"做了一个好的记忆模块"，Nested Learning 就是"解释了为什么记忆和优化是同一件事"。
