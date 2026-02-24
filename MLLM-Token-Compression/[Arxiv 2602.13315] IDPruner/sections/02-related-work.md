[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览

Related Work 按三类方法组织：(1) Importance-based Token Pruning，(2) Diversity-based Token Pruning，(3) Hybrid Strategies。每类都有清晰的代表方法和各自的缺陷分析。这一节也是 IDPruner 定位自身的地方。

---

## Large Multimodal Models and Visual Token Pruning

Recent Multimodal Large Language Models (MLLMs) (Liu et al., 2023a; Wang et al., 2024; Zhu et al., 2025b) have demonstrated impressive capabilities across various visual tasks, yet they encounter significant computational bottlenecks due to the massive volume of visual tokens. Static-resolution models like LLaVA-1.5 (Liu et al., 2023a) and LLaVA-NeXT (Liu et al., 2024a) require 576 and 2,880 input tokens per image, respectively, while newer architectures such as the Qwen-VL (Bai et al., 2025), LLaVA-OneVision (Li et al., 2024a), and InternVL (Zhu et al., 2025b) series demand comparable token budgets for high-resolution processing. Consequently, visual token pruning, which eliminates unnecessary tokens, has emerged as a crucial technique for accelerating MLLM inference. Current research typically falls into two categories: importance-based methods and diversity-based methods.

> 💡 **Token 数量规模**: LLaVA-1.5 = 576 tokens/图，LLaVA-NeXT = 2880 tokens/图。高分辨率模型 token 数急剧增加，剪枝技术的必要性不言而喻。

## Importance-based Token Pruning

Importance-based approaches reduce computational overhead by retaining only the most salient tokens. Early studies rely on attention scores from LLM decoder layers (Chen et al., 2024a; Zhang et al., 2024e; Xing et al., 2024; Zhang et al., 2025b; Ye et al., 2025; Han et al., 2025), while subsequent research discovers that the attention of the [CLS] token in Vision Transformers (ViT) provides a more effective importance measure (Yang et al., 2025b; Liu et al., 2025; Zhang et al., 2024d; Tong et al., 2025). To mitigate limitations such as FlashAttention incompatibility, recent work has introduced alternative metrics, including optimal transport and L2 norms (Yang et al., 2025a; Dhouib et al., 2025). Beyond training-free methods, approaches like VisionSelector (Zhu et al., 2025a) employ learnable modules to estimate token importance, achieving state-of-the-art performance through end-to-end training. Despite their effectiveness in capturing region-specific details, these methods often overlook global context, potentially causing information loss in background areas.

> 💡 **Importance-based 方法进化路线**:
> 1. **LLM decoder attention** → 直觉：语言模型关注的 visual token 更重要。但依赖 attention，FlashAttention 不兼容。
> 2. **ViT [CLS] attention** → 更有效的重要性度量（CLS token 聚合了全局视觉信息）
> 3. **替代度量** (OT, L2 norm) → 解决 FlashAttention 不兼容问题
> 4. **VisionSelector（可学习）** → 端到端训练，SOTA 但需要训练开销
>
> IDPruner 采用 VisionSelector 作为重要性估计器，因此**继承了其需要训练的特性**。这是 IDPruner 不是 training-free 方法的根本原因。

## Diversity-based Token Pruning

Diversity-based approaches aim to preserve information coverage by regarding visual tokens as a collective set, minimizing redundancy to retain a representative subset of visual features. DivPrune (Alvar et al., 2025) formulates this task as a Max-Min Diversity Problem, solving it via a greedy algorithm to maximize semantic coverage, while DART (Wen et al., 2025) employs a parallelizable strategy that selects pivot tokens and eliminates their nearest neighbors to maintain diversity. However, maximizing redundancy reduction often comes at the cost of missing fine-grained details in focal regions, as these methods may indiscriminately retain task-irrelevant noise.

> 💡 **Diversity-based 方法缺陷**: 把 token 当集合而不是个体，最大化语义覆盖。缺陷是：视觉图像中重要区域（前景）往往有大量相似 token（因为该区域有细节），diversity-based 方法会把这些细节 token 视为冗余而丢弃，导致丢失精细信息。这正是需要 hybrid 方法的原因。
>
> 💡 **DART 特点**: 用「pivot tokens + 删除近邻」的策略，可并行化，这是工程上的优势。

## Hybrid Strategies

Synergizing importance and diversity typically yields superior performance compared to single-criterion methods. VisPrune (Zhang et al., 2024c) allocates token budgets based on both [CLS] attention and diversity, while CDPruner (Zhang et al., 2025a) employs Determinantal Point Processes (DPP) to balance these objectives. Other approaches explore alternative integration strategies, such as ensuring spatial coverage via region-based allocation (Zou et al., 2025; Arif et al., 2025) or modeling pruning as a set cover problem to optimize multimodal coverage (Li et al., 2025; Deng et al., 2025). Although effective, these methods typically rely on heuristic integration strategies without a systematic analytical framework. In this work, we address this limitation by introducing a systematic framework that optimally harmonizes importance and diversity.

> 💡 **Hybrid 方法对比**:
> - **VisPruner (VisPrune)**: CLS attention + diversity，两阶段，启发式
> - **CDPruner**: DPP（行列式点过程），有数学依据但计算复杂
> - **SCOPE**: saliency + coverage，区域分配
> - **IDPruner**: MMR（最大边际相关性），有信息检索理论背书，O(KN) 复杂度
>
> IDPruner 的核心差异化：**有原则性的分析框架** + **MMR 提供 Pareto 最优保证** + **计算高效**。
>
> 💡 **STAR-Pro 引用建议**: STAR-Pro 应在这一节的 Hybrid Strategies 部分引用 IDPruner，描述为「用 MMR 算法联合优化重要性和多样性，但依赖可学习的重要性估计模块（VisionSelector），需要训练」。这样可以突显 STAR-Pro 作为 training-free hybrid 方法的优势。

## 🔖 Section 总结

Related Work 用三个层次的方法对比，清楚地定位了 IDPruner 的位置：比纯 importance-based 或 diversity-based 方法更全面，比现有 hybrid 方法有更严格的理论依据。核心批评对手是 DivPrune（diversity-only）、VisionSelector（importance SOTA，但 IDPruner 在此基础上加了 diversity）、以及其他 hybrid 方法（直觉式组合）。
