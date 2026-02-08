[← 返回 README](../README.md)

# 7. Open Challenges and Future Work

## 📌 预览
四大未解决挑战：理论基础缺失、缺乏任务/内容自适应、实际任务性能下降、评估标准不完善。

---

## 7.1 Lack of Theoretical Understanding

Although token compression has achieved notable empirical success, most existing approaches remain largely experience-driven and lack rigorous theoretical grounding. Apart from a few works, such as DeCo [105] and DART [183], which analyze how compression influences representation learning within MLLMs, the majority of methods rely on heuristic intuition and limited empirical validation. Consequently, they often exhibit poor transferability across datasets, architectures, and modalities, as well as insufficient robustness under distribution shift.

A key weakness lies in the absence of a principled theory of token importance. Current practices—such as ranking tokens by attention weights, pairwise similarity, or mutual information—lack causal or generalization-based justification. These metrics indicate correlation rather than necessity.

> 💡 **理论空白**: 为什么某些 tokens 可以被安全删除？目前没有理论解释。Attention-based 排序只是"相关性"而非"因果必要性"。这意味着现有方法可能在某些分布上失效。未来需要从 sufficiency、causality、robustness 角度建立理论。

---

## 7.2 Lack of Task- and Content-Aware Adaptivity

Most existing strategies operate in a task-agnostic and content-agnostic manner, applying a fixed compression ratio regardless of task type or visual complexity. However, the granularity of information required varies substantially. As M3 [91] observed, for most benchmarks crafted from natural scenes (such as COCO), 9 tokens per image suffice. In contrast, dense visual perception tasks such as document understanding or OCR require 144-576 tokens per image.

> 💡 **自适应压缩的必要性**: 自然场景 9 tokens 就够，但 OCR 任务需要 144-576 tokens。固定压缩率不可能同时满足两者。VisionThink [190] 用 RL 让模型自主决定是否需要更高分辨率输入 — 这是一个有前景的方向。

Future research should explore task- and content-aware compression, where the model dynamically determines the degree and manner of token reduction. VisionThink [190] proposes a reinforcement learning-based approach enabling autonomous decision on whether higher-resolution visual input is necessary.

---

## 7.3 Performance Degradation in Practical Tasks

Although many token compression methods demonstrate competitive results on general Visual QA tasks, often maintaining comparable accuracy even when reducing visual tokens to 1/3 or 1/4, this performance stability does not generalize well to real-world applications. Tasks requiring fine-grained perception, such as OCR [293], [294], document understanding [295], and dense reasoning over structured visual layouts, tend to experience substantial accuracy drops after compression.

> 💡 **性能下降的本质**: Token compression 在 general VQA 上看起来很好（token 减到 1/3 还能保持精度），但这可能只是因为 VQA benchmark 本身不需要太多视觉信息。一旦到 OCR、文档理解等需要精细视觉信息的任务，压缩的危害就暴露了。这是当前方法"刷榜"的一个陷阱。

---

## 7.4 Limitations of Existing Evaluation

Three key limitations in current evaluation practices:

1. **Lack of systematic task categorization.** Benchmarks are grouped into broad categories, offering limited insight into how compression affects specific capabilities (e.g., spatial relation reasoning or object motion tracking) and content domains (e.g., table or chart interpretation).

2. **Inefficient evaluation processes.** Current evaluations employ at least ten benchmarks with tens of thousands of examples. Many benchmarks exhibit substantial overlap in evaluation focus, leading to redundant assessments.

3. **Absence of consistent evaluation standards.** The selection of benchmarks and metrics varies widely across studies, each emphasizing different strengths. This inconsistency hinders fair cross-method comparison.

> 💡 **评估标准问题**: 
> - 缺乏细粒度能力拆解（"空间推理"和"运动追踪"被混在一起评）
> - Benchmark 之间重叠多，10 个 benchmark 可能只测了 3 种能力
> - 各方法选择性报告有利的 benchmark，难以公平比较
> 
> **启示**: 需要一个专门面向 token compression 的评测框架，包含细粒度能力维度和统一标准。

---

## 🔖 Section 总结

### 四大挑战速查
| 挑战 | 核心问题 | 未来方向 |
|------|----------|----------|
| 理论缺失 | 没有"为什么能压缩"的理论 | 因果分析、sufficiency 理论 |
| 缺乏自适应 | 固定压缩率不适应不同任务 | Task-aware + Content-aware 动态压缩 |
| 实际性能下降 | OCR/文档等细粒度任务掉点严重 | 任务特定的压缩策略 |
| 评估不完善 | Benchmark 重叠、标准不一 | 统一评测框架 + 细粒度能力维度 |
