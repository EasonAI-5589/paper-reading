[← 返回 README](../README.md)

# Abstract

## 📌 预览
FSR 提出一个受人类视觉认知启发的 **training-free、plug-and-play** 视觉 token 剪枝框架，通过三阶段 Focus-Scan-Refine 动态分配 local evidence 和 global context 的 token 预算。

---

Vision-language models (VLMs) often generate massive visual tokens that greatly increase inference latency and memory footprint; while training-free token pruning offers a practical remedy, existing methods still struggle to balance local evidence and global context under aggressive compression.
We propose Focus-Scan-Refine (FSR), a human-inspired, plug-and-play pruning framework that mimics how humans answer visual questions: focus on key evidence, then scan globally if needed, and refine the scanned context by aggregating relevant details.
FSR first focuses on key evidence by combining visual importance with instruction relevance, avoiding the bias toward visually salient but query-irrelevant regions.
It then scans for complementary context conditioned on the focused set, selecting tokens that are most different from the focused evidence.
Finally, FSR refines the scanned context by aggregating nearby informative tokens into the scan anchors via similarity-based assignment and score-weighted merging, without increasing the token budget.
Extensive experiments across multiple VLM backbones and vision-language benchmarks show that FSR consistently improves the accuracy-efficiency trade-off over existing state-of-the-art pruning methods.

> 💡 **Abstract 批读**:
> - **问题定位**: 现有 training-free 剪枝方法在高压缩比下无法同时保留 local evidence 和 global context
> - **核心方案**: 三阶段人类认知模拟 — Focus（聚焦）→ Scan（扫描）→ Refine（精炼）
> - **Focus**: 双通道评分（visual saliency + instruction relevance），避免只关注视觉显著但与 query 无关的区域
> - **Scan**: 条件采样，选择与 Focus 集合最不同的 token，补充全局上下文
> - **Refine**: 将丢弃 token 通过 similarity-based 加权合并到 Scan anchors，不增加 budget
> - **关键词**: training-free, plug-and-play, human-inspired, dynamic budget allocation

---

## 🔖 Section 总结

### 核心洞察
1. 将 token pruning 问题重新定义为 local vs. global 的动态分配问题
2. 三阶段设计有认知科学背景支撑（focus attention → scan context → ensemble coding）
3. 与 CDPruner 的 DPP formulation 不同，FSR 采用更直觉的贪心策略但有理论保证
