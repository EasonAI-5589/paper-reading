[← 返回 README](../README.md)

# Abstract

## 📌 预览
MedFrameQA 是首个多图医学 VQA benchmark，从教育视频提取时序连贯帧构建 2,851 个 VQA，11 个 MLLM 准确率均低于 55%。

---

Medical education videos capture the systematic, multi-image diagnostic reasoning that clinicians employ in practice—examining series of related scans, comparing views, and synthesizing findings across modalities. To evaluate whether MLLMs can perform this fundamental aspect of clinical reasoning, we introduce MEDFRAMEQA —the first benchmark explicitly designed to test multi-image medical VQA through educationally-validated diagnostic sequences.

> 💡 **动机**: 临床实践中医生看的是一组相关影像（不同视角、不同时间点、不同模态），而非单张图。现有 benchmark 都是单图 VQA，无法评估这种跨图推理能力。

To build MEDFRAMEQA with high-scalability and high-quality, we develop 1) an automated pipeline that extracts temporally coherent frames from medical videos and constructs VQA items whose content evolves logically across images, and 2) a multiple-stage filtering strategy, including model-based and manual review, to preserve data clarity, difficulty, and medical relevance.

> 💡 **方法概览**: 两个关键设计——(1) 自动化 pipeline 从视频提取时序连贯帧并生成 VQA；(2) 多阶段过滤（模型过滤 + 人工审核）保证质量和难度。

The resulting dataset comprises 2,851 VQA pairs (gathered from 9,237 high-quality frames in 3,420 videos), covering nine human body systems and 43 organs; every question is accompanied by two to five images.

> 💡 **数据规模**: 2,851 VQA、9,237 帧、3,420 视频、9 个人体系统、43 个器官。每题 2-5 张图。

We comprehensively benchmark 11 advanced Multimodal LLMs—both proprietary and open source, with and without explicit reasoning modules—on MEDFRAMEQA. The evaluation challengingly reveals that all models perform poorly, with most accuracies below 50%, and accuracy fluctuates as the number of images per question increases.

> 💡 **核心发现**: 所有模型准确率 < 55%，多数 < 50%。帧数增加时准确率波动而非稳定下降，说明模型没有有效利用多帧信息。

Error analysis further shows that models frequently ignore salient findings, mis-aggregate evidence across images, and propagate early mistakes through their reasoning chains; results also vary substantially across body systems, organs, and modalities.

> 💡 **错误模式三类**:
> 1. **忽略关键发现** — 只看部分帧，漏掉重要信息
> 2. **证据聚合错误** — 无法正确综合多图信息
> 3. **错误传播** — 单图误判导致整条推理链崩溃
>
> 这与 Agent Memory 领域的挑战高度相关：agent 在长程任务中也面临信息遗漏和错误累积问题。

These findings highlight a critical gap: while MLLMs may handle single-image medical tasks, they fail at the multi-image comparative reasoning that defines real clinical practice. We hope this work can catalyze research on clinically grounded, multi-image reasoning and accelerate progress toward more capable diagnostic AI systems.

---

## 🔖 Section 总结

### 核心洞察
1. 首个专门测试多图医学推理的 benchmark，填补单图→多图的评估空白
2. 从教育视频构建数据的思路很聪明——视频天然包含时序连贯的多帧序列
3. 所有 MLLM 表现差，揭示了当前模型在跨图推理上的根本缺陷
