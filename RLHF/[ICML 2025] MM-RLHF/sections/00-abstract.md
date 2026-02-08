[← 返回 README](../README.md)

# Abstract

## 📌 预览
MM-RLHF 提出了一个 120k 规模的细粒度人工标注偏好对比数据集，以及 Critique-Based Reward Model 和 Dynamic Reward Scaling 两项创新，全面提升 MLLM 的对齐效果。

---

Despite notable advancements in Multimodal Large Language Models (MLLMs), most state-of-the-art models have not undergone thorough alignment with human preferences. This gap exists because current alignment research has primarily achieved progress in specific areas (e.g., hallucination reduction), while the broader question of whether aligning models with human preferences can systematically enhance MLLM capability remains largely unexplored.

> 💡 **核心问题**: 现有 MLLM 对齐研究只在特定方面（如幻觉减少）有进展，能否全面系统地提升 MLLM 能力？MM-RLHF 给出了肯定的回答。

To this end, we introduce MM-RLHF, a dataset containing **120k** fine-grained, human-annotated preference comparison pairs. This dataset represents a substantial advancement over existing resources, offering superior size, diversity, annotation granularity, and quality.

> 💡 **数据集亮点**: 120k 人工标注偏好对比对，在规模、多样性、标注粒度和质量上都显著超越现有资源。

Leveraging this dataset, we propose several key innovations to improve both the quality of reward models and the efficiency of alignment algorithms. Notably, we introduce a **Critique-Based Reward Model**, which generates critiques of model outputs before assigning scores, offering enhanced interpretability and more informative feedback compared to traditional scalar reward mechanisms. Additionally, we propose **Dynamic Reward Scaling**, a method that adjusts the loss weight of each sample according to the reward signal, thereby optimizing the use of high-quality comparison pairs.

> 💡 **两大创新**:
> 1. **Critique-Based Reward Model**: 先生成评价（critique），再打分，提升可解释性
> 2. **Dynamic Reward Scaling**: 根据 reward margin 动态调整每个样本的训练权重，让高质量样本发挥更大作用

Our approach is rigorously evaluated across 10 distinct dimensions and 27 benchmarks, with results demonstrating significant and consistent improvements in performance. Specifically, fine-tuning LLaVA-ov-7B with MM-RLHF and our alignment algorithm leads to a **19.5%** increase in conversational abilities and a **60%** improvement in safety.

> 💡 **关键数字**: 10 维度、27 benchmarks 上一致提升；对话能力 +19.5%，安全性 +60%。

---

## 🔖 Section 总结

### 核心洞察
1. 现有 MLLM 普遍缺少人类偏好对齐阶段，大多停留在 SFT
2. MM-RLHF 证明：精心设计的对齐流程可以**全面提升** MLLM（而不是只改善某一方面）
3. 三大支柱：高质量数据 + Critique-Based Reward Model + Dynamic Reward Scaling
