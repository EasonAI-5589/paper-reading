# Abstract

> 来源: MM-RLHF: The Next Step Forward in Multimodal LLM Alignment (ICML 2025)
> 作者: Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, et al.
> 机构: KuaiShou, CASIA, NJU, USTC, PKU, Alibaba, Meta AI
> 链接: https://mm-rlhf.github.io/

---

## 📄 原文

Despite notable advancements in Multimodal Large Language Models (MLLMs), most state-of-the-art models have not undergone thorough alignment with human preferences. This gap exists because current alignment research has primarily achieved progress in specific areas (e.g., hallucination reduction), while the broader question of whether aligning models with human preferences can systematically enhance MLLM capability remains largely unexplored.

> 💡 **问题定位**: 当前 MLLM alignment 只在局部领域（如减幻觉）取得进展，没有人系统性地验证过"alignment 能不能全面提升 MLLM"。

To this end, we introduce MM-RLHF, a dataset containing **120k** fine-grained, human-annotated preference comparison pairs. This dataset represents a substantial advancement over existing resources, offering superior size, diversity, annotation granularity, and quality.

> 💡 **数据集**: 120K human-annotated preference pairs，比现有数据集（如 LLaVA-RLHF 的 <10K）大一个数量级，而且是**细粒度人工标注**。

Leveraging this dataset, we propose several key innovations to improve both the quality of reward models and the efficiency of alignment algorithms. Notably, we introduce a **Critique-Based Reward Model**, which generates critiques of model outputs before assigning scores, offering enhanced interpretability and more informative feedback compared to traditional scalar reward mechanisms. Additionally, we propose **Dynamic Reward Scaling**, a method that adjusts the loss weight of each sample according to the reward signal, thereby optimizing the use of high-quality comparison pairs.

> 💡 **两个核心创新**:
> 1. **Critique-Based Reward Model**: 先生成文字评价（critique），再打分 → 可解释性强
> 2. **Dynamic Reward Scaling**: DPO 训练时，根据 reward margin 动态调整每个样本的权重 → 高质量样本影响更大

Our approach is rigorously evaluated across 10 distinct dimensions and 27 benchmarks, with results demonstrating significant and consistent improvements in performance. Specifically, fine-tuning LLaVA-ov-7B with MM-RLHF and our alignment algorithm leads to a **19.5%** increase in conversational abilities and a **60%** improvement in safety.

> 💡 **关键数字**:
> | 指标 | 数值 |
> |------|------|
> | Preference pairs | **120K** |
> | 评估维度 | **10 dimensions, 27 benchmarks** |
> | Conversation 提升 | **+19.5%** |
> | Safety 提升 | **+60%** |

---

## 💡 总结

### 一句话概括
MM-RLHF 是首个系统性验证"MLLM alignment 可以全面提升模型能力"的工作，通过 120K 人工标注数据 + Critique-Based Reward Model + Dynamic Reward Scaling，在 27 个 benchmark 上取得全面提升。

### 对 Apple Assignment 的价值
- **Human annotation protocol**: 50+ annotators, 2 months, fine-grained scoring + ranking + textual explanation
- **Preference data 规模**: 120K pairs，覆盖 image/video/safety 三大域
- **Critique-Based RM**: 从 scalar reward 升级到"先评价再打分"，是 reward modeling 的重要范式创新
- **Dynamic Reward Scaling**: 在 DPO 中引入 reward margin 加权，实用性强
