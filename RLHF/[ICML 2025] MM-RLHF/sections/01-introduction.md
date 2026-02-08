# 1. Introduction

> 来源: MM-RLHF (ICML 2025)

---

## 📄 原文

Although Multimodal Large Language Models (MLLMs) have demonstrated remarkable potential in addressing complex tasks that involve the integration of vision, language, and audio, state-of-the-art models today seldom undergo a rigorous alignment stage [64, 17, 12, 16, 2]. Typically, these models only progress to the Supervised Fine-tuning (SFT) stage, leaving critical aspects such as truthfulness, safety, and alignment with human preferences largely unaddressed. While recent efforts have begun to explore MLLM alignment, they often focus on specific domains, such as mitigating hallucination or enhancing conversational capabilities, which fail to comprehensively improve the model's overall performance and reliability. This raises a critical question:

**Is alignment with human preferences only capable of enhancing MLLMs in a limited set of tasks?**

> 💡 **核心问题**: 现在的 MLLM 大多只做到 SFT 就停了，没有经过 alignment。已有的 alignment 工作只在局部领域（如 hallucination）有效。本文要回答的核心问题：alignment 能不能**全面**提升 MLLM？

In this work, we confidently answer this question with a resounding "No.". We demonstrate that a well-designed alignment pipeline can comprehensively enhance MLLMs along multiple dimensions, including visual perception, reasoning, dialogue, and trustworthiness, thereby significantly broadening their practical applicability. To achieve this, we conduct in-depth investigations into three pivotal areas: data curation, reward modeling, and alignment algorithms.

> 💡 **回答**: 本文用实验证明——精心设计的 alignment pipeline 可以**全面**提升 MLLM。三个支柱：数据、Reward Model、对齐算法。

At first, we introduce MM-RLHF, a dataset designed to advance Multimodal Reinforcement Learning from Human Feedback (RLHF). The dataset spans three domains: image, video understanding, and MLLM safety. Constructed through a rigorous pipeline, MM-RLHF ensures high-quality, fine-grained annotations. Dataset creation process involves the following steps (Figure 1):

• **Data Collection.** We curate a diverse set of multimodal tasks from various sources, totaling 10 million data instances, ensuring broad representation across tasks.
• **Data Selection.** Through rigorous re-sampling, we extract 30k representative queries, ensuring diversity across a wide range of data types, such as real-world scenarios, mathematical reasoning, chart understanding, and other practical domains (Figure 2).
• **Model Response Generation.** We utilize state-of-the-art models, such as Claude 3.5-Sonnet and Qwen2-VL-72B, to generate responses for various tasks.
• **Fine-grained Human Annotation.** We employ a meticulous annotation process, involving over 50 annotators over two months, to score, rank, and provide textual explanations for responses. This results in more than 120k high-quality ranked comparison pairs.

> 💡 **数据构建 Pipeline（重点！Apple Assignment 相关）**:
> ```
> 10M 原始数据
>   ↓ CLIP 聚类 + 重采样
> 30K queries (Long:Short:MCQ = 4:5:1)
>   ↓ SOTA 模型生成 responses (GPT-4o, Claude 3.5, Qwen2-VL-72B, LLaVA-OV-72B)
> 多个 responses per query
>   ↓ 50+ annotators, 2 months
> 120K+ ranked comparison pairs
>   (scoring + ranking + textual explanation per dimension)
> ```
> **关键细节**:
> - 标注三个维度: Helpfulness, Faithfulness, Ethical Considerations
> - 每个 response 都有分维度打分 + 理由
> - 所有 response 做 overall ranking + 排序理由
> - 平手时还有创新策略（构造正/负样本）

![Figure 1](../images/f9a4cacc02cbeb166decb90a91ff089e62d4cec94949afaf4eb3798188e6775d.jpg)
*Figure 1: MM-RLHF Construction Pipeline. (1) Data Collection and Cleaning: 从 10M 样本出发，基于图片相似度聚类，跨类均匀采样。(2) Response Generation: 用 GPT-4o、Qwen2-VL-72B 等生成高质量回答。(3) Human Annotation: 9 个类别的人工标注，包括 scoring、ranking 和 explanations。*

> 💡 **Figure 1 批读**:
> ```
> Pipeline 三阶段:
> ├── Stage 1: Data Collection & Cleaning
> │   ├── 10M instruction samples (多源)
> │   ├── CLIP encode → KNN clustering (100 centers)
> │   └── 均匀采样 → 30K queries
> ├── Stage 2: Response Generation
> │   ├── 闭源: GPT-4o, Claude 3.5-Sonnet
> │   └── 开源: Qwen2-VL-72B, LLaVA-OV-72B, LLaVA-Video-72B
> └── Stage 3: Human Annotation
>     ├── 9 categories 的细粒度评分
>     ├── Scoring (per dimension) + Ranking (overall)
>     └── Textual explanations for each score/rank
> ```

Compared to existing datasets, MM-RLHF significantly advances in diversity, response quality, and annotation granularity, providing a robust foundation for MLLM alignment.

Building on the MM-RLHF dataset, we investigate how human-annotated data can enhance MLLM alignment, with a focus on reward modeling and training optimization. Recognizing the pivotal role of reward models in providing feedback signals to guide the alignment process, we propose a **Critique-Based Reward Model** (Figure 3). Traditional reward models, which output scalar values, often lack interpretability, while directly using MLLMs as reward models place high demands on their instruction-following capabilities, limiting their practicality. To address these limitations, we first transform concise human annotations into detailed, model-friendly formats using MLLMs. These enriched annotations serve as learning targets, guiding the reward model to first generate critiques and then assign scores based on the critiques. This approach enables the model to provide fine-grained scoring explanations, significantly enhancing the quality and interpretability of the reward signals. **MM-RLHF-Reward-7B achieves SOTA performance on several reward model benchmarks, outperforming several 72B-scale models.**

> 💡 **Critique-Based Reward Model 核心思路**:
> ```
> 传统 RM: input → scalar score (不可解释)
> 直接用 MLLM 打分: 依赖 instruction-following (不稳定)
> 
> Critique-Based RM (本文):
> input → critique (文字评价) → score (基于评价的打分)
> ```
> **关键创新**: 用 GPT-4o 扩展人工标注的简短理由为详细 critique，作为训练目标。结果：7B 模型超过多个 72B 模型。

Building on this high-quality reward model, we introduce **Dynamic Reward Scaling** within the Direct Preference Optimization (DPO) framework. Traditional DPO methods [3] use a fixed training weight for all human-preferred and non-preferred training pairs. In contrast, Dynamic Reward Scaling calculates a reward margin for each comparison pair using MM-RLHF-Reward-7B. During training, it assigns higher weights to comparison pairs with larger reward margins. This ensures that the most informative samples have a stronger influence on model updates. As a result, the training process becomes more efficient, leading to improved model performance.

> 💡 **Dynamic Reward Scaling**:
> - 传统 DPO: 所有 pair 权重一样
> - MM-DPO: reward margin 大的 pair → 权重大（更有信息量）
> - 用 MM-RLHF-Reward-7B 计算 reward margin，动态调整 β

Finally, to rigorously evaluate our approach, we construct two specialized benchmarks. The first, **MM-RLHF-RewardBench**, is sampled from our dataset and consists of meticulously human-annotated data for evaluating reward models. The second, **MM-RLHF-SafetyBench**, is curated and filtered from existing benchmarks and focuses on safety-related tasks, including privacy protection, adversarial attacks, jailbreaking, and harmful content detection.

We conduct extensive evaluations across ten key dimensions, covering 27 benchmarks. The results demonstrate that our training algorithm, combined with the high-quality MM-RLHF dataset, leads to significant improvements in model performance. Specifically, models fine-tuned with our approach achieve an average **11%** gain in conversational abilities and a **57%** reduction in unsafe behavior. The integration of our reward model further amplifies these gains, highlighting the effectiveness of our alignment algorithm.

> 💡 **贡献总结**:
> 1. **MM-RLHF 数据集**: 120K human-annotated pairs, 3 domains (image/video/safety), fine-grained
> 2. **Critique-Based Reward Model**: 先 critique 再打分，7B 超 72B
> 3. **MM-DPO with Dynamic Reward Scaling**: reward margin 加权
> 4. **两个 benchmark**: MM-RLHF-RewardBench + MM-RLHF-SafetyBench
> 5. **全面评估**: 10 dimensions, 27 benchmarks, conversation +11%, safety +57%

---

## 💡 Section 总结

### 核心洞察
1. **Gap**: 当前 MLLM 普遍缺少 alignment 阶段，只到 SFT 就停了
2. **Claim**: 精心设计的 alignment 可以**全面**提升 MLLM（不只是减幻觉）
3. **三个支柱**: 高质量数据 + Critique-Based RM + Dynamic Reward Scaling in DPO

### 对 Apple Assignment 的关键信息
- **Human annotation protocol**: 50+ annotators, 2 months, 三维度(Helpfulness/Faithfulness/Ethics)打分 + ranking + explanation
- **Preference data**: 10M → 30K queries → 120K+ pairs
- **Annotation 创新**: 平手时构造正/负样本（避免无用 tie）
- **质量保证**: 8 multimodal research experts 定期 review
