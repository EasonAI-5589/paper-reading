# Abstract

> 来源: ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries (arXiv 2025)
> 机构: 上海人工智能实验室 (Shanghai AI Lab)

---

## 📄 原文

The proliferation of hour-long videos (e.g., lectures, podcasts, documentaries) has intensified demand for efficient content structuring.

> 💡 **问题背景**: 小时级长视频（讲座、播客、纪录片）越来越多，用户需要高效的内容结构化方案。

However, existing approaches are constrained by small-scale training with annotations that are typical short and coarse, restricting generalization to nuanced transitions in long videos.

> 💡 **现有方法问题**: 
> - 训练规模小
> - 标注粗糙、简短
> - 难以捕捉长视频中的细微转换点

We introduce ARC-Chapter, the first large-scale video chaptering model trained on over million-level long video chapters, featuring bilingual, temporally grounded, and hierarchical chapter annotations.

> 💡 **ARC-Chapter 是什么**:
> - 首个**百万级**章节数据训练的模型
> - **双语** (中英文)
> - **时间锚定** (每个章节有精确时间戳)
> - **层级标注** (从短标题到长摘要)

To achieve this goal, we curated a bilingual English-Chinese chapter dataset via a structured pipeline that unifies ASR transcripts, scene texts, visual captions into multi-level annotations, from short title to long summaries.

> 💡 **数据构建方法**:
> ```
> 输入: ASR + 场景文字 + 视觉描述
>        ↓ 结构化流程
> 输出: 短标题 → 结构化章节 → 长摘要 (多层级)
> ```

We demonstrate clear performance improvements with data scaling, both in data volume and label intensity.

> 💡 **Scaling Law 发现**: 数据量↑ + 标注密度↑ → 性能持续提升（没有饱和！）

Moreover, we design a new evaluation metric termed GRACE, which incorporates many-to-one segment overlaps and semantic similarity, better reflecting real-world chaptering flexibility.

> 💡 **新指标 GRACE**:
> - **Many-to-one 匹配**: 允许多个预测章节对应一个 GT（比 SODA 的 one-to-one 更合理）
> - **语义相似度**: 不只看词汇重叠，还看意思是否相近
> - 更符合真实场景的灵活性

Extensive experiments demonstrate that ARC-Chapter establishes a new state-of-the-art by a significant margin, outperforming the previous best by 14.0% in F1 score and 11.3% in SODA score.

> 💡 **SOTA 性能**:
> | 指标 | 提升幅度 |
> |------|----------|
> | F1 | **+14.0%** (45.3 → 59.3) |
> | SODA | **+11.3%** (19.3 → 30.6) |

Moreover, ARC-Chapter shows excellent transferability, improving the state-of-the-art on downstream tasks like dense video captioning on YouCook2.

> 💡 **迁移能力**: 在 YouCook2 dense captioning 任务也达到 SOTA

---

## 💡 总结

### 一句话概括
ARC-Chapter 用百万级双语层级标注数据训练，配合新指标 GRACE，在 VidChapters-7M 上大幅超越前作 (F1 +14%)。

### 核心贡献

| # | 贡献 | 对比 VidChapters-7M |
|---|------|---------------------|
| 1 | **VidAtlas 数据集** (410K 视频) | VidChapters-7M 是 817K，但标注更粗 |
| 2 | **层级标注** (标题→章节→摘要) | VidChapters-7M 只有单层标题 |
| 3 | **双语** (中英文) | VidChapters-7M 93% 英语 |
| 4 | **GRACE 指标** | SODA 的改进版 |
| 5 | **Scaling Law** | 首次证明数据没饱和 |

### vs VidChapters-7M 论文的关系

```
VidChapters-7M (2023)        ARC-Chapter (2025)
├── 首个大规模数据集          ├── 更大规模 + 更精细标注
├── 定义任务和评测            ├── 改进评测指标 (GRACE)
├── Vid2Seq baseline          ├── 新方法 + GRPO 强化学习
└── SODA = 11.4               └── SODA = 30.6 (+170%)
```
