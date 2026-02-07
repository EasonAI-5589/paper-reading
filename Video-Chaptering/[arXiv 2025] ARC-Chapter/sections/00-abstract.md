# 0. Abstract

## 📄 原文

> The proliferation of hour-long videos (e.g., lectures, podcasts, documentaries) has intensified demand for efficient content structuring.
>
> ==小时级长视频（讲座、播客、纪录片）激增 → 需要高效的内容结构化==

> However, existing approaches are constrained by small-scale training with annotations that are typical short and coarse, restricting generalization to nuanced transitions in long videos.
>
> ==现有方法问题：训练规模小、标注粗糙短促 → 难以泛化到长视频的细微转换==

> We introduce ARC-Chapter, the first large-scale video chaptering model trained on over million-level long video chapters, featuring bilingual, temporally grounded, and hierarchical chapter annotations.
>
> ==ARC-Chapter：首个百万级长视频章节模型，双语、时间锚定、层级标注==

> To achieve this goal, we curated a bilingual English-Chinese chapter dataset via a structured pipeline that unifies ASR transcripts, scene texts, visual captions into multi-level annotations, from short title to long summaries.
>
> ==数据构建：结构化流程，融合 ASR、场景文字、视觉描述 → 短标题到长摘要的多层标注==

> We demonstrate clear performance improvements with data scaling, both in data volume and label intensity.
>
> ==关键发现：数据规模和标注密度都能持续提升性能（Scaling Law）==

> Moreover, we design a new evaluation metric termed GRACE, which incorporates many-to-one segment overlaps and semantic similarity, better reflecting real-world chaptering flexibility.
>
> ==新指标 GRACE：多对一匹配 + 语义相似度，更符合真实场景的灵活性==

> Extensive experiments demonstrate that ARC-Chapter establishes a new state-of-the-art by a significant margin, outperforming the previous best by 14.0% in F1 score and 11.3% in SODA score.
>
> ==SOTA 性能：F1 +14.0%，SODA +11.3%==

> Moreover, ARC-Chapter shows excellent transferability, improving the state-of-the-art on downstream tasks like dense video captioning on YouCook2.
>
> ==强迁移性：在 YouCook2 等下游任务也达到 SOTA==

---

## 💡 Key Takeaways

1. **问题**：长视频 + 现有方法规模小/标注粗
2. **方案**：ARC-Chapter = 百万级数据 + 层级标注 + GRACE 指标
3. **结果**：F1 +14%，SODA +11.3%，下游任务也 SOTA

---

## 📊 核心贡献一览

| # | 贡献 | 说明 |
|---|------|------|
| 1 | **VidAtlas 数据集** | 410k+ 视频，115k 小时，中英双语 |
| 2 | **层级标注** | 短标题 → 结构化章节 → 时间戳描述 |
| 3 | **GRACE 指标** | 多对一匹配，解决粒度歧义 |
| 4 | **Scaling Law** | 首次证明章节任务数据规模没有饱和 |

---

*[返回论文目录](../README.md)*
