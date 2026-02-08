# Abstract

> 来源: ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries (arXiv 2025)
> 机构: ARC Lab, Tencent PCG

---

## 📄 原文

# ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries

Junfu $\mathbf { p } _ { \mathbf { u } } { } ^ { * }$ , Teng Wang∗, Yixiao Ge†, Yuying Ge, Chen Li, Ying Shan

ARC Lab, Tencent PCG

∗Core contributors, †Project lead

> 💡 **作者团队**: 腾讯 PCG ARC Lab，核心贡献者是 Junfu Pu 和 Teng Wang，项目负责人是 Yixiao Ge。

The proliferation of hour-long videos (e.g., lectures, podcasts, documentaries) has intensified demand for efficient content structuring. However, existing approaches are constrained by small-scale training with annotations that are typical short and coarse, restricting generalization to nuanced transitions in long videos. We introduce ARC-Chapter, the first large-scale video chaptering model trained on over million-level long video chapters, featuring bilingual, temporally grounded, and hierarchical chapter annotations. To achieve this goal, we curated a bilingual English-Chinese chapter dataset via a structured pipeline that unifies ASR transcripts, scene texts, visual captions into multi-level annotations, from short title to long summaries. We demonstrate clear performance improvements with data scaling, both in data volume and label intensity. Moreover, we design a new evaluation metric termed GRACE, which incorporates many-to-one segment overlaps and semantic similarity, better reflecting real-world chaptering flexibility. Extensive experiments demonstrate that ARC-Chapter establishes a new state-of-the-art by a significant margin, outperforming the previous best by $1 4 . 0 \%$ in F1 score and $1 1 . 3 \%$ in SODA score. Moreover, ARC-Chapter shows excellent transferability, improving the state-of-the-art on downstream tasks like dense video captioning on YouCook2.

> 💡 **摘要核心信息拆解**:
> 1. **问题**: 小时级长视频（讲座、播客、纪录片）越来越多，需要高效的内容结构化
> 2. **现有方法瓶颈**: 训练规模小，标注粗糙简短，难以泛化到长视频细微转换
> 3. **ARC-Chapter 方案**:
>    - 首个**百万级**章节数据训练的视频章节化模型
>    - **双语**（中英文）、**时间锚定**、**层级标注**
>    - 数据构建: ASR + 场景文字 + 视觉描述 → 短标题 → 结构化章节 → 长摘要
> 4. **关键发现**: 数据量↑ + 标注密度↑ → 性能持续提升（Scaling Law）
> 5. **新指标 GRACE**: Many-to-one 匹配 + 语义相似度，比 SODA 更合理
> 6. **SOTA 性能**: F1 +14.0%（45.3→59.3）, SODA +11.3%（19.3→30.6）
> 7. **迁移能力**: YouCook2 dense captioning 也达到 SOTA

Date: November 18, 2025

Github: https://github.com/TencentARC/ARC-Chapter

---

## 💡 Section 总结

### 一句话概括
ARC-Chapter 用百万级双语层级标注数据训练，配合新指标 GRACE，在 VidChapters-7M 上大幅超越前作（F1 +14%）。

### 核心贡献一览

| # | 贡献 | 关键数字 |
|---|------|----------|
| 1 | **VidAtlas 数据集**（410K+ 视频，115K 小时） | 50× 于前作训练规模 |
| 2 | **层级标注**（标题→章节→摘要） | 3 级输出结构 |
| 3 | **双语支持**（中英文） | — |
| 4 | **GRACE 指标** | Many-to-one 匹配 |
| 5 | **Scaling Law** | 首次证明数据未饱和 |
| 6 | **GRPO 强化学习** | 进一步提升时间定位精度 |

### vs Chapter-LLaMA（前 SOTA）

```
Chapter-LLaMA (CVPR 2025)       ARC-Chapter (arXiv 2025)
├── 20K 样本训练                 ├── 100万级章节训练
├── ASR-only                     ├── 多模态（ASR + Video）
├── 单层标题输出                 ├── 层级输出（标题/章节/描述）
├── F1 = 45.3                    ├── F1 = 59.3 (+14.0)
├── SODA = 19.3                  ├── SODA = 30.6 (+11.3)
└── CIDEr = 100.9                └── CIDEr = 186.6 (+85%)
```
