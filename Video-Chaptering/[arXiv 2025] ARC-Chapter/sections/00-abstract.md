[← 返回 README](../README.md)

# Abstract

## 📌 预览
论文提出 ARC-Chapter：首个百万级长视频 chaptering 模型，配套双语层级标注数据集、新评估指标 GRACE，在 VidChapters-7M 上 F1 +14.0%，SODA +11.3%。

---

The proliferation of hour-long videos (e.g., lectures, podcasts, documentaries) has intensified demand for efficient content structuring. However, existing approaches are constrained by small-scale training with annotations that are typical short and coarse, restricting generalization to nuanced transitions in long videos. We introduce ARC-Chapter, the first large-scale video chaptering model trained on over million-level long video chapters, featuring bilingual, temporally grounded, and hierarchical chapter annotations. To achieve this goal, we curated a bilingual English-Chinese chapter dataset via a structured pipeline that unifies ASR transcripts, scene texts, visual captions into multi-level annotations, from short title to long summaries. We demonstrate clear performance improvements with data scaling, both in data volume and label intensity. Moreover, we design a new evaluation metric termed GRACE, which incorporates many-to-one segment overlaps and semantic similarity, better reflecting real-world chaptering flexibility. Extensive experiments demonstrate that ARC-Chapter establishes a new state-of-the-art by a significant margin, outperforming the previous best by 14.0% in F1 score and 11.3% in SODA score. Moreover, ARC-Chapter shows excellent transferability, improving the state-of-the-art on downstream tasks like dense video captioning on YouCook2.

> 💡 **摘要批读**:
> 论文有三个核心卖点：
> 1. **数据规模**：百万级长视频 chapter 标注（VidAtlas），比之前大 50 倍
> 2. **层级标注**：Short Title → Structural Chapter → Timestamp-Aligned Description，三级输出
> 3. **评估指标**：GRACE 采用 many-to-one 匹配，比 SODA 的 one-to-one 更鲁棒
>
> 关键数字：F1 45.3→59.3（+14.0），SODA 19.3→30.6（+11.3），CIDEr 100.9→186.6

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 之前 SOTA | ARC-Chapter | 提升 |
|------|-----------|-------------|------|
| F1 | 45.3 | 59.3 | +14.0 |
| SODA | 19.3 | 30.6 | +11.3 |
| CIDEr | 100.9 | 186.6 | +85.7 |

### 核心洞察
1. 长视频 chaptering 的瓶颈是数据规模和标注粒度，不是模型架构
2. 多模态（视频+ASR）比单模态显著更好
3. 现有指标（SODA）对 chaptering 任务不够合适，需要 many-to-one 匹配
