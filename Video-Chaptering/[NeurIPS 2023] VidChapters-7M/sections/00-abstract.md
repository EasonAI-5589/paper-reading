[← 返回 README](../README.md)

# Abstract

## 📌 预览
VidChapters-7M 的摘要：提出了一个 817K 视频、7M 章节的大规模数据集，定义了三个任务，并展示了在 dense video captioning 上的迁移学习效果。

---

Segmenting long videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation.

> 💡 **数据集核心特点**: 无需人工标注！利用 YouTube 用户自己添加的章节信息，自动化地从网上爬取，因此可以扩展到百万级别。

We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title.

> 💡 **三个任务**:
> 1. **Video Chapter Generation** — 时序分割 + 生成标题（最难，端到端）
> 2. **Chapter Generation given GT Boundaries** — 给定边界，只生成标题（简化版）
> 3. **Video Chapter Grounding** — 给定标题，定位时间段（反向任务）

We benchmark both simple baselines and state-of-the-art video-language models for these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset. Our dataset, code, and models are publicly available at https://antoyang.github.io/vidchapters.html.

> 💡 **关键发现**: 在 VidChapters-7M 上预训练后，迁移到 dense video captioning 任务表现优异，而且性能随预训练数据量增长而提升——这说明数据集的规模优势是实实在在的。

![Figure 1](../images/4a3cb6ce77e5c33483e082c5486a35ee29e772502c561d05f2e7e59f118c2701.jpg)
*Figure 1: A video with user-annotated chapters in VidChapters-7M: the video is temporally segmented into chapters, which are annotated with a chapter title in free-form natural language.*

> 💡 **Figure 1 批读**:
> - 展示了一个实际的 YouTube 视频章节示例
> - 每个章节有起始时间戳 + 自由文本标题
> - 章节是连续的、不重叠的，覆盖整个视频
> - 这种结构化标注正是 VidChapters-7M 的数据形态

---

## 🔖 Section 总结

### 核心洞察
1. **问题定义清晰**: 长视频导航是刚需，但缺乏公开数据集
2. **数据获取巧妙**: 利用用户自愿创建的章节信息，零额外标注成本
3. **任务设计完整**: 三个任务从不同角度考察视频章节理解
4. **迁移价值突出**: 预训练→下游 dense captioning 的范式被验证有效
