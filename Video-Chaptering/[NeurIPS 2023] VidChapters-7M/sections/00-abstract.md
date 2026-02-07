# Abstract

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)
> 作者: Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid
> 链接: https://antoyang.github.io/vidchapters.html

---

## 📄 原文

Segmenting long videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. 

> 💡 **问题与方案**: 长视频导航是刚需，但缺数据。核心创新——利用 YouTube 用户已有的章节标注，自动爬取，无需人工标注。结果：817K 视频，7M 章节。

We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. 

> 💡 **三个任务**:
> | 任务 | 输入 | 输出 | 难度 |
> |------|------|------|------|
> | Chapter Generation | 视频 | 时间+标题 | ⭐⭐⭐ |
> | GT-Boundary Title | 视频+时间 | 标题 | ⭐⭐ |
> | Chapter Grounding | 视频+标题 | 时间 | ⭐⭐ |

We benchmark both simple baselines and state-of-the-art video-language models for these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset. Our dataset, code, and models are publicly available at https://antoyang.github.io/vidchapters.html.

> 💡 **核心结论**:
> - 预训练在 VidChapters-7M 上，迁移到 YouCook2/ViTT 大幅提升 SOTA
> - **Scaling 规律**: 数据越多，效果越好
> - 代码/数据/模型全部开源

![Figure 1](../images/4a3cb6ce77e5c33483e082c5486a35ee29e772502c561d05f2e7e59f118c2701.jpg)
*Figure 1: A video with user-annotated chapters in VidChapters-7M: the video is temporally segmented into chapters, which are annotated with a chapter title in free-form natural language.*

> 💡 **Figure 1 批读**: 
> - 这是一个烹饪教程视频，用户标注了章节
> - 左侧：视频帧序列 | 右侧：章节列表 (时间戳+标题)
> - 章节是**连续、无间隙**的，完全覆盖整个视频
> - 标题是**自由文本**，简洁但有语义 (如 "Adding spices" 而非 "00:05:30")

---

## 💡 总结

### 一句话概括
VidChapters-7M 是首个大规模视频章节数据集，通过爬取 YouTube 用户标注获得，定义了三个任务，验证了预训练价值和 scaling 规律。

### 关键数字
| 指标 | 数值 |
|------|------|
| 视频数 | **817K** |
| 章节数 | **7M** |
| 标注方式 | 自动爬取 (无人工) |
| 下游提升 | YouCook2 +14 CIDEr |
