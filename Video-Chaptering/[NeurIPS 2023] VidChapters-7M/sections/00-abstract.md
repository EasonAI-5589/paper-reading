# Abstract

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)
> 作者: Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid
> 链接: https://antoyang.github.io/vidchapters.html

---

## 📄 原文

Segmenting long videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. 

We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. 

We benchmark both simple baselines and state-of-the-art video-language models for these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset. Our dataset, code, and models are publicly available at https://antoyang.github.io/vidchapters.html.

![Figure 1](../images/4a3cb6ce77e5c33483e082c5486a35ee29e772502c561d05f2e7e59f118c2701.jpg)
*Figure 1: A video with user-annotated chapters in VidChapters-7M: the video is temporally segmented into chapters, which are annotated with a chapter title in free-form natural language.*

---

## 💡 理解

### 核心要点
- [x] **数据集**: VidChapters-7M = 817K 视频 + 7M 章节，全部自动爬取，无需人工标注
- [x] **三个任务定义**:
  1. Video Chapter Generation: 时间分割 + 标题生成 (最难)
  2. Chapter Title Generation (给边界): 给定时间边界，生成标题
  3. Video Chapter Grounding: 给定标题，定位时间段
- [x] **迁移学习**: 预训练后在 YouCook2/ViTT 上大幅提升 SOTA
- [x] **Scaling**: 数据量越大，下游性能越好

### 🖼️ Figure 1 解读
- 展示一个烹饪视频的章节标注示例
- 左边: 视频帧序列，按时间排列
- 右边: 用户标注的章节列表
  - 每个章节: 开始时间戳 + 自由文本标题
  - 章节连续、无缝覆盖整个视频
- 说明章节是用户主动标注的，语义质量高

### 这篇论文要解决什么问题？
1. **用户痛点**: 长视频难以导航，手动浏览耗时
2. **数据瓶颈**: 之前没有大规模公开的视频章节数据集
3. **解决方案**: 利用 YouTube 用户自己标注的章节，自动爬取构建数据集

### 主要贡献
1. **VidChapters-7M 数据集**: 首个大规模视频章节数据集 (817K videos, 7M chapters)
2. **三个任务定义**: Chapter Generation / GT-Boundary Title / Chapter Grounding
3. **Benchmark**: 评估 baseline 和 SOTA 模型，建立评测标准
4. **迁移学习**: 证明预训练价值，在 dense captioning 任务上大幅提升

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 视频数 | 817K |
| 章节数 | 7M |
| 标注方式 | 自动爬取用户标注 |
| 下游任务 | YouCook2, ViTT |

### 我的疑问
- [x] 为什么用户会主动标注章节？→ YouTube 2020 年推出章节功能，创作者为了提升观看体验会主动添加
- [ ] 爬取的章节质量如何保证？→ Section 3.3 会详细分析
- [ ] 三个任务难度排序？→ Task 1 > Task 3 > Task 2
