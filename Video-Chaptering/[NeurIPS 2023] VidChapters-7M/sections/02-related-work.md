[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
Related Work 分两部分：(1) 大规模视觉-语言数据集，定位 VidChapters-7M 的独特性；(2) 相关视频任务，区分 chapter generation 与 dense captioning、temporal grounding 等任务。

---

**Large-scale vision-language datasets.** The development of powerful multi-modal models [3, 15, 23, 35, 37, 38, 46, 48–50, 54, 61, 62, 72, 85, 87, 90, 94, 99, 105, 115, 116, 129] has been made possible by pretraining on large-scale image-caption datasets scraped from the Web such as SBU [68], Conceptual Captions [82], Conceptual-12M [12], LAIT [71], Wikipedia-ImageText [86], RedCaps [18] and LAION-5B [78]. Similarly, many strong video-language models [2, 27, 30, 41, 45, 47, 52, 53, 58, 65, 80, 81, 88, 89, 91, 97, 100, 107, 110–112, 126] have been pretrained on Web-scraped video-text datasets. These datasets are largely composed of short videos paired with captions, e.g. WebVid-10M [5] and VideoCC [66], or narrated videos with speech transcripts aligned over time (ASR), e.g. HowTo100M [64], YT-Temporal-1B [117, 118] and HD-VILA-100M [108]. Our proposed VidChapters-7M dataset is also downloaded from the Web, via a scalable pipeline without the need for expensive manual annotation. Unlike these datasets, VidChapters-7M consists of long videos with user-annotated chapters aligned over time (see Table 1), which significantly differ from ASR (see Section 3.3). Furthermore, most videos in VidChapters-7M also contain ASR. Finally, VidChapters-7M is also related to the recent ChapterGen dataset [10], which also consists of user-annotated chapters. However, ChapterGen is several orders of magnitude smaller than VidChapters-7M (10K vs 817K videos) and is not open-sourced at the time of writing.

> 💡 **VidChapters-7M 的定位**:
> - 和图文数据集（SBU, LAION-5B 等）不同：是视频级别的
> - 和短视频数据集（WebVid-10M）不同：视频更长（23min vs 10s）
> - 和 ASR 数据集（HowTo100M）不同：有结构化的章节标注，不仅是语音转录
> - 和 ChapterGen 不同：大 80 倍（817K vs 10K），且开源
> - **独特优势**: 同时有 ASR + 用户章节标注

**Video tasks.** The video chapter generation task requires temporally segmenting the video into chapters, hence is related to video shot detection [76, 77, 84], movie scene segmentation [14, 75], temporal action localization [13, 16, 59, 83, 120, 121] and temporal action segmentation [8, 21, 26, 43, 55, 104]. However, unlike these tasks, video chapter generation also requires generating a free-form natural language chapter title for each segment. Hence this task is also related to video captioning [25, 57, 63, 69, 98, 102, 125], video title generation [4, 119, 123], generic event boundary captioning [103] and dense video captioning [42, 101, 128]. Most related to video chapter generation, the dense video captioning task requires temporally localizing and captioning all events in an untrimmed video. In contrast, video chapter generation requires temporally segmenting the video (i.e. the start of the chapter $i + 1$ is the end of chapter $i$, and the chapters cover the full video), and involves generating a chapter title that is substantially shorter than a video caption. We study in more detail the transfer learning between these two tasks in Section 4.4. Finally, the video chapter grounding task is related to temporal language grounding [33, 34, 44, 45, 67, 113, 122, 124]. However, we here focus on localizing a chapter starting point and not a start-end window. Furthermore, most temporal language grounding methods represent the video only with visual inputs, while we also exhibit the benefits of using speech inputs for localizing chapters in videos (see Section 4.3).

> 💡 **Chapter Generation vs Dense Video Captioning 的关键区别**:
> - **分割方式**: Chapter 是连续不重叠的完整分割；Dense captioning 的事件可以重叠
> - **标题长度**: Chapter title 很短（平均 5.4 词）；Dense caption 是完整句子
> - **覆盖范围**: Chapter 覆盖整个视频；Dense captioning 可能只覆盖部分
>
> **Chapter Grounding vs Temporal Grounding 的区别**:
> - Chapter grounding 关注起始点，不是起止窗口
> - 本文还利用了语音信息，而多数 temporal grounding 方法只用视觉

---

## 🔖 Section 总结

### 核心洞察
1. VidChapters-7M 填补了「大规模、长视频、结构化标注」的空白
2. Video chapter generation 是 temporal segmentation + captioning 的结合，比单独的分割或描述更有挑战
3. 与 dense video captioning 最相关，但章节标注更简洁、结构化
