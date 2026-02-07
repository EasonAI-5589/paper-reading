# 2. Related Work

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

**Large-scale vision-language datasets.** The development of powerful multi-modal models [3, 15, 23, 35, 37, 38, 46, 48–50, 54, 61, 62, 72, 85, 87, 90, 94, 99, 105, 115, 116, 129] has been made possible by pretraining on large-scale image-caption datasets scraped from the Web such as SBU [68], Conceptual Captions [82], Conceptual-12M [12], LAIT [71], Wikipedia-ImageText [86], RedCaps [18] and LAION-5B [78]. Similarly, many strong video-language models [2, 27, 30, 41, 45, 47, 52, 53, 58, 65, 80, 81, 88, 89, 91, 97, 100, 107, 110–112, 126] have been pretrained on Web-scraped video-text datasets. These datasets are largely composed of short videos paired with captions, e.g. WebVid-10M [5] and VideoCC [66], or narrated videos with speech transcripts aligned over time (ASR), e.g. HowTo100M [64], YT-Temporal-1B [117, 118] and HD-VILA-100M [108]. Our proposed VidChapters-7M dataset is also downloaded from the Web, via a scalable pipeline without the need for expensive manual annotation. Unlike these datasets, VidChapters-7M consists of long videos with user-annotated chapters aligned over time (see Table 1), which significantly differ from ASR (see Section 3.3). Furthermore, most videos in VidChapters-7M also contain ASR. Finally, VidChapters-7M is also related to the recent ChapterGen dataset [10], which also consists of user-annotated chapters. However, ChapterGen is several orders of magnitude smaller than VidChapters-7M (10K vs 817K videos) and is not open-sourced at the time of writing.

> 💡 **视觉-语言数据集现状**:
> ```
> 现有数据集分类:
> ├── 图像-文本
> │   ├── SBU, CC, LAION-5B...
> │   └── 支撑了 CLIP, BLIP 等模型
> │
> └── 视频-文本
>     ├── 短视频+Caption: WebVid-10M, VideoCC
>     │   → 问题: 10秒太短
>     │
>     ├── 长视频+ASR: HowTo100M, YT-Temporal-1B
>     │   → 问题: ASR 语义弱，过度分割
>     │
>     └── 长视频+Chapter: ChapterGen (10K), VidChapters-7M ⭐
>         → 优势: 用户语义标注，高质量
> ```
> 
> **ChapterGen 对比**: 10K vs 817K = **80 倍差距**，且未开源

**Video tasks.** The video chapter generation task requires temporally segmenting the video into chapters, hence is related to video shot detection [76, 77, 84], movie scene segmentation [14, 75], temporal action localization [13, 16, 59, 83, 120, 121] and temporal action segmentation [8, 21, 26, 43, 55, 104]. However, unlike these tasks, video chapter generation also requires generating a free-form natural language chapter title for each segment. Hence this task is also related to video captioning [25, 57, 63, 69, 98, 102, 125], video title generation [4, 119, 123], generic event boundary captioning [103] and dense video captioning [42, 101, 128]. Most related to video chapter generation, the dense video captioning task requires temporally localizing and captioning all events in an untrimmed video. In contrast, video chapter generation requires temporally segmenting the video (i.e. the start of the chapter i+1 is the end of chapter i, and the chapters cover the full video), and involves generating a chapter title that is substantially shorter than a video caption. We study in more detail the transfer learning between these two tasks in Section 4.4. Finally, the video chapter grounding task is related to temporal language grounding [33, 34, 44, 45, 67, 113, 122, 124]. However, we here focus on localizing a chapter starting point and not a start-end window. Furthermore, most temporal language grounding methods represent the video only with visual inputs, while we also exhibit the benefits of using speech inputs for localizing chapters in videos (see Section 4.3).

> 💡 **相关任务谱系**:
> ```
> Video Chapter Generation 的任务定位
> │
> ├── 继承自 "时间分割"
> │   ├── Shot Detection (镜头边界)
> │   ├── Scene Segmentation (场景分割)
> │   └── Action Segmentation (动作分割)
> │   → 但这些不需要生成文字
> │
> ├── 继承自 "文字生成"
> │   ├── Video Captioning (整视频描述)
> │   ├── Video Title Generation (标题)
> │   └── Dense Video Captioning (稠密描述)
> │   → 但这些不需要精确分割
> │
> └── Video Chapter Generation = 分割 + 生成 ⭐
> ```
> 
> **vs Dense Captioning 的关键区别**:
> | 维度 | Dense Captioning | Chapter Generation |
> |------|-----------------|-------------------|
> | 分段 | 独立事件，可重叠 | **连续无间隙** |
> | 覆盖 | 部分"有趣"事件 | **整个视频** |
> | 描述 | ~15 词句子 | **~5 词标题** |
> | 目的 | 描述内容 | **帮助导航** |

---

## 💡 Section 2 总结

### 本文的定位

VidChapters-7M 填补了一个重要空白：**大规模 + 长视频 + 高质量语义标注**。

### 与最相关工作的对比

| 工作 | 规模 | 开源 | 标注类型 |
|------|------|------|----------|
| ChapterGen | 10K | ❌ | Chapter |
| **VidChapters-7M** | **817K** | ✅ | **Chapter + ASR** |

### Chapter Generation 的任务创新

1. **分割约束更强**: 连续、无间隙、完全覆盖
2. **描述更简洁**: 5.4 词标题 vs 15 词 caption
3. **应用更直接**: 帮助用户导航，而非描述内容
