# 2. Related Work

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

**Large-scale vision-language datasets.** The development of powerful multi-modal models [3, 15, 23, 35, 37, 38, 46, 48–50, 54, 61, 62, 72, 85, 87, 90, 94, 99, 105, 115, 116, 129] has been made possible by pretraining on large-scale image-caption datasets scraped from the Web such as SBU [68], Conceptual Captions [82], Conceptual-12M [12], LAIT [71], Wikipedia-ImageText [86], RedCaps [18] and LAION-5B [78]. Similarly, many strong video-language models [2, 27, 30, 41, 45, 47, 52, 53, 58, 65, 80, 81, 88, 89, 91, 97, 100, 107, 110–112, 126] have been pretrained on Web-scraped video-text datasets. These datasets are largely composed of short videos paired with captions, e.g. WebVid-10M [5] and VideoCC [66], or narrated videos with speech transcripts aligned over time (ASR), e.g. HowTo100M [64], YT-Temporal-1B [117, 118] and HD-VILA-100M [108]. Our proposed VidChapters-7M dataset is also downloaded from the Web, via a scalable pipeline without the need for expensive manual annotation. Unlike these datasets, VidChapters-7M consists of long videos with user-annotated chapters aligned over time (see Table 1), which significantly differ from ASR (see Section 3.3). Furthermore, most videos in VidChapters-7M also contain ASR. Finally, VidChapters-7M is also related to the recent ChapterGen dataset [10], which also consists of user-annotated chapters. However, ChapterGen is several orders of magnitude smaller than VidChapters-7M (10K vs 817K videos) and is not open-sourced at the time of writing.

**Video tasks.** The video chapter generation task requires temporally segmenting the video into chapters, hence is related to video shot detection [76, 77, 84], movie scene segmentation [14, 75], temporal action localization [13, 16, 59, 83, 120, 121] and temporal action segmentation [8, 21, 26, 43, 55, 104]. However, unlike these tasks, video chapter generation also requires generating a free-form natural language chapter title for each segment. Hence this task is also related to video captioning [25, 57, 63, 69, 98, 102, 125], video title generation [4, 119, 123], generic event boundary captioning [103] and dense video captioning [42, 101, 128]. Most related to video chapter generation, the dense video captioning task requires temporally localizing and captioning all events in an untrimmed video. In contrast, video chapter generation requires temporally segmenting the video (i.e. the start of the chapter i+1 is the end of chapter i, and the chapters cover the full video), and involves generating a chapter title that is substantially shorter than a video caption. We study in more detail the transfer learning between these two tasks in Section 4.4. Finally, the video chapter grounding task is related to temporal language grounding [33, 34, 44, 45, 67, 113, 122, 124]. However, we here focus on localizing a chapter starting point and not a start-end window. Furthermore, most temporal language grounding methods represent the video only with visual inputs, while we also exhibit the benefits of using speech inputs for localizing chapters in videos (see Section 4.3).

---

## 💡 理解

### 核心要点
- [x] 现有视频-语言数据集的两大类：短视频+Caption / 长视频+ASR
- [x] VidChapters-7M 的独特性：长视频 + 用户章节 (非 ASR)
- [x] ChapterGen 是最相关工作，但规模差 80 倍 (10K vs 817K)
- [x] Video Chapter Generation 与 Dense Captioning 的区别

### 相关任务谱系图

```
视频内容理解任务
├── 时间分割类 (不需要文字)
│   ├── Shot Detection - 镜头边界检测
│   ├── Scene Segmentation - 场景分割 (电影)
│   ├── Action Localization - 动作定位
│   └── Action Segmentation - 动作分割
│
├── 描述生成类 (不需要分割)
│   ├── Video Captioning - 整视频描述
│   └── Video Title Generation - 标题生成
│
└── 分割+描述类 (需要两者) ⭐ 本文关注
    ├── Dense Video Captioning - 稠密描述
    ├── Event Boundary Captioning - 事件边界
    └── **Video Chapter Generation** - 章节生成 (本文)
```

### Chapter Generation vs Dense Captioning (关键区别)

| 对比维度 | Dense Video Captioning | Video Chapter Generation |
|---------|----------------------|-------------------------|
| 分段关系 | 事件独立，可重叠 | **连续、无间隙** |
| 时间覆盖 | 仅"有趣"事件 | **整个视频** |
| 描述长度 | 完整句子 (~15词) | **简短标题** (5.4词) |
| 描述目的 | 描述事件内容 | **帮助用户导航** |
| 边界定义 | start + end | **只需 start** (end = 下一个 start) |

### Chapter Generation vs Video Grounding

| 对比维度 | Temporal Grounding | Chapter Grounding |
|---------|-------------------|-------------------|
| 输入 | 视频 + 自由查询 | 视频 + 章节标题 |
| 输出 | start-end 窗口 | **只需 start** |
| 模态 | 通常仅视觉 | **视觉 + 语音** |

### VidChapters-7M 与现有数据集对比

| 数据集 | 视频数 | 描述类型 | 是否开源 | 长度 |
|--------|--------|----------|----------|------|
| WebVid-10M | 10M | Caption | ✅ | 短 (~10s) |
| HowTo100M | 1M | ASR | ✅ | 中 (7min) |
| ChapterGen | 10K | Chapter | ❌ | 长 |
| **VidChapters-7M** | **817K** | **Chapter + ASR** | ✅ | 长 (23min) |

### 我的疑问
- [x] 为什么 ChapterGen 没开源？→ 论文发表时还未开源，可能后续开源
- [x] Dense Captioning 和 Chapter 能互相迁移吗？→ Section 4.4 证明可以
- [x] 为什么要强调"用户标注"？→ 因为这意味着高质量语义，而非自动生成
