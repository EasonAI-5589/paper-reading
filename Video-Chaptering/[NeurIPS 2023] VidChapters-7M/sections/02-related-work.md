# 2. Related Work

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

### Large-scale vision-language datasets

The development of powerful multi-modal models has been made possible by pretraining on large-scale image-caption datasets scraped from the Web such as SBU, Conceptual Captions, Conceptual-12M, LAIT, Wikipedia-ImageText, RedCaps and LAION-5B. 

Similarly, many strong video-language models have been pretrained on Web-scraped video-text datasets. These datasets are largely composed of:
- **Short videos paired with captions**: WebVid-10M, VideoCC
- **Narrated videos with ASR**: HowTo100M, YT-Temporal-1B, HD-VILA-100M

Our proposed VidChapters-7M dataset is also downloaded from the Web, via a scalable pipeline without expensive manual annotation. Unlike these datasets, VidChapters-7M consists of **long videos with user-annotated chapters aligned over time**, which significantly differ from ASR.

VidChapters-7M is also related to the recent ChapterGen dataset, which also consists of user-annotated chapters. However, ChapterGen is several orders of magnitude smaller (10K vs 817K videos) and is not open-sourced.

### Video tasks

The video chapter generation task requires temporally segmenting the video into chapters, hence is related to:
- Video shot detection
- Movie scene segmentation
- Temporal action localization
- Temporal action segmentation

However, unlike these tasks, video chapter generation also requires **generating a free-form natural language chapter title** for each segment. Hence this task is also related to:
- Video captioning
- Video title generation
- Generic event boundary captioning
- **Dense video captioning** (most related)

**Key difference from dense video captioning:**
- Dense captioning: temporally localizes and captions all events
- Video chaptering: temporally **segments** the video (chapters are contiguous, covering full video) and generates **short titles** (not long captions)

---

## 💡 理解

### 核心要点
- [ ] 

### 相关任务对比

| 任务 | 输出 | 与 Chapter Generation 的区别 |
|------|------|------------------------------|
| Shot Detection | 镜头边界 | |
| Scene Segmentation | 场景边界 | |
| Action Segmentation | 动作标签 | |
| Dense Captioning | 时间段+描述 | |
| **Chapter Generation** | 时间边界+标题 | |

### 现有数据集对比

| 数据集 | 特点 | 局限 |
|--------|------|------|
| WebVid-10M | 短视频+caption | |
| HowTo100M | 长视频+ASR | |
| ChapterGen | 用户章节 | |
| **VidChapters-7M** | | |

### 我的疑问
- [ ] 
