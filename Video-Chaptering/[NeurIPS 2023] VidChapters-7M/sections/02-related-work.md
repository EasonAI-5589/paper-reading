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
- [x] 视频-语言数据集分两类: 短视频+caption vs 长视频+ASR
- [x] VidChapters-7M 独特之处: 长视频 + 用户章节标注 (非 ASR)
- [x] ChapterGen 是唯一类似数据集，但规模小 80 倍且未开源
- [x] Video Chaptering 结合了时序分割和文本生成两个方向

### 相关任务谱系图

```
                    Video Understanding
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    时序分割任务      描述生成任务      定位任务
          │               │               │
    ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
    │           │   │           │   │           │
  Shot      Action  Video    Dense  Moment   Temporal
Detection  Segment  Caption  Caption Retrieval Grounding
    │           │       │       │       │         │
    └─────┬─────┘       └───┬───┘       └────┬────┘
          │                 │                │
          └────────────────┬┴────────────────┘
                           │
                  Video Chapter Generation
                  (时序分割 + 标题生成)
```

### 相关任务对比

| 任务 | 输入 | 输出 | 与 Chapter 的区别 |
|------|------|------|------------------|
| **Shot Detection** | 视频 | 镜头边界 | 只看视觉变化，无语义 |
| **Scene Segmentation** | 电影 | 场景边界 | 针对电影，粒度不同 |
| **Action Segmentation** | 视频 | 动作标签序列 | 预定义类别，非自由文本 |
| **Video Captioning** | 短视频 | 描述句 | 单句，无时序分割 |
| **Dense Captioning** | 视频 | 多个(时间段, 描述) | 事件可重叠，非覆盖全视频 |
| **Temporal Grounding** | 视频+查询 | 时间段 | 给定文本找时间，反向任务 |
| **Chapter Generation** | 长视频 | (边界, 标题)序列 | **连续分割 + 简短标题** |

### 现有视频-语言数据集

| 数据集 | 类型 | 规模 | 视频长度 | 标注 |
|--------|------|------|----------|------|
| WebVid-10M | 短视频 | 10M | ~10s | Caption |
| VideoCC | 短视频 | 10M | ~10s | Caption |
| HowTo100M | 长视频 | 1M | ~7min | ASR |
| YT-Temporal-1B | 长视频 | 19M | ~6min | ASR |
| ActivityNet Captions | 长视频 | 20K | ~3min | Dense Caption |
| ChapterGen | 长视频 | 10K | ? | 用户章节 (未开源) |
| **VidChapters-7M** | 长视频 | **817K** | **23min** | **用户章节** |

### Dense Captioning vs Chapter Generation

| 特性 | Dense Captioning | Chapter Generation |
|------|------------------|-------------------|
| 分段关系 | 事件独立，可重叠 | **连续、无间隙** |
| 覆盖范围 | 仅描述"有趣"事件 | **覆盖整个视频** |
| 描述类型 | 一句话完整描述 | **简短标题** |
| 粒度 | 细粒度动作 | **粗粒度主题** |
| 目的 | 描述发生了什么 | **帮助导航** |

### 我的疑问
- [x] ChapterGen 为什么不开源？→ 可能是商业考虑或数据版权问题
- [x] 为什么 Dense Captioning 不能直接用于章节生成？→ 因为 Dense Caption 描述的是独立事件，不保证连续覆盖整个视频
