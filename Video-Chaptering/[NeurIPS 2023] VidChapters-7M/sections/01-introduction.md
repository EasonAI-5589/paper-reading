# 1. Introduction

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

As online media consumption grows, the volume of video content available is increasing rapidly. While searching for specific videos is already a challenging problem, searching within a long video is an even less explored task. Manual navigation can often be time consuming, particularly for long videos. A compelling solution for organizing content online is to segment long videos into chapters (see Figure 1). Chapters are contiguous, non-overlapping segments, completely partitioning a video. Each chapter is also labeled with a short description of the chapter content, enabling users to quickly navigate to areas of interest and easily replay different parts of a video.

![Figure 2](../images/52e5446400ff7e972a74089879f9386da10915fc4ac1e3e87f57c0632c6c76ee.jpg)
*Figure 2: Illustration of the three tasks defined for VidChapters-7M*

Given the plethora of content already online, our goal is to explore automatic solutions related to video chaptering - generating chapters automatically, and grounding chapter titles temporally in long videos. While the benefits of automatically chaptering videos are obvious, data for this task is scarce:

- Video captioning datasets (WebVid-10M, VideoCC) consist of short videos (10s in length) → unsuitable
- Web datasets with longer videos (HowTo100M, YT-Temporal-1B) have ASR which is weakly related to visual content → would over-segment videos
- Moment retrieval or dense video captioning datasets describe low-level actions comprehensively but don't focus on creating explicit structure → also manually annotated, not scalable

To remedy this, we curate **VidChapters-7M**, a large-scale dataset of user-annotated video chapters automatically scraped from the Web:
- 7M chapters for over 817K videos
- Long videos (23 minutes on average)
- Rich chapter annotations (starting timestamp + title)
- Diverse: 12 different video categories with at least 20K videos each

On top of this dataset we define 3 video tasks (Figure 2):
1. **Video chapter generation**: temporally segmenting + generating chapter title
2. **Video chapter generation given GT boundaries**: generating title for annotated segment
3. **Video chapter grounding**: localizing chapter given title

### Contributions

(i) We present VidChapters-7M, a large-scale dataset of user-annotated video chapters (817K videos, 7M chapters)

(ii) We evaluate baselines and SOTA video-language models on video chapter generation (with/without GT boundaries) and video chapter grounding

(iii) We show video chapter generation models transfer well to dense video captioning (YouCook2, ViTT), outperforming prior methods and showing scaling behavior

---

## 💡 理解

### 核心要点
- [x] **问题动机**: 视频内容爆炸式增长，但视频内搜索是未充分探索的任务
- [x] **Chapter 定义**: 连续、不重叠、完全覆盖整个视频的片段，每段有简短标题
- [x] **现有数据集不适用**: 要么太短、要么是 ASR、要么是低级动作描述
- [x] **解决方案**: 爬取用户自己标注的章节

### 🖼️ Figure 2 解读 (三个任务)

```
Task 1: Video Chapter Generation (最难)
┌─────────────────────────────────────┐
│  输入: 完整视频                      │
│  输出: 时间边界 + 章节标题            │
│  [00:00-02:30] Intro                │
│  [02:30-05:00] Setup                │
│  [05:00-08:00] Main Content         │
└─────────────────────────────────────┘

Task 2: Chapter Title Generation (给边界)
┌─────────────────────────────────────┐
│  输入: 视频 + GT 时间边界            │
│  输出: 每段的章节标题                 │
│  [00:00-02:30] → "???"  → "Intro"   │
└─────────────────────────────────────┘

Task 3: Video Chapter Grounding (反向)
┌─────────────────────────────────────┐
│  输入: 视频 + 章节标题               │
│  输出: 对应的时间段                   │
│  "Setup" → [02:30-05:00]            │
└─────────────────────────────────────┘
```

### 现有数据集为什么不行？

| 数据集类型 | 代表 | 问题 |
|-----------|------|------|
| 短视频 caption | WebVid-10M, VideoCC | 只有 10s，太短 |
| 长视频 + ASR | HowTo100M, YT-Temporal-1B | ASR 与视觉弱相关，会过度分割 |
| Dense captioning | ActivityNet Captions | 描述低级动作，非结构化，规模小 |
| Moment retrieval | Charades-STA | 描述低级动作，手动标注 |

### VidChapters-7M 的优势
- ✅ 大规模 (817K videos vs ActivityNet 20K)
- ✅ 长视频 (23 min avg)
- ✅ 用户标注 (高质量语义)
- ✅ 无需人工标注 (自动爬取)
- ✅ 多样性 (12 个视频类别)

### Chapter vs Dense Captioning 的区别

| 特性 | Dense Captioning | Chapter Generation |
|------|------------------|-------------------|
| 分段方式 | 事件可重叠 | **连续、无缝** |
| 覆盖范围 | 部分事件 | **整个视频** |
| 描述长度 | 一句话描述 | **简短标题** (5.4 词) |
| 粒度 | 低级动作 | **语义主题** |

### 我的疑问
- [x] 三个任务难度排序？→ Task 1 > Task 3 > Task 2 (Task 1 需要同时做分割和生成)
- [x] 为什么 ASR 不能直接当章节？→ ASR 太细碎 (269.8 句/视频 vs 8.3 章节/视频)，且与视觉内容弱相关
