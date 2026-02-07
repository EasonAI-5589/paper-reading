# 1. Introduction

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

As online media consumption grows, the volume of video content available is increasing rapidly. While searching for specific videos is already a challenging problem, searching within a long video is an even less explored task. Manual navigation can often be time consuming, particularly for long videos. A compelling solution for organizing content online is to segment long videos into chapters (see Figure 1). Chapters are contiguous, non-overlapping segments, completely partitioning a video. Each chapter is also labeled with a short description of the chapter content, enabling users to quickly navigate to areas of interest and easily replay different parts of a video. Chapters also give structure to a video, which is useful for long videos that contain inherently listed content, such as listicles, instructional videos, music compilations and so on.

![Figure 2](../images/52e5446400ff7e972a74089879f9386da10915fc4ac1e3e87f57c0632c6c76ee.jpg)
*Figure 2: Illustration of the three tasks defined for VidChapters-7M.*

Given the plethora of content already online, our goal is to explore automatic solutions related to video chaptering - generating chapters automatically, and grounding chapter titles temporally in long videos. While the benefits of automatically chaptering videos are obvious, data for this task is scarce. Video captioning datasets (such as WebVid-10M and VideoCC) consist of short videos (10s in length), and hence are unsuitable. Web datasets consisting of longer videos (HowTo100M, YT-Temporal-1B) come with aligned speech transcripts (ASR), which are only weakly related to visual content, and if used as chapter titles would tend to over-segment videos. Moment retrieval or dense video captioning datasets are perhaps the most useful, but do not focus on creating explicit structure, and instead describe low-level actions comprehensively. Such datasets are also manually annotated, and hence not scalable and small in size (see Table 1).

To remedy this, we curate VidChapters-7M, a large-scale dataset of user-annotated video chapters automatically scraped from the Web. Our dataset consists of 7M chapters for over 817K videos. Compared to existing datasets, videos in VidChapters-7M are long (23 minutes on average) and contain rich chapter annotations consisting of a starting timestamp and a title per chapter. Our dataset is also diverse, with 12 different video categories having at least 20K videos each, which itself is the size of existing dense video captioning datasets. 

On top of this dataset we also define 3 video tasks (see Figure 2): 
(i) **video chapter generation** which requires temporally segmenting the video and generating a chapter title for each segment; 
(ii) **video chapter generation given ground-truth boundaries**, which requires generating a chapter title given an annotated video segment; and 
(iii) **video chapter grounding**, which requires temporally localizing a chapter given the chapter title. 

All three tasks involve parsing and understanding long videos, and multi-modal reasoning (video and text), and hence are valuable steps towards story understanding.

For all three tasks, we implement simple baselines as well as recent, state-of-the-art video-text methods. We find that the tasks are far from being solved, demonstrating the value of this problem. Interestingly, we also show that our video chapter generation models trained on VidChapters-7M transfer well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Moreover, we show that pretraining using both speech transcripts and chapter annotations significantly outperforms the widely used pretraining method based only on speech transcripts. This demonstrates the additional value of our dataset as a generic video-language pretraining set. Interestingly, we also find that the transfer performance scales with the size of the chapter dataset.

**In summary, our contributions are:**

(i) We present VidChapters-7M, a large-scale dataset of user-annotated video chapters obtained from the Web consisting of 817K videos and 7M chapters;

(ii) Based on this dataset, we evaluate a range of simple baselines and state-of-the-art video-language models on the tasks of video chapter generation with and without ground-truth boundaries, and video chapter grounding;

(iii) We show that video chapter generation models trained on VidChapters-7M transfer well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks, outperforming prior pretraining methods based on narrated videos, and showing promising scaling behavior.

Our dataset, code and models are publicly available on our website.

**Table 1: Comparison of VidChapters-7M with existing datasets.**

| Dataset | # Videos | Duration (min) | # Descriptions | Annotations |
|---------|----------|----------------|----------------|-------------|
| HowTo100M | 1M | 7 | 136M | Speech transcripts |
| YT-Temporal-1B | 19M | 6 | ~900M | Speech transcripts |
| HD-VILA-100M | 3M | 7 | 103M | Speech transcripts |
| ActivityNet Captions | 20K | 3 | 100K | Dense Captions |
| YouCook2 | 2K | 6 | 15K | Dense Captions |
| ViTT | 8K | 4 | 56K | Dense Captions |
| Ego4D | 10K | 23 | 4M | Dense Captions |
| **VidChapters-7M (Ours)** | **817K** | **23** | **7M** | **Speech + User Chapters** |

---

## 💡 理解

### 核心要点
- [x] **问题动机**: 视频内容爆炸增长，但"视频内搜索"是未充分探索的任务
- [x] **Chapter 定义**: 
  - 连续 (contiguous)
  - 不重叠 (non-overlapping)
  - 完全覆盖整个视频 (completely partitioning)
  - 每段有简短描述性标题
- [x] **现有数据集不适用**: 要么太短、要么只有 ASR、要么是低级动作描述
- [x] **解决方案**: 爬取 YouTube 用户自己标注的章节

### 🖼️ Figure 2 解读 (三个任务)

```
┌─────────────────────────────────────────────────────────────┐
│  Task 1: Video Chapter Generation (最难，完整任务)          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  输入: 完整视频 (帧 + 音频)                          │   │
│  │  输出: [(t1, "Intro"), (t2, "Setup"), (t3, "Demo")]  │   │
│  │  需要: 同时做时间分割 + 标题生成                     │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Task 2: Chapter Title Generation (简化版，给边界)          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  输入: 视频 + GT 时间边界 [0:00-2:30, 2:30-5:00...]  │   │
│  │  输出: 每段的标题 ["Intro", "Setup", ...]            │   │
│  │  只需: 理解内容 + 生成标题                           │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Task 3: Video Chapter Grounding (反向任务)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  输入: 视频 + 章节标题 "Setup"                       │   │
│  │  输出: 对应时间段 [2:30-5:00]                        │   │
│  │  类似: Temporal grounding / Moment retrieval        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Table 1 解读：现有数据集为什么不行？

| 数据集类型 | 代表 | 视频数 | 时长 | 问题 |
|-----------|------|--------|------|------|
| **长视频 + ASR** | HowTo100M | 1M | 7min | ASR 与视觉弱相关，会过度分割 |
| **长视频 + ASR** | YT-Temporal-1B | 19M | 6min | 同上 |
| **短视频 + Caption** | WebVid-10M | - | ~10s | 太短，不适合章节任务 |
| **Dense Captioning** | ActivityNet | 20K | 3min | 描述低级动作，规模小，手动标注 |
| **Dense Captioning** | YouCook2 | 2K | 6min | 规模太小 |
| **VidChapters-7M** | Ours | **817K** | **23min** | ✅ 大规模 + 长视频 + 用户语义标注 |

**关键洞察**: VidChapters-7M 是唯一同时满足"大规模"+"长视频"+"语义标注"的数据集

### VidChapters-7M 的优势总结
- ✅ **规模大**: 817K videos (vs ActivityNet 20K)
- ✅ **视频长**: 23 min avg (vs WebVid 10s)
- ✅ **标注质量高**: 用户主动标注的语义章节 (vs ASR)
- ✅ **无需人工标注**: 自动爬取 (scalable)
- ✅ **多样性好**: 12 个视频类别，每类 >20K

### Dense Captioning vs Chapter Generation

| 特性 | Dense Captioning | Chapter Generation |
|------|------------------|-------------------|
| 分段关系 | 事件独立，可重叠 | **连续、无间隙** |
| 覆盖范围 | 仅描述"有趣"事件 | **覆盖整个视频** |
| 描述类型 | 一句话完整描述 | **简短标题** (5.4 词) |
| 粒度 | 细粒度动作 | **粗粒度语义主题** |
| 目的 | 描述发生了什么 | **帮助用户导航** |

### 论文三大贡献

1. **数据集贡献**: VidChapters-7M (817K videos, 7M chapters)
2. **任务贡献**: 定义 3 个任务 + 评测协议
3. **方法贡献**: 
   - Baseline 评测
   - 迁移学习验证 (YouCook2/ViTT 提升)
   - Scaling 规律 (数据越多越好)

### 我的疑问
- [x] 为什么 ASR 不能直接当章节？→ ASR 太细碎 (269.8 句/视频 vs 8.3 章节)，且语义不够精炼
- [x] 三个任务的应用场景？
  - Task 1: 自动给视频加章节 (最实用)
  - Task 2: 辅助创作者写标题
  - Task 3: 视频内搜索/跳转
- [x] 为什么选择这三个任务？→ 覆盖了章节相关的主要能力：分割、生成、定位
