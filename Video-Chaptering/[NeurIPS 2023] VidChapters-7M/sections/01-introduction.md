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
- [ ] 

### 🖼️ Figure 2 解读
- 左边 Task 1: 
- 中间 Task 2: 
- 右边 Task 3: 

### 现有数据集为什么不行？
| 数据集类型 | 问题 |
|-----------|------|
| 短视频 caption (WebVid) | |
| 长视频 ASR (HowTo100M) | |
| Dense captioning | |

### 这篇论文的定位
- 

### 我的疑问
- [ ] 
