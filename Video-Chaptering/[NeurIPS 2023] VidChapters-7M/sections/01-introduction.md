# 1. Introduction

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

As online media consumption grows, the volume of video content available is increasing rapidly. While searching for specific videos is already a challenging problem, searching within a long video is an even less explored task. Manual navigation can often be time consuming, particularly for long videos. A compelling solution for organizing content online is to segment long videos into chapters (see Figure 1). Chapters are contiguous, non-overlapping segments, completely partitioning a video. Each chapter is also labeled with a short description of the chapter content, enabling users to quickly navigate to areas of interest and easily replay different parts of a video. Chapters also give structure to a video, which is useful for long videos that contain inherently listed content, such as listicles, instructional videos, music compilations and so on.

> 💡 **问题动机**: 
> - 视频内容爆炸，但"视频内搜索"是未充分探索的任务
> - 手动浏览长视频很痛苦
> - **Chapter 定义**: 连续、不重叠、完全覆盖、有简短标题
> - **应用场景**: 教程、合集、列表类视频

![Figure 2](../images/52e5446400ff7e972a74089879f9386da10915fc4ac1e3e87f57c0632c6c76ee.jpg)
*Figure 2: Illustration of the three tasks defined for VidChapters-7M.*

> 💡 **Figure 2 批读 (三个任务)**:
> ```
> Task 1: Video Chapter Generation (完整任务)
> ├── 输入: 视频
> ├── 输出: [(时间1, 标题1), (时间2, 标题2), ...]
> └── 难度: ⭐⭐⭐ (需要分割+生成)
> 
> Task 2: Chapter Title Given Boundaries (简化)
> ├── 输入: 视频 + GT 时间边界
> ├── 输出: 每段的标题
> └── 难度: ⭐⭐ (只需生成)
> 
> Task 3: Chapter Grounding (反向)
> ├── 输入: 视频 + 标题
> ├── 输出: 对应时间段
> └── 难度: ⭐⭐ (类似 moment retrieval)
> ```

Given the plethora of content already online, our goal is to explore automatic solutions related to video chaptering - generating chapters automatically, and grounding chapter titles temporally in long videos. While the benefits of automatically chaptering videos are obvious, data for this task is scarce. Video captioning datasets (such as WebVid-10M and VideoCC) consist of short videos (10s in length), and hence are unsuitable. Web datasets consisting of longer videos (HowTo100M, YT-Temporal-1B) come with aligned speech transcripts (ASR), which are only weakly related to visual content, and if used as chapter titles would tend to over-segment videos. Moment retrieval or dense video captioning datasets are perhaps the most useful, but do not focus on creating explicit structure, and instead describe low-level actions comprehensively. Such datasets are also manually annotated, and hence not scalable and small in size (see Table 1).

> 💡 **现有数据集为什么不行？**
> | 类型 | 代表 | 问题 |
> |------|------|------|
> | 短视频 Caption | WebVid-10M | 10秒太短 |
> | 长视频 + ASR | HowTo100M | ASR 与视觉弱相关，会过度分割 |
> | Dense Captioning | ActivityNet | 描述低级动作，规模小，需人工标注 |
> 
> **核心矛盾**: 要么太短、要么太细碎、要么太小

To remedy this, we curate VidChapters-7M, a large-scale dataset of user-annotated video chapters automatically scraped from the Web. Our dataset consists of 7M chapters for over 817K videos. Compared to existing datasets, videos in VidChapters-7M are long (23 minutes on average) and contain rich chapter annotations consisting of a starting timestamp and a title per chapter. Our dataset is also diverse, with 12 different video categories having at least 20K videos each, which itself is the size of existing dense video captioning datasets. 

> 💡 **VidChapters-7M 优势**:
> - ✅ 规模大: 817K videos (vs ActivityNet 20K)
> - ✅ 视频长: 23 min avg (vs WebVid 10s)
> - ✅ 标注质量高: 用户主动标注
> - ✅ 无需人工: 自动爬取
> - ✅ 多样性: 12 类，每类 >20K

On top of this dataset we also define 3 video tasks (see Figure 2): 
(i) **video chapter generation** which requires temporally segmenting the video and generating a chapter title for each segment; 
(ii) **video chapter generation given ground-truth boundaries**, which requires generating a chapter title given an annotated video segment; and 
(iii) **video chapter grounding**, which requires temporally localizing a chapter given the chapter title. 

All three tasks involve parsing and understanding long videos, and multi-modal reasoning (video and text), and hence are valuable steps towards story understanding.

> 💡 **任务设计思路**: 把"章节生成"分解成三个子任务：
> - Task 1 = 分割 + 生成 (端到端)
> - Task 2 = 只生成 (ablation: 分割能力)
> - Task 3 = 反向定位 (验证对齐能力)
> 
> 这样可以 disentangle 不同能力，更好地分析模型。

For all three tasks, we implement simple baselines as well as recent, state-of-the-art video-text methods. We find that the tasks are far from being solved, demonstrating the value of this problem. Interestingly, we also show that our video chapter generation models trained on VidChapters-7M transfer well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Moreover, we show that pretraining using both speech transcripts and chapter annotations significantly outperforms the widely used pretraining method based only on speech transcripts. This demonstrates the additional value of our dataset as a generic video-language pretraining set. Interestingly, we also find that the transfer performance scales with the size of the chapter dataset.

> 💡 **关键实验结论**:
> 1. 任务远未解决 → 有研究价值
> 2. 预训练迁移效果好 → YouCook2/ViTT SOTA
> 3. **ASR + Chapter > ASR only** → 章节标注有额外价值
> 4. **Scaling 有效** → 数据越多越好

**In summary, our contributions are:**

(i) We present VidChapters-7M, a large-scale dataset of user-annotated video chapters obtained from the Web consisting of 817K videos and 7M chapters;

(ii) Based on this dataset, we evaluate a range of simple baselines and state-of-the-art video-language models on the tasks of video chapter generation with and without ground-truth boundaries, and video chapter grounding;

(iii) We show that video chapter generation models trained on VidChapters-7M transfer well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks, outperforming prior pretraining methods based on narrated videos, and showing promising scaling behavior.

Our dataset, code and models are publicly available on our website.

> 💡 **三大贡献**:
> 1. **数据**: VidChapters-7M (817K videos, 7M chapters)
> 2. **评测**: 三个任务 + baseline + SOTA 模型
> 3. **预训练**: 验证迁移价值 + scaling 规律

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

> 💡 **Table 1 批读**:
> ```
> 数据集规模对比:
> ├── ASR 类 (大但弱监督)
> │   ├── HowTo100M: 1M videos, 136M ASR
> │   └── YT-Temporal-1B: 19M videos, 900M ASR
> │   问题: ASR 太细碎，语义弱
> │
> ├── Dense Captioning 类 (小但强监督)
> │   ├── ActivityNet: 20K videos
> │   ├── YouCook2: 2K videos
> │   └── ViTT: 8K videos
> │   问题: 规模太小，需人工
> │
> └── VidChapters-7M ⭐
>     ├── 817K videos (介于两者之间)
>     ├── 23 min avg (最长！)
>     └── 7M 用户章节标注 (高质量语义)
> ```
> 
> **独特优势**: VidChapters-7M 是唯一同时满足"大规模 + 长视频 + 语义标注"的数据集

---

## 💡 Section 1 总结

### 核心问题
长视频导航是刚需，但缺乏大规模、高质量的章节标注数据集。

### 创新点
利用 YouTube 用户自己标注的章节，自动爬取构建数据集，无需人工标注。

### 技术贡献
1. **VidChapters-7M 数据集**: 817K videos, 7M chapters
2. **三个任务定义**: Generation / GT-Boundary Title / Grounding
3. **预训练价值验证**: YouCook2/ViTT SOTA + Scaling 规律
