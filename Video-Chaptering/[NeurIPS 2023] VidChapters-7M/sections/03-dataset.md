# 3. VidChapters-7M: A Large-Scale Dataset

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

Our goal is to build a large and diverse set of videos annotated with temporally localized chapter information, consisting of chapter titles and chapter start times. 

**Key properties of chapters:**
- Contiguous, non-overlapping segments
- Completely partitioning a video

Manual annotation is time consuming and expensive → we automatically scrape chapter information from videos online.

### 3.1 Data Collection

Since early 2020, YouTube users can create chapters for uploaded videos by annotating them in the YouTube description. The YouTube API does not enable explicit search for user-chaptered videos. Hence our data collection procedure:

1. **Collect video candidates**: Start from YT-Temporal-180M dataset (92 million video IDs)
2. **Extract chapters from descriptions**: Use regex to find format `<Timestamp>: <Chapter Title>` or `<Chapter Title>: <Timestamp>`
3. **Filter**: Videos must contain at least two timestamps in ascending order
4. **Result**: 817K user-chaptered videos (0.9% of all candidates)

Note: YouTube's auto-generated chapters are excluded (they don't appear in descriptions).

### 3.2 Data Processing

**ASR extraction:**
- Use Whisper-Large-V2 model for speech transcription
- Use WhisperX for accurate word-level timestamps
- 97.3% of videos contain speech

**Visual feature extraction:**
- CLIP ViT-L/14 backbone
- Resolution: 224×224 pixels
- Frame rate: 1 FPS

### 3.3 Data Analysis

**Dataset Statistics:**
| 指标 | 数值 |
|------|------|
| 视频数量 | 817,076 |
| 章节总数 | 6,813,732 |
| 不同章节标题数 | 4,894,855 |
| 平均章节数/视频 | 8.3 |
| 平均章节间隔 | 142.0s |
| 平均标题词数 | 5.4 |
| 平均视频时长 | 1354s (≈23 min) |

![Figure 3](../images/4e921976857e122fea207c15afe0e2aa1e04b86bc6e68a62711080447d177805.jpg)
*Figure 3: Statistics of the VidChapters-7M dataset*

**ASR vs Chapters:**
| | ASR | Chapters |
|---|-----|----------|
| 平均数量/视频 | 269.8 sentences | 8.3 titles |
| 平均时长 | 3.9s | 142.0s |
| 平均词数 | 11.5 | 5.4 |

**Language distribution:**
- 92.9% English chapter titles
- 93.9% English ASR
- 13 languages appear in >1K videos

**Manual quality assessment (100 videos):**
| Chapter Title 类型 | 百分比 |
|-------------------|--------|
| Speech + Visual | 49% |
| Speech-only | 26% |
| Structure-only (step 1, step 2...) | 14% |
| Visual-only | 3% |
| Audio-only | 3% |
| Audio + Visual | 2% |
| Unrelated | 3% |

---

## 💡 理解

### 核心要点
- [x] **数据来源**: YouTube 用户自己在视频描述中标注的章节 (2020 年 YouTube 推出此功能)
- [x] **爬取流程**: 从 92M 候选 → 正则匹配 → 过滤 → 817K 视频 (仅 0.9%)
- [x] **处理流程**: Whisper ASR + CLIP 视觉特征 (1 FPS)
- [x] **数据质量**: 83% 章节与视频内容相关，3% 无关

### 🖼️ Figure 3 解读 (数据集统计)

```
┌─────────────────────────────────────────────────────────────┐
│  左上: 章节数分布                右上: 章节时长分布            │
│  - 大多数视频 5-15 个章节         - 大多数章节 60-180 秒        │
│  - 长尾分布                       - 长尾分布                   │
├─────────────────────────────────────────────────────────────┤
│  左下: 标题长度分布               右下: 视频类别分布            │
│  - 大多数 3-7 个词                - HowTo & Style 最多 (17%)   │
│  - 简短精炼                       - 12 个类别 > 20K 视频       │
└─────────────────────────────────────────────────────────────┘
```

**视频类别分布** (从图中读取):
1. HowTo & Style: 17%
2. Gaming: ~15%
3. Entertainment: ~12%
4. Music: ~10%
5. People & Blogs: ~8%
6. ...其他类别

### 数据收集流程图

```
YT-Temporal-180M (92M video IDs)
            │
            ▼
    下载视频 description
            │
            ▼
   ┌────────────────────────────┐
   │  正则匹配章节格式:          │
   │  "00:00 - Introduction"    │
   │  或 "Introduction - 00:00" │
   └────────────────────────────┘
            │
            ▼
   ┌────────────────────────────┐
   │  过滤条件:                  │
   │  - 至少 2 个时间戳          │
   │  - 时间戳递增               │
   │  - 排除 YouTube 自动生成    │
   └────────────────────────────┘
            │
            ▼
    VidChapters-7M (817K videos)
         (0.9% 留存率)
```

### ASR vs Chapters 的关键区别

| 维度 | ASR | Chapters | 差异倍数 |
|------|-----|----------|---------|
| 数量/视频 | 269.8 句 | 8.3 个 | **32x** |
| 平均时长 | 3.9s | 142.0s | **36x** |
| 平均词数 | 11.5 词 | 5.4 词 | **0.5x** |

**结论**: ASR 太细碎，不能直接当章节用！

### 章节标题来源分析 (人工评估 100 视频)

```
                    Speech + Visual (49%)
                    ████████████████████
                    
                    Speech-only (26%)
                    ██████████
                    
                    Structure-only (14%)
                    █████
                    
                    Visual/Audio/Unrelated (11%)
                    ████
```

**关键发现**:
- **75% 需要 Speech**: 说明 ASR 对章节生成很重要
- **49% 需要 Visual**: 说明纯文本方法有局限
- **14% 仅结构**: "Step 1", "Part 2" 这类标题没有语义
- **3% 无关**: 存在噪声

### 数据质量与偏差

**优点**:
- ✅ 用户标注质量高 (83% 有意义)
- ✅ 多样性好 (12 个类别)
- ✅ 规模大 (817K videos)

**偏差/问题**:
- ⚠️ 92.9% 英语 (语言偏差)
- ⚠️ 性别词汇偏差 (男性词 39.7% vs 女性词 19.7%)
- ⚠️ 0.7% NSFW 内容
- ⚠️ 14% 章节仅有结构信息无语义

### 我的疑问
- [x] 为什么只有 0.9% 的视频有章节？→ 因为用户主动标注章节需要额外工作，大多数创作者不会做
- [x] 为什么排除 YouTube 自动生成的章节？→ 因为自动生成的章节不在 description 中，而且质量可能不如人工标注
- [x] 1 FPS 的视觉特征够用吗？→ 对于分钟级的章节来说，1 FPS 已经足够捕捉语义变化
