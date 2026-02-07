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
- [ ] 

### 🖼️ Figure 3 解读
- 左上: 
- 右上: 
- 左下: 
- 右下: 

### 数据收集流程图
```
YT-Temporal-180M (92M videos)
        ↓
   下载 description
        ↓
   正则匹配章节格式
        ↓
   过滤 (≥2 时间戳)
        ↓
VidChapters-7M (817K videos, 0.9%)
```

### 为什么 ASR ≠ Chapters？
- 
- 
- 

### 数据质量问题
- 

### 我的疑问
- [ ] 
