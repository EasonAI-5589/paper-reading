# 2. Related Works

> 来源: ARC-Chapter (arXiv 2025)

---

## 📄 原文

**Video Understanding.** Recent advances in video understanding have been driven by large-scale pretraining and multimodal architectures. Models like CLIP, VideoCLIP, and Video-LLaMA have shown remarkable capabilities in understanding short video clips.

> 💡 **视频理解进展**: CLIP/VideoCLIP/Video-LLaMA 在短视频理解上取得突破

However, scaling these models to hour-long videos remains challenging due to memory constraints and the need to capture long-range temporal dependencies.

> 💡 **挑战**: 扩展到小时级长视频仍困难（内存限制 + 长程依赖）

**Video Chaptering.** Video chaptering aims to segment long videos into semantically coherent chapters. VidChapters-7M introduced the first large-scale benchmark for this task. Chapter-Llama extended this work by leveraging large language models for improved chapter generation.

> 💡 **Video Chaptering 发展**:
> ```
> 时间线:
> ├── 2023: VidChapters-7M - 首个大规模 benchmark
> ├── 2024: Chapter-Llama - 用 LLM 提升性能
> └── 2025: ARC-Chapter - 更大数据 + 层级标注 + GRACE
> ```

**Dense Video Captioning.** Related tasks include dense video captioning, which produces temporally localized descriptions for video events. Unlike chaptering, dense captioning typically focuses on overlapping events rather than sequential, non-overlapping chapters.

> 💡 **Dense Captioning vs Chaptering**:
> | 维度 | Dense Captioning | Video Chaptering |
> |------|-----------------|------------------|
> | 事件关系 | 可重叠 | **连续不重叠** |
> | 覆盖范围 | 部分事件 | **整个视频** |
> | 描述长度 | 完整句子 | **简短标题** |
> | 评价指标 | SODA | SODA + **GRACE** |

**Video Summarization.** Video summarization creates condensed versions of videos. While related, it focuses on content selection rather than structural segmentation.

> 💡 **Video Summarization vs Chaptering**:
> - Summarization: 选择关键帧/片段，生成浓缩版
> - Chaptering: 保留完整视频，添加导航结构

---

## 💡 Section 2 总结

### Video Chaptering 在相关任务中的定位

```
视频内容理解任务谱系:

├── 短视频理解 (<5min)
│   ├── Action Recognition
│   ├── Video Captioning
│   └── Video QA
│
├── 长视频理解 (>5min)
│   ├── Movie Understanding
│   ├── Video Summarization
│   └── Video Chaptering ⭐ (本文关注)
│
└── 时间定位任务
    ├── Temporal Grounding
    ├── Dense Video Captioning
    └── Video Chaptering ⭐
```

### ARC-Chapter 相对前作的改进

| 前作 | 局限 | ARC-Chapter 改进 |
|------|------|-----------------|
| VidChapters-7M | 数据标注粗糙 | 层级标注 |
| Chapter-Llama | 数据规模有限 | 50x 数据 |
| SODA 指标 | 一对一匹配 | GRACE 指标 |
