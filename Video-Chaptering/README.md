# Video Chaptering 视频章节生成

> 📁 研究方向: 自动将长视频分割成语义连贯的章节，并生成章节标题

---

## 🎯 任务定义

**Video Chaptering** = 时间分割 + 标题生成

给定一个长视频，目标是:
1. **Temporal Segmentation**: 找到章节边界时间戳
2. **Title Generation**: 为每个章节生成描述性标题

---

## 📚 核心论文

| 论文 | 会议 | 年份 | 贡献 |
|------|------|------|------|
| [ARC-Chapter](./ARC-Chapter/) | arXiv | 2025.11 | **最新 SOTA**, GRACE 指标 |
| Chapter-Llama | CVPR | 2025.03 | 小时级视频章节, LLM 方法 |
| YTSEG | EACL | 2024 | YouTube 章节分割 benchmark |
| VidChapters-7M | NeurIPS | 2023 | **THE benchmark** (817K 视频) |

---

## 📊 主要 Benchmarks

| Dataset | 规模 | 平均时长 | 语言 |
|---------|------|----------|------|
| VidChapters-7M | 817K 视频, 7M 章节 | ~12 min | EN |
| VidAtlas | 410K 视频 | 16.8 min | EN + ZH |
| YTSEG | 19K 视频 | - | EN |

---

## 📈 评估指标

| 指标 | 评估内容 | 备注 |
|------|----------|------|
| **F1** | 分割准确度 | 主要指标 |
| **SODA** | 整体质量 (分割+标题) | 一对一匹配 |
| **GRACE** | 整体质量 | 多对一匹配 (更灵活) |
| CIDEr | 标题质量 | 文本相似度 |
| mIoU | 时间重叠 | 辅助指标 |

---

## 🔗 相关任务

- **Dense Video Captioning**: 密集视频描述
- **Temporal Grounding**: 给定文本找时间段
- **Video Summarization**: 视频摘要
- **Action Segmentation**: 动作分割

---

## 🍎 Apple Assignment 相关

这个方向直接对应 Apple Data Scientist Assignment Part B:
> LLM 自动生成视频章节 (标题 + 时间戳) 的质量评估框架

---

*Created: 2026-02-06*
