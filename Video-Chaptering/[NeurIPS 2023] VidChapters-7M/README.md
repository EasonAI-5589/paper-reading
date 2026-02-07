# VidChapters-7M: Video Chapters at Scale

> **会议**: NeurIPS 2023  
> **作者**: Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid  
> **机构**: Inria Paris, VGG Oxford, Czech Technical University  
> **链接**: https://antoyang.github.io/vidchapters.html  
> **阅读状态**: ✅ 深度阅读完成

---

## 📖 一句话总结

VidChapters-7M 是首个大规模视频章节数据集 (817K 视频, 7M 章节)，通过爬取 YouTube 用户标注的章节构建，定义了三个任务（生成、给边界生成、定位），并证明了在 dense captioning 任务上的迁移学习价值。

---

## 🎯 核心贡献

| 贡献 | 详情 |
|------|------|
| 📊 **数据集** | 817K videos, 7M chapters, 自动爬取无需人工标注 |
| 🎯 **三个任务** | Chapter Generation / GT-Boundary Title / Grounding |
| 📈 **SOTA 结果** | SODA=11.4 (best), Speech > Visual 2x |
| 🚀 **迁移学习** | YouCook2 +14 CIDEr, ViTT +6 CIDEr |

---

## 📁 阅读笔记

| Section | 文件 | 状态 |
|---------|------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | ✅ |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | ✅ |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | ✅ |
| 3. VidChapters-7M | [03-dataset.md](sections/03-dataset.md) | ✅ |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | ✅ |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | ✅ |

---

## 💡 关键洞察

### 为什么这篇论文重要？

1. **填补数据空白**: 之前没有大规模视频章节数据集
2. **任务定义清晰**: 区分于 Dense Captioning，更符合用户导航需求
3. **可扩展方法**: 利用用户标注，无需人工成本
4. **强迁移能力**: 预训练后显著提升下游任务

### 核心发现

```
Speech > Visual 2x
├── 75% 章节需要理解语音内容
├── 仅 3% 章节纯视觉
└── 最佳: Speech + Visual 融合

ASR ≠ Chapters
├── ASR: 269.8 句/视频, 3.9秒/句
├── Chapter: 8.3 章/视频, 142秒/章
└── 差距 32x → 需要学习"总结"能力
```

### 任务难度排序

```
Chapter Generation (Full) > Chapter Grounding > Chapter Title (GT boundary)
     SODA=11.4                 R@5s=15.5          CIDEr=120.5
```

---

## 📊 关键数据

| 指标 | 数值 |
|------|------|
| 视频数 | 817K |
| 章节数 | 7M |
| 平均视频时长 | 23分钟 |
| 平均章节数 | 8.3/视频 |
| 平均章节时长 | 142秒 |
| 章节标题词数 | 5.4词 |
| 内容相关率 | 83% |
| 语音覆盖率 | 97.3% |

---

## 🔗 相关论文

- **Chapter-Llama (CVPR 2025)**: LLM-based video chaptering
- **SODA (ECCV 2020)**: Story-oriented evaluation metric
- **Vid2Seq (CVPR 2023)**: Dense video captioning model
- **HowTo100M (ICCV 2019)**: Narrated video pretraining dataset

---

## 📝 对 Apple 面试的启示

1. VidChapters-7M 是 **THE benchmark** for video chaptering
2. **语音模态非常重要** → 需要 ASR 集成
3. **评估指标**: SODA > CIDEr (考虑时间对齐)
4. **数据获取**: 用户标注是可扩展的解决方案
5. **任务差异**: Chapter ≠ Dense Caption (连续、无间隙、短标题)
