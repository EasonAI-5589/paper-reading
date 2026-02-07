# VidChapters-7M: Video Chapters at Scale

> **NeurIPS 2023 Datasets & Benchmarks Track**

📄 **Paper**: [arxiv:2309.13952](https://arxiv.org/abs/2309.13952)  
🌐 **Project**: https://antoyang.github.io/vidchapters.html  
💻 **Code**: https://github.com/antoyang/VidChapters

---

## 📌 核心贡献

1. **VidChapters-7M 数据集**: 817K 视频, 7M 章节 (用户标注，自动爬取)
2. **三个任务定义**:
   - Video Chapter Generation (分割 + 生成标题)
   - Chapter Generation w/ GT Boundaries (给边界，生成标题)
   - Video Chapter Grounding (给标题，找时间)
3. **Baseline 方法**: CLIP + GPT-2, Vid2Seq, PDVC
4. **迁移学习**: 在 YouCook2, ViTT 上达到 SOTA

---

## 📊 数据集统计

| 指标 | 数值 |
|------|------|
| 视频数量 | 817K |
| 章节总数 | 7M |
| 平均章节数/视频 | 8.3 |
| 平均视频时长 | 23 min |
| 平均章节时长 | 142s |
| 平均标题词数 | 5.4 |
| 语言 | 92.9% 英语 |

---

## 🤖 Baseline 模型

### 分割方法 (Segmentation)
| 方法 | 类型 | 说明 |
|------|------|------|
| **Text Tiling** | 文本 | 基于词汇共现检测话题转换 |
| **Shot Detection** | 视觉 | 基于帧差检测场景变化 |

### 端到端方法
| 方法 | 架构 | F1 | SODA |
|------|------|-----|------|
| **PDVC** | DETR-style (visual only) | 18.3 | 8.5 |
| **Vid2Seq** | T5 + CLIP (speech + visual) | 23.1 | 10.8 |
| **Vid2Seq + HowTo100M** | 预训练版 | **25.0** | **12.1** |

> 💡 后续论文常说的 "Baseline" 就是指 **Vid2Seq + HowTo100M 预训练** (F1=25.0)

---

## 📈 评估指标

### 分割指标
- **R@Ks / P@Ks**: 距离阈值下的召回/精确率 (K=1s, 3s, 5s)
- **R@IoU / P@IoU**: IoU 阈值下的召回/精确率 (0.3, 0.5, 0.7, 0.9)

### 标题生成指标
- **BLEU** (B1-B4): N-gram 匹配
- **CIDEr** (C): 共识度
- **METEOR** (M): 同义词匹配
- **ROUGE-L** (RL): 最长公共子序列

### 综合指标
- **SODA_c** (S): 时序最优匹配 + F-measure (惩罚冗余)

---

## 🔬 关键发现

1. **多模态很重要**: Speech + Visual > Speech only > Visual only
2. **预训练有效**: HowTo100M 预训练提升性能
3. **迁移学习**: VidChapters-7M 预训练在 YouCook2 上达到 SOTA
4. **数据规模**: 性能随数据量增加而提升

---

## 📂 文件结构

```
VidChapters-7M/
├── README.md           # 本文件
├── full.md             # MinerU 解析的完整论文
├── paper.pdf           # 原始 PDF
├── content_list.json   # 结构化内容
├── layout.json         # 版面分析
└── images/             # 论文图片
```

---

## 📝 引用

```bibtex
@inproceedings{yang2023vidchapters,
  title={VidChapters-7M: Video Chapters at Scale},
  author={Yang, Antoine and Nagrani, Arsha and Laptev, Ivan and Sivic, Josef and Schmid, Cordelia},
  booktitle={NeurIPS},
  year={2023}
}
```

---

*解析时间: 2026-02-07*
