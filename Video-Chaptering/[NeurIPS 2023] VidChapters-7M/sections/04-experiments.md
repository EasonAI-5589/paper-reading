# 4. Experiments

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

### Evaluation Metrics

**Chapter title quality (without position):**
- BLEU (B): N-gram 精确匹配
- CIDEr (C): TF-IDF 加权，惩罚通用描述
- METEOR (M): 同义词 + 词干匹配
- ROUGE-L (RL): 最长公共子序列

**Video chapter generation (with location):**
- Standard evaluation tool: matched pairs at IoU thresholds {0.3, 0.5, 0.7, 0.9}
- **SODA_c (S)**: 
  - 先找时序最优匹配 (捕捉视频故事)
  - 再计算 METEOR
  - 用 F-measure 惩罚冗余章节
- Recall (R) / Precision (P): 跨 IoU 阈值平均

### 4.1 Video Chapter Generation (完整任务)

**Zero-shot baselines:**
| Method | Modalities | SODA |
|--------|------------|------|
| Text tiling + Random | Speech | 0.4 |
| Text tiling + LLaMA | Speech | 0.2 |
| Shot detect + BLIP-2 | Visual | 0.6 |
| Vid2Seq (pretrained) | Speech+Visual | 0.1 |

**Finetuned models:**
| Method | Modalities | SODA | F1 |
|--------|------------|------|-----|
| PDVC | Visual | 6.8 | 18.3 |
| Vid2Seq (Speech) | Speech | 10.5 | - |
| Vid2Seq (Visual) | Visual | 5.5 | - |
| **Vid2Seq (Both)** | Speech+Visual | **11.4** | **25.0** |

### 4.2 Chapter Title Generation (给定边界)

| Method | Modalities | CIDEr |
|--------|------------|-------|
| Random | Speech | 10.4 |
| LLaMA | Speech | 0.0 |
| BLIP-2 | Visual | 12.4 |
| Vid2Seq (Speech) | Speech | 105.3 |
| Vid2Seq (Visual) | Visual | 47.1 |
| **Vid2Seq (Both)** | Speech+Visual | **120.5** |

### 4.3 Video Chapter Grounding

| Method | Modalities | R@5s |
|--------|------------|------|
| Random | Speech | 1.8 |
| BERT | Speech | 6.8 |
| CLIP | Visual | 5.2 |
| **Moment-DETR (finetuned)** | Visual | **15.5** |

### 4.4 Transfer Learning

**Pretraining on VidChapters-7M → Dense Captioning:**
| Pretraining | YouCook2 SODA | ViTT SODA |
|-------------|---------------|-----------|
| C4 + HowTo100M | 8.6 | 14.1 |
| C4 + VidChapters (ASR+Chap) | 9.8 | 15.1 |
| **C4 + HowTo100M + VidChapters** | **10.3** | **15.0** |

**Scaling behavior:** 1% → 10% → 100% 数据，性能持续提升

---

## 💡 理解

### 核心要点
- [x] **Finetuning 至关重要**: Zero-shot 方法 SODA < 1，finetuned Vid2Seq 达到 11.4
- [x] **Speech > Visual**: 纯 Speech 模型 (10.5) 优于纯 Visual 模型 (5.5)
- [x] **多模态最佳**: Speech + Visual (11.4) 比单模态都好
- [x] **迁移学习有效**: VidChapters 预训练提升下游 dense captioning

### 评估指标详解

| 指标 | 评估什么 | 特点 |
|------|---------|------|
| **BLEU** | N-gram 匹配 | 严格，容易低分 |
| **CIDEr** | TF-IDF 加权 | 惩罚 "the", "a" 等通用词 |
| **METEOR** | 同义词匹配 | 比 BLEU 更宽松 |
| **SODA** | 时序匹配 + F-measure | **最重要**，惩罚冗余 |
| **F1** | 边界检测 | 综合 Precision/Recall |

### 为什么 SODA 比传统指标更好？

```
传统指标问题:
┌─────────────────────────────────────┐
│  生成 200 个章节，覆盖所有 GT       │
│  → 很多匹配 → 高分                  │
│  但人类无法阅读 200 个章节！❌       │
└─────────────────────────────────────┘

SODA 解决方案:
┌─────────────────────────────────────┐
│  1. 时序最优匹配 (保持故事顺序)     │
│  2. F-measure (惩罚冗余)            │
│  → 生成太多会降低 Precision ✅      │
└─────────────────────────────────────┘
```

### 实验结论总结

#### 4.1 完整章节生成
```
性能排序: Vid2Seq(Both) > Vid2Seq(Speech) > PDVC > Vid2Seq(Visual)
         11.4 SODA      10.5 SODA       6.8    5.5

关键发现:
- Zero-shot 几乎不工作 (SODA < 1)
- HowTo100M 预训练有帮助 (+0.8 SODA)
- Speech 模态比 Visual 重要 2 倍
```

#### 4.2 给定边界的标题生成
```
性能排序: Vid2Seq(Both) > Vid2Seq(Speech) >> Vid2Seq(Visual)
         120.5 CIDEr    105.3 CIDEr      47.1 CIDEr

关键发现:
- LLaMA zero-shot 完全失败 (CIDEr=0)
- Speech 信息更重要 (105.3 vs 47.1)
```

#### 4.3 章节定位
```
Moment-DETR finetuned: R@5s = 15.5
比 BERT baseline (6.8) 好 2 倍多

注意: Moment-DETR 只用 Visual，没有用 Speech
→ 说明还有提升空间
```

#### 4.4 迁移学习
```
VidChapters-7M 预训练的价值:
- YouCook2: 8.6 → 10.3 SODA (+20%)
- ViTT: 14.1 → 15.0 SODA (+6%)

Scaling 规律:
1% 数据 → 10% → 100%
性能持续提升，没有饱和
```

### Baseline 模型架构理解

**Vid2Seq:**
- 基于 T5 的 seq2seq 模型
- 输入: 视频帧 + ASR
- 输出: 时间戳 + 标题 的序列
- 预训练: C4 文本 + HowTo100M 视频

**PDVC (Parallel Dense Video Captioning):**
- DETR 风格架构
- 只用 Visual 输入
- 并行预测事件和描述

**Moment-DETR:**
- DETR 用于时序定位
- 给定文本查询，预测时间段

### 模态重要性分析

```
Speech vs Visual 消融实验:

Task 1 (完整生成):
Speech only:  SODA = 10.5  ████████████████████
Visual only:  SODA = 5.5   ██████████
Both:         SODA = 11.4  ██████████████████████

Task 2 (标题生成):
Speech only:  CIDEr = 105.3 █████████████████████
Visual only:  CIDEr = 47.1  █████████
Both:         CIDEr = 120.5 ████████████████████████

结论: Speech 贡献 ~2x Visual
原因: 75% 的章节标题与 Speech 相关 (来自 Section 3.3)
```

### 我的疑问
- [x] 为什么 LLaMA zero-shot 完全失败？→ 因为章节标题有特定格式，LLaMA 生成的是完整句子，风格不匹配
- [x] 为什么 Visual-only 效果差？→ 因为大多数章节标题基于 Speech 内容，纯视觉缺少语义信息
- [x] SODA 和 F1 的关系？→ SODA 关注故事完整性，F1 关注边界准确性，两者互补
- [x] 为什么迁移学习有效？→ 因为 Chapter Generation 和 Dense Captioning 都需要时序理解 + 文本生成能力
