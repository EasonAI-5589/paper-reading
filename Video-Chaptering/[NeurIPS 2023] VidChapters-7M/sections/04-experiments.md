# 4. Experiments

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

### Evaluation Metrics

To evaluate the quality of the generated chapter titles (without their positions), we use standard metrics used for visual captioning: 
- **BLEU** (B) - N-gram precision
- **CIDEr** (C) - Consensus-based image description evaluation
- **METEOR** (M) - Alignment-based metric
- **ROUGE-L** (RL) - Longest common subsequence

To evaluate video chapter generation as a whole, including the locations of the generated chapters, we follow standard protocols used for dense video captioning:
- Standard evaluation tool which calculates matched pairs between generated events and the ground truth across IoU thresholds of {0.3, 0.5, 0.7, 0.9}
- **SODA_c** (S) - Story-oriented dense captioning metric which tries to find a temporally optimal matching between generated and reference chapters to capture the story of a video

To separately evaluate chapter localization:
- **R@Ks, R@K** - Recall across various thresholds
- **P@Ks, P@K** - Precision across various thresholds
- **R, P** - Average recall/precision across IoU thresholds {0.3, 0.5, 0.7, 0.9}

### Implementation Details

- **Speech transcripts and visual features** extracted as explained in Section 3.2
- **Adam optimizer** for training
- **Model selection** based on best validation performance
- **Hardware**: 8 NVIDIA A100 80GB GPUs

---

## 4.1 Video Chapter Generation

This task requires temporally segmenting the video and generating a chapter title for each segment.

### Models Evaluated

**Zero-shot approaches:**
- **Text tiling** [32] - Detects subtopic shifts based on lexical co-occurrence patterns
- **Shot detection** [92] - Visual scene change detection based on sum of absolute differences

**Combined zero-shot baselines:**
- Text tiling + Random (random speech sentence from predicted boundaries)
- Text tiling + LLaMA-7B (summarize speech in predicted boundaries)
- Shot detection + BLIP-2 (describe middle frame of predicted segment)

**End-to-end trained models:**
- **PDVC** [101] - DETR-style visual-only architecture
- **Vid2Seq** [114] - Multi-modal sequence-to-sequence model pretrained on C4 + HowTo100M

### Results (Table 3 & 4)

**Table 3: Video chapter generation (global metrics)**

| Method | Modalities | Pretraining | Finetuned | SODA | CIDEr |
|--------|------------|-------------|-----------|------|-------|
| Text tiling + Random | Speech | - | ✗ | 0.4 | 0.8 |
| Text tiling + LLaMA | Speech | Text | ✗ | 0.2 | 0.5 |
| Shot detect + BLIP-2 | Visual | 129M | ✗ | 0.6 | 0.2 |
| Vid2Seq (zero-shot) | S+V | C4+HTM | ✗ | 0.1 | 0.1 |
| PDVC | Visual | - | ✓ | 6.8 | 35.8 |
| Vid2Seq (S) | Speech | C4+HTM | ✓ | 10.5 | 50.7 |
| Vid2Seq (V) | Visual | C4+HTM | ✓ | 5.5 | 21.1 |
| **Vid2Seq (S+V)** | **S+V** | **C4+HTM** | ✓ | **11.4** | **55.7** |

**Key findings:**
- Models trained on VidChapters-7M **significantly outperform** zero-shot baselines
- **Speech-only > Visual-only** for Vid2Seq
- **Speech + Visual** gives best performance → multi-modal task confirmed
- Pretraining on HowTo100M improves performance, especially for vision-aware models

---

## 4.2 Video Chapter Generation Given Ground-Truth Boundaries

This simplified task assumes perfect temporal segmentation.

### Results (Table 5)

| Method | Modalities | Pretraining | Finetuned | CIDEr |
|--------|------------|-------------|-----------|-------|
| Random | Speech | - | ✗ | 10.4 |
| LLaMA | Speech | Text | ✗ | 0.0 |
| BLIP-2 | Visual | 129M | ✗ | 12.4 |
| Vid2Seq (zero-shot) | S+V | C4+HTM | ✗ | 0.9 |
| Vid2Seq (S) | Speech | C4+HTM | ✓ | 105.3 |
| Vid2Seq (V) | Visual | C4+HTM | ✓ | 47.1 |
| **Vid2Seq (S+V)** | **S+V** | **C4+HTM** | ✓ | **120.5** |

**Key findings:**
- LLaMA struggles to summarize speech into chapter titles → underperforms random!
- Zero-shot Vid2Seq underperforms random due to large domain gap between ASR and chapter titles
- Trained models achieve much higher performance (120.5 vs 12.4 CIDEr)

---

## 4.3 Video Chapter Grounding

This task requires localizing a chapter start time (or start-end window) given an annotated chapter title.

### Models Evaluated

**Zero-shot:**
- Random - randomly pick speech sentence timestamps
- BERT [19] - pick speech sentence with closest embedding to query
- CLIP [72] - pick frames where query-frame similarity is highest

**Trained:**
- Moment-DETR [45] - DETR-style video grounding model

### Results (Table 6)

| Method | Modalities | Pretraining | Finetuned | R@5s | R@0.5 |
|--------|------------|-------------|-----------|------|-------|
| Random | Speech | - | ✗ | 1.8 | 0.3 |
| BERT | Speech | Books+Wiki | ✗ | 6.8 | 0.3 |
| CLIP | Visual | 400M | ✗ | 5.2 | 5.2 |
| Moment-DETR | Visual | 5.4K | ✗ | 1.6 | 3.6 |
| **Moment-DETR** | **Visual** | - | ✓ | **15.5** | **27.3** |

**Key findings:**
- ASR-based baselines can find start times but struggle with start-end windows
- CLIP slightly worse at start times but much better at windows
- Trained Moment-DETR significantly outperforms all zero-shot baselines
- **Limitation**: Moment-DETR cannot use speech input → future research needed

---

## 4.4 Transfer Learning on Dense Video Captioning

### Datasets
- **YouCook2** [127]: 2K cooking videos, avg 320s, 7.7 sentences/video
- **ViTT** [36]: 8K instructional videos, avg 250s, 7.1 tags/video

### Results after Finetuning (Table 7)

| Method | Pretraining | YouCook2 SODA | YouCook2 CIDEr |
|--------|-------------|---------------|----------------|
| PDVC | - | 4.8 | 28.8 |
| PDVC | VC (Chap.) | 5.9 | 34.7 |
| Vid2Seq | C4+HTM | 8.6 | 53.2 |
| Vid2Seq | C4+VC (ASR+Chap.) | 9.8 | 62.9 |
| **Vid2Seq** | **C4+HTM+VC (ASR+Chap.)** | **10.3** | **67.2** |

**Key findings:**
- Pretraining on VidChapters-7M **greatly improves** downstream dense video captioning
- Combining HowTo100M + VidChapters-7M gives best results
- **+14 CIDEr** improvement on YouCook2 from VidChapters-7M pretraining
- **Scaling behavior**: More chapter data → better downstream performance

### Zero-shot Dense Video Captioning (Table 8)

| Method | Pretraining | YouCook2 SODA | YouCook2 CIDEr |
|--------|-------------|---------------|----------------|
| Text tiling + Random | - | 0.3 | 0.9 |
| Shot detect + BLIP-2 | 129M | 0.6 | 1.0 |
| Vid2Seq | C4+VC (ASR+Chap.) | **3.2** | **10.2** |
| Vid2Seq | C4+HTM+VC (ASR+Chap.) | **3.9** | **13.3** |

**Key findings:**
- First to explore this challenging zero-shot setting (no manual dense caption annotation)
- Using both ASR + chapter annotations → largely better zero-shot performance
- ASR and chapters are complementary for zero-shot transfer

---

## 💡 理解

### 核心实验结论速查

| 实验 | 最佳方法 | 最佳模态 | 关键发现 |
|------|---------|---------|----------|
| 4.1 Chapter Generation | Vid2Seq | S+V | Speech > Visual 2x |
| 4.2 GT-Boundary Title | Vid2Seq | S+V | LLaMA 失败！ |
| 4.3 Chapter Grounding | Moment-DETR | Visual | 需要多模态扩展 |
| 4.4 Dense Captioning | Vid2Seq | S+V | +14 CIDEr |

### 🖼️ 实验结果可视化

```
Task 1: Video Chapter Generation
┌──────────────────────────────────────────────────────┐
│  SODA Score 比较                                     │
│  ├── Zero-shot baselines: 0.1-0.6                    │
│  ├── PDVC (visual): 6.8                              │
│  ├── Vid2Seq (speech): 10.5                          │
│  ├── Vid2Seq (visual): 5.5                           │
│  └── Vid2Seq (S+V): 11.4 ⭐ SOTA                     │
│                                                      │
│  → Speech is more important than Visual!             │
│  → Multi-modal fusion helps                          │
└──────────────────────────────────────────────────────┘

Task 4.4: Transfer to Dense Captioning
┌──────────────────────────────────────────────────────┐
│  YouCook2 CIDEr Score                                │
│  ├── PDVC (scratch): 28.8                            │
│  ├── Vid2Seq (HTM): 53.2                             │
│  ├── Vid2Seq (VC): 62.9 (+9.7)                       │
│  └── Vid2Seq (HTM+VC): 67.2 (+14) ⭐                 │
│                                                      │
│  → VidChapters-7M is valuable for pretraining!       │
│  → Scaling: more data → better transfer              │
└──────────────────────────────────────────────────────┘
```

### 为什么 Speech > Visual？

1. **Table 2 证据**: 75% 章节需要语音理解
2. **信息密度**: 23分钟语音 > 23分钟静态帧
3. **语义层次**: 语音已经是高层语义，而视觉需要更多抽象

### 为什么 LLaMA 在 Task 2 失败？

```
LLaMA 输入: "Right, we're gonna do the Synthetics Dirty Race. 
             No we're not. So we're gonna put two t-shirts..."
             
LLaMA 输出: "The video discusses the process of doing a 
             synthetic dirty race involving two t-shirts..."
             
GT 标题: "Laundry Tips"

问题: LLaMA 生成详细描述，而非简洁标题！
     需要 VidChapters-7M 训练才能学会"标题风格"
```

### Scaling 规律验证

| Pretraining Data | YouCook2 CIDEr |
|-----------------|----------------|
| HTM only | 53.2 |
| HTM + 1% VC | 52.7 |
| HTM + 10% VC | 63.9 |
| HTM + 100% VC | **67.2** |

→ 数据量越大，迁移效果越好

### 任务难度排序

```
难度: Task 1 > Task 3 > Task 2

Task 1 (Full Generation): 
  需要同时做分割 + 生成，误差叠加

Task 3 (Grounding):
  给定标题找位置，比生成标题容易

Task 2 (GT-Boundary Title):
  给定边界生成标题，最简单的子任务
```

### 我的疑问
- [x] 为什么 SODA 比 CIDEr 低那么多？→ SODA 考虑时间对齐，更严格
- [x] Moment-DETR 为什么不能用语音？→ 原始设计只支持视觉输入
- [x] HowTo100M 和 VidChapters-7M 有重叠吗？→ 可能有，但目标不同 (ASR vs Chapter)
