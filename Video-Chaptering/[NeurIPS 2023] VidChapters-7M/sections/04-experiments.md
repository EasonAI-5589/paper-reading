# 4. Experiments

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

### Evaluation Metrics

**Chapter title quality (without position):**
- BLEU (B)
- CIDEr (C)  
- METEOR (M)
- ROUGE-L (RL)

**Video chapter generation (with location):**
- Standard evaluation tool: matched pairs at IoU thresholds {0.3, 0.5, 0.7, 0.9}
- **SODA_c (S)**: finds temporally optimal matching → captures story → F-measure to penalize redundant chapters
- Recall (R@Ks, R@K) and Precision (P@Ks, P@K) at various thresholds

### 4.1 Video Chapter Generation

**Models evaluated:**

| Method | Modalities | Pretraining | Finetuned | SODA |
|--------|------------|-------------|-----------|------|
| Text tiling + Random | Speech | - | ✗ | 0.4 |
| Text tiling + LLaMA | Speech | Text | ✗ | 0.2 |
| Shot detect + BLIP-2 | Visual | 129M img-txt | ✗ | 0.6 |
| Vid2Seq | Speech+Visual | C4 + HowTo100M | ✗ | 0.1 |
| PDVC | Visual | - | ✓ | 6.8 |
| **Vid2Seq** | Speech+Visual | C4 + HowTo100M | ✓ | **11.4** |

**Key findings:**
- Finetuning on VidChapters-7M is crucial
- Speech-only > Visual-only (Vid2Seq: 10.5 vs 5.5 SODA)
- Speech + Visual is best (11.4 SODA)
- HowTo100M pretraining helps

### 4.2 Video Chapter Generation Given GT Boundaries

When given ground-truth boundaries:
- Zero-shot baselines (LLaMA, BLIP-2) struggle
- Vid2Seq finetuned achieves 120.5 CIDEr
- Speech-only (105.3 CIDEr) > Visual-only (47.1 CIDEr)

### 4.3 Video Chapter Grounding

**Models:**

| Method | Modalities | R@5s |
|--------|------------|------|
| Random | Speech | 1.8 |
| BERT | Speech | 6.8 |
| CLIP | Visual | 5.2 |
| Moment-DETR | Visual (finetuned) | **15.5** |

**Findings:**
- ASR-based baselines can find start times but struggle with start-end windows
- Finetuning on VidChapters-7M significantly improves performance

### 4.4 Transfer Learning to Dense Video Captioning

**Pretraining on VidChapters-7M improves downstream:**

| Pretraining | YouCook2 SODA | ViTT SODA |
|-------------|---------------|-----------|
| C4 + HowTo100M | 8.6 | 14.1 |
| C4 + VidChapters-7M (ASR+Chap) | 9.8 | 15.1 |
| C4 + HowTo100M + VidChapters-7M | **10.3** | **15.0** |

**Key insight:** Performance scales with dataset size (1% → 10% → 100% of VidChapters-7M)

---

## 💡 理解

### 核心要点
- [ ] 

### 评估指标总结

| 指标 | 评估什么 | 特点 |
|------|---------|------|
| SODA | | |
| F1 | | |
| CIDEr | | |
| R@K / P@K | | |

### 实验结论

1. **模态重要性**: Speech > Visual, 但两者结合最好
2. **Finetuning**: 必须在 VidChapters-7M 上 finetune
3. **预训练**: HowTo100M 预训练有帮助
4. **迁移学习**: VidChapters-7M 预训练提升下游任务

### Baseline 架构理解

**Vid2Seq:**
- 

**PDVC:**
- 

### 我的疑问
- [ ] 为什么 Speech > Visual？
- [ ] SODA 具体怎么计算？
