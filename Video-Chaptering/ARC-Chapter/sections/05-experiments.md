# 5. Experiments

## 📄 原文逐段解析

---

## 5.1 Evaluation Benchmark

### 5.1.1 三个评估数据集

> To comprehensively assess our model's capabilities in video chaptering, we evaluate it on **three distinct benchmarks** covering different languages, scales, and data modalities.
>
> ==三个 benchmark：覆盖不同语言、规模、模态==

| Benchmark | 语言 | 规模 | 模态 | 用途 |
|-----------|------|------|------|------|
| VidChapters7M-test | EN | 8.2k 样本 | ASR-only | 大规模测试 |
| VidChapters7M-sml300val | EN | 300 样本 | Video + ASR | 快速评估/消融 |
| VidAtlas-test | ZH | 1.5k+ 样本 | Video + ASR | 中文泛化 |

### 5.1.2 评估维度

> The evaluation targets two key criteria:
> 1. The precision of **temporal boundary localization**
> 2. **Semantic relevance** of the generated chapter titles/descriptions
>
> ==两个评估维度：时间边界精度 + 语义相关性==

---

## 5.2 Comparison with the State of the Art

### 5.2.1 VidChapters7M-test 结果 (Table 1)

**ASR-only 设置（与 Chapter-Llama 公平对比）：**

| Method | Finetune | F1 | tIoU | SODA | CIDEr |
|--------|----------|-----|------|------|-------|
| GPT-4o-mini | ✗ | 31.2 | 63.6 | 6.8 | 37.8 |
| GPT-4o | ✗ | 37.6 | 68.0 | 8.1 | 51.0 |
| Gemini-2.0-Flash | ✗ | 40.2 | 69.3 | 11.4 | 69.7 |
| Gemini-1.5-Pro | ✗ | 42.2 | 70.9 | 11.4 | 63.2 |
| Llama 3.1-8B | ✗ | 29.5 | 62.5 | 6.2 | 30.7 |
| Vid2Seq | ✓ | 26.7 | 58.6 | 11.6 | 55.8 |
| Chapter-Llama | ✓ | 45.3 | 71.8 | 19.3 | 100.9 |
| **ARC-Chapter-asr** | ✓ | **54.5** | **76.7** | **25.3** | **144.0** |

> Our model achieves a new state-of-the-art result in the ASR-only regime, with absolute gains of **+9.2 in F1**, **+4.9 in tIoU**, and **+6.0 in SODA** over Chapter-Llama.
>
> ==相比 Chapter-Llama：F1 +9.2，tIoU +4.9，SODA +6.0==

**多模态设置：**

| Method | F1 | tIoU | SODA | CIDEr |
|--------|-----|------|------|-------|
| ARC-Chapter-asr | 54.5 | 76.7 | 25.3 | 144.0 |
| ARC-Chapter-vid | 50.2 | 74.3 | 22.9 | 138.3 |
| **ARC-Chapter-vidasr** | **59.3** | **79.6** | **30.6** | **186.6** |

> The performance gain **enlarges as video duration increases**. For long videos (30-60 min), the evaluation metrics of SODA and CIDEr for ARC-Chapter are remarkably higher than Chapter-LLama.
>
> ==长视频（30-60分钟）性能优势更明显==

### 5.2.2 VidChapters7M-sml300 结果 (Table 2)

**模态组合消融：**

| Method | Speech | Video | F1 | SODA | CIDEr |
|--------|--------|-------|-----|------|-------|
| Chapter-LLaMA | ✓ | ✗ | 38.5 | 13.9 | 67.3 |
| Chapter-LLaMA | ✓ | ✓ (embed) | 40.4 | 15.3 | 74.9 |
| Chapter-LLaMA | ✓ | ✓ (both) | 44.4 | 16.3 | 84.2 |
| **ARC-Chapter** | ✓ | ✗ | **56.5** | **25.9** | **148.5** |
| **ARC-Chapter** | ✗ | ✓ | 50.0 | 21.6 | 130.8 |
| **ARC-Chapter** | ✓ | ✓ | **62.4** | **30.1** | **190.7** |

> ARC-Chapter demonstrates superior performance by effectively integrating both speech and video information.
>
> ==多模态融合显著提升性能==

### 5.2.3 VidAtlas-test 结果 (Table 3)

**中文泛化性：**

| Method | Modality | F1 | SODA | CIDEr | GRACE |
|--------|----------|-----|------|-------|-------|
| Claude-Sonnet | A | 37.8 | 7.1 | 36.9 | 11.1 |
| DeepSeek-R1 | A | 38.9 | 10.0 | 44.8 | 13.4 |
| Gemini-2.5-Pro | A+V | 48.7 | 13.5 | 75.8 | 19.8 |
| **ARC-Chapter-asr** | A | **58.8** | **24.8** | **111.3** | **28.0** |
| **ARC-Chapter-vid** | V | 57.6 | 21.6 | 98.1 | 25.0 |
| **ARC-Chapter-vidasr** | A+V | **66.2** | **30.2** | **141.5** | **34.1** |

> Our full multimodal model achieves an overall F1 score of 66.2, marking a significant leap over Gemini-2.5-Pro with an absolute improvement of **+17.5 in F1** and more than doubling the SODA score.
>
> ==比 Gemini-2.5-Pro：F1 +17.5，SODA 翻倍==

---

## 5.3 Transferability

### 5.3.1 Dense Video Captioning (Table 4)

> We pre-trained ARC-Chapter on our dataset before fine-tuning and testing it on the dense video captioning benchmarks.
>
> ==VidAtlas 预训练 → 下游任务微调==

| Method | YouCook2 F1 | YouCook2 SODA | ActivityNet F1 |
|--------|-------------|---------------|----------------|
| Vid2Seq | 27.3 | 7.9 | 52.4 |
| TRACE | 31.8 | 6.7 | 39.3 |
| TimeExpert | 33.5 | 7.2 | 40.5 |
| **ARC-Chapter** | **37.9** | **12.5** | **55.9** |

> ARC-Chapter achieves an F1/SODA Score of **37.9/12.5** on YouCook2, a substantial improvement over the previous best of 33.5/7.9.
>
> ==YouCook2：F1 37.9 (+4.4)，SODA 12.5 (+5.3)==

> This demonstrates that the knowledge acquired during pre-training effectively transfers and enhances performance on downstream tasks.
>
> ==预训练知识有效迁移到下游任务==

---

## 5.4 Ablation Studies

### 5.4.1 Scaling Property (Figure 6)

> We analyze how ARC-Chapter scales with the amount of training data.
>
> ==分析训练数据量对性能的影响==

**实验设置：**
- 训练集采样：20%, 40%, 60%, 80%, 100%
- 三种输入模态：ASR-only, Video-only, ASR+Video
- 两个 benchmark：VidChapters-7M, VidAtlas

**关键发现：**

> The performance across all metrics (F1, tIOU, SODA, and CIDEr) and input modalities demonstrates a **clear positive correlation** with the amount of training data.
>
> ==所有指标和模态都随数据量正向相关==

> ARC-Chapter is **highly data-efficient**, achieving strong performance with as little as 20% of the training data.
>
> ==数据高效：20% 数据就能达到较强性能==

> Furthermore, it is **data-scalable**, continuing to benefit from larger corpora for even better results.
>
> ==可扩展：更多数据持续提升性能==

| 数据比例 | F1 (approx) | 趋势 |
|----------|-------------|------|
| 20% | ~48 | 已超 Chapter-Llama |
| 40% | ~52 | ↑ |
| 60% | ~55 | ↑ |
| 80% | ~57 | ↑ |
| 100% | 59.3 | **未饱和！** |

### 5.4.2 Hierarchical Annotations (Table 5)

> We evaluate our model's capability to generate outputs of varying complexity.
>
> ==评估不同复杂度输出的生成能力==

**Short Title vs Structural Info：**

| 数据集 | 输出格式 | F1 | tIoU | SODA | CIDEr |
|--------|----------|-----|------|------|-------|
| VidChapter-sml300 | Short Title | 62.4 | 81.6 | 30.1 | 190.7 |
| VidChapter-sml300 | Structural | 61.4 | 80.6 | 30.8 | 194.5 |
| VidAtlas-test | Short Title | 66.2 | 84.0 | 30.2 | 141.5 |
| VidAtlas-test | Structural | 65.9 | 83.8 | 30.8 | 143.5 |

> When comparing the segmentation metrics (F1 and tIoU) for Short Title versus Structural Info, we observe only a **negligible difference**.
>
> ==复杂输出（Structural）几乎不影响分割准确度==

> The model can perform complex, multi-part generation in a single forward pass **without compromising** its core ability to accurately segment the video.
>
> ==单次推理生成复杂层级输出，不牺牲分割能力==

### 5.4.3 Performance with GRPO (Table 6)

> We compare the performance of our models before (SFT-base) and after (+RL) GRPO optimization.
>
> ==对比 SFT vs SFT+GRPO==

| Model | Stage | F1 | tIoU | CIDEr | GRACE |
|-------|-------|-----|------|-------|-------|
| Base-vidasr | SFT | 59.3 | 79.6 | 186.6 | 34.3 |
| **GRPO-vidasr** | +RL | **60.8** | **80.7** | **190.7** | **34.6** |

**三个关键结论：**

> **1. GRPO directly improves temporal metrics:**
> We observe a clear performance boost in F1 and tIoU scores across all configurations.
>
> ==GRPO 直接提升时间指标（F1, tIoU）==

> **2. Cross-modal transferability:**
> Despite GRPO training being conducted exclusively on the video modality, the temporal localization performance of the ASR and Video+ASR inputs also improves.
>
> ==跨模态迁移：仅用 Video 训练，ASR 和 Video+ASR 也提升==

> **3. No sacrifice in semantic quality:**
> Semantic metrics such as CIDEr remain highly comparable to the SFT baseline, and in some cases even improve.
>
> ==语义质量不降反升（CIDEr 提升）==

---

## 5.5 Qualitative Visualization

### 5.5.1 英文视频示例 (Figure 7)

**视频主题**：US Debt & Stablecoins

**Short Title 输出：**
```
00:00:00 - Intro
00:00:48 - US Debt Problem
00:05:35 - Stablecoins & US Bonds
00:09:22 - Refilling The TGA
00:14:24 - Stablecoin Regulation
00:17:08 - Which Cryptos Will Win
```

**Structural Chapter 输出示例：**
```
▷ US Debt Problem [00:00:48]
Title: The US Debt Ceiling Crisis and Market Impact
Intro: This section details the US debt ceiling situation, 
       explaining that the US government hit its debt ceiling 
       in January and cannot issue more debt...
```

> The model successfully navigates complex financial terminology. The generated title, abstract, and introduction are **distinct yet complementary**, providing a rich, layered understanding.
>
> ==能处理复杂金融术语，层级输出互补==

### 5.5.2 中文视频示例 (Figure 8)

**视频主题**：稳定币投资机遇与挑战

> The model exhibits a **comparable level of understanding and generation quality** in Chinese.
>
> ==中文理解和生成质量与英文相当==

> This strong cross-lingual performance underscores the model's ability to generalize the learned chaptering and summarization skills.
>
> ==强跨语言泛化能力==

---

## 💡 Key Takeaways

1. **SOTA 性能**：VidChapters-7M 上 F1 59.3（+14%），SODA 30.6（+58%）
2. **Scaling Law**：数据量从 20% → 100%，性能持续提升，未饱和
3. **多模态融合**：ASR+Video 比单模态高 5-8 SODA
4. **层级标注**：复杂输出不牺牲分割准确度
5. **GRPO 有效**：时间指标提升，语义不降反升
6. **迁移性强**：YouCook2 上 F1 37.9，SODA 12.5
7. **双语泛化**：中英文质量相当

---

*[返回论文目录](../README.md)*
