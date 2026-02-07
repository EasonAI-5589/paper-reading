# 4. Experiments

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

In this Section, we present the results of models on VidChapters-7M for the full video chapter generation task in Section 4.1, the task of video chapter generation given ground-truth boundaries in Section 4.2 and the video chapter grounding task in Section 4.3. Finally, we study transfer learning from video chapter generation to dense video captioning tasks in Section 4.4.

> 💡 **Section 4 概览**: 四个实验，难度递减：
> - 4.1 完整生成（分割+标题）← 最难
> - 4.2 给边界生成标题 ← 简化版
> - 4.3 给标题找位置 ← 反向任务
> - 4.4 迁移到 dense captioning ← 验证预训练价值

---

**Evaluation metrics.** To evaluate the quality of the generated chapter titles (without their positions), we use standard metrics used for visual captioning: BLEU [70] (B), CIDEr [95] (C), METEOR [7] (M) and ROUGE-L [56] (RL). To evaluate video chapter generation as a whole, including the locations of the generated chapters, we follow standard protocols used for dense video captioning, given the similar nature of the two tasks. We use the standard evaluation tool [42] which calculates matched pairs between generated events and the ground truth across IoU thresholds of {0.3, 0.5, 0.7, 0.9}, and compute captioning metrics over the matched pairs. However, these metrics do not take into account the story of the video and give high scores to methods generating many redundant chapters. Hence for an overall evaluation, we also use SODA_c [22] (S) which first tries to find a temporally optimal matching between generated and reference chapters to capture the story of a video, then computes METEOR scores for the matching and derives F-measure scores from the METEOR scores to penalize redundant chapters. To separately evaluate chapter localization, we report the recall (R@Ks, R@K) and the precision (P@Ks, P@K) across various thresholds in terms of the distance to the ground-truth start time or IoU with the ground-truth start-end window. We also report the average recall (R) and average precision (P) across IoU thresholds of {0.3, 0.5, 0.7, 0.9}.

> 💡 **评价指标详解**:
>
> 指标分两大类：**文本质量指标**（评价标题写得好不好）和 **时间定位指标**（评价时间找得准不准）
>
> ---
>
> **一、文本质量指标（评价章节标题）**
>
> | 指标 | 全称 | 大白话解释 | 例子 |
> |------|------|-----------|------|
> | **BLEU** | Bilingual Evaluation Understudy | 数"重叠词"有多少 | 预测"炒鸡蛋教程" vs GT"鸡蛋炒法" → 重叠词"鸡蛋"，BLEU有分 |
> | **CIDEr** | Consensus-based Image Description Evaluation | 数重叠词，但**稀有词加分更多** | "番茄"比"的"更重要，因为"的"到处都有 |
> | **METEOR** | Metric for Evaluation of Translation with Explicit Ordering | 考虑**同义词和词干** | "cooking"和"cook"算匹配，"film"和"movie"也算 |
> | **ROUGE-L** | Recall-Oriented Understudy for Gisting Evaluation | 找**最长公共子序列** | 顺序也重要，不只是词袋匹配 |
>
> ```
> 例子：
> GT 标题: "Making scrambled eggs"
> 预测标题: "How to cook eggs"
>
> BLEU: "eggs" 重叠 → 有分
> CIDEr: "eggs" 是关键词，加权高 → 分更高
> METEOR: "cook" ≈ "making" (同义) → 额外加分
> ROUGE-L: 最长公共子序列 "eggs" → 有分
> ```
>
> ---
>
> **二、时间定位指标（评价章节边界）**
>
> | 指标 | 含义 | 大白话解释 |
> |------|------|-----------|
> | **R@5s** | Recall @ 5秒 | 预测的起点和真实起点**差距在5秒内**，算找对了 |
> | **R@3s** | Recall @ 3秒 | 更严格，要求差距在3秒内 |
> | **R@0.5** | Recall @ IoU=0.5 | 预测时间段和真实时间段的**重叠率≥50%** |
> | **R@0.7** | Recall @ IoU=0.7 | 更严格，重叠率要≥70% |
> | **P@Ks** | Precision | 同上，但从精确率角度算 |
>
> ```
> 例子：IoU (Intersection over Union) 怎么算？
>
> GT 时间段:     |████████████|        (0:00 - 2:00)
> 预测时间段:        |████████████████|  (0:30 - 3:00)
>                    |████████|         ← 重叠部分 (0:30 - 2:00) = 1.5分钟
>                |████████████████████| ← 并集 (0:00 - 3:00) = 3分钟
>
> IoU = 重叠 / 并集 = 1.5 / 3 = 0.5
> → R@0.5 ✅ (≥0.5)
> → R@0.7 ❌ (<0.7)
> ```
>
> ---
>
> **三、最重要的指标：SODA ⭐**
>
> | 特点 | 解释 |
> |------|------|
> | **考虑故事线** | 不只看单个章节对不对，看整体顺序是否合理 |
> | **惩罚冗余** | 生成10个重复章节？扣分！ |
> | **一对多匹配** | GT有3章，你预测5章，会找最优匹配再算分 |
>
> ```
> 为什么 SODA 最重要？
>
> 普通指标的问题：
> - 模型生成100个章节，总有几个能蒙对 → 高分！
> - 但用户体验很差（太多冗余）
>
> SODA 的解决方案：
> - 先做"最优匹配"（匈牙利算法）
> - 多余的章节不参与计分
> - 最后用 F-measure 惩罚冗余
> ```
>
> ---
>
> **四、所有指标都是越高越好！**
>
> | 指标 | 范围 | 方向 | 备注 |
> |------|------|------|------|
> | SODA | 0-100 | ↑ 越高越好 | 本文最高 11.4 |
> | BLEU | 0-100 | ↑ 越高越好 | |
> | CIDEr | 0-∞ | ↑ 越高越好 | 可以超过100 |
> | METEOR | 0-100 | ↑ 越高越好 | |
> | ROUGE-L | 0-100 | ↑ 越高越好 | |
> | R@Ks | 0-100 | ↑ 越高越好 | 召回率 |
> | P@Ks | 0-100 | ↑ 越高越好 | 精确率 |
>
> **没有"越低越好"的指标**，这些都是衡量"做得多好"的正向指标。
>
> ---
>
> **总结**：看论文结果时，**主要看 SODA (S)**，其次看 CIDEr (C)。时间定位看 R@0.5。

**Implementation details.** Unless stated otherwise, for all models, we use the speech transcripts (ASR) and visual features extracted as explained in Section 3.2. By default, each model is taken from the corresponding official implementation, and all model hyper-parameters are set according to the original papers. We use the Adam optimizer [39] for training and select the final model based on the best validation performance. Our experiments are run on 8 NVIDIA A100 80GB GPUs. More details are included in Appendix Section D.

> 💡 **实验设置**: 8×A100 80GB，用官方实现和默认超参，保证公平比较。

---

### 4.1 Video chapter generation

> 💡 **4.1 要点预览**: 最完整的任务——给一个视频，输出所有章节的 (时间边界, 标题)。这是端到端评测。

In this Section, we study the task of video chapter generation that requires temporally segmenting the video and generating a chapter title for each segment.

**Table 3: Video chapter generation (global metrics) on VidChapters-7M test set.** Here, finetuned refers to finetuning on the VidChapters-7M train set, and speech refers to transcribed speech (ASR).

| Method | Modalities | Pretraining Data | Finetuned | S | B1 | B2 | B3 | B4 | C | M | RL |
|--------|------------|------------------|-----------|-----|-----|-----|-----|-----|------|-----|------|
| Text tiling [32] + Random | Speech | ∅ | ✗ | 0.4 | 0.6 | 0.2 | 0.1 | 0.0 | 0.8 | 0.7 | 0.6 |
| Text tiling [32] + LLaMA [93] | Speech | Text mixture | ✗ | 0.2 | 0.4 | 0.1 | 0.1 | 0.0 | 0.5 | 0.3 | 0.4 |
| Shot detect [92] + BLIP-2 [51] | Visual | 129M image-texts | ✗ | 0.6 | 0.7 | 0.3 | 0.1 | 0.1 | 0.2 | 0.6 | 0.8 |
| Vid2Seq [114] | Speech+Visual | C4 + HowTo100M | ✗ | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.1 |
| PDVC [101] | Visual | ∅ | ✓ | 6.8 | 9.4 | 3.7 | 1.4 | 0.9 | 35.8 | 9.4 | 11.4 |
| Vid2Seq [114] | Speech | C4 | ✓ | 10.2 | 9.5 | 6.7 | 4.0 | 2.7 | 48.8 | 8.5 | 11.0 |
| Vid2Seq [114] | Speech | C4 + HowTo100M | ✓ | 10.5 | 9.9 | 7.0 | 4.2 | 2.9 | 50.7 | 8.7 | 11.4 |
| Vid2Seq [114] | Visual | C4 | ✓ | 3.1 | 2.3 | 1.5 | 0.6 | 0.5 | 10.9 | 2.2 | 2.9 |
| Vid2Seq [114] | Visual | C4 + HowTo100M | ✓ | 5.5 | 4.5 | 2.8 | 1.2 | 0.9 | 21.1 | 4.1 | 5.5 |
| Vid2Seq [114] | Speech+Visual | C4 | ✓ | 10.6 | 9.9 | 7.0 | 4.2 | 2.8 | 51.3 | 8.8 | 11.6 |
| **Vid2Seq [114]** | **Speech+Visual** | **C4 + HowTo100M** | ✓ | **11.4** | **10.9** | **7.7** | **4.6** | **3.1** | **55.7** | **9.5** | **12.6** |

> 💡 **Table 3 批读 (整体指标)**:
> ```
> SODA 得分排行:
> ├── Zero-shot: 0.1-0.6 (几乎不工作)
> ├── PDVC (Visual): 6.8
> ├── Vid2Seq (Speech): 10.5
> ├── Vid2Seq (Visual): 5.5  ← Speech 是 Visual 的 2 倍！
> └── Vid2Seq (S+V): 11.4 ⭐ SOTA
> ```
> **关键发现**: 
> - Zero-shot 完全失败 (SODA < 1)
> - 必须在 VidChapters-7M 上 finetune
> - Speech >> Visual (10.5 vs 5.5)
> - 多模态融合有小幅提升 (10.5 → 11.4)

**Table 4: Video chapter generation (segmentation metrics) on VidChapters-7M test set.**

| Method | Modalities | Pretraining Data | Finetuned | R@5s | R@3s | R@0.5 | R@0.7 | P@5s | P@3s | P@0.5 | P@0.7 |
|--------|------------|------------------|-----------|------|------|-------|-------|------|------|-------|-------|
| Text tiling [32] | Speech | ∅ | ✗ | 9.4 | 5.8 | 23.6 | 8.9 | 12.6 | 7.9 | 26.0 | 8.8 |
| Shot detect [92] | Visual | ∅ | ✗ | 31.2 | 27.4 | 24.9 | 12.5 | 33.2 | 29.7 | 18.0 | 8.7 |
| Vid2Seq [114] | Speech+Visual | C4 + HowTo100M | ✗ | 10.7 | 9.5 | 5.8 | 0.2 | 23.3 | 18.5 | 1.9 | 0.8 |
| PDVC [101] | Visual | ∅ | ✓ | 21.1 | 17.8 | 31.2 | 22.5 | 45.3 | 40.2 | 47.2 | 26.9 |
| Vid2Seq [114] | Speech | C4 | ✓ | 37.8 | 29.5 | 44.6 | 26.1 | 29.0 | 23.0 | 38.0 | 23.4 |
| Vid2Seq [114] | Speech | C4 + HowTo100M | ✓ | 36.7 | 28.9 | 46.5 | 27.2 | 29.5 | 23.3 | 40.4 | 24.8 |
| Vid2Seq [114] | Visual | C4 | ✓ | 35.3 | 26.4 | 23.6 | 8.7 | 17.9 | 13.6 | 17.2 | 7.1 |
| Vid2Seq [114] | Visual | C4 + HowTo100M | ✓ | 33.5 | 25.0 | 33.0 | 14.5 | 19.5 | 14.7 | 26.2 | 12.5 |
| Vid2Seq [114] | Speech+Visual | C4 | ✓ | 36.3 | 28.6 | 45.8 | 26.9 | 29.9 | 23.8 | 40.9 | 24.9 |
| **Vid2Seq [114]** | **Speech+Visual** | **C4 + HowTo100M** | ✓ | **36.4** | **28.5** | **48.2** | **28.5** | **30.3** | **24.0** | **43.1** | **26.4** |

> 💡 **Table 4 批读 (分割指标)**:
> - **PDVC 精度最高** (P@0.5=47.2)，但召回低——保守预测，漏掉很多章节
> - **Vid2Seq 召回最高** (R@0.5=48.2)——更激进，找到更多章节
> - **Shot detect** 在纯分割任务上还行 (R@5s=31.2)，但标题生成差
> - **权衡**: Vid2Seq 在 recall-precision 上更均衡

**Models.** For the video chapter segmentation subtask, we evaluate two zero-shot approaches (i.e., that are not trained on VidChapters-7M): speech text tiling [32], which detects subtopic shifts based on the analysis of lexical co-occurrence patterns, and a visual scene change detection algorithm [92] based on the sum of absolute differences. To derive zero-shot baselines for the full video chapter generation task, we combine text tiling and shot detection with various alternatives that can generate text given text or visual input: a random baseline that predicts a random speech sentence spoken inside the predicted boundaries, LLaMA-7B [93] (prompted to summarize the speech transcript spoken inside the predicted boundaries) and BLIP-2 [51] (prompted to describe the middle video frame of the predicted segment). Finally, we also train and evaluate two state-of-the-art end-to-end dense video captioning models on VidChapters-7M: PDVC [101] which consists of a visual-only DETR-style [11] architecture and Vid2Seq [114] which is a multi-modal sequence-to-sequence model pretrained on the C4 text corpus [74] and on narrated videos with ASR (e.g., YT-Temporal-1B [118]). For Vid2Seq, we also report zero-shot results after pretraining on narrated videos without finetuning on VidChapters-7M.

> 💡 **Baseline 方法总结**:
> | 方法 | 分割 | 标题生成 | 特点 |
> |------|------|----------|------|
> | Text tiling + Random | 语音主题切换 | 随机选句子 | 最简单 baseline |
> | Text tiling + LLaMA | 语音主题切换 | LLM 总结 | LLM 不会写"标题" |
> | Shot detect + BLIP-2 | 镜头切换 | 图像描述 | 纯视觉方案 |
> | PDVC | 端到端 | 端到端 | DETR 风格，纯视觉 |
> | Vid2Seq | 端到端 | 端到端 | T5 + 多模态，⭐最强 |

**Implementation details.** We use the text tiling implementation from the NLTK library [9] which tokenizes the text into pseudosentences of size 50. We use the shot detection software from the FFMPEG library [92] with a confidence threshold of 0.7. For BLIP-2, we use the 3.4B-parameter variant with FLAN-T5-XL [106] and CLIP ViT-L/14 [20, 72]. We reimplement Vid2Seq [114] (originally released in Jax) in PyTorch, use T5-Base pretrained on C4 [74] for initialization and pretrain Vid2Seq on HowTo100M [64]. More details are included in Appendix Section D.

**Results.** We report the results for video chapter generation using global metrics and localization-only metrics in Tables 3 and 4, respectively. We observe that models trained on VidChapters-7M outperform zero-shot baselines, demonstrating the effectiveness of training on VidChapters-7M. In particular, PDVC [101] has the best precision and Vid2Seq [114] achieves the best results in terms of overall generation and recall. We also find that Vid2Seq's speech-only mode outperforms its visual-only mode and that using both speech and visual inputs leads to the best performance. This demonstrates that video chapter generation is a multi-modal task. Finally, we observe that pretraining using ASR in narrated videos from HowTo100M [64] improves the video chapter generation performance of the Vid2Seq model. Specifically, pretraining on HowTo100M is more beneficial for vision-aware models than for the speech-only model.

> 💡 **4.1 小结**:
> 1. **必须 finetune**: Zero-shot 完全失败
> 2. **Speech >> Visual**: 语音模态重要性是视觉的 2 倍
> 3. **多模态有帮助**: S+V > S > V
> 4. **HowTo100M 预训练有用**: 特别是对视觉模态

---

### 4.2 Video chapter generation given ground-truth boundaries

> 💡 **4.2 要点预览**: 简化版任务——假设时间边界已知，只需要生成标题。测试模型的"语言生成"能力。

In this Section, we study the task of generating chapter titles provided correct temporal boundaries of video chapters. This task is a simplification of the previously studied task where we assume perfect temporal segmentation. We adopt the same models and implementation details as previously introduced in Section 4.1.

**Table 5: Chapter title generation given ground-truth boundaries on VidChapters-7M test set.**

| Method | Modalities | Pretraining Data | Finetuned | B1 | B2 | B3 | B4 | C | M | RL |
|--------|------------|------------------|-----------|-----|-----|-----|-----|-------|-----|------|
| Random | Speech | ∅ | ✗ | 2.4 | 1.3 | 0.9 | 0.7 | 10.4 | 2.2 | 4.4 |
| LLaMA [93] | Speech | Text mixture | ✗ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.2 |
| BLIP-2 [51] | Visual | 129M image-texts | ✗ | 3.1 | 1.5 | 0.9 | 0.7 | 12.4 | 2.2 | 4.5 |
| Vid2Seq [114] | Speech+Visual | C4 + HowTo100M | ✗ | 2.0 | 1.2 | 0.9 | 0.6 | 0.9 | 0.3 | 0.6 |
| Vid2Seq [114] | Speech | C4 + HowTo100M | ✓ | 21.0 | 15.5 | 12.1 | 10.0 | 105.3 | 11.5 | 24.5 |
| Vid2Seq [114] | Visual | C4 + HowTo100M | ✓ | 10.1 | 5.6 | 3.5 | 2.4 | 47.1 | 5.1 | 14.7 |
| Vid2Seq [114] | Speech+Visual | C4 | ✓ | 21.6 | 15.7 | 12.3 | 10.0 | 110.8 | 11.5 | 26.0 |
| **Vid2Seq [114]** | **Speech+Visual** | **C4 + HowTo100M** | ✓ | **23.5** | **17.2** | **13.4** | **11.0** | **120.5** | **12.6** | **28.3** |

> 💡 **Table 5 批读**:
> ```
> 惊人发现: LLaMA 比随机还差！
> 
> CIDEr 得分:
> ├── Random: 10.4
> ├── LLaMA: 0.0  ← 完全失败！
> ├── BLIP-2: 12.4
> ├── Vid2Seq (S): 105.3
> ├── Vid2Seq (V): 47.1
> └── Vid2Seq (S+V): 120.5 ⭐
> ```
> **为什么 LLaMA 失败？**
> - LLaMA 生成的是**详细描述**，不是**简洁标题**
> - 输入: "Right, we're gonna do the Synthetics Dirty Race..."
> - LLaMA 输出: "The video discusses the process of doing a synthetic dirty race..."
> - GT 标题: "Laundry Tips"
> - **结论**: 需要在 VidChapters-7M 上学习"标题风格"

**Results.** We report results for video chapter generation given ground-truth boundaries in Table 5. Similar to the full video chapter generation task, we observe that solving the task without training on VidChapters-7M is hard. Indeed, LLaMA [93] struggles to summarize the speech content into a chapter title and underperforms the random baseline. Furthermore, BLIP-2 [51] slightly improves over the random baseline. In addition, Vid2Seq [114] in zero-shot mode underperforms the random baseline due to the large domain gap between ASR and chapter titles (see Section 3.3). In comparison, the performance of models trained on VidChapters-7M is significantly higher. Moreover, Vid2Seq's speech-only mode outperforms its visual-only mode, and using both speech and visual inputs is beneficial, confirming the benefit of multi-modal reasoning for the task of generating chapter titles. Finally, pretraining on narrated videos from HowTo100M [64] improves the performance of the Vid2Seq model on VidChapters-7M.

> 💡 **4.2 小结**:
> 1. **LLM ≠ 章节生成器**: 通用 LLM 不会写"章节标题"这种格式
> 2. **领域差距大**: ASR 是冗长口语，章节标题是精炼书面语
> 3. **Speech 仍然 > Visual**: 105.3 vs 47.1 (CIDEr)

---

### 4.3 Video chapter grounding

> 💡 **4.3 要点预览**: 反向任务——给定章节标题，找到对应的时间位置。类似"视频搜索"。

In this Section, we study the task of video chapter grounding that requires a model to temporally localize a chapter start time (or start-end window) given an annotated chapter title (query). Hence, compared to the video chapter generation task, we here assume chapter titles to be given and focus on the temporal chapter localization only.

**Table 6: Video chapter grounding on VidChapters-7M test set.**

| Method | Modalities | Pretraining Data | Finetuned | R@10s | R@5s | R@3s | R@1s | R@0.3 | R@0.5 | R@0.7 | R@0.9 |
|--------|------------|------------------|-----------|-------|------|------|------|-------|-------|-------|-------|
| Random | Speech | ∅ | ✗ | 3.1 | 1.8 | 1.2 | 0.6 | 0.7 | 0.3 | 0.1 | 0.0 |
| BERT [19] | Speech | BookCorpus + Wikipedia | ✗ | 9.0 | 6.8 | 5.4 | 2.9 | 0.6 | 0.3 | 0.1 | 0.0 |
| CLIP [72] | Visual | 400M image-texts | ✗ | 8.1 | 5.2 | 3.7 | 1.4 | 10.7 | 5.2 | 2.3 | 0.5 |
| Moment-DETR [45] | Visual | 5.4K narrated videos [45] | ✗ | 3.2 | 1.6 | 1.1 | 0.5 | 11.3 | 3.6 | 0.8 | 0.1 |
| **Moment-DETR [45]** | **Visual** | **∅** | ✓ | **21.8** | **15.5** | **12.4** | **8.3** | **37.4** | **27.3** | **17.6** | **6.4** |

> 💡 **Table 6 批读**:
> ```
> 找起点 (R@Ks) vs 找区间 (R@IoU):
> 
> BERT (语音匹配):
> ├── R@5s: 6.8  ← 找起点还行
> └── R@0.5: 0.3 ← 找区间很差
> 
> CLIP (视觉匹配):
> ├── R@5s: 5.2  ← 找起点差
> └── R@0.5: 5.2 ← 找区间好一点
> 
> Moment-DETR (训练后):
> ├── R@5s: 15.5 ⭐
> └── R@0.5: 27.3 ⭐
> ```
> **洞察**:
> - BERT 用语音嵌入匹配，能找到大概位置，但边界不准
> - CLIP 用视觉匹配，边界估计更好
> - 训练后的 Moment-DETR 两项都最好
> - **遗憾**: Moment-DETR 只支持视觉，不能用语音

**Models.** We evaluate three zero-shot alternatives: a random baseline that randomly picks the timestamps of a speech sentence in the video, a BERT [19] baseline that picks the timestamps of the speech sentence that has the closest text embedding with the queried chapter title, and a CLIP [72] baseline picking the frames where the query-frame similarity score drops from the highest scoring frame by a certain threshold ε. We also train and evaluate on VidChapters-7M a state-of-the-art end-to-end video grounding model: Moment-DETR [45] which is designed for moment retrieval based on visual inputs. Furthermore, we report zero-shot performance of Moment-DETR obtained with the model checkpoint from Lei et al. [45] pretrained on 5.4K narrated videos with ASR from the QVHighlights dataset [45].

**Implementation details.** We use the [CLS] token sequence embedding for the BERT baseline and a threshold of ε = 0.05 for the CLIP baseline. More details are provided in Appendix Section D.

**Results.** We report results for the video chapter grounding task in Table 6. We first observe that the simple zero-shot baselines based on ASR can decently find start times, but struggle to predict start-end windows due to the important domain gap between ASR and video chapters (see Section 3.3). The CLIP [72] baseline slightly underperforms the BERT baseline [19] at retrieving start times, but is much better at finding start-end windows. Furthermore, the Moment-DETR model [45] trained on VidChapters-7M outperform the zero-shot baselines for both localization of start times and start-end windows, which further demonstrates the effectiveness of training on VidChapters-7M. Finally, we note that Moment-DETR cannot handle speech inputs, but hope that our results showing the benefit of this modality on other tasks in VidChapters-7M will foster research in the localization of language queries in untrimmed videos using multi-modal inputs (vision and speech transcripts).

> 💡 **4.3 小结**:
> 1. **起点 vs 区间**: 找起点容易，找准确区间难
> 2. **语音 vs 视觉互补**: BERT 找起点好，CLIP 找区间好
> 3. **研究空白**: 缺乏多模态 grounding 模型

---

### 4.4 Transfer learning on dense video captioning

> 💡 **4.4 要点预览**: 验证 VidChapters-7M 的预训练价值——在 YouCook2/ViTT 上能提升多少？

In this Section, we investigate the pretraining of video-language models on our new VidChapters-7M. To this end, we adopt video chapter generation models trained on VidChapters-7M (see Section 4.1) to the tasks of dense video captioning with or without finetuning.

**Datasets.** We use two dense video captioning datasets. YouCook2 [127] has 2K untrimmed videos of cooking procedures. On average, each video lasts 320s and is annotated with 7.7 temporally-localized sentences. ViTT [36] was created to better reflect the distribution of instructional videos in the wild compared to YouCook2, and consists of 8K untrimmed instructional videos. On average, each video lasts 250s and is annotated with 7.1 temporally-localized short tags. For both datasets, we extract speech transcripts and visual features as described in Section 3.2, and follow the standard splits for training, validation and testing. Note that we only use videos available on YouTube at the time of the work, resulting in 10 to 20% less videos than in the original datasets.

> 💡 **下游数据集**:
> | 数据集 | 视频数 | 平均时长 | 标注数 | 特点 |
> |--------|--------|----------|--------|------|
> | YouCook2 | 2K | 5.3 min | 7.7 句 | 烹饪教程 |
> | ViTT | 8K | 4.2 min | 7.1 tags | 更多样的教程 |

**Implementation details.** See Section 4.1 and Appendix Section D.

**Table 7: Comparison with the state of the art on the YouCook2 and ViTT dense video captioning benchmarks.** T: Transcribed speech, V: Visual, HTM: HowTo100M [64], VC: VidChapters-7M, Chap.: Chapters. † denote results of our experiments.

| Method | Modalities | Pretraining Data | YouCook2 S | YouCook2 C | ViTT S | ViTT C |
|--------|------------|------------------|------------|------------|--------|--------|
| PDVC [101] | V | ∅ | 4.4 | 22.7 | - | - |
| Vid2Seq [114] | T+V | C4 + HTM | 8.3 | 48.3 | - | - |
| PDVC† | V | ∅ | 4.8 | 28.8 | 9.4 | 40.6 |
| PDVC† | V | VC (Chap.) | 5.9 | 34.7 | 10.1 | 41.5 |
| Vid2Seq† | T+V | C4 + HTM | 8.6 | 53.2 | 14.1 | 44.8 |
| Vid2Seq† | T+V | C4 + VC (ASR+Chap.) | 9.8 | 62.9 | 15.1 | 50.9 |
| Vid2Seq† | T+V | C4 + HTM + 10% VC | 9.9 | 63.9 | 14.5 | 47.4 |
| **Vid2Seq†** | **T+V** | **C4 + HTM + VC** | **10.3** | **67.2** | **15.0** | **50.0** |

> 💡 **Table 7 批读 (Finetune 迁移)**:
> ```
> YouCook2 CIDEr 提升路径:
> ├── PDVC (scratch): 28.8
> ├── Vid2Seq (HTM): 53.2
> ├── + VidChapters: 62.9 (+9.7)
> └── + HTM + VC: 67.2 (+14) ⭐ 新 SOTA
> 
> 提升来源:
> ├── HowTo100M: 学习 ASR 理解
> └── VidChapters: 学习语义分割 + 标题生成
> ```
> **Scaling 规律**:
> | 预训练数据 | YouCook2 CIDEr |
> |-----------|----------------|
> | HTM only | 53.2 |
> | HTM + 1% VC | 52.7 |
> | HTM + 10% VC | 63.9 |
> | HTM + 100% VC | **67.2** |
> 
> → 数据越多，迁移效果越好！

**Results after finetuning.** In Table 7, we show that pretraining for video chapter generation on VidChapters-7M greatly improves the downstream dense video captioning performance compared to training from scratch or pretraining only with ASR data as done in previous work [114]. We also find that pretraining both on HowTo100M [64] and VidChapters-7M results in the best overall performance. In particular, the Vid2Seq model pretrained on both HowTo100M and VidChapters-7M largely improves the state of the art on both the YouCook2 and ViTT benchmarks. In detail, on the YouCook2 benchmark, in the setting with C4 + HowTo100M pretraining, we observe that a boost of about 4.9 points in CIDEr is obtained with our reimplementation of Vid2Seq, and that 14.0 additional points in CIDEr are obtained by pretraining on VidChapters-7M. Finally, we report the results of the Vid2Seq model after pretraining on different fractions of VidChapters-7M for a fixed number of iterations. We construct these subsets such that larger subsets include the smaller ones. These results suggest that the scale of the chapter dataset is an important factor in the downstream dense video captioning performance. We conclude that VidChapters-7M opens a promising avenue for multi-modal pretraining. We further show qualitative examples of dense video captioning in Appendix Section B.

**Table 8: Zero-shot dense video captioning on the YouCook2 and ViTT benchmarks.**

| Method | Modalities | Pretraining Data | YouCook2 S | YouCook2 C | ViTT S | ViTT C |
|--------|------------|------------------|------------|------------|--------|--------|
| Text tiling + Random | T | ∅ | 0.3 | 0.9 | 0.3 | 0.6 |
| Shot detect + BLIP-2 | V | 129M | 0.6 | 1.0 | 0.2 | 0.1 |
| Vid2Seq | T+V | C4 + HTM | 0.0 | 0.1 | 0.0 | 0.0 |
| Vid2Seq | T+V | C4 + VC (ASR+Chap.) | 3.2 | 10.2 | 9.1 | 30.2 |
| **Vid2Seq** | **T+V** | **C4 + HTM + VC** | **3.9** | **13.3** | **9.0** | **28.0** |

> 💡 **Table 8 批读 (Zero-shot 迁移)**:
> ```
> Zero-shot = 不在 YouCook2/ViTT 上训练，直接测试
> 
> 关键发现:
> ├── HTM 预训练: 0.1 CIDEr (几乎不工作)
> ├── VC (ASR only): 也不工作
> ├── VC (Chap only): 也不工作  
> └── VC (ASR + Chap): 10.2 CIDEr ⭐
> ```
> **为什么必须同时用 ASR + Chapter？**
> - 只用 ASR: 模型不知道怎么"分割"
> - 只用 Chapter: 模型不知道怎么用语音输入
> - 两者结合: 学会"听语音→分割→生成标题"

**Zero-shot dense video captioning.** In Table 8, we report results obtained by directly applying video chapter generation models trained on VidChapters-7M for dense video captioning without finetuning for this task. As far as we know, our work is the first to explore this challenging zero-shot setting where no manual annotation of dense video captions is used for training. The Vid2Seq model trained only using ASR data underperforms the random baseline, due to the large domain difference between speech transcripts and dense captions [114]. In the visual-only setting, the variant trained on chapter annotations is better than the variant trained on ASR annotations. In the visual+speech settings, only using chapter annotations does not perform well, as training only on chapters (i.e., without speech) does not enable the model to learn how to use the input speech modality at inference. However, using both ASR and chapter annotations results in a largely better zero-shot dense video captioning performance and outperforms all baselines not trained on VidChapters-7M, demonstrating the complementary nature of the ASR and chapters annotations. Finally, we also observe the benefits of increasing the size of the pretraining dataset of chapters in this setting.

> 💡 **4.4 小结**:
> 1. **VidChapters-7M 预训练价值巨大**: YouCook2 +14 CIDEr
> 2. **ASR + Chapter 互补**: 缺一不可
> 3. **Scaling 有效**: 数据越多越好
> 4. **首个 zero-shot dense captioning 探索**

---

## 💡 Section 4 总结

### 四个任务的核心结论

| Task | 最佳方法 | 关键发现 |
|------|---------|----------|
| 4.1 完整生成 | Vid2Seq (S+V) | Speech >> Visual (2x) |
| 4.2 给边界生标题 | Vid2Seq (S+V) | LLaMA 不会写"标题" |
| 4.3 给标题找位置 | Moment-DETR | 缺多模态 grounding 模型 |
| 4.4 迁移学习 | Vid2Seq (HTM+VC) | +14 CIDEr on YouCook2 |

### 为什么 Speech > Visual？

1. **数据证据** (Table 2): 75% 章节需要理解语音
2. **实验证据** (Table 3): SODA 10.5 vs 5.5
3. **直觉解释**: 23 分钟语音 >> 23 分钟 1FPS 静态帧

### 预训练数据的组合效果

```
C4 (文本理解)
  + HowTo100M (ASR 对齐)
    + VidChapters-7M (章节标注)
      = 最佳迁移性能
```

### 任务难度排序

```
难 ← Task 1 (Full) > Task 3 (Grounding) > Task 2 (GT Title) → 易
     需要分割+生成   需要定位           只需要生成
```
