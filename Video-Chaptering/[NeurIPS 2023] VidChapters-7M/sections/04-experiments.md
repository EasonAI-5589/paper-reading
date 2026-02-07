# 4. Experiments

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

In this Section, we present the results of models on VidChapters-7M for the full video chapter generation task in Section 4.1, the task of video chapter generation given ground-truth boundaries in Section 4.2 and the video chapter grounding task in Section 4.3. Finally, we study transfer learning from video chapter generation to dense video captioning tasks in Section 4.4.

**Evaluation metrics.** To evaluate the quality of the generated chapter titles (without their positions), we use standard metrics used for visual captioning: BLEU [70] (B), CIDEr [95] (C), METEOR [7] (M) and ROUGE-L [56] (RL). To evaluate video chapter generation as a whole, including the locations of the generated chapters, we follow standard protocols used for dense video captioning, given the similar nature of the two tasks. We use the standard evaluation tool [42] which calculates matched pairs between generated events and the ground truth across IoU thresholds of {0.3, 0.5, 0.7, 0.9}, and compute captioning metrics over the matched pairs. However, these metrics do not take into account the story of the video and give high scores to methods generating many redundant chapters. Hence for an overall evaluation, we also use SODA_c [22] (S) which first tries to find a temporally optimal matching between generated and reference chapters to capture the story of a video, then computes METEOR scores for the matching and derives F-measure scores from the METEOR scores to penalize redundant chapters. To separately evaluate chapter localization, we report the recall (R@Ks, R@K) and the precision (P@Ks, P@K) across various thresholds in terms of the distance to the ground-truth start time or IoU with the ground-truth start-end window. We also report the average recall (R) and average precision (P) across IoU thresholds of {0.3, 0.5, 0.7, 0.9}.

**Implementation details.** Unless stated otherwise, for all models, we use the speech transcripts (ASR) and visual features extracted as explained in Section 3.2. By default, each model is taken from the corresponding official implementation, and all model hyper-parameters are set according to the original papers. We use the Adam optimizer [39] for training and select the final model based on the best validation performance. Our experiments are run on 8 NVIDIA A100 80GB GPUs. More details are included in Appendix Section D.

### 4.1 Video chapter generation

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

**Models.** For the video chapter segmentation subtask, we evaluate two zero-shot approaches (i.e., that are not trained on VidChapters-7M): speech text tiling [32], which detects subtopic shifts based on the analysis of lexical co-occurrence patterns, and a visual scene change detection algorithm [92] based on the sum of absolute differences. To derive zero-shot baselines for the full video chapter generation task, we combine text tiling and shot detection with various alternatives that can generate text given text or visual input: a random baseline that predicts a random speech sentence spoken inside the predicted boundaries, LLaMA-7B [93] (prompted to summarize the speech transcript spoken inside the predicted boundaries) and BLIP-2 [51] (prompted to describe the middle video frame of the predicted segment). Finally, we also train and evaluate two state-of-the-art end-to-end dense video captioning models on VidChapters-7M: PDVC [101] which consists of a visual-only DETR-style [11] architecture and Vid2Seq [114] which is a multi-modal sequence-to-sequence model pretrained on the C4 text corpus [74] and on narrated videos with ASR (e.g., YT-Temporal-1B [118]). For Vid2Seq, we also report zero-shot results after pretraining on narrated videos without finetuning on VidChapters-7M.

**Implementation details.** We use the text tiling implementation from the NLTK library [9] which tokenizes the text into pseudosentences of size 50. We use the shot detection software from the FFMPEG library [92] with a confidence threshold of 0.7. For BLIP-2, we use the 3.4B-parameter variant with FLAN-T5-XL [106] and CLIP ViT-L/14 [20, 72]. We reimplement Vid2Seq [114] (originally released in Jax) in PyTorch, use T5-Base pretrained on C4 [74] for initialization and pretrain Vid2Seq on HowTo100M [64]. More details are included in Appendix Section D.

**Results.** We report the results for video chapter generation using global metrics and localization-only metrics in Tables 3 and 4, respectively. We observe that models trained on VidChapters-7M outperform zero-shot baselines, demonstrating the effectiveness of training on VidChapters-7M. In particular, PDVC [101] has the best precision and Vid2Seq [114] achieves the best results in terms of overall generation and recall. We also find that Vid2Seq's speech-only mode outperforms its visual-only mode and that using both speech and visual inputs leads to the best performance. This demonstrates that video chapter generation is a multi-modal task. Finally, we observe that pretraining using ASR in narrated videos from HowTo100M [64] improves the video chapter generation performance of the Vid2Seq model. Specifically, pretraining on HowTo100M is more beneficial for vision-aware models than for the speech-only model.

Qualitative examples. See Appendix Section B.

### 4.2 Video chapter generation given ground-truth boundaries

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

**Results.** We report results for video chapter generation given ground-truth boundaries in Table 5. Similar to the full video chapter generation task, we observe that solving the task without training on VidChapters-7M is hard. Indeed, LLaMA [93] struggles to summarize the speech content into a chapter title and underperforms the random baseline. Furthermore, BLIP-2 [51] slightly improves over the random baseline. In addition, Vid2Seq [114] in zero-shot mode underperforms the random baseline due to the large domain gap between ASR and chapter titles (see Section 3.3). In comparison, the performance of models trained on VidChapters-7M is significantly higher. Moreover, Vid2Seq's speech-only mode outperforms its visual-only mode, and using both speech and visual inputs is beneficial, confirming the benefit of multi-modal reasoning for the task of generating chapter titles. Finally, pretraining on narrated videos from HowTo100M [64] improves the performance of the Vid2Seq model on VidChapters-7M.

### 4.3 Video chapter grounding

In this Section, we study the task of video chapter grounding that requires a model to temporally localize a chapter start time (or start-end window) given an annotated chapter title (query). Hence, compared to the video chapter generation task, we here assume chapter titles to be given and focus on the temporal chapter localization only.

**Table 6: Video chapter grounding on VidChapters-7M test set.**

| Method | Modalities | Pretraining Data | Finetuned | R@10s | R@5s | R@3s | R@1s | R@0.3 | R@0.5 | R@0.7 | R@0.9 |
|--------|------------|------------------|-----------|-------|------|------|------|-------|-------|-------|-------|
| Random | Speech | ∅ | ✗ | 3.1 | 1.8 | 1.2 | 0.6 | 0.7 | 0.3 | 0.1 | 0.0 |
| BERT [19] | Speech | BookCorpus + Wikipedia | ✗ | 9.0 | 6.8 | 5.4 | 2.9 | 0.6 | 0.3 | 0.1 | 0.0 |
| CLIP [72] | Visual | 400M image-texts | ✗ | 8.1 | 5.2 | 3.7 | 1.4 | 10.7 | 5.2 | 2.3 | 0.5 |
| Moment-DETR [45] | Visual | 5.4K narrated videos [45] | ✗ | 3.2 | 1.6 | 1.1 | 0.5 | 11.3 | 3.6 | 0.8 | 0.1 |
| **Moment-DETR [45]** | **Visual** | **∅** | ✓ | **21.8** | **15.5** | **12.4** | **8.3** | **37.4** | **27.3** | **17.6** | **6.4** |

**Models.** We evaluate three zero-shot alternatives: a random baseline that randomly picks the timestamps of a speech sentence in the video, a BERT [19] baseline that picks the timestamps of the speech sentence that has the closest text embedding with the queried chapter title, and a CLIP [72] baseline picking the frames where the query-frame similarity score drops from the highest scoring frame by a certain threshold ε. We also train and evaluate on VidChapters-7M a state-of-the-art end-to-end video grounding model: Moment-DETR [45] which is designed for moment retrieval based on visual inputs. Furthermore, we report zero-shot performance of Moment-DETR obtained with the model checkpoint from Lei et al. [45] pretrained on 5.4K narrated videos with ASR from the QVHighlights dataset [45].

**Implementation details.** We use the [CLS] token sequence embedding for the BERT baseline and a threshold of ε = 0.05 for the CLIP baseline. More details are provided in Appendix Section D.

**Results.** We report results for the video chapter grounding task in Table 6. We first observe that the simple zero-shot baselines based on ASR can decently find start times, but struggle to predict start-end windows due to the important domain gap between ASR and video chapters (see Section 3.3). The CLIP [72] baseline slightly underperforms the BERT baseline [19] at retrieving start times, but is much better at finding start-end windows. Furthermore, the Moment-DETR model [45] trained on VidChapters-7M outperform the zero-shot baselines for both localization of start times and start-end windows, which further demonstrates the effectiveness of training on VidChapters-7M. Finally, we note that Moment-DETR cannot handle speech inputs, but hope that our results showing the benefit of this modality on other tasks in VidChapters-7M will foster research in the localization of language queries in untrimmed videos using multi-modal inputs (vision and speech transcripts).

### 4.4 Transfer learning on dense video captioning

In this Section, we investigate the pretraining of video-language models on our new VidChapters-7M. To this end, we adopt video chapter generation models trained on VidChapters-7M (see Section 4.1) to the tasks of dense video captioning with or without finetuning.

**Datasets.** We use two dense video captioning datasets. YouCook2 [127] has 2K untrimmed videos of cooking procedures. On average, each video lasts 320s and is annotated with 7.7 temporally-localized sentences. ViTT [36] was created to better reflect the distribution of instructional videos in the wild compared to YouCook2, and consists of 8K untrimmed instructional videos. On average, each video lasts 250s and is annotated with 7.1 temporally-localized short tags. For both datasets, we extract speech transcripts and visual features as described in Section 3.2, and follow the standard splits for training, validation and testing. Note that we only use videos available on YouTube at the time of the work, resulting in 10 to 20% less videos than in the original datasets.

**Implementation details.** See Section 4.1 and Appendix Section D.

**Table 7: Comparison with the state of the art on the YouCook2 and ViTT dense video captioning benchmarks.** T: Transcribed speech, V: Visual, HTM: HowTo100M [64], VC: VidChapters-7M, Chap.: Chapters. † denote results of our experiments.

| Method | Modalities | Pretraining Data | YouCook2 S | YouCook2 C | YouCook2 M | YouCook2 R | YouCook2 P | ViTT S | ViTT C | ViTT M | ViTT R | ViTT P |
|--------|------------|------------------|------------|------------|------------|------------|------------|--------|--------|--------|--------|--------|
| PDVC [101] | V | ∅ | 4.4 | 22.7 | 4.7 | - | - | - | - | - | - | - |
| E2ESG [130] | T+V | C4 + WikiHow | - | 25.0 | 3.5 | 20.7 | 20.6 | - | 25.0 | 8.1 | 32.2 | 32.1 |
| Vid2Seq [114] | T+V | C4 + HTM | 8.3 | 48.3 | 9.5 | 27.1 | 27.0 | - | - | - | - | - |
| Vid2Seq [114] | T+V | C4 + YT-Temporal-1B | 7.9 | 47.1 | 9.3 | 27.9 | 27.8 | 13.5 | 43.5 | 8.5 | 42.6 | 46.2 |
| PDVC† | V | ∅ | 4.8 | 28.8 | 5.8 | 22.6 | 33.1 | 9.4 | 40.6 | 16.5 | 19.2 | 37.4 |
| PDVC† | V | VC (Chap.) | 5.9 | 34.7 | 7.5 | 28.8 | 36.4 | 10.1 | 41.5 | 16.1 | 21.3 | 37.2 |
| Vid2Seq† | T+V | C4 + HTM | 8.6 | 53.2 | 10.5 | 29.2 | 26.2 | 14.1 | 44.8 | 8.7 | 43.8 | 44.5 |
| Vid2Seq† | T+V | C4 + VC (ASR+Chap.) | 9.8 | 62.9 | 11.7 | 32.5 | 30.1 | 15.1 | 50.9 | 9.6 | 45.1 | 46.7 |
| Vid2Seq† | T+V | C4 + HTM + VC (ASR) | 8.4 | 50.1 | 10.3 | 29.7 | 26.3 | 14.3 | 45.6 | 8.8 | 43.7 | 44.9 |
| Vid2Seq† | T+V | C4 + HTM + 1% of VC (ASR+Chap) | 8.8 | 52.7 | 10.4 | 29.3 | 27.6 | 13.5 | 41.6 | 8.2 | 44.7 | 42.1 |
| Vid2Seq† | T+V | C4 + HTM + 10% of VC (ASR+Chap.) | 9.9 | 63.9 | 12.1 | 32.4 | 31.4 | 14.5 | 47.4 | 9.2 | 45.3 | 45.9 |
| **Vid2Seq†** | **T+V** | **C4 + HTM + VC (ASR+Chap.)** | **10.3** | **67.2** | **12.3** | **34.0** | **31.2** | **15.0** | **50.0** | **9.5** | **45.5** | **46.9** |

**Results after finetuning.** In Table 7, we show that pretraining for video chapter generation on VidChapters-7M greatly improves the downstream dense video captioning performance compared to training from scratch or pretraining only with ASR data as done in previous work [114]. We also find that pretraining both on HowTo100M [64] and VidChapters-7M results in the best overall performance. In particular, the Vid2Seq model pretrained on both HowTo100M and VidChapters-7M largely improves the state of the art on both the YouCook2 and ViTT benchmarks. In detail, on the YouCook2 benchmark, in the setting with C4 + HowTo100M pretraining, we observe that a boost of about 4.9 points in CIDEr is obtained with our reimplementation of Vid2Seq, and that 14.0 additional points in CIDEr are obtained by pretraining on VidChapters-7M. Finally, we report the results of the Vid2Seq model after pretraining on different fractions of VidChapters-7M for a fixed number of iterations. We construct these subsets such that larger subsets include the smaller ones. These results suggest that the scale of the chapter dataset is an important factor in the downstream dense video captioning performance. We conclude that VidChapters-7M opens a promising avenue for multi-modal pretraining. We further show qualitative examples of dense video captioning in Appendix Section B.

**Table 8: Zero-shot dense video captioning on the YouCook2 and ViTT benchmarks.** T: Transcribed speech, V: Visual, HTM: HowTo100M [64], VC: VidChapters-7M, Chap.: Chapters.

| Method | Modalities | Pretraining Data | YouCook2 S | YouCook2 C | YouCook2 M | YouCook2 R | YouCook2 P | ViTT S | ViTT C | ViTT M | ViTT R | ViTT P |
|--------|------------|------------------|------------|------------|------------|------------|------------|--------|--------|--------|--------|--------|
| Text tiling [32] + Random | T | ∅ | 0.3 | 0.9 | 0.3 | 3.8 | 6.6 | 0.3 | 0.6 | 0.6 | 11.6 | 24.4 |
| Text tiling [32] + LLaMA [93] | T | Text mixture | 0.2 | 0.6 | 0.2 | 3.8 | 6.6 | 0.2 | 0.6 | 0.5 | 11.6 | 24.4 |
| Shot detect [92] + BLIP-2 [51] | V | 129M image-texts | 0.6 | 1.0 | 0.5 | 8.9 | 5.5 | 0.2 | 0.1 | 0.2 | 3.1 | 13.7 |
| Vid2Seq [114] | V | C4 + VC (ASR) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.8 |
| Vid2Seq [114] | V | C4 + VC (Chap.) | 0.7 | 1.1 | 0.5 | 21.3 | 8.6 | 1.5 | 1.9 | 0.6 | 18.9 | 10.4 |
| Vid2Seq [114] | T+V | C4 + HTM | 0.0 | 0.1 | 0.0 | 0.5 | 0.6 | 0.0 | 0.0 | 0.0 | 0.5 | 1.0 |
| Vid2Seq [114] | T+V | C4 + VC (ASR) | 0.1 | 0.1 | 0.0 | 1.1 | 0.9 | 0.0 | 0.0 | 0.0 | 0.7 | 0.6 |
| Vid2Seq [114] | T+V | C4 + VC (Chap.) | 0.1 | 0.2 | 0.1 | 0.7 | 1.4 | 0.7 | 1.1 | 0.3 | 14.3 | 12.8 |
| Vid2Seq [114] | T+V | C4 + VC (ASR+Chap.) | 3.2 | 10.2 | 2.9 | 20.6 | 19.7 | 9.1 | 30.2 | 6.7 | 33.8 | 40.8 |
| Vid2Seq [114] | T+V | C4 + HTM + VC (ASR) | 0.0 | 0.1 | 0.0 | 1.2 | 0.9 | 0.0 | 0.0 | 0.0 | 0.8 | 0.7 |
| Vid2Seq [114] | T+V | C4 + HTM + 1% of VC (ASR+Chap.) | 2.7 | 7.2 | 2.1 | 18.1 | 17.3 | 5.5 | 15.5 | 4.3 | 31.3 | 37.1 |
| Vid2Seq [114] | T+V | C4 + HTM + 10% of VC (ASR+Chap.) | 3.2 | 11.5 | 3.0 | 19.4 | 19.2 | 6.4 | 21.6 | 5.3 | 31.0 | 38.2 |
| **Vid2Seq [114]** | **T+V** | **C4 + HTM + VC (ASR+Chap.)** | **3.9** | **13.3** | **3.4** | **22.3** | **20.1** | **9.0** | **28.0** | **6.5** | **33.7** | **40.1** |

**Zero-shot dense video captioning.** In Table 8, we report results obtained by directly applying video chapter generation models trained on VidChapters-7M for dense video captioning without finetuning for this task. As far as we know, our work is the first to explore this challenging zero-shot setting where no manual annotation of dense video captions is used for training. The Vid2Seq model trained only using ASR data underperforms the random baseline, due to the large domain difference between speech transcripts and dense captions [114]. In the visual-only setting, the variant trained on chapter annotations is better than the variant trained on ASR annotations. In the visual+speech settings, only using chapter annotations does not perform well, as training only on chapters (i.e., without speech) does not enable the model to learn how to use the input speech modality at inference. However, using both ASR and chapter annotations results in a largely better zero-shot dense video captioning performance and outperforms all baselines not trained on VidChapters-7M, demonstrating the complementary nature of the ASR and chapters annotations. Finally, we also observe the benefits of increasing the size of the pretraining dataset of chapters in this setting.

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
