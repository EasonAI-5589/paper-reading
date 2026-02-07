# 4.3. Ablation studies

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In the following, we experiment with (i) the contribution of speech and caption modalities, along with the effect of LLM finetuning, (ii) the effect of our frame selection method for captioning, (iii) the amount of training data, and (iv) the use of frame embeddings instead of captions. As mentioned above, we use 1k training and 300 validation videos for these ablations.

Modalities and LLM finetuning. In Tab. 2, we ablate the impact of finetuning the LLM and the contribution of each of the speech and caption modalities. In the top block, we run our baselines in zero-shot setting as introduced in the previous section. The speech-only baseline outperforms the captions-only baseline by a large margin in the zero-shot setting. This suggests that speech contains more relevant information for chaptering, as was previously observed by [112].

As shown in the bottom block of Tab. 2, we observe large performance improvements when finetuning the LLM, as opposed to zero-shot. We hypothesize that zero-shot prompting with a long multi-modal text, potentially containing redundant and irrelevant information, may overwhelm the LLM [82, 104]. We obtain our best model by combining the two modalities, which performs better than the individual speech-only or caption-only models. This demonstrates the multi-modal capabilities of our model.

Speech-based frame selection. In Tab. 3, we examine a number of strategies to sample frames at which we extract captions. In addition to previously described metrics, for each of the frame sampling approaches, we report the average number of captions per video and the average number of text tokens per minute. For reference, we also report an off-the-shelf shot detection [12] and Vid2Seq [112, 113].

We compare our speech-based frame selection strategy to various baselines. We experiment with sampling (i) uniformly

<table><tr><td colspan="2">Method</td><td>Frame selection average for captions</td><td>#frames</td><td>#tokens per min.</td><td colspan="2">|Segmentation F1 tIoU</td><td colspan="2">Titles s</td></tr><tr><td colspan="9">BasELInES</td></tr><tr><td colspan="2">Shot detection [12] n/a</td><td></td><td>49.4 100.0</td><td>n/a 128.6</td><td>6.2 25.4</td><td>37.6 57.8</td><td>-</td><td>- 11.2 55.0</td></tr><tr><td colspan="9">Vid2Seq [112, 113] 100 equidistant Chapter-Llama Variants</td></tr><tr><td colspan="2">Speech Caption X</td><td>n/a</td><td>n/a</td><td>248.6</td><td>38.5</td><td>68.1</td><td></td><td>13.9 67.3</td></tr><tr><td colspan="9">✓</td></tr><tr><td rowspan="5">X</td><td rowspan="5">✓</td><td>100 equidistant</td><td>100.0</td><td>449.1</td><td>| 21.0</td><td>53.8</td><td>8.4</td><td>36.0</td></tr><tr><td>Every 10 sec.</td><td>83.1</td><td>280.3</td><td>12.8</td><td>45.9</td><td></td><td>4.3 13.0</td></tr><tr><td>Shot boundaries</td><td>49.4</td><td>193.2</td><td>16.2</td><td>50.7</td><td>3.9</td><td>12.4</td></tr><tr><td>10 equidistant</td><td>10.0</td><td>41.8</td><td>11.0</td><td>46.4</td><td>3.6</td><td>9.0</td></tr><tr><td>Speech-based</td><td>10.3</td><td>36.2</td><td>39.1</td><td>67.7</td><td></td><td>5.9 20.2</td></tr><tr><td rowspan="5">✓</td><td rowspan="5">✓</td><td>100 equidistant</td><td>100.0</td><td>746.2</td><td>| 39.2</td><td>67.4</td><td>16.1</td><td>83.8</td></tr><tr><td>Every 10 sec.</td><td>83.1</td><td>570.1</td><td>41.0</td><td>69.3</td><td></td><td>15.4 77.3</td></tr><tr><td>Shot boundaries</td><td>40.4</td><td>481.7</td><td>40.6</td><td>69.1</td><td></td><td>15.8 79.3</td></tr><tr><td>10 equidistant</td><td>10.0</td><td>326.1</td><td>40.1</td><td>67.9</td><td></td><td>15.8 77.5</td></tr><tr><td>Speech-based</td><td>10.3</td><td>320.4</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr></table>

Table 3. Frame selection strategies for captioning: We evaluate different approaches for selecting frames to extract captions from, comparing our speech-based selection method against baselines. The table shows results for models trained on 1k videos and evaluated on 300 validation videos. We experiment with using speech only, captions only, and both modalities (bottom section). For caption extraction, we compare our speech-based approach to other alternatives such as equidistant sampling (100 or 10 frames), uniformly sampling every 10 seconds, or sampling at shot boundaries using [12]. Our speech-based frame selection achieves the best overall performance (F1: 42.6, tIoU: 70.6) while requiring significantly fewer number of frames on average (10.3) compared to other sampling approaches. The tokens-per-minute statistic shows the total input length including both speech transcriptions and captions, excluding the fixed prompt template.

100 frames as in Vid2Seq, (ii) every 10 seconds, (iii) at shot boundaries detected by an off-the-shelf shot detector [12], (iv) 10 equidistant frames to be similar to our speech-based locations (i.e., 10.0 vs 10.3 number of frames on average), and (v) sampling at frames predicted as chapter boundaries by our LLM that inputs only speech. In all cases, we limit the maximum number of frames to 100 as in [112, 113] to handle extreme durations.

In both caption-only and caption+speech settings, our speechbased frame selection approach achieves better segmentation results than the more frame-expensive baselines ‘100 equidistant’, ‘every $1 0 \mathrm { s e c } ^ { \prime }$ , and ‘shot boundaries’, while using much less frames, and also improves over the ‘10 equidistant’ baseline which uses a similar number of frames. This demonstrates the effectiveness of our speech-based frame selection strategy.

For reference, we also report positive comparison against shot detection and Vid2Seq [112, 113]. Note Vid2Seq has less #tokens per min. compared to our 100 equidistant variants, because Vid2Seq uses a different timestamp tokenizer in the input.

Amount of training data. Given the large-scale nature of the VidChapters-7M training set, we investigate how much chapter data is needed for LoRA finetuning the LLM. We plot the performance against the number of training videos in Fig. 4. We start by the zero-shot baseline as the first data point, and report our method with 1k, 5k, 7k, and 10k videos, split evenly between three durations. We see that after increasing above several thousand training videos starts to bring diminishing returns. We therefore keep 10k training videos for our final LLM, which makes our approach highly efficient to train (40min on 4 H100 GPUs). Note that here we focus on the chaptering LLM and always use frame sampling locations from a speech-based module trained on 10k separate videos.

![](images/4caa2834498d5b63bf68558dd8184414cd0efacabc3aad878d8517146bd791ac.jpg)  
Figure 4. Amount of training data: Our experiments show a substantial improvement when moving from zero-shot to training with 1k videos. Beyond 1k videos, performance continues to improve but at a much slower rate, motivating our choice of using only 10k training videos for our final LLM.

Frame embeddings vs captions. In Tab. 4, we investigate whether raw visual embeddings could serve as an alternative to textual captions. To this end, we experiment with replacing the captions with frame embeddings. Specifically, for each frame, we extract the 1152-dimensional output embedding corresponding to the [CLS] token from a frozen SigLIP model [122], and feed through a 2-layer MLP mapping network. We initialize the MLP weights from MANTIS [42] and train jointly with the LLM during finetuning. The results with ‘Speech+Embeddings’ are better than ‘Speech’ alone (38.5 vs 40.4 F1), but worse than ‘Speech+Captions’ (42.6 vs 40.4 F1). The performance gap between ‘Speech+Embeddings’ and ‘Speech+Captions’ may be due to the richer information provided by captions, which use multiple tokens per frame, directly in text form, compared to the single [CLS] token frame embedding, requiring a mapping network to be ingested by an LLM. Finally, while combining all modalities achieves the best performance (44.4 F1), we exclude frame embeddings from our final model due to practical considerations, e.g., they add complexity, increase processing time by $2 . 5 \mathrm { x }$ , and require $3 0 0 0 \mathrm { x }$ more storage space.

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- Figure: 4caa2834498d5b63bf68558dd8184414cd0efacabc3aad878d8517146bd791ac.jpg

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
