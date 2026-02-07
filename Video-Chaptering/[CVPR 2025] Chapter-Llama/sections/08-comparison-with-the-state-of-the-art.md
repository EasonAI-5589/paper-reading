# 4.2. Comparison with the state of the art

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In Tab. 1, we report the performance of our model on the full VidChapters-7M test set [112] (‘All’ columns), and compare to the state of the art reported in [112], which uses Vid2Seq [113]. Moreover, we evaluate four proprietary models using our speechbased frame selection and captioning in a zero-shot manner.

We observe that our finetuned Chapter-Llama achieves substantial performance improvements across all metrics and video duration categories. (e.g., 45.3 vs 26.7 F1 and 19.3 vs 11.6 SODA compared to Vid2Seq). Notably, our improvement over Vid2Seq is more important for medium and long videos compared to short videos. Note that our final approach was trained using the subset of data detailed in the previous section, specifically $2 0 \mathrm { k }$ videos, which constitutes only $2 . 5 \%$ of the total available training data. In contrast, the baseline Vid2Seq model [113] was trained on a considerably larger dataset, utilizing both HowTo100M [59] and the entire VidChapters-7M training set.

Additionally, we report performances of our model without training on any chapter annotations (i.e., both the speech-based frame selector and the LLM are not finetuned, and run with the same prompt as in the finetuned setting). We see that our zero-shot method also achieves competitive performance (e.g., 29.5 F1), whereas Vid2Seq only trained on HowTo100M does not generalize (3.0 F1).

Finally, when zero-shot evaluating the proprietary models, GPT4-o [64] and Gemini variants [28], with our speech-based frame selection and captioning inputs, we observe competitive performances (e.g., $4 2 . 2 \ \mathrm { F } 1$ with Gemini- $1 . 5 – \mathrm { P r o } _ { , }$ ); however, our Chapter-Llama still surpasses on all metrics. Note that, due to API costs of the proprietary models, we performed their evaluation on a random $10 \%$ subset of the test set; however, we verified that the scores are similar between $10 \%$ and $100 \%$ of the test set when evaluating with Chapter-Llama.

Qualitative comparison. In Fig. 3, we provide qualitative examples comparing our method against Vid2Seq [112, 113] and our zero-shot baseline. Our predictions align well with the ground truth chapters, accurately capturing both the temporal boundaries and generating relevant titles. In contrast, Vid2Seq segments tend to be less accurate, and we also observe that it often produces repetitive titles (bottom example). The zero-shot Chapter-Llama baseline tends to generate relatively longer and

<table><tr><td>Backbone</td><td rowspan="2">Frame selection</td><td rowspan="2">Ft.</td><td colspan="4">Short</td><td colspan="4">Medium</td><td colspan="4"></td><td colspan="4">All</td></tr><tr><td></td><td>F1</td><td></td><td>tIoU</td><td>C</td><td></td><td>F1</td><td>tIoU</td><td>S</td><td>C</td><td>F1</td><td>Long tIoU</td><td>S</td><td>C</td><td>F1</td><td>tIoU S</td><td>C</td></tr><tr><td>GPT-4o-mini [64]†</td><td>Ours</td><td>X</td><td>32.1</td><td>64.5</td><td>7.2</td><td>42.4</td><td>30.5</td><td>62.3</td><td>6.1</td><td>30.6</td><td>28.0</td><td>61.0</td><td>6.0</td><td>27.3</td><td>31.2</td><td>63.6</td><td>6.8</td><td>37.8</td></tr><tr><td>GPT-4o [64]t</td><td>Ours</td><td>X</td><td>37.7</td><td>68.0</td><td>8.4</td><td>53.8</td><td>38.1</td><td>68.8</td><td>8.1</td><td>51.4</td><td>36.5</td><td>66.2</td><td></td><td>6.6 34.8</td><td>37.6</td><td>68.0</td><td>8.1</td><td>51.0</td></tr><tr><td>Gemini-2.0-Flash [28]†</td><td>Ours</td><td>X</td><td>39.9</td><td>69.2</td><td>12.0</td><td>72.8</td><td>43.8</td><td>71.4</td><td>11.2</td><td>70.3</td><td>34.9</td><td>66.2</td><td></td><td>9.0 51.6</td><td>40.2</td><td>69.3</td><td>11.4</td><td>69.7</td></tr><tr><td>Gemini-1.5-Pro [28]†</td><td>Ours</td><td>×</td><td>41.7</td><td>70.6</td><td>11.7</td><td>65.3</td><td>43.8</td><td>71.8</td><td>11.2</td><td>61.4</td><td>41.3</td><td>70.6</td><td>10.1</td><td>55.3</td><td>42.2</td><td>70.9</td><td>11.4</td><td>63.2</td></tr><tr><td>Vid2Seq [112, 113]</td><td>Equidistant</td><td>X</td><td>2.5</td><td>28.6</td><td>0.3</td><td>0.3</td><td></td><td>3.2 29.7</td><td>0.3</td><td>0.4</td><td></td><td>4.6 32.0</td><td>0.3</td><td>0.5</td><td>3.0</td><td>29.3</td><td>0.3</td><td>0.4</td></tr><tr><td>Llama 3.1-8B</td><td>Ours</td><td>X</td><td>29.9</td><td>63.4</td><td>7.1</td><td>34.5</td><td>30.6</td><td>62.7</td><td></td><td>5.4 28.1</td><td></td><td>26.6 59.3</td><td></td><td>3.6 18.9</td><td>29.5</td><td>62.5</td><td>6.2</td><td>30.7</td></tr><tr><td>Vid2Seq [112, 113]</td><td>Equidistant</td><td>✓</td><td></td><td>|33.4 63.7</td><td>15.2</td><td>74.9</td><td>19.0</td><td>53.3</td><td>7.5 31.9 |</td><td></td><td>16.7</td><td>50.8</td><td></td><td>5.9 28.4 |</td><td>| 26.7</td><td>58.6</td><td>11.6</td><td>55.8</td></tr><tr><td>Llama 3.1-8B (Chapter-Llama)</td><td>Ours</td><td>✓</td><td></td><td>45.5 72.2</td><td>20.2</td><td>103.5</td><td>46.7</td><td>72.3</td><td>18.8 98.7</td><td></td><td></td><td>41.3 69.2</td><td>15.8 91.2</td><td></td><td>45.3</td><td>71.8</td><td>19.3</td><td>100.9</td></tr></table>

Table 1. Comparison to the state of the art on VidChapters-7M test set: We split the table into (bottom) the comparison between Chapter-Llama and the state-of-the-art method Vid2Seq [113], and (top) the evaluation of proprietary models. Chapter-Llama significantly outperforms Vid2Seq trained and reported by [112] (45.3 vs 26.7 F1). Our method also achieves strong performance in zero-shot mode – without finetuning (Ft.) on any chapter annotation (29.5 F1). Furthermore, we report performance of proprietary models in such zero-shot setting, using our speech-based frame selection and captioning, and observe inferior results than Chapter-Llama (42.2 F1 with Gemini-1.5-Pro). Note that we use the full official $8 . 1 \mathrm { k }$ test set videos (‘All’), unlike in the remaining experiments that report on the validation subset. We also report the performance breakdown into short (4891), medium (1736), and long (892) test videos. Our model was trained on 10k videos balanced across short, medium and long durations. $^ \dagger$ denotes evaluation on a random $10 \%$ subset of the test set due to API costs of proprietary models.

![](images/e822fc8106c6e6554e6f028c31276f9215e03c860bcf0051f09d06e31c7aa14a.jpg)  
Figure 3. Qualitative results: We display two examples and compare our Chapter-Llama results against the ground truth (GT), as well as the zero-shot (ZS) and Vid2Seq (VS) baselines. For each example, we show the corresponding SODA (S) and CIDEr (C) scores. Our method overall shows the highest similarity with the GT, while Vid2Seq can suffer from repeated chapter titles, and zero-shot generations tend to over-segment.

Table 2. Contribution of different modalities and finetuning: Finetuning the LLM with 1k videos largely improves chaptering performance on 300 validation videos, see bottom block vs top block. In the finetuned setting, we further demonstrate the advantages of combining both modalities, i.e., transcribed speech from ASR and automatic captions extracted from video frames.   

<table><tr><td colspan="2">Modalities</td><td colspan="2">Segmentation</td><td colspan="2">Titles</td></tr><tr><td colspan="2">Speech</td><td>Captions</td><td>F1</td><td>tIoU</td><td>S C</td></tr><tr><td rowspan="3">ee-oz</td><td>X</td><td>✓</td><td>12.6</td><td>48.6 1.9 57.3</td><td>6.4</td></tr><tr><td>✓</td><td>×</td><td>22.7</td><td>4.4 6.9</td><td>19.7</td></tr><tr><td>✓</td><td>✓</td><td>29.9</td><td>63.0</td><td>33.7 20.2</td></tr><tr><td rowspan="2">Riumd</td><td>X</td><td>✓</td><td>39.1</td><td>67.7</td><td>5.9</td></tr><tr><td>✓</td><td>×</td><td>38.5</td><td>68.1 13.9</td><td>67.3</td></tr><tr><td></td><td>✓</td><td>✓</td><td>42.6</td><td>70.6</td><td>16.4 82.4</td></tr></table>

verbose chapter titles and often generates chapters that appear to be continuations of previous chapters rather than distinct segments, while also exhibiting over-segmentation issues. We provide more examples in Appendix D.

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- Figure: e822fc8106c6e6554e6f028c31276f9215e03c860bcf0051f09d06e31c7aa14a.jpg

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
