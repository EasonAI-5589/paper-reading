# D.1. Evaluation metrics

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In Sec. 4.1, we introduced our primary evaluation metrics for video chaptering: tIoU and F1 scores. Here, we illustrate how these metrics are calculated using concrete examples, as shown in Fig. A.4.

For tIoU (temporal Intersection over Union), we first match predicted and ground truth segments by greedily selecting pairs with the highest IoU scores. In the top example of Fig. A.4, we have 5 ground truth chapters and 4 predicted chapters. The matching process starts with chapters having the most overlap, and each chapter can only be used once. The tIoU score (84.7) is then calculated as the mean IoU across all matched pairs (97.6, 53.6, 89.3, 98.3). Similarly, for the bottom example, the tIoU score of 49.4 is the mean of 60.7, 47.14, and 40.3.

Table A.13. Video chapter generation (segmentation metrics) on VidChapters [112] test set: Comparison of segmentation metrics between Vid2Seq and our best model from Tab. 1. Metrics include precision and recall at 3-second and 5-second thresholds, as well as at 0.5 and $0 . 7 \mathrm { I o U }$ thresholds. Our method consistently outperforms Vid2Seq across all metrics.   

<table><tr><td>Method</td><td>P@5s</td><td>R@5s</td><td>P@3s</td><td>R@3s</td><td>P@0.5</td><td>R@0.5</td><td>P@0.7</td><td>R@0.7</td></tr><tr><td>Vid2Seq [113]</td><td>30.6</td><td>36.4</td><td>24.4</td><td>28.7</td><td>46.3</td><td>51.1</td><td>28.7</td><td>30.6</td></tr><tr><td>Chapter-Llama</td><td>52.0</td><td>51.7</td><td>45.1</td><td>44.7</td><td>66.3</td><td>63.4</td><td>49.9</td><td>47.8</td></tr></table>

Table A.14. Full metrics used by VidChapters [112]: We report the full metrics (referred to as ‘global metrics’ in [112]) on the test set of VidChapters. We compare Vid2Seq and our best model from Tab. 1. Metrics include SODA [26] (S), BLEU [67] (B1-B4), CIDEr [97] (C), METEOR [7] (M), and ROUGE-L [51] (RL). Our method consistently outperforms Vid2Seq across all metrics.   

<table><tr><td>Method</td><td>S</td><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>C</td><td>M</td><td>RL</td></tr><tr><td>Vid2Seq [113]</td><td>11.6</td><td>11.1</td><td>7.7</td><td>4.5</td><td>3.1</td><td>55.8</td><td>9.6</td><td>12.8</td></tr><tr><td>Chapter-Llama</td><td>19.3</td><td>19.5</td><td>14.3</td><td>8.7</td><td>5.6</td><td>100.9</td><td>15.4</td><td>22.2</td></tr></table>

![](images/28f22e44cf818cfc4be931997bdcf447f7da0e98fab051a7d2167a036a5c900e.jpg)  
Figure A.3. Accuracy of number of chapter predictions: The violin plot shows the distribution of differences between the predicted and ground truth number of chapters for three video chaptering models: Chapter-Llama, Zero-shot, and Vid2Seq. The Chapter-Llama model exhibits the most concentrated distribution centered around 0, indicating accurate number of chapter prediction. The Zero-shot model tends to slightly overpredict the number of chapters, while the Vid2Seq model often significantly overpredicts the number of chapters. The median differences are 0, 1, and 2 for Chapter-Llama, Zero-shot, and Vid2Seq, respectively, with mean number of chapter differences of -0.2, 0.5, and 4.5 (not shown).

For the F1 score, we compute precision and recall at different IoU thresholds (from 0.5 to 0.95 with a step of 0.05). In the top example, at a threshold of 0.5, all predicted chapters have a ground truth match with an overlap higher than $50 \%$ , resulting in a precision of $100 \%$ . However, one ground truth chapter out of 5 is left without a prediction, leading to a recall of $80 \%$ . The F1 score is then computed as the harmonic mean of precision and recall. This process is repeated for all thresholds, and the final F1 metric is the average across these thresholds.

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- Figure: 28f22e44cf818cfc4be931997bdcf447f7da0e98fab051a7d2167a036a5c900e.jpg

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
