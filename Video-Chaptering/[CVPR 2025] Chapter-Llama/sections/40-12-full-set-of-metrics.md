# C.12. Full set of metrics

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In Sec. 4.1 of the main paper, we adopted the evaluation metrics (F1, tIoU, SODA, and CIDEr), which we consider more suitable for assessing video chapter generation. For completeness and direct comparison with VidChapters [112], we also report results using their full set of metrics in Tabs. A.13 and A.14. The segmentation metrics include precision and recall at 3-second and 5-second thresholds, as well as at 0.5 and 0.7 IoU thresholds. The full metrics (referred to as ‘global metrics’ by [112]) comprise SODA (S) [26], BLEU (B1-B4) [67], CIDEr (C) [97], METEOR (M) [7], and ROUGE-L (RL) [51]. Our model consistently outperforms Vid2Seq [113] across all metrics.

Table A.10. Including long videos at training improves results: Training with 1k videos balanced across short, medium, and long durations (last row, ‘All’) improves performance compared to training with just 1k short videos (first row). The improvement is most pronounced for long videos $( + 2 . 5 \mathrm { F } 1 )$ ). When averaging across short/medium/long validation splits, training with all videos improves all metrics: F1 $_ { ( + 0 . 6 ) }$ , tIoU $( + 0 . 8 )$ , S $_ { ( + 0 . 6 ) }$ , and C $( + 3 . 2 )$ .   

<table><tr><td rowspan="2">Training videos</td><td colspan="4">Short (val)</td><td colspan="4">Medium (val)</td><td colspan="4">Long (val)</td><td colspan="4">All (val)</td></tr><tr><td>F1</td><td>tIoU</td><td>s</td><td>C</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>Short</td><td>49.7</td><td>75.0</td><td>21.4</td><td>112.9</td><td>38.3</td><td>67.6</td><td>13.2</td><td>61.4</td><td>37.9</td><td>66.7</td><td>12.8</td><td>63.3</td><td>42.0</td><td>69.8</td><td>15.8</td><td>79.2</td></tr><tr><td>Medium</td><td>47.5</td><td>74.6</td><td>21.3</td><td>109.8</td><td>37.9</td><td>67.5</td><td>13.2</td><td>55.6</td><td>38.3</td><td>67.0</td><td>13.3</td><td>63.5</td><td>41.2</td><td>69.7</td><td>15.9</td><td>76.3</td></tr><tr><td>Long</td><td>46.6</td><td>74.0</td><td>19.5</td><td>104.9</td><td>39.3</td><td>68.1</td><td>13.4</td><td>62.0</td><td>38.1</td><td>66.9</td><td>14.3</td><td>75.1</td><td>41.3</td><td>69.7</td><td>15.8</td><td>80.8</td></tr><tr><td>All</td><td>48.4</td><td>74.4</td><td>21.2</td><td>110.8</td><td>38.9</td><td>68.0</td><td>13.1</td><td>57.3</td><td>40.4</td><td>69.3</td><td>14.9</td><td>79.1</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr></table>

Table A.11. Oracle experiment with partial ground truth input: We evaluate the capability of Chapter-Llama in predicting chapters when provided with ground truth chapter boundaries or titles. The first scenario represents an oracle experiment for title metrics, as it predicts chapters based on known timestamps (second row). The second scenario serves as a form of video chapter grounding, i.e., given known titles to segment the boundaries (last row). The model was trained with 1k videos and evaluated with 300 videos.   

<table><tr><td>Boundaries</td><td>Titles</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>X</td><td>X</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr><tr><td>✓</td><td>X</td><td>99.1</td><td>99.7</td><td>23.8</td><td>121.4</td></tr><tr><td>X</td><td>✓</td><td>64.0</td><td>80.1</td><td>71.5</td><td>506.3</td></tr></table>

Table A.12. Performance on validation videos without ASR: We evaluate the performance of our best performing model in videos without ASR predictions (190 videos in validation). We observe that the Chapter-Llama outperforms Vid2Seq in all metrics, but the performance of both models is worse than when ASR is available.   

<table><tr><td>Method</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>Vid2Seq [113]</td><td>12.6</td><td>45.5</td><td>5.5</td><td>18.0</td></tr><tr><td>Chapter-Llama (ours)</td><td>15.5</td><td>49.6</td><td>5.0</td><td>26.3</td></tr></table>

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- 无图表

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
