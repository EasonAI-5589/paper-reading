# 4.1. Data and evaluation

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

Data. We train and evaluate on the recently released VidChapters-7M [112] dataset that includes user-annotated chaptered videos sourced from YouTube. Speech transcripts are obtained using Whisper [73] as the ASR method. In the original release, there is a total of 817k videos, spanning 8M chapters, with 2.4 minutes per chapter and 5.4 words per chapter title, totaling to 23 minutes and 8.3 chapters per video on average. Data is split into 801k training, 8.2k validation, and $8 . 2 \mathrm { k }$ test videos. To measure performance at different video lengths, we define three categories depending on video duration: ‘short’ (0-15min), ‘medium’ (15-30min), and ‘long’ (30-60min) videos. In this work, we use a subset of the training data as we observe increasing the training set brings diminishing returns at the cost of extended training times (see Fig. 4). Specifically, we use about $2 0 \mathrm { k }$ training videos (10k short videos used for the speech-based frame selection model and another 10k videos evenly split across short, medium and long durations for the final model). For stateof-the-art comparisons (Sec. 4.2), we employ the full official test set, which also contains videos without any speech $2 . 5 \%$ of the videos), and videos longer than 60 minutes (e.g., there are few videos that last about 12 hours). In ablations (Sec. 4.3), both for faster experimentation, and to limit the use of the test set during experimentation, we train on a randomly sampled subset of 1k videos (evenly split between short, medium, and long) and report results on a randomly sampled subset of 300 validation videos (100 from each duration) that have at least one speech utterance.

Evaluation metrics. We primarily monitor temporal segmentation metrics to evaluate our chapter boundary detections. In particular, we employ tIoU and F1 scores. For tIoU (temporal Intersection over Union), we first compute the optimal matching between predicted and ground truth segments by greedily selecting pairs with highest IoU scores. The tIoU score is then calculated as the mean IoU across all matched pairs, multiplied by 100 to obtain a percentage. For F1 score, we first compute precision and recall at different IoU thresholds (ranging from 0.5 to 0.95 with a step of 0.05). At each threshold, a prediction is considered correct if it has IoU above the threshold with a ground truth segment. The precision is the ratio of correct predictions to total predictions, while recall is the ratio of matched ground truth segments to total ground truth segments. The F1 score is then computed as the harmonic mean of precision and recall. The final F1 metric is the average across all thresholds, multiplied by 100 to obtain a percentage. Note that [112] uses recall and precision metrics in two ways: (1)

by considering timestamps within 3 or 5 second thresholds as matches, and (2) by considering segments with IoU above 0.5 or 0.7 as matches. While these metrics provide point estimates at specific thresholds, we find that tIoU and F1 scores offer several advantages: they evaluate performance continuously across multiple thresholds, are more interpretable, and provide a more comprehensive evaluation of the model. For completeness, we also report the metrics used in [112] in Appendix C.

For chapter title evaluation, we follow [112] and report SODA (S) [26] and CIDEr (C) [97], which measure the quality of the titles for the predicted segments that match to the ground segments (see [112] for details).

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
