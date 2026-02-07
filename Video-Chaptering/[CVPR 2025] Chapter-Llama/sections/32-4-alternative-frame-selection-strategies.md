# C.4. Alternative frame selection strategies

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In the main paper, given a detected chapter boundary from our speech-only model, we select frames at the boundary location itself. In Tab. A.5, we explore alternative frame sampling strategies, including: (1) shot boundaries or midpoints detected with PySceneDetect [12], $( 2 ) \pm 1$ sec before and after speech-based chapter boundary predictions, (3) speechbased Chapter-Llama (CL) predicted boundary locations and midpoints between these locations. See the caption for comments.

Table A.4. Effect of modality prefixes: Adding prefixes to the ASR and captions modalities improves performance.   

<table><tr><td>Has prefix?</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td></td><td>41.9</td><td>69.6</td><td>16.0</td><td>78.5</td></tr><tr><td>✗</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr></table>

Table A.5. Alternative frame selection strategies: We evaluate alternative frame sampling strategies including: (1) shot boundaries and midpoints detected with PySceneDetect [12], (2) frames sampled $\pm 1$ second around chapter boundaries predicted by our speech-based Chapter-Llama (CL) model, (3) frames at CL predicted boundaries and midpoints between them. Results show that sampling at CL boundaries achieves competitive performance across all metrics while requiring significantly fewer frames (10.3 vs 20.6-49.4 frames per video).   

<table><tr><td>Frame selection for captions</td><td>#frames ↓</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>Shot midpoints</td><td>49.4</td><td>40.8</td><td>69.1</td><td>15.6</td><td>77.0</td></tr><tr><td>Shot boundaries</td><td>49.4</td><td>40.6</td><td>69.1</td><td>15.8</td><td>79.3</td></tr><tr><td>Speech-based CL ±1 sec</td><td>20.6</td><td>42.7</td><td>69.5</td><td>16.5</td><td>83.2</td></tr><tr><td>Speech-based CL midpoints</td><td>10.3</td><td>41.2</td><td>69.0</td><td>15.6</td><td>73.7</td></tr><tr><td>Speech-based CL boundaries</td><td>10.3</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr></table>

Table A.6. Effect of training data size on speech-based frame selector: We analyze how the amount of training data used for the speechonly frame selector (first column) affects downstream performance of our Chapter-Llama (CL) model. The frame selector is trained on either 1k or 10k videos to predict frame locations where captions should be extracted, while the CL is trained on either 1k or 10k different videos for chapter generation. Comparing rows 1 vs 3 and 2 vs 4, we observe that increasing frame selector training data from 1k to 10k videos has minimal impact on segmentation metrics, but slightly improves title generation. In contrast, increasing CL training data from 1k to 10k videos (rows 1 vs 2 and 3 vs 4) improves both segmentation and title metrics.   

<table><tr><td colspan="2"># videos F. selector</td><td colspan="2">Segmentation</td><td colspan="2">Titles</td></tr><tr><td rowspan="2">1k</td><td>CL 1k</td><td>F1 42.7</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>10k</td><td>46.9</td><td>70.8 72.9</td><td>15.6 17.5</td><td>78.1 86.8</td></tr><tr><td rowspan="2">10k</td><td>1k</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr><tr><td>10k</td><td>46.7</td><td>72.2</td><td>18.6</td><td>96.4</td></tr></table>

Throughout our experiments, we train the speech-only model using 10k videos to obtain frame locations for caption extraction (and 1k videos in most of our experiments to train our Chapter-Llama model). In Tab. A.6, we analyze how the amount of training data in the speech-only model affects downstream performance on our Chapter-Llama model using both speech and captions.

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
