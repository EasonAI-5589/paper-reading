# C.5. Training data size on the frame selection model

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

The second to last row (42.6 F1) represents our main result reported in our ablations, and the last row (46.7 F1) shows results when using 10k videos for speech-only model training and 10k videos for Chapter-Llama (CL) model training, corresponding to the final point in the number of training videos vs performance plot in Fig. 4 of the main paper. The first two rows show new results using only 1k videos to train the speech-only model. We observe that increasing training data for the speech-only frame selector model from 1k to 10k videos has minimal impact on segmentation metrics but improves title generation performance in both cases – from 17.5 to 18.6 SODA when using 10k videos for Chapter-Llama training, and from 15.6 to 16.4 SODA when using 1k videos for Chapter-Llama training. Increasing the training data from 1k to 10k videos for our Chapter-Llama model improves performance on both segmentation and title benchmarks, with F1 scores improving from 42.7 to 46.9 and from 42.6 to 46.7, respectively.

Table A.7. Frame selector and Chapter-Llama training data overlap: Given the set of videos used to train the speech-based frame selector model $( V _ { F . S . } )$ and and the Chapter-Llama model $( V _ { C , L . } ) _ { \mathrm { { \Omega } } }$ we compare the performance of Chapter-Llama when using different subsets of videos $( V _ { F . S . } \neq V _ { C . L . } )$ , and when using the same, already seen, videos $( V _ { F . S . } = V _ { C . L . } )$ . We see that using the same 1k set of videos for both models decreases performance.   

<table><tr><td>Training data</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>VF.S.=VC.L.</td><td>41.4</td><td>70.1</td><td>15.1</td><td>77.5</td></tr><tr><td>VF.S.6=VC.L.</td><td>42.7</td><td>70.8</td><td>15.6</td><td>78.1</td></tr></table>

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
