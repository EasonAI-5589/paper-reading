# C.1. Predicting timestamps without chapter titles

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In our experiments, the Chapter-Llama model was trained to predict both chapter times and titles together. An alternative approach could involve training the model to predict chapter times exclusively, subsequently using another model to derive chapter titles from these times. However, as depicted in Tab. A.2, this approach underperforms compared to our current method. Therefore, we choose to continue training the Chapter-Llama model to predict both elements together, as the inclusion of chapter titles appears to enhance the accuracy of chapter time predictions.

![](images/bfae9eff319ef1d65b5a2f54c2246209ef095af6548171d484f30f8b9a268c77.jpg)  
Figure A.1. Video duration distribution: Distribution of video durations in our training set (bars, left axis) and average number of chapters per duration bin (gray line, right axis). Most videos are less than 15 minutes long, with progressively fewer videos at longer durations. The average number of chapters increases with video duration but plateaus around 13 chapters for videos longer than one hour.

![](images/3d232c125f6bca909f5b1117f40dcb4102ee995d6b50f414938554a99c56e179.jpg)  
Figure A.2. Video category distribution: We compare the distribution of video categories between the training set of the full VidChapters-7M dataset and our 20k training subset. We observe similar distributions given our uniform sampling from the original training set.

Table A.2. Effect of chapter titles on timestamp prediction: We evaluate training Chapter-Llama with only timestamps or with timestamps and chapter titles, and observe that adding chapter titles slightly improves the segmentation metrics (F1: $+ 0 . 6$ , tIoU: $+ 0 . 2 )$ .   

<table><tr><td>Ground Truth Format</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>HH:MM:SS</td><td>42.0</td><td>70.4</td><td>-</td><td>-</td></tr><tr><td>HH:MM:SS - Title</td><td>42.6</td><td>70.6</td><td>16.4</td><td>82.4</td></tr></table>

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- Figure: bfae9eff319ef1d65b5a2f54c2246209ef095af6548171d484f30f8b9a268c77.jpg
- Figure: 3d232c125f6bca909f5b1117f40dcb4102ee995d6b50f414938554a99c56e179.jpg

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
