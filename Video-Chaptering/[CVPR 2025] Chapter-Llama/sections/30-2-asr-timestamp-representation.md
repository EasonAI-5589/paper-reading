# C.2. ASR timestamp representation

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

As mentioned in Sec. 3, we use ASR outputs obtained with WhisperX [6], which contain start and end timestamps of each ASR segment. For our experiments, we only use the start timestamps, as opposed to using start and end timestamps of each ASR segment. In Tab. A.3, we analyze the impact of including end timestamps from ASR segments in addition to start timestamps. When using only speech inputs, including end timestamps improves performance (e.g., 41.4 vs 38.5 F1). However, when training with speech and captions, using only start timestamps performs better, particularly for title generation metrics (e.g., 82.4 vs 19.9 CIDEr). We hypothesize this is because captions only have single timestamps, so having ASR segments with both start and end times creates an inconsistency between modalities that degrades performance. Therefore, in our final model we use only start timestamps for ASR segments.

Table A.3. Adding end timestamps to ASR input: Adding end timestamps to ASR transcripts improves performance when using only speech $\left( + 2 . 9 \mathrm { F } 1 \right)$ ). However, when combining speech with captions, including end timestamps decreases performance significantly, especially on title metrics (e.g., 19.9 vs 82.4 CIDEr). We hypothesize this may be due to the inconsistency between modalities, where captions have single timestamps while speech segments have start and end times.   

<table><tr><td colspan="2">Modalities</td><td>ASR</td><td colspan="2">Segmentation</td><td colspan="2">Titles</td></tr><tr><td>Speech</td><td>Capt.</td><td>timestamp</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>√</td><td>-</td><td>start end start</td><td>41.4 38.5</td><td>69.7 68.1</td><td>15.8 13.9</td><td>77.9 67.3</td></tr><tr><td>✓</td><td>L</td><td>start end start</td><td>39.1 42.6</td><td>67.6 70.6</td><td>6.0 16.4</td><td>19.9 82.4</td></tr></table>

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
