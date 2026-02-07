# 4.4. Iterative prediction on longer videos

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In our ablation studies, our experimental setting considered training and evaluating with videos that fit within the LLM context window. In Tab. 5, we evaluate the benefit of our iterative prediction procedure for handling videos that exceed the LLM context window. For this, we identify videos in the validation set whose inputs exceed the LLM inference context window $\mathrm { \Omega } > 3 5 k \mathrm { \Omega }$ tokens), resulting in 110 videos. On this challenging subset, we find that our iterative prediction procedure improves chaptering results compared to the baseline that only runs the LLM once by cropping the input to the first input window, across various context windows (10k, 15k, and 20k). We refer to Appendix $\mathrm { B }$ for details on the video lengths and statistics of videos that exceed the LLM context window.

Table 4. Frame embeddings vs captions: We compare using frame captions versus visual features from a frozen SigLIP model projected through a learned 2-layer MLP mapping network (‘Embeddings’). While the ‘Speech+Embeddings’ combination performs better than speech alone (40.4 vs 38.5 F1), it underperforms compared to the ‘Speech $^ +$ Captions’ combination (42.6 vs 40.4 F1). All models are trained with 1k videos and evaluated on 300 videos.   

<table><tr><td colspan="2">Modalities</td><td colspan="2">Segmentation</td><td colspan="2">Titles</td></tr><tr><td></td><td>Speech Embeddings</td><td>Captions</td><td>F1</td><td>tIoU S</td><td>C</td></tr><tr><td>✓ -</td><td>;</td><td>- -</td><td>38.5 38.4</td><td>68.1 66.5</td><td>13.9 67.3 3.4 7.3</td></tr><tr><td>-</td><td></td><td>✓</td><td>39.1</td><td>67.7</td><td>5.9 20.2</td></tr><tr><td>✓ ✓</td><td>✓</td><td>-</td><td>40.4</td><td>68.2</td><td>15.3 74.9</td></tr><tr><td>✓</td><td>-</td><td></td><td>42.6</td><td>70.6</td><td>16.4 82.4</td></tr><tr><td></td><td>J</td><td>√</td><td>44.4</td><td>71.5</td><td>16.3 84.2</td></tr></table>

Table 5. Iterative prediction: Our iterative prediction procedure improves chaptering results on the subset of 110 videos which exceed 35k tokens compared to the baseline that only runs the LLM once (by only taking the first window, and discarding the rest of the input sequence), across various context windows. As we increase the context window in the iterative prediction, the performance gradually improves and the average number of iterations decreases. The model is trained with 1k videos.   

<table><tr><td>Window</td><td># tok.</td><td>avg # iter.</td><td colspan="3">Subset exceeding 35k tokens F1 tIoU S</td></tr><tr><td rowspan="3">First</td><td>10k</td><td>1</td><td>13.1</td><td>50.5 4.0</td><td>31.2</td></tr><tr><td>15k</td><td>1</td><td>16.6</td><td>54.9 5.4</td><td>43.3</td></tr><tr><td>20k</td><td>1</td><td>18.7</td><td>56.7 6.6</td><td>47.5</td></tr><tr><td rowspan="3">Iterative</td><td>10k</td><td>8.5</td><td>18.5</td><td>57.1</td><td>6.9 25.1</td></tr><tr><td>15k</td><td>5.4</td><td>23.6</td><td>60.1</td><td>8.7 35.2</td></tr><tr><td>20k</td><td>4.1</td><td>25.3</td><td>61.4</td><td>10.3 44.0</td></tr></table>

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
