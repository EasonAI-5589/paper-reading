# C.9. Training on videos of various durations

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In most of our experiments, we have trained our model on 1k videos balanced across duration categories, i.e., 333 short videos $( < 1 5 \ \mathrm { m i n } )$ ), 333 medium-length videos $( 1 5 { - } 3 0 \mathrm { m i n } )$ , and 334 long videos $( 3 0 \mathrm { - } 6 0 \mathrm { m i n } )$ ). In Tab. A.10, we show the benefit of such training on videos of various durations. For this experiment, we train new models only on 1k short videos, on 1k medium videos, and on 1k long videos. For evaluation, we use the same 300 validation videos as before, with 100 videos sampled from each duration category. As expected, training on short videos performs best on short videos (49.7 F1), while training on long videos performs best on long videos (40.4 F1). Training with a balanced mix of all three durations achieves the best overall performance across all categories (42.6 F1).

Table A.9. LoRA rank: Comparing LoRA ranks $\mathrm { r } { = } 8$ and $_ { \mathrm { r = 1 6 } }$ , we find that with 1k training videos, the lower rank performs better. With $^ { 5 \mathrm { k } }$ videos, $_ { \mathrm { r = 1 6 } }$ slightly outperforms $\mathrm { r } { = } 8$ . At 10k videos, both ranks achieve similar results, suggesting that with sufficient training data, model capacity becomes less important.   

<table><tr><td>#videos</td><td>rank</td><td>F1</td><td>tIoU</td><td>S</td><td>C</td></tr><tr><td>1k</td><td>8 16</td><td>42.6 39.9</td><td>70.6 68.5</td><td>16.4 15.6</td><td>82.4 78.4</td></tr><tr><td>5k</td><td>8 16</td><td>45.6 46.5</td><td>72.3 72.8</td><td>18.3 18.5</td><td>90.0 92.8</td></tr><tr><td>10k</td><td>8 16</td><td>46.7 46.6</td><td>72.2 72.4</td><td>18.6 18.6</td><td>96.4 92.5</td></tr></table>

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
