# C.8. LoRA rank

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

In Tab. A.9, we conduct experiments comparing LoRA ranks $r = 8$ and $r = 1 6$ across different training data sizes. With 1k training videos, the lower rank $r = 8$ performs notably better (42.6 vs $3 9 . 9 \mathrm { F } 1$ score). As we increase to 5k videos, $r = 1 6$ shows a slight advantage (46.5 vs 45.6 F1), while at 10k videos both ranks achieve comparable performance (46.7 vs 46.6 F1). This suggests that with limited training data, a lower rank helps prevent overfitting, while with more data the model capacity becomes less critical. Based on these findings and considering efficiency, we use $r { = } 8$ as our default LoRA rank throughout all experiments in the paper.

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
