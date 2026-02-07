# D.3. Chapter-Llama prediction examples

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

Similar to Fig. 3 of the main paper, in Fig. A.6, we present two additional examples comparing our method against Vid2Seq and our zero-shot baseline.

In Fig. A.7, we show three examples of our Chapter-Llama predictions compared to the ground truth (GT) for videos without speech $3 \%$ of the data). We observe that many of the completely ‘speechless’ videos contain OCR-readable text to help the viewer follow the video (top and bottom examples), in which cases the captioners tend to perform OCR, leading to satisfactory chaptering results. Otherwise, in case of no onscreen text and no speech (e.g., only music), the result is inferior, though still acceptable (middle example). As also evaluated in Tab. A.12, our model still achieves reasonable quantitative performance, even if speech indeed tends to be more informative for chaptering than visual modality [112].

![](images/591c8164e682ed79fae9a373b13d94f897c85f7ae1523d6cf8c61e343560cd35.jpg)  
Figure A.4. Segmentation metrics visualization: We illustrate with examples how tIoU and F1 scores are calculated for video chaptering. The top example shows a high-quality prediction with good overlap, while the bottom example demonstrates a lower-quality prediction with more misalignments. We additionally show the corresponding SODA (S) and CIDEr (C) scores.

![](images/6a0b9bf2c8b4df479777c70ad368939888ac24fb64d7f9afa8ba131f741e3ead.jpg)

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- Figure: 591c8164e682ed79fae9a373b13d94f897c85f7ae1523d6cf8c61e343560cd35.jpg
- Figure: 6a0b9bf2c8b4df479777c70ad368939888ac24fb64d7f9afa8ba131f741e3ead.jpg

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
