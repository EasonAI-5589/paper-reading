# Abstract

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

Segmenting long videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. 

We introduce the following three tasks based on this data:
1. **Video chapter generation**: temporally segmenting the video and generating a chapter title for each segment
2. **Video chapter generation given ground-truth boundaries**: generating a chapter title given an annotated video segment
3. **Video chapter grounding**: temporally localizing a chapter given its annotated title

We benchmark both simple baselines and state-of-the-art video-language models for these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks in both zero-shot and finetuning settings, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset.

![Figure 1](../images/4a3cb6ce77e5c33483e082c5486a35ee29e772502c561d05f2e7e59f118c2701.jpg)
*Figure 1: A video with user-annotated chapters in VidChapters-7M*

---

## 💡 理解

### 核心要点
- [ ] 

### 这篇论文要解决什么问题？
- 

### 主要贡献是什么？
1. 
2. 
3. 

### 我的疑问
- [ ] 
