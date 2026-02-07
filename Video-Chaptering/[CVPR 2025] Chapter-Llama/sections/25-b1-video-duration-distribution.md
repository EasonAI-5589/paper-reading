# B.1. Video duration distribution

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

Figure A.1 shows the distribution of video durations in our training set. The majority of videos $( 5 8 . 4 \% )$ are short videos less than 15 minutes long, while $2 1 . 9 \%$ are medium-length (15-30 minutes), $1 1 . 4 \%$ are long (30-60 minutes), and $8 . 3 \%$ exceed one hour. Interestingly, we observe that the average number of chapters per video increases with video duration up to about 60 minutes, where it plateaus at approximately 13 chapters. This plateau suggests a practical limit to manual chapter annotation, as annotators may be reluctant to segment videos into more than 13 chapters regardless of duration. The median video duration is 12:46 minutes.

<table><tr><td>Category</td><td>&lt;15k tokens</td></tr><tr><td>Short</td><td>466k 100 %</td></tr><tr><td>Medium</td><td>175k 100 %</td></tr><tr><td>Long</td><td>71k 79 %</td></tr></table>

Table A.1. Videos in each category with fewer than 15k tokens: We show the number of videos and proportion of short, medium, and long videos in the training set that do not exceed the 15k token limit of our training context window, from among 817k original training set videos of VidChapters. For videos without extracted captions, the caption token length are estimated by multiplying the average number of tokens per caption by the number of ground truth chapters.

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
