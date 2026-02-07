# 5. Conclusion, Limitations, and Societal Impacts

> 来源: VidChapters-7M: Video Chapters at Scale (NeurIPS 2023)

---

## 📄 原文

We introduced VidChapters-7M, a large-scale dataset of 817K user-chaptered videos, with over 7M chapters scraped from the Web. We used this dataset to benchmark simple baselines and strong video-text models on the tasks of video chapter generation, video chapter generation given ground-truth boundaries, and video chapter grounding. 

We showed that these tasks are far from being solved, demonstrating the value of VidChapters-7M for video understanding research. We further showed that video chapter generation models pretrained on VidChapters-7M transfer well to dense video captioning benchmarks, largely improving state of the art.

### Limitations

1. **Visual features are pre-extracted** at 1 FPS resolution → prevents the model from operating on the original video directly

2. **User-annotated chapters can be noisy** → 3% of videos have chapter titles unrelated to video content; others have titles only related to video structure (step 1, step 2...)

3. **Tasks are not fully solved** → need better models

### Broader Impacts

**Positive:**
- Help users navigate long videos more efficiently
- Improve accessibility for people with hearing/visual impairments

**Negative:**
- Dataset may contain harmful content (we flag but don't remove NSFW/toxic content)
- Biases in the data (92.9% English, gender imbalance)

---

## 💡 理解

### 核心要点
- [ ] 

### 论文贡献总结

| 贡献 | 具体内容 |
|------|---------|
| 数据集 | |
| 任务定义 | |
| Baseline | |
| 迁移学习 | |

### 局限性分析

1. **1 FPS 特征提取**
   - 问题: 
   - 影响: 

2. **标注噪声**
   - 问题: 
   - 影响: 

### 未来方向
- [ ] 端到端训练
- [ ] 更长视频
- [ ] 多语言扩展
- [ ] 

### 我的疑问
- [ ] 
