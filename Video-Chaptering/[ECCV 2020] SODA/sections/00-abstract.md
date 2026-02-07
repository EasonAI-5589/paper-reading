# Abstract

## 原文

Dense Video Captioning (DVC) is a challenging task that localizes all events in a short video and describes them with natural language sentences. The main goal of DVC is video story description, that is, to generate a concise video story that supports human video comprehension without watching it. In recent years, DVC has attracted increasing attention in the vision and language research community, and has been employed as a task of the workshop, ActivityNet Challenge. In the current research community, the official scorer provided by ActivityNet Challenge is the de-facto standard evaluation framework for DVC systems. It computes averaged METEOR scores for matched pairs between generated and reference captions whose Intersection over Union (IoU) exceeds a specific threshold value. However, the current framework does not take into account the story of the video or the ordering of captions. It also tends to give high scores to systems that generate several hundred redundant captions, that humans cannot read. This paper proposes a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems. SODA first tries to find temporally optimal matching between generated and reference captions to capture the story of a video. Then, it computes METEOR scores for the matching and derives F-measure scores from the METEOR scores to penalize redundant captions. To demonstrate that SODA gives low scores for inadequate captions in terms of video story description, we evaluate two state-of-the-art systems with it, varying the number of captions. The results show that SODA gives low scores against too many or too few captions and high scores against captions whose number equals to that of a reference, while the current framework gives good scores for all the cases. Furthermore, we show that SODA tends to give lower scores than the current evaluation framework in evaluating captions in the incorrect order.

**Keywords**: Automatic Evaluation, Dense Video Captioning, Video Story Description

---

## 理解与批注

### 任务定义
- **Dense Video Captioning (DVC)**: 定位视频中所有事件 + 用自然语言描述
- **核心目标**: 生成简洁的视频故事，帮助人类理解视频（无需观看）

### 现有评测问题 ⚠️
ActivityNet Challenge 的官方评测存在两个关键问题：

| 问题 | 描述 |
|------|------|
| **忽略故事性** | 不考虑 caption 的顺序和故事结构 |
| **奖励冗余** | 生成几百个 caption 反而得高分 |

### SODA 的解决方案 ✅
1. **时序最优匹配**: 用 DP 找保持时序的最佳匹配
2. **F-measure**: 惩罚过多/过少的 caption

### 实验结论
- SODA 对不合适数量的 caption 给低分
- SODA 对顺序错误的 caption 给更低分
- 现有框架对所有情况都给差不多的分数（无法区分好坏）
