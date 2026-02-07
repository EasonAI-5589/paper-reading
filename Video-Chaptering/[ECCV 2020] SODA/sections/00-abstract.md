# Abstract

## 原文

Dense Video Captioning (DVC) is a challenging task that localizes all events in a short video and describes them with natural language sentences. The main goal of DVC is video story description, that is, to generate a concise video story that supports human video comprehension without watching it. In recent years, DVC has attracted increasing attention in the vision and language research community, and has been employed as a task of the workshop, ActivityNet Challenge. In the current research community, the official scorer provided by ActivityNet Challenge is the de-facto standard evaluation framework for DVC systems. It computes averaged METEOR scores for matched pairs between generated and reference captions whose Intersection over Union (IoU) exceeds a specific threshold value. However, the current framework does not take into account the story of the video or the ordering of captions. It also tends to give high scores to systems that generate several hundred redundant captions, that humans cannot read. This paper proposes a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems. SODA first tries to find temporally optimal matching between generated and reference captions to capture the story of a video. Then, it computes METEOR scores for the matching and derives F-measure scores from the METEOR scores to penalize redundant captions. To demonstrate that SODA gives low scores for inadequate captions in terms of video story description, we evaluate two state-of-the-art systems with it, varying the number of captions. The results show that SODA gives low scores against too many or too few captions and high scores against captions whose number equals to that of a reference, while the current framework gives good scores for all the cases. Furthermore, we show that SODA tends to give lower scores than the current evaluation framework in evaluating captions in the incorrect order.

**Keywords**: Automatic Evaluation, Dense Video Captioning, Video Story Description

---

## 译文

密集视频描述（DVC）是一项具有挑战性的任务，需要定位短视频中的所有事件并用自然语言句子描述它们。DVC 的主要目标是视频故事描述，即生成简洁的视频故事，帮助人类在不观看视频的情况下理解其内容。近年来，DVC 在视觉与语言研究社区中受到越来越多的关注，并被采纳为 ActivityNet Challenge 研讨会的任务之一。在当前研究社区中，ActivityNet Challenge 提供的官方评分器是 DVC 系统的事实标准评估框架。它计算生成描述与参考描述之间匹配对的平均 METEOR 分数，其中匹配对的交并比（IoU）需超过特定阈值。然而，当前框架没有考虑视频的故事性或描述的顺序。它还倾向于给生成数百个冗余描述的系统高分，而人类无法阅读这些描述。本文提出了一种新的评估框架——面向故事的密集视频描述评估框架（SODA），用于衡量视频故事描述系统的性能。SODA 首先尝试找到生成描述与参考描述之间的时序最优匹配，以捕捉视频的故事。然后，它计算匹配的 METEOR 分数，并从 METEOR 分数中推导 F 值分数，以惩罚冗余描述。为了证明 SODA 对视频故事描述而言不充分的描述给出低分，我们用它评估了两个最先进的系统，并改变描述的数量。结果表明，SODA 对过多或过少的描述给出低分，对数量与参考相等的描述给出高分，而当前框架对所有情况都给出高分。此外，我们表明 SODA 在评估顺序不正确的描述时倾向于给出比当前评估框架更低的分数。

**关键词**：自动评估、密集视频描述、视频故事描述

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
