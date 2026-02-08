[← 返回 README](../README.md)

# Abstract

## 📌 预览
论文提出 SODA（Story Oriented Dense video cAptioning evaluation framework），一个面向视频故事描述的 Dense Video Captioning 评估框架。核心动机：现有 ActivityNet Challenge 评估框架忽略了字幕的时序顺序和故事性，且对冗余字幕给出不合理的高分。

---

Dense Video Captioning (DVC) is a challenging task that localizes all events in a short video and describes them with natural language sentences. The main goal of DVC is video story description, that is, to generate a concise video story that supports human video comprehension without watching it. In recent years, DVC has attracted increasing attention in the vision and language research community, and has been employed as a task of the workshop, ActivityNet Challenge.

> 💡 **背景**: DVC 的核心目标不只是"给事件配字幕"，而是生成一个**连贯的视频故事**，让人不用看视频就能理解内容。

In the current research community, the official scorer provided by ActivityNet Challenge is the de-facto standard evaluation framework for DVC systems. It computes averaged METEOR scores for matched pairs between generated and reference captions whose Intersection over Union (IoU) exceeds a specific threshold value. However, the current framework does not take into account the story of the video or the ordering of captions. It also tends to give high scores to systems that generate several hundred redundant captions, that humans cannot read.

> 💡 **问题**: 现有框架两大缺陷：(1) 不考虑字幕顺序（故事性），(2) 冗余字幕反而得高分。系统生成几百条字幕，人根本读不完，但评分还很高。

This paper proposes a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems. SODA first tries to find temporally optimal matching between generated and reference captions to capture the story of a video. Then, it computes METEOR scores for the matching and derives F-measure scores from the METEOR scores to penalize redundant captions.

> 💡 **SODA 核心**: 两步走 — (1) 用动态规划找**时序最优匹配**（考虑顺序），(2) 用 **F-measure** 替代简单平均（惩罚冗余）。

To demonstrate that SODA gives low scores for inadequate captions in terms of video story description, we evaluate two state-of-the-art systems with it, varying the number of captions. The results show that SODA gives low scores against too many or too few captions and high scores against captions whose number equals to that of a reference, while the current framework gives good scores for all the cases. Furthermore, we show that SODA tends to give lower scores than the current evaluation framework in evaluating captions in the incorrect order.

> 💡 **实验验证**: SODA 对字幕数量敏感（太多太少都扣分），对顺序敏感（乱序扣分更狠），而现有框架对这些都不敏感。

**Keywords**: Automatic Evaluation, Dense Video Captioning, Video Story Description

---

## 🔖 Section 总结

### 核心洞察
1. DVC 的本质目标是**视频故事描述**，评估框架应体现故事性
2. 现有框架的两个致命缺陷：忽略顺序 + 不惩罚冗余
3. SODA 通过**时序最优匹配 + F-measure** 解决这两个问题
