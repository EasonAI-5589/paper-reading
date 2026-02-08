[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
详细阐述 DVC 的背景、现有评估框架的问题（松散匹配 + 平均 METEOR），以及 SODA 的解决思路（动态规划最优匹配 + F-measure）。最后列出三点贡献。

---

Dense Video Captioning (DVC) [6] mainly involves two tasks: event detection to identify all events in a short video, and caption generation to describe the event proposals using natural language sentences. DVC is one of the major tasks in vision and language research and has attracted more attention in recent years. In fact, it has been adopted as a task of ActivityNet Challenge since 2017. Its main goal is to generate concise captions that describe the story of a video to help humans understand it. Actually, humans describe the story of a video using 3-4 captions on average. Thus, the generated captions are utilized for grasping an overview of the video without having to watch the entire video [3].

> 💡 **关键数字**: 人类平均只用 **3-4 条字幕**描述一个视频的故事，而现有系统动辄生成几百条。这个数量级差距是本文的核心出发点。

However, the current de-facto standard evaluation framework for DVC systems, which is the official evaluation framework in ActivityNet Challenge, is inappropriate for measuring the performance of a video story description since it disregards the story of the video and the ordering of captions. The framework first matches generated and reference captions when the Intersection over Union (IoU) between them exceeds a specific threshold value. Then, it computes METEOR [2] scores for all matched pairs between the generated and reference captions, and averages them by the number of the pairs. That is, the framework evaluates captions for events without considering the order of their proposals.

> 💡 **现有框架流程**: IoU 超阈值 → 配对 → 算 METEOR → 按配对数平均。问题在于：配对是"松散"的（一对多），且平均分母是配对数而非字幕数。

In addition, another problem with the current framework is that it often obtains a high score by producing several hundred captions that are inadequate as video story descriptions since the scores rely only on the number of matched pairs. As the result, as we will point out in Section 4.2, systems that produce more redundant captions are more advantageous. Most current DVC systems generate several hundred captions for a video, while the number of reference captions is only 3-4.

> 💡 **冗余问题**: 生成越多字幕 → 越多配对 → 平均分不降反升。系统被"激励"生成冗余字幕。

To appropriately and correctly evaluate video story description systems, we need a framework that can consider a video story, the ordering of captions, and can penalize redundant captions. This paper proposes a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems. SODA first applies dynamic programming, that finds the optimal matching between generated and reference captions that maximizes the sum of the IoU by considering the temporal ordering of captions. Thus, it finds the best sequence of generated proposals that maximizes the sum of the IoU against reference proposals. Then, it computes METEOR scores for the matched pairs and derives precision and recall scores on the basis of the calculated METEOR scores. Finally, our framework evaluates generated captions with F-measure scores to consider both the numbers of generated and reference captions.

> 💡 **SODA 三步走**:
> 1. **动态规划** → 找时序最优一对一匹配（解决松散匹配）
> 2. **Precision/Recall** → 分别除以 |P| 和 |G|（解决冗余不惩罚）
> 3. **F-measure** → 综合考虑生成数量和参考数量

To demonstrate the effectiveness of our framework, we evaluate two state-of-the-art systems with it, varying the number of captions. Experimental results on the ActivityNet Captions dataset [6] show that our framework gives low scores to too many or too few captions, inadequate captions as video story description, and gives high scores to captions whose number equals to that of a reference, while the current framework gives almost the same scores to all the cases. Furthermore, we demonstrate that SODA gives lower scores to captions with incorrect order, inconsistent story description, than the current evaluation framework. In addition to the above automatic evaluation, our simple manual evaluation also shows that SODA is superior to the current framework.

Our main contributions are as follows:

– We demonstrate that the current evaluation framework, utilized in ActivityNet Challenge, is insufficient for evaluating video story descriptions.
– We propose a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems by considering the ordering of captions. We introduce F-measure into the evaluation metric to prevent redundant captions from obtaining good scores.
– Our source code will be available on https://github.com/fujiso/SODA.

> 💡 **三点贡献**: (1) 揭示现有框架不足，(2) 提出 SODA（时序匹配 + F-measure），(3) 开源代码。

---

## 🔖 Section 总结

### 核心洞察
1. DVC 系统的评估应反映"视频故事描述"质量，而非单纯的事件检测+字幕生成
2. 现有框架的两个结构性问题：**松散匹配**（不考虑顺序）和**平均方式不当**（不惩罚冗余）
3. SODA 的核心思想：用 DP 做时序最优匹配 + 用 F-measure 同时考虑精度和召回
