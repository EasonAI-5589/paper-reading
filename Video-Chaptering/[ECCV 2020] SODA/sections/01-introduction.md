# 1. Introduction

## 原文

Dense Video Captioning (DVC) [6] mainly involves two tasks: event detection to identify all events in a short video, and caption generation to describe the event proposals using natural language sentences. DVC is one of the major tasks in vision and language research and has attracted more attention in recent years. In fact, it has been adopted as a task of ActivityNet Challenge since 2017. Its main goal is to generate concise captions that describe the story of a video to help humans understand it. Actually, humans describe the story of a video using 3-4 captions on average. Thus, the generated captions are utilized for grasping an overview of the video without having to watch the entire video [3].

However, the current de-facto standard evaluation framework for DVC systems, which is the official evaluation framework in ActivityNet Challenge, is inappropriate for measuring the performance of a video story description since it disregards the story of the video and the ordering of captions. The framework first matches generated and reference captions when the Intersection over Union (IoU) between them exceeds a specific threshold value. Then, it computes METEOR [2] scores for all matched pairs between the generated and reference captions, and averages them by the number of the pairs. That is, the framework evaluates captions for events without considering the order of their proposals.

In addition, another problem with the current framework is that it often obtains a high score by producing several hundred captions that are inadequate as video story descriptions since the scores rely only on the number of matched pairs. As the result, as we will point out in Section 4.2, systems that produce more redundant captions are more advantageous. Most current DVC systems generate several hundred captions for a video, while the number of reference captions is only 3-4.

To appropriately and correctly evaluate video story description systems, we need a framework that can consider a video story, the ordering of captions, and can penalize redundant captions. This paper proposes a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems. SODA first applies dynamic programming, that finds the optimal matching between generated and reference captions that maximizes the sum of the IoU by considering the temporal ordering of captions. Thus, it finds the best sequence of generated proposals that maximizes the sum of the IoU against reference proposals. Then, it computes METEOR scores for the matched pairs and derives precision and recall scores on the basis of the calculated METEOR scores. Finally, our framework evaluates generated captions with F-measure scores to consider both the numbers of generated and reference captions.

To demonstrate the effectiveness of our framework, we evaluate two state-of-the-art systems with it, varying the number of captions. Experimental results on the ActivityNet Captions dataset [6] show that our framework gives low scores to too many or too few captions, inadequate captions as video story description, and gives high scores to captions whose number equals to that of a reference, while the current framework gives almost the same scores to all the cases. Furthermore, we demonstrate that SODA gives lower scores to captions with incorrect order, inconsistent story description, than the current evaluation framework. In addition to the above automatic evaluation, our simple manual evaluation also shows that SODA is superior to the current framework.

Our main contributions are as follows:

- We demonstrate that the current evaluation framework, utilized in ActivityNet Challenge, is insufficient for evaluating video story descriptions.
- We propose a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems by considering the ordering of captions. We introduce F-measure into the evaluation metric to prevent redundant captions from obtaining good scores.
- Our source code will be available on https://github.com/fujiso/SODA.

---

## 理解与批注

### DVC 任务的两个子任务
1. **Event Detection**: 识别视频中的所有事件
2. **Caption Generation**: 用自然语言描述事件

> 💡 人类平均用 **3-4 个 caption** 描述一个视频，而现有系统生成**几百个**

### 现有评测框架的问题

#### 问题 1: 忽略顺序
```
匹配规则: IoU(generated, reference) > τ
↓
只看时间重叠，不管顺序
```

#### 问题 2: 冗余得高分
```
Reference: 3-4 captions
现有系统: 200+ captions
结果: 生成更多反而分数更高 ❌
```

### SODA 的三步解决方案

```
Step 1: 动态规划
找最大化 Σ IoU 的一对一匹配，保持时序

Step 2: 计算 METEOR
对匹配对计算 METEOR 分数

Step 3: F-measure
Precision = Σ METEOR / |P|
Recall = Σ METEOR / |G|
F1 = 2PR / (P+R)
```

### 论文贡献
1. ⚠️ 证明现有评测不足
2. ✅ 提出 SODA（考虑顺序 + F-measure）
3. 📦 开源代码
