# 1. Introduction

## 原文

Dense Video Captioning (DVC) mainly involves two tasks: event detection to identify all events in a short video, and caption generation to describe the event proposals using natural language sentences. DVC is one of the major tasks in vision and language research and has attracted more attention in recent years. In fact, it has been adopted as a task of ActivityNet Challenge since 2017. Its main goal is to generate concise captions that describe the story of a video to help humans understand it. Actually, humans describe the story of a video using 3-4 captions on average. Thus, the generated captions are utilized for grasping an overview of the video without having to watch the entire video.

However, the current de-facto standard evaluation framework for DVC systems, which is the official evaluation framework in ActivityNet Challenge, is inappropriate for measuring the performance of a video story description since it disregards the story of the video and the ordering of captions. The framework first matches generated and reference captions when the Intersection over Union (IoU) between them exceeds a specific threshold value. Then, it computes METEOR scores for all matched pairs between the generated and reference captions, and averages them by the number of the pairs. That is, the framework evaluates captions for events without considering the order of their proposals.

In addition, another problem with the current framework is that it often obtains a high score by producing several hundred captions that are inadequate as video story descriptions since the scores rely only on the number of matched pairs. As the result, as we will point out in Section 4.2, systems that produce more redundant captions are more advantageous. Most current DVC systems generate several hundred captions for a video, while the number of reference captions is only 3-4.

To appropriately and correctly evaluate video story description systems, we need a framework that can consider a video story, the ordering of captions, and can penalize redundant captions. This paper proposes a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems. SODA first applies dynamic programming, that finds the optimal matching between generated and reference captions that maximizes the sum of the IoU by considering the temporal ordering of captions. Thus, it finds the best sequence of generated proposals that maximizes the sum of the IoU against reference proposals. Then, it computes METEOR scores for the matched pairs and derives precision and recall scores on the basis of the calculated METEOR scores. Finally, our framework evaluates generated captions with F-measure scores to consider both the numbers of generated and reference captions.

To demonstrate the effectiveness of our framework, we evaluate two state-of-the-art systems with it, varying the number of captions. Experimental results on the ActivityNet Captions dataset show that our framework gives low scores to too many or too few captions, inadequate captions as video story description, and gives high scores to captions whose number equals to that of a reference, while the current framework gives almost the same scores to all the cases. Furthermore, we demonstrate that SODA gives lower scores to captions with incorrect order, inconsistent story description, than the current evaluation framework. In addition to the above automatic evaluation, our simple manual evaluation also shows that SODA is superior to the current framework.

Our main contributions are as follows:

- We demonstrate that the current evaluation framework, utilized in ActivityNet Challenge, is insufficient for evaluating video story descriptions.
- We propose a new evaluation framework, Story Oriented Dense video cAptioning evaluation framework (SODA), for measuring the performance of video story description systems by considering the ordering of captions. We introduce F-measure into the evaluation metric to prevent redundant captions from obtaining good scores.
- Our source code will be available on https://github.com/fujiso/SODA.

---

## 译文

密集视频描述（DVC）主要涉及两个任务：事件检测——识别短视频中的所有事件，以及描述生成——用自然语言句子描述事件提案。DVC 是视觉与语言研究中的主要任务之一，近年来受到越来越多的关注。事实上，它自 2017 年以来被采纳为 ActivityNet Challenge 的任务之一。其主要目标是生成简洁的描述来描述视频的故事，帮助人类理解它。实际上，人类平均使用 3-4 个描述来描述视频的故事。因此，生成的描述被用于在不观看整个视频的情况下把握视频概览。

然而，当前 DVC 系统的事实标准评估框架，即 ActivityNet Challenge 的官方评估框架，不适合衡量视频故事描述的性能，因为它忽视了视频的故事性和描述的顺序。该框架首先在生成描述与参考描述之间的交并比（IoU）超过特定阈值时进行匹配。然后，它计算生成描述与参考描述之间所有匹配对的 METEOR 分数，并按匹配对数量进行平均。也就是说，该框架在评估事件描述时不考虑其提案的顺序。

此外，当前框架的另一个问题是，它经常通过生成数百个作为视频故事描述不充分的描述而获得高分，因为分数仅依赖于匹配对的数量。结果，正如我们将在第 4.2 节指出的，生成更多冗余描述的系统更具优势。大多数当前的 DVC 系统为一个视频生成数百个描述，而参考描述的数量仅为 3-4 个。

为了正确适当地评估视频故事描述系统，我们需要一个能够考虑视频故事、描述顺序并能惩罚冗余描述的框架。本文提出了一种新的评估框架——面向故事的密集视频描述评估框架（SODA），用于衡量视频故事描述系统的性能。SODA 首先应用动态规划，通过考虑描述的时序顺序，找到生成描述与参考描述之间最大化 IoU 之和的最优匹配。因此，它找到了针对参考提案最大化 IoU 之和的最佳生成提案序列。然后，它计算匹配对的 METEOR 分数，并基于计算的 METEOR 分数推导精确率和召回率分数。最后，我们的框架用 F 值分数评估生成的描述，以同时考虑生成描述和参考描述的数量。

为了证明我们框架的有效性，我们用它评估了两个最先进的系统，并改变描述的数量。在 ActivityNet Captions 数据集上的实验结果表明，我们的框架对过多或过少的描述（作为视频故事描述不充分的描述）给出低分，对数量与参考相等的描述给出高分，而当前框架对所有情况给出几乎相同的分数。此外，我们证明 SODA 对顺序不正确、故事描述不一致的描述给出比当前评估框架更低的分数。除了上述自动评估，我们简单的人工评估也表明 SODA 优于当前框架。

我们的主要贡献如下：

- 我们证明了 ActivityNet Challenge 中使用的当前评估框架不足以评估视频故事描述。
- 我们提出了一种新的评估框架——面向故事的密集视频描述评估框架（SODA），通过考虑描述的顺序来衡量视频故事描述系统的性能。我们将 F 值引入评估指标，以防止冗余描述获得高分。
- 我们的源代码将在 https://github.com/fujiso/SODA 上提供。

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
