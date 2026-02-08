[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
两部分：(1) Dense Video Captioning 的方法进展和数据集，(2) Dense Caption 评估方法的现状。核心论点：现有评估框架源自 Dense Image Captioning，不适合需要故事性的 DVC。

---

## 2.1 Dense Video Captioning

The goal of DVC is to obtain concise and coherent description of all events in a video. It requires understanding the entire video contents and contextual reasoning of individual events. Recent researches [6, 17, 20, 8] handled this challenge by dividing it into two subtasks: event proposal detection and caption generation for the events. For example, Wang et al. [17] proposed a bidirectional LSTM-based encoder-decoder model with a context gating mechanism. The mechanism reflects both past and future contexts to the event proposals and the captions. Zhou et al. [20] proposed a self-attention [14] based end-to-end model. The end-to-end architecture could bridge the event detection and the captioning modules, hence it tended to generate a consistent caption for each individual event. However, these models did not explicitly consider the dependency or relationship among the individual events. Mun et al. [10] challenged to generate brief and consistent captions by reducing the number of event proposals with pointer networks [16].

> 💡 **DVC 方法演进**:
> - Wang et al. [17]: BiLSTM + context gating（考虑上下文）
> - Zhou et al. [20]: Self-attention end-to-end（端到端，事件检测+字幕生成联合优化）
> - Mun et al. [10]: Pointer network 减少冗余 proposal
>
> 关键问题：这些模型都没有**显式建模事件间的依赖关系**。

There are several existing datasets for video-to-text generation other than ActivityNet Captions [6]: Youcook II [19], VideoStory [3], TACoS [12], and TACoS Multi-Sentence [13]. Youcook II, TACoS, and TACoS Multi-Sentence datasets were constructed to evaluate the captioning of cooking videos. As these types of captions temporally depend on each other, their order is an important factor to evaluate the systems. However, since Youcook II employed the same evaluation framework as ActivityNet Challenge, and TaCoS and TaCoS Multi-Sentence employed BLEU, the systems might not be evaluated correctly on the datasets. The VideoStory dataset was constructed to evaluate video story description systems for short videos on an social networking service. However, the systems are also evaluated on the dataset with the same framework as ActivityNet Challenge, which is insufficient to evaluate the story of a video. We believe that SODA is useful not only for the ActivityNet Captions dataset but also for the other datasets constructed to evaluate system captions that convey the story.

> 💡 **数据集总结**:
> | 数据集 | 领域 | 评估方法 | 问题 |
> |--------|------|----------|------|
> | ActivityNet Captions | 通用视频 | 官方 scorer | 不考虑故事性 |
> | Youcook II | 烹饪 | 同 ActivityNet | 同上 |
> | TACoS / Multi-Sentence | 烹饪 | BLEU | 不考虑时序 |
> | VideoStory | 社交视频 | 同 ActivityNet | 同上 |
>
> 所有数据集都面临评估不充分的问题 → SODA 有广泛适用性。

---

## 2.2 Dense Caption Evaluation

The automatic evaluation of video description/captioning is a long term and unsolved problem. The evaluation of DVC is required to measure two aspects: 1) the accuracy of localized events, and 2) that of generated captions for each event. The current evaluation framework of DVC is inspired by that of dense image captioning (DIC) [4], which generates captions that describe localized objects in an image comprehensively. In this evaluation framework, each generated caption is separately evaluated using some metrics (See Section 3 for details.) because the captions independently describe each localized object.

> 💡 **关键区别**: DIC 中各字幕是**独立的**（描述不同物体），但 DVC 中字幕应构成**连贯故事**（有时序依赖）。用 DIC 的框架评 DVC 是根本性的不匹配。

Thus, there is a significant difference between DVC and DIC in whether generated captions should consist of a story or not. However, the current evaluation framework of DVC, which is a simple extension of that of DIC, does not consider the temporal dependency between captions explicitly, which causes the potential risk of overestimation (See Section 4.2 for details.).

In contrast, SODA solves this problem through optimal matching with ground-truth events and penalizing redundant events, as we will explain in Section 5. It would be more difficult to obtain a factitiously high score with SODA compared with the current evaluation framework because SODA requires systems to detect the exact number of events and captions, that we believe will lead to further progress of DVC tasks.

> 💡 **SODA 的激励效果**: 现有框架激励系统生成冗余字幕（多生成不扣分），SODA 激励系统生成**数量正确**的字幕 → 推动 DVC 研究往正确方向发展。

The research community of DVC has mainly used the following six different evaluation metrics for caption sentences: ROUGE-L [9], METEOR[2], BLEU [11], CIDEr [15], SPICE [1], and WMD [7]. These metrics were originated from text generation tasks in natural language processing such as machine translation, text summarization, and image captioning. There have been several experiments to make clear which metrics are better for caption evaluation [18, 5] because of too many metrics. They showed that evaluation metrics being relatively less sensitive to word order and synonym changes in a sentence, like CIDEr and METEOR, can provide a high correlation with human judgments. Therefore, METEOR was adopted as the main evaluation metric in DVC.

> 💡 **为什么用 METEOR**: 在六种指标中，METEOR 对词序和同义词变化不太敏感，与人类判断相关性高。注意 SODA 改的是**匹配框架**，底层文本指标仍用 METEOR。

---

## 🔖 Section 总结

### 核心洞察
1. DVC 方法在进步，但评估框架没跟上 — 从 DIC 直接搬来，忽略了故事性
2. DVC ≠ DIC：字幕有时序依赖，不能独立评估
3. SODA 的改进不在文本指标层面（仍用 METEOR），而在**匹配框架层面**（时序最优匹配 + F-measure）
