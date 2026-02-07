# 2. Related Work

## 2.1 Dense Video Captioning

### 原文

The goal of DVC is to obtain concise and coherent description of all events in a video. It requires understanding the entire video contents and contextual reasoning of individual events. Recent researches handled this challenge by dividing it into two subtasks: event proposal detection and caption generation for the events.

- **Wang et al.** proposed a bidirectional LSTM-based encoder-decoder model with a context gating mechanism. The mechanism reflects both past and future contexts to the event proposals and the captions.
- **Zhou et al.** proposed a self-attention based end-to-end model. The end-to-end architecture could bridge the event detection and the captioning modules, hence it tended to generate a consistent caption for each individual event.
- **Mun et al.** challenged to generate brief and consistent captions by reducing the number of event proposals with pointer networks.

There are several existing datasets for video-to-text generation other than ActivityNet Captions: YouCook II, VideoStory, TACoS, and TACoS Multi-Sentence. YouCook II, TACoS, and TACoS Multi-Sentence datasets were constructed to evaluate the captioning of cooking videos. As these types of captions temporally depend on each other, their order is an important factor to evaluate the systems. However, since YouCook II employed the same evaluation framework as ActivityNet Challenge, and TaCoS and TaCoS Multi-Sentence employed BLEU, the systems might not be evaluated correctly on the datasets. The VideoStory dataset was constructed to evaluate video story description systems for short videos on a social networking service. However, the systems are also evaluated on the dataset with the same framework as ActivityNet Challenge, which is insufficient to evaluate the story of a video. We believe that SODA is useful not only for the ActivityNet Captions dataset but also for the other datasets constructed to evaluate system captions that convey the story.

### 译文

DVC 的目标是获取视频中所有事件的简洁且连贯的描述。它需要理解整个视频内容并对各个事件进行上下文推理。近期研究通过将这一挑战分解为两个子任务来处理：事件提案检测和事件描述生成。

- **Wang 等人**提出了一种基于双向 LSTM 的编码器-解码器模型，带有上下文门控机制。该机制将过去和未来的上下文反映到事件提案和描述中。
- **Zhou 等人**提出了一种基于自注意力的端到端模型。端到端架构可以桥接事件检测和描述模块，因此它倾向于为每个单独的事件生成一致的描述。
- **Mun 等人**尝试通过使用指针网络减少事件提案的数量来生成简洁一致的描述。

除了 ActivityNet Captions 之外，还有几个用于视频到文本生成的现有数据集：YouCook II、VideoStory、TACoS 和 TACoS Multi-Sentence。YouCook II、TACoS 和 TACoS Multi-Sentence 数据集是为评估烹饪视频的描述而构建的。由于这些类型的描述在时间上相互依赖，它们的顺序是评估系统的重要因素。然而，由于 YouCook II 采用了与 ActivityNet Challenge 相同的评估框架，而 TaCoS 和 TaCoS Multi-Sentence 采用了 BLEU，这些系统可能无法在这些数据集上得到正确评估。VideoStory 数据集是为评估社交网络服务上短视频的视频故事描述系统而构建的。然而，这些系统也在该数据集上使用与 ActivityNet Challenge 相同的框架进行评估，这不足以评估视频的故事。我们相信 SODA 不仅对 ActivityNet Captions 数据集有用，而且对其他用于评估传达故事的系统描述的数据集也有用。

---

### 理解与批注

#### DVC 的两个子任务
```
Video → Event Proposal Detection → Caption Generation → Descriptions
        (检测事件边界)              (生成描述)
```

#### 主要方法演进

| 方法 | 架构 | 特点 |
|------|------|------|
| Wang et al. | BiLSTM + Context Gating | 利用过去/未来上下文 |
| Zhou et al. | Self-Attention E2E | 事件检测+描述联合优化 |
| Mun et al. | Pointer Networks | 减少 proposal 数量 |

> 💡 这些方法都**没有显式考虑事件间的依赖关系**

---

## 2.2 Dense Caption Evaluation

### 原文

The automatic evaluation of video description/captioning is a long term and unsolved problem. The evaluation of DVC is required to measure two aspects: 1) the accuracy of localized events, and 2) that of generated captions for each event. The current evaluation framework of DVC is inspired by that of dense image captioning (DIC), which generates captions that describe localized objects in an image comprehensively. In this evaluation framework, each generated caption is separately evaluated using some metrics because the captions independently describe each localized object.

Thus, there is a significant difference between DVC and DIC in whether generated captions should consist of a story or not. However, the current evaluation framework of DVC, which is a simple extension of that of DIC, does not consider the temporal dependency between captions explicitly, which causes the potential risk of overestimation.

In contrast, SODA solves this problem through optimal matching with ground-truth events and penalizing redundant events. It would be more difficult to obtain a factitiously high score with SODA compared with the current evaluation framework because SODA requires systems to detect the exact number of events and captions, that we believe will lead to further progress of DVC tasks.

The research community of DVC has mainly used the following six different evaluation metrics for caption sentences: ROUGE-L, METEOR, BLEU, CIDEr, SPICE, and WMD. These metrics were originated from text generation tasks in natural language processing such as machine translation, text summarization, and image captioning. There have been several experiments to make clear which metrics are better for caption evaluation. They showed that evaluation metrics being relatively less sensitive to word order and synonym changes in a sentence, like CIDEr and METEOR, can provide a high correlation with human judgments. Therefore, METEOR was adopted as the main evaluation metric in DVC.

### 译文

视频描述/视频字幕的自动评估是一个长期未解决的问题。DVC 的评估需要衡量两个方面：1）定位事件的准确性，2）每个事件生成描述的准确性。当前 DVC 的评估框架受到密集图像描述（DIC）的启发，后者生成全面描述图像中定位对象的描述。在这个评估框架中，每个生成的描述使用一些指标单独评估，因为这些描述独立描述每个定位的对象。

因此，DVC 和 DIC 之间存在显著差异，即生成的描述是否应该构成一个故事。然而，当前 DVC 的评估框架，作为 DIC 评估框架的简单扩展，没有明确考虑描述之间的时序依赖，这导致了高估的潜在风险。

相比之下，SODA 通过与真实事件的最优匹配和惩罚冗余事件来解决这个问题。与当前评估框架相比，使用 SODA 更难获得虚假的高分，因为 SODA 要求系统检测准确数量的事件和描述，我们相信这将推动 DVC 任务的进一步进展。

DVC 研究社区主要使用以下六种不同的评估指标来评估描述句子：ROUGE-L、METEOR、BLEU、CIDEr、SPICE 和 WMD。这些指标起源于自然语言处理中的文本生成任务，如机器翻译、文本摘要和图像描述。已有多项实验明确哪些指标更适合描述评估。它们表明，对句子中词序和同义词变化相对不敏感的评估指标，如 CIDEr 和 METEOR，可以与人工判断提供高相关性。因此，METEOR 被采纳为 DVC 的主要评估指标。

---

### 理解与批注

#### DVC vs DIC 的关键区别

| 任务 | Caption 关系 | 评测需求 |
|------|-------------|---------|
| **DIC** (图像) | 独立描述物体 | 独立评估每个 caption |
| **DVC** (视频) | 组成故事 | 需要考虑**时序依赖** |

> ⚠️ 当前 DVC 评测是 DIC 评测的简单扩展，忽略了时序！

#### 为什么选择 METEOR？

实验表明 METEOR 和 CIDEr 与人工评估相关性最高：
- 对词序变化不太敏感
- 考虑同义词匹配
- ActivityNet Challenge 因此采用 METEOR
