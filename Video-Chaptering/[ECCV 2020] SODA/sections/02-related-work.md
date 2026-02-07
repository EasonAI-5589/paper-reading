# 2. Related Work

## 2.1 Dense Video Captioning

### 原文

The goal of DVC is to obtain concise and coherent description of all events in a video. It requires understanding the entire video contents and contextual reasoning of individual events. Recent researches handled this challenge by dividing it into two subtasks: event proposal detection and caption generation for the events.

- **Wang et al.** proposed a bidirectional LSTM-based encoder-decoder model with a context gating mechanism. The mechanism reflects both past and future contexts to the event proposals and the captions.
- **Zhou et al.** proposed a self-attention based end-to-end model. The end-to-end architecture could bridge the event detection and the captioning modules, hence it tended to generate a consistent caption for each individual event.
- **Mun et al.** challenged to generate brief and consistent captions by reducing the number of event proposals with pointer networks.

There are several existing datasets for video-to-text generation:
- **ActivityNet Captions**: Most widely used
- **YouCook II**: Cooking videos
- **VideoStory**: Social media short videos
- **TACoS / TACoS Multi-Sentence**: Cooking videos with multi-sentence descriptions

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

The automatic evaluation of video description/captioning is a long term and unsolved problem. The evaluation of DVC is required to measure two aspects:
1. the accuracy of localized events
2. that of generated captions for each event

The current evaluation framework of DVC is inspired by that of **dense image captioning (DIC)**, which generates captions that describe localized objects in an image comprehensively. In this evaluation framework, each generated caption is separately evaluated because the captions independently describe each localized object.

Thus, there is a significant difference between DVC and DIC in whether generated captions should consist of a story or not. However, the current evaluation framework of DVC, which is a simple extension of that of DIC, does not consider the temporal dependency between captions explicitly.

The research community has mainly used the following evaluation metrics:
- **ROUGE-L**: 最长公共子序列
- **METEOR**: 同义词匹配 ← ActivityNet 官方采用
- **BLEU**: N-gram 匹配
- **CIDEr**: 共识度
- **SPICE**: 语义图匹配
- **WMD**: 词向量距离

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
