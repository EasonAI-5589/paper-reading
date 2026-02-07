# 6. Experiments

## 6.1 Experimental Settings

### 原文

We used the ActivityNet Captions dataset, which contains 20k YouTube videos. The dataset consists of 10,024, 4,915 and 5,044 videos for training, validation and test data, respectively. We evaluated our evaluation framework only on the validation data because the test data is not publicly available. Each video in the validation data has on average 3.52 human-written captions with start/end time annotations. The average number of words in a caption is 13.54.

We evaluated the following two state-of-the-art DVC systems:

- **End-to-end transformer-based system [Zhou et al.]**: The end-to-end transformer-based models could detect events by considering the whole video information and generate consistent captions for the events simultaneously. The number of output captions per video is **228.21** on average.
- **LSTM-based system [Wang et al.]**: The bidirectional LSTM-based encoder-decoder models with a context gating mechanism. The number of output captions per video is **97.10** on average.

**Detecting inappropriate captions**: We randomly selected int(m × |G|) captions and evaluated with m = 0.1, 0.5, 1.0, 2.0, 10, and "all".

**Detecting incorrect ordering**: We evaluated (a) Swap (swapping adjacent captions), and (b) Shuffle (randomly shuffling order).

### 译文

我们使用了 ActivityNet Captions 数据集，该数据集包含 20k 个 YouTube 视频。数据集分别由 10,024、4,915 和 5,044 个视频组成，用于训练、验证和测试数据。我们仅在验证数据上评估我们的评估框架，因为测试数据不公开可用。验证数据中的每个视频平均有 3.52 个人工编写的描述，带有开始/结束时间标注。描述中的平均词数为 13.54。

我们评估了以下两个最先进的 DVC 系统：

- **端到端 Transformer 系统 [Zhou 等人]**：端到端 Transformer 模型可以通过考虑整个视频信息同时检测事件并为事件生成一致的描述。每个视频输出描述的平均数量为 **228.21**。
- **LSTM 系统 [Wang 等人]**：带有上下文门控机制的双向 LSTM 编码器-解码器模型。每个视频输出描述的平均数量为 **97.10**。

**检测不合适的描述**：我们随机选择 int(m × |G|) 个描述，并用 m = 0.1, 0.5, 1.0, 2.0, 10 和 "all" 进行评估。

**检测错误顺序**：我们评估了 (a) Swap（交换相邻描述），和 (b) Shuffle（随机打乱顺序）。

---

### 理解与批注

#### 数据集统计

| 指标 | 数值 |
|------|------|
| 视频总数 | 20K |
| 验证集 | 4,915 |
| 平均 caption 数/视频 | 3.52 |
| 平均词数/caption | 13.54 |

#### 评估的系统对比

| 系统 | 架构 | 平均生成 caption 数 |
|------|------|-------------------|
| E2E Transformer | 端到端 Transformer | **228.21** |
| LSTM | BiLSTM + Context Gating | **97.10** |

> ⚠️ 两个系统都生成了远超参考（3-4个）的 caption 数量

---

## 6.2 Results

### 6.2.1 Detecting Inappropriate Caption Number

#### 原文 (Table 1 关键数据)

From the results, the scores of the current evaluation framework do not change significantly even when the number of captions changes. In particular, there is only a slight difference between m = 1 (appropriate) and "all" (inappropriate). In contrast, SODA gave low scores for too many and too few captions. When we utilized a small m, precision became high, while recall became low. Precision became low and recall became high when we utilized a large m. Thus, SODA can penalize inadequate captions.

#### 译文

从结果来看，即使描述数量发生变化，当前评估框架的分数也没有显著变化。特别是，m = 1（合适）和 "all"（不合适）之间只有轻微差异。相比之下，SODA 对过多和过少的描述给出了低分。当我们使用较小的 m 时，精确率变高，而召回率变低。当我们使用较大的 m 时，精确率变低，召回率变高。因此，SODA 可以惩罚不充分的描述。

---

### 理解与批注

#### 关键结果表

| m | Current (E2E) | SODA(c) F1 (E2E) |
|---|---------------|------------------|
| 0.1 | 3.78 | 1.47 |
| 0.5 | 4.04 | 3.41 |
| **1.0** | 4.10 | **4.02** |
| 2.0 | 4.14 | 3.83 |
| 10 | 4.18 | 1.70 |
| All | 4.19 | 0.63 |

#### 关键发现 1: Current 无法区分好坏

```
Current 分数变化: 3.78 → 4.19 (只差 0.41)
                  ↑ m=0.1   ↑ m=All
```

> 💡 生成多少 caption 分数都差不多，这是**严重的问题**

#### 关键发现 2: SODA 正确惩罚

```
SODA(c) 分数变化:
  m=0.1  → 1.47  (太少，低分 ✓)
  m=1.0  → 4.02  (刚好，最高分 ✓)
  m=All  → 0.63  (太多，最低分 ✓)
```

---

### 6.2.2 Oracle Comparison

#### 原文 (Table 2)

E2E Transformer outperformed LSTM in terms of both METEOR and Self-BLEU scores. Although LSTM outperformed E2E Transformer with the current evaluation framework in Table 1, SODA correctly ranks E2E Transformer higher.

| System | METEOR | Self-BLEU |
|------|--------|-----------|
| E2E Transformer | **21.3** | **79.5** |
| LSTM | 13.43 | 90.6 |

#### 译文

E2E Transformer 在 METEOR 和 Self-BLEU 分数方面都优于 LSTM。尽管在表 1 中使用当前评估框架时 LSTM 优于 E2E Transformer，但 SODA 正确地将 E2E Transformer 排名更高。

---

### 理解与批注

#### Current 的错误排名

| 评测 | E2E Transformer | LSTM | 正确？ |
|------|-----------------|------|--------|
| Current | 4.19 | **4.97** | ❌ LSTM 更高 |
| SODA | **4.02** | 3.15 | ✅ E2E 更高 |
| Oracle METEOR | **21.3** | 13.43 | 真实能力 |

> 💡 SODA 的排名与 Oracle 一致，Current 的排名是错误的

---

### 6.2.3 Detecting Incorrect Ordering

#### 原文 (Table 3)

The percentage decreases for Shuffle with SODA(c) are in range of 47-57%, while those with Current are in range of 18-33%. SODA is more sensitive to the incorrect ordering.

| Setting | Current | SODA(c) |
|------|---------|---------|
| Correct | 16.1 | 17.8 |
| Swap | 14.5 (-10.2%) | 14.5 (-18.9%) |
| Shuffle | 10.8 (-33.1%) | 7.66 (**-57.0%**) |

#### 译文

使用 SODA(c) 时 Shuffle 的百分比下降在 47-57% 范围内，而使用 Current 时在 18-33% 范围内。SODA 对错误顺序更敏感。

---

### 理解与批注

#### SODA 对顺序更敏感

| 操作 | Current 下降 | SODA 下降 |
|------|-------------|-----------|
| Swap (交换相邻) | -10.2% | **-18.9%** |
| Shuffle (随机打乱) | -33.1% | **-57.0%** |

> 💡 SODA 对顺序错误的惩罚几乎是 Current 的 2 倍

---

## 6.3 Manual Evaluation

### 原文

We randomly selected 50 videos and showed them to 12 crowdsourced workers. The accuracies of SODA and the current evaluation framework against human judgment are:

| Task | SODA Accuracy | Current Accuracy |
|---------|------------|----------------|
| E2E vs LSTM | **0.76** | 0.66 |
| Swap vs Shuffle | **0.94** | 0.72 |

In the former human judgment, E2E Transformer obtained better results for 80% of the 50 videos. In the latter human judgment, Swap obtained better results for 94% of the 50 videos.

### 译文

我们随机选择了 50 个视频并展示给 12 名众包工作者。SODA 和当前评估框架相对于人工判断的准确率如下：

| 任务 | SODA 准确率 | Current 准确率 |
|---------|------------|----------------|
| E2E vs LSTM | **0.76** | 0.66 |
| Swap vs Shuffle | **0.94** | 0.72 |

在前一个人工判断中，E2E Transformer 在 50 个视频中的 80% 获得了更好的结果。在后一个人工判断中，Swap 在 50 个视频中的 94% 获得了更好的结果。

---

### 理解与批注

#### 人工评估验证

| 任务 | 人工判断 | SODA 一致性 | Current 一致性 |
|------|---------|------------|---------------|
| E2E vs LSTM | E2E 80% 更好 | 76% | 66% |
| Swap vs Shuffle | Swap 94% 更好 | **94%** | 72% |

> ✅ SODA 与人工评估的一致性显著高于 Current
