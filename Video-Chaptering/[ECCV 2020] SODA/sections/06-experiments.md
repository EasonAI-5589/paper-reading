# 6. Experiments

## 6.1 Experimental Settings

### 原文

We used the ActivityNet Captions dataset, which contains 20k YouTube videos. The dataset consists of 10,024, 4,915 and 5,044 videos for training, validation and test data, respectively. We evaluated our evaluation framework only on the validation data because the test data is not publicly available. Each video in the validation data has on average 3.52 human-written captions with start/end time annotations. The average number of words in a caption is 13.54.

We evaluated the following two state-of-the-art DVC systems:

- **End-to-end transformer-based system [Zhou et al.]**: The end-to-end transformer-based models could detect events by considering the whole video information and generate consistent captions for the events simultaneously. The number of output captions per video is **228.21** on average.
- **LSTM-based system [Wang et al.]**: The bidirectional LSTM-based encoder-decoder models with a context gating mechanism. The number of output captions per video is **97.10** on average.

**Detecting inappropriate captions**: We randomly selected int(m × |G|) captions and evaluated with m = 0.1, 0.5, 1.0, 2.0, 10, and "all".

**Detecting incorrect ordering**: We evaluated (a) Swap (swapping adjacent captions), and (b) Shuffle (randomly shuffling order).

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

### 6.2.1 检测不合适 Caption 数量

#### 原文 (Table 1 关键数据)

| m | Current (E2E) | SODA(c) F1 (E2E) |
|---|---------------|------------------|
| 0.1 | 3.78 | 1.47 |
| 0.5 | 4.04 | 3.41 |
| **1.0** | 4.10 | **4.02** |
| 2.0 | 4.14 | 3.83 |
| 10 | 4.18 | 1.70 |
| All | 4.19 | 0.63 |

From the results, the scores of the current evaluation framework do not change significantly even when the number of captions changes. In particular, there is only a slight difference between m = 1 (appropriate) and "all" (inappropriate). In contrast, SODA gave low scores for too many and too few captions.

---

### 理解与批注

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

### 6.2.2 Oracle 对比

#### 原文 (Table 2)

| 系统 | METEOR | Self-BLEU |
|------|--------|-----------|
| E2E Transformer | **21.3** | **79.5** |
| LSTM | 13.43 | 90.6 |

E2E Transformer outperformed LSTM in terms of both METEOR and Self-BLEU scores. Although LSTM outperformed E2E Transformer with the current evaluation framework in Table 1, SODA correctly ranks E2E Transformer higher.

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

### 6.2.3 检测错误顺序

#### 原文 (Table 3)

| 设置 | Current | SODA(c) |
|------|---------|---------|
| Correct | 16.1 | 17.8 |
| Swap | 14.5 (-10.2%) | 14.5 (-18.9%) |
| Shuffle | 10.8 (-33.1%) | 7.66 (**-57.0%**) |

The percentage decreases for Shuffle with SODA(c) are in range of 47-57%, while those with Current are in range of 18-33%. SODA is more sensitive to the incorrect ordering.

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

| 比较任务 | SODA 准确率 | Current 准确率 |
|---------|------------|----------------|
| E2E vs LSTM | **0.76** | 0.66 |
| Swap vs Shuffle | **0.94** | 0.72 |

---

### 理解与批注

#### 人工评估验证

| 任务 | 人工判断 | SODA 一致性 | Current 一致性 |
|------|---------|------------|---------------|
| E2E vs LSTM | E2E 80% 更好 | 76% | 66% |
| Swap vs Shuffle | Swap 94% 更好 | **94%** | 72% |

> ✅ SODA 与人工评估的一致性显著高于 Current
