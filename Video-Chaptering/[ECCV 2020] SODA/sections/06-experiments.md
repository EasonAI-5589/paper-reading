# 6. Experiments

## 6.1 实验设置

### 数据集

**ActivityNet Captions**:
- 20K YouTube 视频
- 训练/验证/测试: 10,024 / 4,915 / 5,044
- 平均 3.52 captions/视频
- 平均 13.54 词/caption

### 评估的系统

| 系统 | 架构 | 平均 caption 数 |
|------|------|----------------|
| **E2E Transformer** | 端到端 Transformer | 228.21 |
| **LSTM** | BiLSTM + Context Gating | 97.10 |

### 评测指标

| 指标 | 说明 |
|------|------|
| **Current** | ActivityNet 官方评测 |
| **SODA (a)** | $\tau = 0.9, 0.7, 0.5, 0.3$ 平均 |
| **SODA (b)** | $\tau = 0$ |
| **SODA (c)** | IoU 加权 cost |

### 实验设计

1. **检测不合适 caption 数量**: 随机采样 $m \times |\mathcal{G}|$ 个 caption，测试 $m = 0.1, 0.5, 1.0, 2.0, 10, \text{all}$
2. **检测错误顺序**: 测试原始 caption、Swap（交换相邻）、Shuffle（随机打乱）

## 6.2 检测不合适 Caption 数量

### E2E Transformer 结果

| m | Current | SODA (c) F1 |
|---|---------|-------------|
| 0.1 | 3.78 | 1.47 |
| 0.5 | 4.04 | 3.41 |
| **1.0** | 4.10 | **4.02** |
| 2.0 | 4.14 | 3.83 |
| 10 | 4.18 | 1.70 |
| All | 4.19 | 0.63 |

### 关键发现

1. **Current**: 几乎不变（3.78 → 4.19），无法区分好坏
2. **SODA**: 只有 $m=1.0$ 时分数最高，太多/太少都被惩罚

### Oracle 比较

| 系统 | METEOR | Self-BLEU |
|------|--------|-----------|
| E2E Transformer | 21.3 | 79.5 |
| LSTM | 13.43 | 90.6 |

> E2E Transformer 潜力更强，但用 Current 评测时 LSTM 分数更高（不合理）
> SODA 正确反映了 E2E Transformer > LSTM

## 6.3 检测错误顺序

| 设置 | Current | SODA (c) |
|------|---------|----------|
| **Correct** | 16.1 | 17.8 |
| **Swap** | 14.5 (-10.2%) | 14.5 (-18.9%) |
| **Shuffle** | 10.8 (-33.1%) | 7.66 (-57.0%) |

### 关键发现

- **Shuffle** 时 SODA 下降 **57%**，Current 只下降 33%
- SODA 对顺序错误**更敏感**

## 6.4 人工评估

### 实验设计

- 50 个随机视频
- 12 个众包工作者
- 打分 -2 到 2

### 结果

| 比较 | 人工结果 | SODA 准确率 | Current 准确率 |
|------|---------|-------------|----------------|
| E2E vs LSTM | E2E 80% 更好 | **0.76** | 0.66 |
| Swap vs Shuffle | Swap 94% 更好 | **0.94** | 0.72 |

> ✅ SODA 与人工评估更一致
