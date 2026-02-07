# 2. Related Work

## 2.1 Dense Video Captioning

DVC 的目标是获取视频中所有事件的**简洁且连贯**的描述，需要理解整个视频内容并进行上下文推理。

### 现有方法

| 方法 | 架构 | 特点 |
|------|------|------|
| Wang et al. | 双向 LSTM + Context Gating | 利用过去和未来上下文 |
| Zhou et al. | Self-Attention 端到端 | 桥接事件检测和 captioning |
| Mun et al. | Pointer Networks | 减少事件 proposal 数量 |

### 相关数据集

| 数据集 | 领域 | 特点 |
|--------|------|------|
| ActivityNet Captions | 通用 | 最大规模 |
| YouCook II | 烹饪 | caption 时序依赖强 |
| VideoStory | 社交视频 | 短视频故事描述 |
| TACoS | 烹饪 | 多句描述 |

> 💡 这些数据集都用类似的评测框架，可能导致评估不准确

## 2.2 Dense Caption Evaluation

### DVC vs DIC (Dense Image Captioning)

| 任务 | 目标 | 评测需求 |
|------|------|---------|
| **DIC** | 描述图像中的物体 | 独立评估每个 caption |
| **DVC** | 描述视频的故事 | 需要考虑时序依赖 |

现有 DVC 评测只是 DIC 评测的简单扩展，**没有考虑时序依赖**。

### 常用评测指标

| 指标 | 来源 | 特点 |
|------|------|------|
| METEOR | MT | 同义词匹配，与人工评估相关性高 |
| BLEU | MT | N-gram 匹配 |
| CIDEr | Image Captioning | 共识度 |
| ROUGE-L | Summarization | 最长公共子序列 |
| SPICE | Image Captioning | 语义图匹配 |

> 💡 实验表明 **METEOR** 和 **CIDEr** 与人工评估相关性最高，因此 ActivityNet Challenge 采用 METEOR
