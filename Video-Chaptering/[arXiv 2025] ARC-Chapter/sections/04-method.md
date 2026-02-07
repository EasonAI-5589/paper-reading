# 4. ARC-Chapter Method

## 📄 原文逐段解析

---

## 4.1 Overall Framework

### 4.1.1 Base Model

> We leverage **Qwen2.5-VL-7B** as our base model, enhancing its capabilities to process and structure video content into chapters.
>
> ==Base Model：Qwen2.5-VL-7B==

### 4.1.2 三种输入

> The model unifies three inputs:
> 1. An **instruction prompt** that specifies the task of input modalities and output schema
> 2. A sequence of **sampled video frames** that provide appearance, layout and on-screen text (including subtitles)
> 3. A **timestamp-aligned ASR transcript** from audio
>
> ==三输入：Prompt + Video Frames + ASR Transcript==

> While both the video and ASR transcript inputs are optional, the model requires at least one modality to be provided.
>
> ==Video 和 ASR 可选，但至少提供一个==

### 4.1.3 编码方式

> Frames are embedded with **Qwen2.5-VL vision encoder** and translated into visual tokens, while ASR transcript is tokenized as plain text with explicit timestamps.
>
> ==Vision Encoder 编码帧 → visual tokens；ASR 直接 tokenize==

> The **vision encoder is kept frozen** and the language model is instruction tuned on VidAtlas to specialize in video chaptering.
>
> ==Vision Encoder 冻结，只微调 LLM==

---

### 4.1.4 Prompt Design

> To handle the diverse requirements of different inputs and outputs of the model, we design a set of **18 distinct prompt templates**.
>
> ==18 种 Prompt 模板==

**三个维度：**

| 维度 | 选项 |
|------|------|
| **Language** | English / Chinese |
| **Input Modality** | ASR-only / Video-only / ASR+Video |
| **Output Format** | Short Titles / Structural Chapters / Video Descriptions |

> This allows for ablation studies and adaptation to scenarios where one modality may be absent or noisy.
>
> ==灵活适配：某个模态缺失或有噪声时可用==

---

### 4.1.5 Video Input 处理

> To balance temporal coverage and context budget, we follow the setup of Qwen2.5-VL and **cap the visual stream at 768 frames** sampled at up to 1 fps.
>
> ==最多 768 帧，采样率 ≤1 fps==

**采样策略：**
| 视频时长 | 采样策略 |
|----------|----------|
| < 12.8 分钟 | 1 fps |
| ≥ 12.8 分钟 | 均匀降采样到 768 帧 |

> The sampling strategy retains coarse global coverage for hour-long content, ensuring sufficient representation to capture the high-level semantic shifts necessary for the chaptering task.
>
> ==策略目的：保留粗粒度全局覆盖，捕捉高层语义转换==

**动态 Token 分配：**

> Since the model context length is shared across modalities, we **dynamically adjust the per-frame token allowance** according to the input of ASR transcript.
>
> ==根据 ASR 输入动态调整每帧的 token 数==

| 输入模式 | 帧分辨率 | 原因 |
|----------|----------|------|
| Video-only | 高分辨率 | 保留 OCR、字幕等细节 |
| Video+ASR | 低分辨率 | 给 ASR 留 context 空间 |

**时间戳增强：**

> To enhance temporal awareness, we randomly **overlay timestamps onto the video frames**, making the model more sensitive to the video timeline.
>
> ==随机在帧上叠加时间戳 → 增强时间感知==

---

### 4.1.6 ASR Input 处理

**为什么用文本而非音频特征？**

> Although integrating raw audio features or learned audio embeddings from pretrained ASR models (e.g. Whisper) is attractive, it presents severe scalability challenges for long-form video.
>
> ==问题：原始音频特征可扩展性差==

**具体数字：**

> Whisper-style audio encoder produces **50 audio tokens per second**, a 60-minute audio therefore produces **180k tokens**, far exceeding feasible LLM context budgets.
>
> ==60分钟音频 → 180k tokens，超出 LLM context 限制==

> Furthermore, synchronizing fixed-rate audio features with dynamically sampled video frames poses an additional alignment problem.
>
> ==同步问题：固定速率音频 vs 动态采样视频帧==

**解决方案：**

> To address these practical constraints, we opt to use **ASR transcripts as a highly effective proxy** for the audio modality.
>
> ==用 ASR 文本替代原始音频==

> Text is significantly more information-dense. Therefore, the ASR transcript of a long audio segment occupies far fewer tokens than its raw feature representation.
>
> ==文本信息密度更高，tokens 更少==

**具体实现：**

> We use **Whisper-large-v3** to generate timestamped ASR transcripts. The model provides sentence-level segments with corresponding start timestamps.
>
> ==Whisper-large-v3，句子级分段==

> We formulate the ASR text and timestamp of each segment as:
> ```
> start time (hh:mm:ss): <ASR text>
> ```
>
> ==格式：`时间戳: ASR文本`==

---

## 4.2 Training Strategy

### 4.2.1 Training Objective

> We perform **supervised instruction tuning** on VidAtlas and VidChapter-7M using all prompt templates.
>
> ==SFT on VidAtlas + VidChapter-7M==

> The training objective is the standard **autoregressive next-token prediction loss** over the target sequence.
>
> ==标准自回归 next-token loss==

**公式：**
$$\mathcal{L} = -\sum_{i=1}^{n} \log P(y_i | y_{<n}, X_{prompt}, X_{video}, X_{asr})$$

> During training, the **vision encoder is frozen** to enable a larger context length, while **all parameters of the large language model are optimized**.
>
> ==Vision Encoder 冻结，LLM 全参数优化==

---

### 4.2.2 Adaptive Modality Dropping

> To enable a single model to perform well under various deployment conditions, we adopt an **adaptive modality dropping strategy** during training.
>
> ==训练时随机丢弃模态 → 单模型适应多场景==

**三种配置：**
| 配置 | 输入 | 目的 |
|------|------|------|
| Video + ASR | 两者都有 | 完整多模态 |
| Video-only | 仅视频 | 处理无 ASR 场景 |
| ASR-only | 仅 ASR | 处理无视频场景 |

> This strategy **prevents the model from becoming overly reliant on a single modality** and ensures it develops a comprehensive understanding from all available input modalities.
>
> ==避免模型过度依赖单一模态==

> Consequently, a single trained model can be deployed to handle videos under various conditions during inference.
>
> ==单一模型适应多种推理场景==

---

## 4.3 Evaluation Metrics

### 4.3.1 现有指标的问题

> We observe that the primary metrics such as **SODA**, originally developed for dense video captioning, are **not well-suited** for the video chaptering task.
>
> ==SODA 不适合章节任务==

**问题1：一对一匹配**

> While SODA enforces a **one-to-one matching** between predicted and ground-truth events to suppress redundancy in overlapping event detection, video chaptering requires segmenting videos into sequential, **non-overlapping chapters**.
>
> ==SODA 用于检测重叠事件，但章节是连续非重叠的==

**问题2：粒度歧义**

> Chaptering annotations often exhibit **granularity ambiguity**: different annotators may segment the same video at varying levels of detail.
>
> ==不同标注者可能用不同粒度标注同一视频==

> - Some may annotate **coarse-grained chapters** (e.g., by day in a travel vlog)
> - Others may provide **fine-grained chapters** (e.g., by each visited site within a day)
>
> ==例：旅行 vlog 可以按"天"或按"景点"划分==

> This results in multiple valid annotation granularities for the same content.
>
> ==同一内容可能有多种有效的粒度==

---

### 4.3.2 GRACE 指标

> We propose **GRACE**, a metric tailored for video chaptering. It introduces a **many-to-one (set-to-one) matching paradigm**.
>
> ==GRACE = 多对一匹配范式==

**核心思想：**

> Each ground-truth (predicted) chapter can be matched with **a set of** predicted (ground-truth) chapters.
>
> ==一个 GT 章节可以匹配多个预测章节，反之亦然==

**公式：**

$$\text{GRACE} = \sum_{(P_i, G_i) \in M(P,G)} \varphi(P_i, G_i) \cdot \text{BERTScore}(P_i, G_i)$$

$$\varphi(P_i, G_i) = \frac{1}{|P_i||G_i|} \sum_{p \in P_i, g \in G_i} \text{IOU}(p, g)$$

**约束条件：**
- $P_i \cap P_j = \emptyset$ （预测组不重叠）
- $\cup(P_i) = P$ （覆盖所有预测）
- $G_i \cap G_j = \emptyset$ （GT组不重叠）
- $\cup(G_i) = G$ （覆盖所有GT）
- $\min(|P_i|, |G_i|) = 1$ （至少一侧是单个章节）

**最优匹配：**

> We adopt the **dynamic time warping algorithm (DTW)** to achieve the optimal matching $M(P,G)$, with IOU between two chapters being used as the matching criteria.
>
> ==用 DTW 动态规划找最优匹配==

---

### 4.3.3 GRACE 优势

| 优势 | 说明 |
|------|------|
| **粒度鲁棒** | 不同标注风格都能公平评估 |
| **语义保真** | 奖励捕获完整内容的模型 |
| **人类对齐** | 更符合人类对章节边界的判断 |

---

## 4.4 Reinforcement Learning with GRPO

### 4.4.1 动机

> While supervised fine-tuning (SFT) achieves strong performance, the standard cross-entropy loss **does not directly optimize** for the primary objective of video chaptering: **temporal accuracy**.
>
> ==SFT 的交叉熵 loss 无法直接优化时间准确性==

> To further enhance the model's temporal localization capabilities, we introduce a subsequent reinforcement learning phase using the **GRPO algorithm**.
>
> ==引入 GRPO 强化学习优化时间定位==

### 4.4.2 奖励函数

> We leverage our proposed GRACE metric. However, to specifically sharpen the model's ability to predict accurate timestamps, we formulate a **simplified, temporal-only reward** by **omitting the semantic BERTscore** component.
>
> ==奖励函数 = GRACE 的时间部分（去掉 BERTScore）==

$$R = \sum_{(P_i, G_i) \in M(P,G)} \varphi(P_i, G_i)$$

> This reward directly reflects the quality of the temporal segmentation.
>
> ==直接反映时间分割质量==

### 4.4.3 训练设置

| 设置项 | 值 | 原因 |
|--------|-----|------|
| 输入模态 | **仅 Video** | 强化视觉推理能力 |
| 训练数据 | **90k 视频**（中英混合） | 多样性 |
| 输出格式 | 三种全覆盖 | 全面优化 |
| 初始化 | SFT best model | 保留语言能力 |
| KL 系数 | **0.01** | 防止偏离 SFT 太远 |

> The KL divergence coefficient is set to 0.01 to ensure that the policy does not stray far from the robust language generation capabilities learned during SFT, thereby balancing temporal refinement with descriptive quality.
>
> ==KL 正则化：平衡时间优化和语言质量==

---

## 💡 Key Takeaways

1. **架构**：Qwen2.5-VL-7B，冻结 Vision Encoder，微调 LLM
2. **输入处理**：768 帧上限，ASR 用文本（避免 180k tokens 爆炸）
3. **Prompt 设计**：18 种模板，覆盖语言×模态×输出格式
4. **训练策略**：SFT + Adaptive Modality Dropping + GRPO
5. **GRACE 指标**：多对一匹配，DTW 找最优，解决粒度歧义
6. **GRPO**：仅优化时间准确性，KL=0.01 防止语言退化

---

*[返回论文目录](../README.md)*
