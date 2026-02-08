[← 返回 README](../README.md)

# 4 ARC-Chapter

## 📌 预览
模型架构：基于 Qwen2.5-VL-7B，冻结 vision encoder，instruction tuning LLM。输入：prompt + 视频帧 + ASR 转录。训练策略：SFT + 自适应模态 dropout + GRPO 强化学习。评估指标：GRACE（many-to-one 匹配）。

---

## 4.1 Overall Framework

We leverage Qwen2.5-VL-7B [5] as our base model, enhancing its capabilities to process and structure video content into chapters. The architecture of our model is illustrated in Fig. 4. The model unifies three inputs: 1) an instruction prompt that specifies the task of input modalities and output schema. 2) a sequence of sampled video frames that provide appearance, layout and on-screen text (including subtitles which often align with the ASR transcript), and 3) a timestamp-aligned ASR transcript from audio. While both the video and ASR transcript inputs are optional, the model requires at least one modality to be provided. Frames are embedded with Qwen2.5-VL vision encoder and translated into visual tokens, while ASR transcript is tokenized as plain text with explicit timestamps. The vision encoder is kept frozen and the language model is instruction tuned on VidAtlas to specialize in video chaptering.

> 💡 **模型架构要点**:
> - 基座模型：Qwen2.5-VL-7B（7B 参数，不算大）
> - Vision encoder 冻结（为了支持更长的上下文）
> - 三种输入可选组合：Video-only / ASR-only / Video+ASR
> - 训练方式：instruction tuning（SFT）

---

![Figure 4](../images/02c2243528c120860794a2198f238a6c1bbd52ae51d8740bb6ae570866500cdd.jpg)
*Figure 4: 视频 chaptering 模型架构概览。输入包括任务 prompt、采样视频帧和带时间戳的 ASR 转录。视频帧经过冻结的 vision encoder 处理。生成的视觉特征与 tokenized prompt 和 ASR 文本一起送入可训练的 MLLM，输出不同格式的 chapter。*

> 💡 **Figure 4 批读**:
> - 标准的 MLLM 架构：frozen vision encoder + trainable LLM
> - 亮点在于输入设计：prompt 指定语言、模态、输出格式（共 18 种 prompt 模板）
> - 输出灵活：Short Title / Structural Chapter / Video Description

---

**Prompt Design.** The model's behavior is guided by carefully designed prompts that specify the desired task and output format. To handle the diverse requirements of different inputs and outputs of the model, we design a set of 18 distinct prompt templates. These prompts are constructed based on three axes: language in source video, input modality, and desired output format.

• Language: We support English and Chinese to match the language of the source video.
• Input Modality: The prompt specifies whether the model should rely on ASR-only, video-only, or both video and ASR inputs. This allows for ablation studies and adaptation to scenarios where one modality may be absent or noisy.

> 💡 **18 种 Prompt 模板**:
> - 2 (语言: EN/ZH) × 3 (模态: ASR/Video/Both) × 3 (输出: Title/Structural/Description) = 18
> - 这种多模板设计让单一模型适应各种部署场景

• Output Format: We define three distinct output structures: (a) Short Titles for concise chapter markers, (b) Structured Chapters that include a title, abstract, and introduction for each chapter, and (c) Video Descriptions that provide a dense, timestamp-aligned summary of the entire video.

---

**Video Input.** To balance temporal coverage and context budget, we follow the setup of Qwen2.5-VL and cap the visual stream at 768 frames sampled at up to 1 fps. That is to say, videos shorter than 12.8 minutes are sampled with 1 fps, while longer videos are uniformly down-sampled to 768 frames with a lower fps. The sampling strategy retains coarse global coverage for hour-long content, ensuring sufficient representation to capture the high-level semantic shifts necessary for the chaptering task. Since the model context length is shared across modalities, we dynamically adjust the per-frame token allowance according to the input of ASR transcript. For video-only inputs we use a higher frame resolution (higher token budget per frame) so that small text (OCR and subtitles) and fine-grained visual cues are preserved. When ASR is provided alongside video, we reduce frame resolution (thus reducing the number of visual tokens) so that the combined input of visual tokens and ASR text fits the maximum context length of MLLM. This dynamic allocation is implemented by adjusting image scaling and patch-tokenization parameters at preprocessing time. Moreover, to enhance temporal awareness, we randomly overlay timestamps onto the video frames, making the model more sensitive to the video timeline.

> 💡 **视频输入设计**:
> - 最多 768 帧，≤1 fps → 12.8 min 内的视频用 1fps，更长的均匀降采样
> - **动态分辨率**：Video-only 时用高分辨率（保留 OCR/字幕），Video+ASR 时降低分辨率（给 ASR 留 token 预算）
> - **Timestamp overlay**: 随机在帧上叠加时间戳，增强时间感知——简单但有效的 trick

---

**ASR Input.** Although integrating raw audio features or learned audio embeddings from pretrained ASR models (e.g.Whisper [29]) is attractive, it presents severe scalability challenges for long-form video. For example, while Whisper-style audio encoder produces 50 audio tokens per second, a 60-minute audio therefore produces 180k tokens, far exceeding feasible LLM context budgets without aggressive compression or specialized audio-to-token aggregation. Furthermore, synchronizing fixed-rate audio features with dynamically sampled video frames poses an additional alignment problem. To address these practical constraints, we opt to use ASR transcripts as a highly effective proxy for the audio modality. Text is significantly more information-dense. Therefore,

the ASR transcript of a long audio segment occupies far fewer tokens than its raw feature representation. This makes processing hour-long videos computationally feasible for both training and inference. Although such a paradigm introduces an extra step for offline ASR transcription, we believe that trading a modest amount of offline processing time for the ability to handle long-form audio under strict context-length budgets is worthwhile. In our implementation, we use Whisper-large-v3 [29] to generate timestamped ASR transcripts. The model provides sentence-level segments with corresponding start timestamps. We formulate the ASR text and timestamp of each segment as start time (hh:mm:ss): \<ASR text\>. The normalized ASR transcript is then passed to the model either alone (ASR-only) or together with visual tokens (ASR+Video), providing dense semantic information that is particularly useful for temporal boundary detection and chaptering.

> 💡 **为什么用 ASR 文本而非 audio embeddings**:
> - 60 min 音频 → 180k audio tokens（Whisper encoder 50 tokens/s），远超 LLM 上下文
> - ASR 文本信息密度高得多，token 数少几个数量级
> - 代价：需要离线 ASR 转录（用 Whisper-large-v3）
> - 格式：`start_time (hh:mm:ss): <ASR text>`，保留时间戳信息

---

## 4.2 Training Strategy

**Training Objective.** We perform supervised instruction tuning on VidAtlas and VidChapter-7M using all prompt templates. The training objective is the standard autoregressive next-token prediction loss over the target sequence. Given a multimodal input sequence consisting of a prompt X_prompt, video frames X_video, and an ASR transcript X_asr (video stream X_video and ASR streams X_asr are optional), the model is trained to maximize the log-likelihood of the target output sequence Y = (y₁, y₂, ..., yₙ) (e.g., a list of chapter titles, a structured chapter object, or a timestamped description):

![Training Loss](../images/abbbff843122b0bd48c34b8a85af3513bce357daa19b01789e25008da54dea34.jpg)

where y_{<i} represents the preceding ground-truth tokens. During training, the vision encoder is frozen to enable a larger context length, while all parameters of the large language model are optimized with the training objective.

> 💡 **训练目标**:
> - 标准 autoregressive next-token prediction loss
> - 冻结 vision encoder，只训练 LLM 参数
> - 在 VidAtlas + VidChapter-7M 上用全部 18 种 prompt 模板训练

---

**Adaptive Modality Dropping.** To enable a single model to perform well under various deployment conditions, we adopt an adaptive modality dropping strategy during training. For each training sample, we randomly configure the input with a certain probability to be one of three types: 1) Video + ASR: Both modalities are provided to the model. 2) Video-only: The ASR transcript is omitted, forcing the model to rely solely on visual information. and 3) ASR-only: The video frames are omitted, requiring the model to understand the content based on the transcript alone. This strategy prevents the model from becoming overly reliant on a single modality and ensures it develops a comprehensive understanding from all available input modalities. Consequently, a single trained model can be deployed to handle videos under various conditions during inference (whether only a video is available, only transcript is provided, or both are present), without requiring specialized models for each scenario.

> 💡 **自适应模态 Dropout**:
> - 训练时随机 drop 一种模态，让模型学会在任意模态组合下工作
> - 类似于 Dropout 的思想，但作用在模态层面
> - 好处：单一模型支持 Video-only / ASR-only / Both，部署灵活

---

## 4.3 Evaluation Metrics

Evaluation metrics can be divided into two aspects: (1) the accuracy of segmentation (e.g., Precision, Recall, and tIOU [20]), and (2) joint metrics that assess both segmentation and chapter captioning (e.g., CIDEr [20], SODA [10]). However, we observe that the primary metrics such as SODA, originally developed for dense video captioning, are not well-suited for the video chaptering task. While SODA enforces a one-to-one matching between predicted and ground-truth events to suppress redundancy in overlapping event detection, video chaptering requires segmenting videos into sequential, non-overlapping chapters. Furthermore, chaptering annotations often exhibit granularity ambiguity: different annotators may segment the same video at varying levels of detail—some may annotate coarse-grained chapters (e.g., by day in a travel vlog), while others may provide fine-grained chapters (e.g., by each visited site within a day). This results in multiple valid annotation granularities for the same content.

> 💡 **SODA 的问题**:
> - SODA 用 one-to-one 匹配，设计初衷是 dense video captioning（事件可能重叠）
> - 但 chaptering 是 non-overlapping 的顺序分割，而且存在"粒度模糊"
> - 例如旅行 vlog：按天分 vs 按景点分，都是合理的 chaptering

---

To address these challenges, we propose GRACE, a metric tailored for video chaptering. It introduces a many-to-one (set-to-one) matching paradigm, allowing each ground-truth (predicted) chapter to be matched with a set of predicted (ground-truth) chapters. As illustrated in Fig. 5, for each ground-truth chapter, GRACE evaluates the temporal overlap and semantic similarity between the chapter and its matched prediction set, using established language similarity metrics (e.g., BERTscore [51]) for textual comparison. Specifically, we aim to find a best many-to-one mapping M which splits both ground-truth set G and prediction set P into several pairs of groups {(Pᵢ, Gᵢ)}ᵢ₌₁ᴷ, followed by group-based similarity calculation:

![Figure 5](../images/figure5_full.jpg)
*Figure 5: SODA (one-to-one) 和 GRACE (many-to-one) 匹配策略对比。One-to-one 匹配可能漏掉 p₂ 和 g₂ 等重要事件，而 many-to-one 策略考虑所有预测和 GT 事件。*

> 💡 **Figure 5 批读**:
> - 左边 SODA：每个 GT 只能匹配一个 prediction，如果粒度不同就会漏掉
> - 右边 GRACE：允许多个 prediction 匹配一个 GT（或反过来），更灵活
> - 这解决了"标注者粗分、模型细分"（或反过来）的问题

---

![GRACE Equation 1](../images/ecb3996a03557cb203c31a73b2c9df2400ae4f9cceabd6b88d92979e91b7aa58.jpg)

![Phi Equation 2](../images/6882a43e96265b04e4dc3767c67d52e56564ff89d986d1d96def039d00bc68b4.jpg)

![Constraint Equation 3](../images/c4a3b62878297ee3efecd61d0a9343edf3763214ccc0a762429ab32839ca7763.jpg)

> 💡 **GRACE 公式解读**:
> - **Eq. 1**: GRACE = Σ (时间重叠度 φ × BERTscore)，对所有匹配组求和
> - **Eq. 2**: φ 是组内所有 prediction-GT 对的平均 IoU
> - **Eq. 3**: 约束条件——组之间不重叠，所有 chapter 都被分配，每组至少有一端是单个 chapter（many-to-one，不是 many-to-many）
> - 用 **DTW（动态时间规整）** 找最优匹配 M

where Pᵢ and Gᵢ represent groups of chapters. When calculating the BERTScore between two groups, we first concatenate all captions within each group into a single sentence, then compute the BERTScore between the two merged sentences. We adopt the dynamic time warping algorithm (DTW) [6; 31] to achieve the optimal matching M(P, G), with IOU between two chapters being used as the matching criteria.

GRACE provides a more accurate and human-aligned assessment of chaptering models. This design confers several advantages: (1) robustness to annotation granularity, enabling fair evaluation across diverse annotation styles; (2) improved semantic fidelity, rewarding models that capture the full scope of ground-truth chapters; and (3) closer alignment with human judgment of chapter boundaries and content.

> 💡 **GRACE 的三个优势**:
> 1. 对粒度鲁棒：粗分和细分都能公平评估
> 2. 语义保真：奖励覆盖所有 GT 内容的模型
> 3. 与人类判断更一致

---

## 4.4 Reinforcement Learning with GRPO

While supervised fine-tuning (SFT) achieves strong performance, the standard cross-entropy loss does not directly optimize for the primary objective of video chaptering: temporal accuracy. To further enhance the model's temporal localization capabilities, we introduce a subsequent reinforcement learning phase using the GRPO algorithm [12].

The core of this phase is a reward function designed to directly incentivize precise chapter boundary prediction. We leverage our proposed GRACE metric, which holistically evaluates both temporal alignment and semantic content. However, to specifically sharpen the model's ability to predict accurate timestamps of segmented chapters, we formulate a simplified, temporal-only reward by omitting the semantic BERTscore component from Equation (1). For a given ground-truth chapter set G and a model-generated set P, the reward R is calculated by summing the temporal alignment scores φ over the optimal matching M(P, G) found via DTW:

![Reward Equation 4](../images/7b8124edd476232dbe1b15b2784856b470a1ae82f5aaf682601089abfd0dfee1.jpg)

This reward directly reflects the quality of the temporal segmentation, providing a clear and targeted optimization objective.

> 💡 **GRPO 强化学习**:
> - SFT 的 cross-entropy loss 不直接优化时间精度
> - GRPO 用 GRACE 的时间部分（去掉 BERTscore）作为 reward → 直接优化时间对齐
> - KL 系数 0.01：防止偏离 SFT 学到的语言生成能力

---

Due to the significant context length required for multimodal inputs, and to specifically bolster the model's ability to reason from visual cues, we conduct this RL training phase using only the video modality. We select a diverse subset of 90k videos from both Chinese and English SFT data, ensuring that training samples cover all three output formats: short titles, structural chapters, and timestamped video description. We initialize the model with the weights from our best-performing SFT model and further optimize it using GRPO. The KL divergence coefficient is set to 0.01 to ensure that the policy does not stray far from the robust language generation capabilities learned during SFT, thereby balancing temporal refinement with descriptive quality.

> 💡 **GRPO 训练设置**:
> - 只用 video modality 训练 RL（上下文长度限制）
> - 90k 视频子集，中英双语，覆盖三种输出格式
> - 有趣发现（见 Section 5）：虽然只用 video 训练 RL，ASR 和 Video+ASR 的性能也提升了 → 跨模态迁移

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 基座模型 | Qwen2.5-VL-7B |
| 最大帧数 | 768 frames |
| 采样率 | ≤1 fps |
| Prompt 模板数 | 18 |
| GRPO 训练数据 | 90k videos |
| KL 系数 | 0.01 |

### 核心洞察
1. **架构简洁**：不需要复杂的新架构，Qwen2.5-VL + instruction tuning 即可
2. **模态 Dropout 是关键**：让单一模型适应所有模态组合
3. **GRACE 指标**：many-to-one 匹配 + DTW 寻优，解决 chaptering 的粒度模糊问题
4. **GRPO 提升时间精度**：SFT → RL 的两阶段训练，temporal reward 直接优化边界预测
