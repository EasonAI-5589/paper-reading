[← 返回 README](../README.md)

# 5 Audio-centric Token Compression

## 📌 预览
音频 token 压缩方法沿用视觉压缩的四大类框架，但针对音频 1D 时间信号和频谱表示的特点做了适配。还讨论了音频特有的频谱/时间冗余和静音/噪声问题。

---

For audio LLMs, the demand for longer context arises from the need to process higher sampling rates and extended durations of audio.

The extraction of information from the audio modality can be categorized according to the format of audio representation: (1) continuous sequence modeling: this approach utilizes a pre-trained audio encoder, typically models like Whisper (Radford et al., 2023) or Conformer (Gulati et al., 2020), to produce continuous audio embeddings; (2) discrete sequence modeling: this method transforms the input audio signal into discrete audio tokens, usually via vector quantization, where continuous audio features are encoded into a learnable codebook. Mainstream methods include HuBERT (Hsu et al., 2021) and EnCodec (Défossez et al., 2022; Zeghidour et al., 2021).

> 💡 **音频两种建模方式**:
> 1. **连续序列**: Whisper/Conformer → 连续 embedding
> 2. **离散序列**: HuBERT/EnCodec → 向量量化 → 离散 codebook token
> 
> 第二种天然通过 tokenizer 设计减少 token 数，但不是本综述的重点。

The second category inherently reduces the number of tokens by the design of the tokenizer structure and the codebook. Nevertheless, detailed exploration of these specific design considerations falls outside the purview of this survey.

Audio, a 1D signal representing amplitude over time, must be transformed into a suitable format for deep learning models, especially when integrating with MLLMs. MLLMs often leverage architectures designed for 2D data (like images) or general sequences. While the raw waveform is the source, spectrograms (especially Mel-spectrograms) are frequently the preferred representation for audio in MLLMs. This preference arises because spectrograms allow the application of processing techniques similar to those used for images, thereby facilitating multimodal learning.

> 💡 **音频表示**: 虽然音频是 1D 信号，但实践中常转为 Mel-spectrogram（2D 表示），这样可以复用图像处理技术。

Consequently, much like the visual modality, we categorize audio token compression methods as follows:

---

## 5.1 Transformation-based audio-centric Compression

Following the visual modality's categories, we can classify methods based on their downsampling operations:

### 5.1.1 Token Stacking

Similar to the pixel unshuffle operation in 2D image processing, this approach for audio LLMs token compression involves stacking multiple consecutive tokens along the hidden dimension of the token. This effectively reduces the total number of tokens. Notably, HTS-AT (Chen et al., 2022), an early example of audio token stacking for classification tasks within audio transformers, utilized 2D pixel-unshuffling on the 2D features extracted from Mel spectrograms to reduce audio tokens. More recent methods such as SLAM-ASR (Ma et al., 2024c), LLaMA-Omni (Fang et al., 2024), Llama-AVSR (Cappellazzo et al., 2025a) and others (Fathullah et al., 2024) stack the audio token. Since these token stacking operations alter the hidden dimension, an MLP is typically used to realign the dimension for compatibility with other modalities.

> 💡 **Token Stacking ≈ 音频版 Pixel Unshuffle**: 把连续多个音频 token 沿隐藏维度堆叠 → token 数减少、维度增加 → MLP 对齐维度。代表: SLAM-ASR、LLaMA-Omni。

### 5.1.2 Pooling

Another common technique for reducing the number of audio tokens is pooling. Models like Qwen2-audio (Chu et al., 2024b) and Qwen2.5-Omni (Xu et al., 2025b) leverage pooling layers with a stride of 2 to directly decrease the length of the audio representation in a parameter-free manner. This effectively downsamples the audio features, leading to a more compact token sequence. Extending this concept, Llama-MTSK (Cappellazzo et al., 2025b) employs a matryoshka-based training approach for flexible token compression. It trains the model with multi-scale audio and video information by applying average pooling or token stacking at different rates to the initial tokens. This enables Llama-MTSK to dynamically adjust the number of tokens processed during inference, balancing compression and performance within a single model.

> 💡 **音频池化**:
> - **Qwen2-Audio / Qwen2.5-Omni**: stride=2 池化，无参数，简洁高效
> - **Llama-MTSK**: Matryoshka 式训练——不同压缩率训练同一模型 → 推理时灵活调整 token 数

### 5.1.3 Temporal Convolution

For audio tokens, 1D convolutions applied across the temporal dimension can reduce the number of tokens. This method simultaneously allows for the alignment of the hidden dimension for subsequent LLM. Approaches like SpeechVerse (Das et al., 2024), Baichuan-Audio (Li et al., 2025c), OSUM (Geng et al., 2025), and LUCY (Gao et al., 2025) have employed this technique, often resulting in a downsampled audio representation with an effective sampling rate of 12.5 Hz.

> 💡 **1D 时间卷积**: 类比图像的空间卷积，但在时间维度操作。多个方法最终达到 12.5 Hz 的有效采样率。

These methods demonstrate how insights from image compression, particularly involving transformations, can be effectively applied to the audio domain to achieve more efficient token representation for large models.

---

## 5.2 Similarity-based audio-centric Compression

Similarity-based compression methods aim for each audio token to carry unique information rather than being overly redundant. Similar to the ToMe (Bolya et al., 2022) method used in vision transformers (ViT), A-ToMe (Li et al., 2023f) inserts a token merge module between the multihead self-attention (MHSA) and feed-forward network (FFN). This module merges adjacent audio tokens that have high cosine similarity.

> 💡 **A-ToMe**: 音频版 ToMe。在 MHSA 和 FFN 之间插入 merge 模块，合并高余弦相似度的相邻音频 token。直接迁移了视觉领域的 idea。

---

## 5.3 Attention-based audio-centric Compression

For audio tasks, attention-based methods are also effectively utilized to compress tokens.

### 5.3.1 Attention in Encoder

Top-K (Lee & Lee, 2025) is a token selection method operating within the audio spectrogram transformer block. It retains only the top K audio tokens ranked by the magnitude of their attention scores. This prunes less attentive tokens, focusing on those with higher relevance as determined by the self-attention mechanism.

> 💡 **音频 Encoder 内注意力剪枝**: Top-K 方法在音频频谱 Transformer 中按 attention 分数排序，保留 Top-K token。

### 5.3.2 Attention in Decoder

SpeechPrune (Lin et al., 2025b), works in the LLM backbone. It prunes audio tokens based on attention scores provided by the first transformer layer. By utilizing the initial layer's attention, SpeechPrune efficiently identifies and discards less crucial tokens early in the processing pipeline, aiming to reduce computational load and improve efficiency for subsequent layers without significant loss of information.

> 💡 **SpeechPrune**: 在 LLM 第一层的 attention 分数指导下剪枝音频 token。早期剪枝 → 后续所有层都受益。

---

## 5.4 Query-based audio-centric Compression

Audio feature representations can also be compressed using other modalities or learned query mechanisms. Analogous to image LLMs, these methods can be broadly categorized into token distillation and cross-modal selection, based on whether learned queries are explicitly employed.

### 5.4.1 Token Distillation

This category leverages learnable query tokens to distill comprehensive audio information into a compact, fixed-length representation.

Video-LLaMA (Zhang et al., 2023a) and SALMONN series (Tang et al., 2024; Sun et al., 2024b) employ an audio Q-former to transform variable-length audio inputs into a fixed-length sequence of learnable queries, thereby condensing audio information for the LLM. MMCE-Qformer (Xue et al., 2024) compresses acoustic information by utilizing learnable queries to extract global acoustic context from contextual audio embeddings. Concurrently, a cross-attention mechanism, guided by input text embeddings, captures local acoustic context relevant to each text token. This dual approach distills both broad and specific audio features into compact, text-relevant representations. MMS-LLaVA (Yeo et al., 2025) reduces multimodal token length for efficient speech LLMs. It first halves the sequence length with an Early AV-Fusion Module, which combines visual and audio features. Subsequently, an AV Q-Former further compresses these fused features into a fixed number of queries, effectively capturing full speech context to bridge the token gap with text.

> 💡 **音频 Token Distillation 方法**:
> - **Video-LLaMA / SALMONN**: Audio Q-Former → 变长音频 → 固定长度 learnable queries
> - **MMCE-Qformer**: 双通道——learnable queries 提取全局声学上下文 + 文本引导的 cross-attention 捕获局部上下文
> - **MMS-LLaVA**: AV-Fusion 先融合视觉+音频 → AV Q-Former 再压缩

### 5.4.2 Cross-Modal Selection

Similar to the visual modality, audio token compression can also be guided by information from other modalities. Speechprune (Lin et al., 2025b), for example, leverages audio-text correlation to identify semantically important audio segments. This is achieved by calculating a cross-modal similarity matrix based on cosine similarity, which then guides the compression of audio tokens. This approach ensures that the most relevant audio information is retained.

> 💡 **SpeechPrune 的双重身份**: 既是 attention-based（用 LLM 第一层 attention）也是 cross-modal selection（用音频-文本余弦相似度矩阵）。

---

## 5.5 Discussion about Specific Redundancy of Audio

Distinct from visual modalities, audio signals exhibit high sampling rates and significant spectro-temporal correlations. Even brief speech segments yield hundreds of tokens, a substantial portion of which encapsulate overlapping or redundant information. This section delineates redundancy patterns inherent to audio—specifically spectral redundancy, temporal redundancy, and silence or repetitive noise—to establish a foundation for efficient token compression in audio LLMs.

### 5.5.1 Spectral and Temporal Redundancy

Like video, audio exhibits intrinsic temporal structure. Consequently, compressing tokens along the temporal dimension is a well-founded strategy (Someki et al., 2025). Concurrently, given that the high sampling rate of audio generates dense token sequences that burden computational efficiency, it is imperative to mitigate spectral redundancy while preserving semantic integrity. Recently, Bhati et al. (2025) pioneered token pruning for audio LLMs by utilizing spectral features for segmentation before addressing temporal redundancy. Their method achieves a substantial reduction in token density with minimal fine-tuning requirements.

> 💡 **音频特有冗余**:
> - **时间冗余**: 类似视频帧间冗余，连续音频段信息重叠
> - **频谱冗余**: 高采样率产生密集 token，但语义信息集中在特定频段
> - **Bhati et al. (2025)**: 先用频谱特征分段 → 再处理时间冗余，开创了音频 LLM token 剪枝

### 5.5.2 Silence and Audio Noise

Many ASR pipelines explicitly remove long pauses and noise, effectively performing coarse-grained pruning at the waveform level. Nevertheless, end-to-end systems still receive audio-token sequences with muted or noisy segments. Although some tokens are redundant, others carry contextual cues beneficial to downstream tasks; consequently, developing principled audio-token pruning remains a promising yet challenging avenue for future work.

> 💡 **静音和噪声**: 传统 ASR 会在波形层面去除静音/噪声（粗粒度剪枝），但端到端系统仍需处理这些段落。挑战在于：有些「看似冗余」的 token 实际包含有用的上下文线索。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 1D 卷积后有效采样率 | 12.5 Hz |
| Qwen2-Audio 池化 stride | 2 |

### 核心洞察
1. 音频 token 压缩方法大量借鉴视觉领域（ToMe → A-ToMe, Pixel Unshuffle → Token Stacking）
2. 音频有独特的频谱冗余和静音/噪声问题，需要专门处理
3. 音频 Q-Former 是最主流的蒸馏方法，用于将变长音频压缩为固定长度
4. 音频 token 压缩研究相比图像/视频还很初期，是一个有潜力的方向
