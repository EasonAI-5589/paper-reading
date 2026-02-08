[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 阐述长视频理解的三大挑战（跨模态建模、数据稀缺、评估指标不足），并提出 ARC-Chapter 的三大贡献：百万级数据集、半自动层级标注管线、GRACE 指标。

---

The exponential proliferation of long-form video content, including educational lectures, vlogs, live streams, and meeting recordings—poses significant challenges for automatic content understanding. Video chaptering [35; 44] has emerged as a promising solution, segmenting videos into navigable and semantically coherent chapters. This enables efficient content retrieval, summarization, and enhanced user interaction, which are critical for managing and consuming large-scale video data.

> 💡 **背景**: Video chaptering = 把长视频分成有语义的 chapter，每个 chapter 有时间戳+标题。核心应用场景：YouTube 视频章节、会议记录导航。

---

Despite notable advances in segmenting short videos (usually within five minutes) for tasks such as action segmentation [8; 22; 27; 32; 39], temporal event localization [16; 54], and dense video captioning [19; 38; 46], the structuring of hour-long videos remains a formidable challenge. First, modeling sophisticated semantics across multimodal inputs, including visual and audio streams—over extended temporal horizons requires robust and scalable architectures. Second, the scarcity of large-scale datasets with fine-grained annotations hinders the development and evaluation of effective chaptering models. Third, existing evaluation metrics [10; 19] often fail to capture the semantic granularity of chapter boundaries, leading to suboptimal matching and similarity scoring between predicted and ground-truth segments [10].

> 💡 **三大挑战**:
> 1. **长时序建模**：小时级视频的多模态语义理解
> 2. **数据稀缺**：缺少大规模细粒度标注
> 3. **评估缺陷**：现有指标（如 SODA）的 one-to-one 匹配不适合 chaptering 的粒度灵活性

---

In this technical report, we introduce ARC-Chapter, a comprehensive framework designed to address the unique challenges of long-form video structuring. As illustrated in Fig. 1, ARC-Chapter enables the segmentation of lengthy videos into navigable chapters and generates hierarchical summaries that capture both coarse and fine-grained content structure. Our work makes three primary contributions. First, we advance the scalability of video chaptering by developing the first large-scale model trained on one million long videos, totaling 400,000 hours of content. This dataset is fifty times larger than those used in previous studies [35], allowing our model to generalize across diverse video domains and formats. Second, we propose a semi-automatic annotation pipeline for hierarchical summaries, which leverages easily accessible human-annotated coarse labels. This pipeline integrates automatic speech recognition (ASR) derived transcripts with timestamped visual

> 💡 **三大贡献**:
> 1. **规模**：100 万长视频，40 万小时，比之前（~20k）大 50 倍
> 2. **标注管线**：利用用户提供的粗标注（chapter markers）+ ASR + 视觉 caption → LLM 生成层级标注
> 3. **GRACE 指标**：many-to-one 匹配，容忍粒度差异

---

![Figure 1](../images/2a72d77e0c5249282bb10f53b978aa4515a296d1a978577042ce495ecc54ccfb.jpg)
*Figure 1: ARC-Chapter 的能力展示。给定视频，模型生成三级输出：1) Short Title — 简短标题；2) Structural Chapter — 包含 title、abstract、introduction 的结构化标注；3) Timestamp-Aligned Video Description — 时间对齐的细粒度描述。*

> 💡 **Figure 1 批读**:
> - 这个三级层级设计很聪明：Short Title 用于快速导航，Structural Chapter 用于深度理解，Video Description 用于精确检索
> - 这种层级结构类似于书的目录(Short Title) → 章节摘要(Structural) → 正文(Description)

---

elements, enabling a holistic and multimodal understanding of video content. Third, we introduce GRACE, a novel granularity-robust evaluation metric designed to address the semantic misalignment issues prevalent in existing chaptering benchmarks. GRACE provides a more accurate assessment of chapter boundary quality by accounting for varying levels of semantic granularity.

Our extensive experiments demonstrate the effectiveness of ARC-Chapter, which establishes a new state-of-the-art on both Chinese and English long-form video chaptering benchmarks. Specifically, ARC-Chapter substantially outperforms previous methods on the VidChapters-7M test sets (e.g., CIDEr: 100.9→186.6; F1: 45.3→59.3; SODA: 19.3→30.6). We validate the importance of multimodality, showing that our full model surpasses video-only and audio-only variants by 7.7 and 5.3 points on SODA, respectively. Furthermore, pretraining on our large-scale dataset significantly enhances transferability, evidenced by notable performance gains on downstream tasks like YouCook2 and ActivityNet Captions. Crucially, our work is the first to identify a clear scaling law in video chaptering: model performance consistently improves with increased training data and label density. This finding refutes previous observations that performance saturates on smaller datasets (~20k samples) [35] and suggests a promising direction for future research.

> 💡 **关键发现**:
> - **多模态优势**：Video+ASR 比 Video-only 高 7.7 SODA，比 ASR-only 高 5.3 SODA
> - **Scaling Law**：性能随数据量持续提升，推翻了 Chapter-Llama 在 ~20k 样本就饱和的结论
> - **迁移性**：预训练后在 YouCook2、ActivityNet Captions 上也创新 SOTA

---

The remainder of this report is structured as follows: Section 2 reviews related works; Section 3 describes the dataset and annotation pipeline; Section 4 details our methodology and model architecture; Section 5 presents experimental results and analysis; Section 6 concludes.

---

## 🔖 Section 总结

### 核心洞察
1. 长视频 chaptering 之前被数据规模限制（~20k 样本），ARC-Chapter 用 100 万视频打破这个瓶颈
2. 半自动标注管线的关键：利用视频平台已有的用户 chapter markers 作为种子标注
3. 评估指标 GRACE 解决了 chaptering 特有的"粒度模糊"问题——同一视频可以粗分或细分
