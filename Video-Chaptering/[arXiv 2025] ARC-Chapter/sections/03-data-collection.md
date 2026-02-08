# 3. Data Collection and Annotation

> 来源: ARC-Chapter (arXiv 2025)

---

## 📄 原文

A significant challenge in developing strong video chaptering models is the scarcity of publicly available datasets with detailed, multi-level annotations. Existing datasets typically provide only sparse labels, such as video-level categories for video classification or coarse temporal segments with brief titles such as VidChapters-7M. To address this limitation and to facilitate research on hierarchical video chaptering and summarization, we introduce a new, richly annotated video chaptering dataset. This section details our data curation and annotation pipeline.

> 💡 **数据问题**: 现有数据集要么只有视频级标签（分类任务），要么只有粗略时间段+简短标题（如 VidChapters-7M）。ARC-Chapter 需要更丰富的多层级标注。

## 3.1 Data Curation

One of the key contributions of our work is the introduction of a new large-scale dataset, named VidAtlas, which is designed for the task of hierarchical video chaptering and summarization. Our primary goal is to construct a dataset that not only provides accurate chapter boundaries but also offers dense, multi-granularity textual descriptions for both individual chapters and the entire video.

> 💡 **VidAtlas 数据集定位**: 不仅要有准确的章节边界，还要有多粒度的文本描述（从章节级到视频级）。

Data Sourcing. We begin by sourcing videos from the video platform. The primary selection criterion is the presence of author-provided chapter markers. These markers, which include the start/end timestamps and a short title for each chapter, are manually defined by the video uploader. This approach provides us with a highly accurate human-verified ground truth for the temporal segmentation of videos, which is a significant foundation for our subsequent annotation efforts. The collected videos, which are long, well-structured, and information-dense, are ideal candidates for video chaptering.

> 💡 **数据来源巧妙之处**: 利用视频上传者自己标注的章节标记（YouTube/B站等平台支持的功能）作为 ground truth。这些标记是人工验证的准确时间分割，省去了昂贵的人工标注成本。

Filtering and Refinement. Starting with this initial collection, we apply several filtering criteria to guarantee the quality and diversity of our dataset for video understanding and chaptering. First, we retain videos whose durations lie between 2 minutes and 3 hours. This range excludes trivial short clips, which are unnecessary for chaptering, as well as overly long videos, which are often unstructured (e.g., live streams) and difficult to process due to the context-length limitations of our model. Second, we curate videos across a wide range of domains, including educational lectures, DIY tutorials, reviews & unboxings, interviews & podcasts, webinars & presentations, gaming & music albums, fitness & cooking and documentaries. This wide distribution of domains ensures that the dataset is not biased towards any specific genre and supports the development of more generalizable models.

> 💡 **数据筛选策略**:
> - **时长过滤**: 2 分钟 ~ 3 小时（排除过短的无需章节化的片段，和过长的直播）
> - **领域多样性**: 教育、DIY、评测、访谈、游戏、烹饪、纪录片等
> - **目的**: 避免领域偏差，提升模型泛化能力

![Figure 3](../images/figure3_full.jpg)
*Figure 3: 数据集统计：(a) VidAtlas 中视频时长（上）和章节时长（下）的分布。(b) VidAtlas 中视频主题分布。*

> 💡 **数据统计要点**:
> - 视频平均时长 16.8 分钟，章节平均时长 182 秒（~3 分钟）
> - 16 个主要类别，100+ 子类别
> - 包含 Games、Knowledge、Technology、Music、Life 等多样类别

## 3.2 Hierarchical Annotation

To generate high-quality video chaptering annotations, we design an automated annotation pipeline that leverages both multimodal content extraction and large language model (LLM)-based reasoning based on the videos with user-provided chapter makers, i.e. timestamps and brief title of each chapter. The illustration of our annotation pipeline is shown in Fig. 2.

> 💡 **标注流水线核心思路**: 利用用户提供的粗标注（时间戳+简短标题）作为锚点，通过多模态信息提取 + LLM 推理来生成丰富的层级标注。

Multimodal Information Extraction. Considering efficiency and cost, we avoid directly using multimodal large language models (MLLMs) for video annotation. Instead, we first extract multimodal information from video frames and audio, integrate this content, and then feed the result into text-only LLM for reasoning and annotation. Specifically, we use Whisper-v3 [29] to transcribe speech into text, segmented into sentences with the corresponding timestamps. In parallel, we uniformly sample video frames with a fixed sampling frame rate and employ Qwen2.5-VL-7B [4] to extract visual captions and on-screen text (OCR) for better understanding of the video content. Subsequently, the visual captions and ASR transcripts are temporally aligned based on their respective timestamps. This process allows us to interleave the textual content from both modalities into a unified chronologically ordered sequence. This multimodal transcript, together with

the original user-provided chapter timestamps and short titles, is fed into LLM for reasoning and structural segmentation.

> 💡 **多模态信息提取流程**:
> ```
> 视频 → Whisper-v3 → ASR 文本（带时间戳）
>       → Qwen2.5-VL-7B → 视觉描述 + OCR 文字
>                    ↓
>         按时间戳交错排列 → 统一的多模态文本序列
>                    ↓
>         + 用户章节标记 → 送入 text-only LLM
> ```
> **为什么不直接用 MLLM 处理视频？** 效率和成本考量。先提取信息转成文本，再用纯文本 LLM 推理，大幅降低计算开销。

LLM Reasoning and Chaptering. The LLM is prompted to analyze the transcript and reorganize the content into a structured set of chapters, each containing a comprehensive title, an abstract, an introduction, and precise temporal boundaries. Following this, we perform a verification step on the LLM's output to ensure that the generated chapter boundaries strictly adhere to the original timestamps. Building upon the verified structured chapter information, we further prompt the LLM to produce a comprehensive, timestamped narrative description for the entire video. Through this annotation pipeline, we can efficiently obtain accurate, multi-level video chapter segmentation and descriptive annotations. The resulting annotations form a dense, hierarchically organized representation of long-form videos, supporting a wide range of research tasks in video understanding, temporal reasoning, chaptering, and summarization.

> 💡 **LLM 标注的两步走**:
> 1. **第一步 — 结构化章节**: LLM 分析多模态文本，生成每个章节的 title + abstract + introduction + 时间边界 → 验证时间边界是否与原始标记一致
> 2. **第二步 — 视频描述**: 基于第一步结果，LLM 生成完整的时间对齐叙述描述
>
> **关键设计**: 时间边界始终锚定在用户原始标注上，LLM 只负责内容细化，不修改时间。

## 3.3 Dataset Statistics

We summarize the key statistics of our VidAtlas dataset and highlight the properties that make it suited for research on video chaptering and summarization. The dataset comprises 410k+ videos with an average duration of 16.8 minutes, amounting to more than 115k hours of diverse content. On average, each video is segmented into 5.5 chapters, with an average chapter duration of 182 seconds (approximately 3 minutes). Fig. 3a provides a detailed statistic of the duration distributions for both videos and chapters. Our dataset contains a wide spectrum of video and chapter lengths to ensure models are trained on a diverse temporal structures. This comprehensive video/chapter length distribution makes the models exposed to a variety of content length, from concise segments to hour-long narratives, forcing models to resolve both rapid topic shifts and sustained thematic segments. To mitigate genre bias, VidAtlas covers a wide array of topics, including 16 primary categories with over 100 subcategories, as shown in Fig. 3b. The categories of VidAtlas include Games, Knowledge, Technology, Music, Life, Animation, and Sports, together with other variety that captures long-tail topics. Videos in these categories are typically well-structured and information-dense, making them ideal for chaptering.

> 💡 **VidAtlas vs VidChapters-7M 对比**:
> | 维度 | VidAtlas | VidChapters-7M |
> |------|----------|----------------|
> | 视频数量 | 410K+ | 817K |
> | 总时长 | 115K 小时 | — |
> | 平均视频时长 | 16.8 分钟 | — |
> | 平均章节数/视频 | 5.5 | — |
> | 标注层级 | 3 级（标题/章节/描述） | 1 级（仅标题） |
> | 语言 | 中英双语 | 主要英语 |
> | 类别数 | 16 主类 + 100 子类 | — |
>
> VidAtlas 虽然视频数量少于 VidChapters-7M，但标注质量和丰富度远超。

---

## 💡 Section 总结

数据收集部分的核心创新在于**低成本高质量的标注流水线**：

1. **数据来源**: 利用平台用户自标注的章节标记，获得免费的人工时间分割 GT
2. **多模态提取**: Whisper-v3（ASR）+ Qwen2.5-VL-7B（视觉+OCR）→ 统一文本序列
3. **LLM 细化**: 纯文本 LLM 将粗标注扩展为三级层级标注
4. **时间锚定**: LLM 只细化内容，不修改时间边界，保证时间准确性

**启发**: 这种"人工粗标注 + 自动细化"的范式成本低、可扩展，值得在其他任务中借鉴。
