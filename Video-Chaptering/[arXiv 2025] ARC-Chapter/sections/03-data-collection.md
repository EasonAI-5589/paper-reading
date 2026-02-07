# 3. Data Collection and Annotation

## 📄 原文逐段解析

### 3.0 Overview

> A significant challenge in developing strong video chaptering models is the scarcity of publicly available datasets with detailed, multi-level annotations.
>
> ==核心挑战：缺乏公开的多层级详细标注数据集==

> Existing datasets typically provide only sparse labels, such as video-level categories for video classification or coarse temporal segments with brief titles such as VidChapters-7M.
>
> ==现有数据集问题：标签稀疏（只有视频级分类或粗粒度章节短标题）==

> To address this limitation and to facilitate research on hierarchical video chaptering and summarization, we introduce a new, richly annotated video chaptering dataset.
>
> ==解决方案：引入新的丰富标注数据集==

---

## 3.1 Data Curation

### 3.1.1 数据集命名

> One of the key contributions of our work is the introduction of a new large-scale dataset, named **VidAtlas**, which is designed for the task of hierarchical video chaptering and summarization.
>
> ==VidAtlas = 专为层级章节化和摘要设计的大规模数据集==

### 3.1.2 数据集目标

> Our primary goal is to construct a dataset that not only provides accurate chapter boundaries but also offers dense, multi-granularity textual descriptions for both individual chapters and the entire video.
>
> ==目标：准确章节边界 + 密集多粒度文本描述（章节+整体视频）==

### 3.1.3 数据来源

> We begin by sourcing videos from the video platform. The primary selection criterion is the presence of **author-provided chapter markers**.
>
> ==来源：视频平台上带有作者标注章节的视频==

> These markers, which include the start/end timestamps and a short title for each chapter, are manually defined by the video uploader.
>
> ==原始标注：上传者手动定义的时间戳 + 短标题==

> This approach provides us with a highly accurate human-verified ground truth for the temporal segmentation of videos.
>
> ==优势：人工验证的高精度 ground truth==

### 3.1.4 筛选和精炼

> **时长筛选：**
> We retain videos whose durations lie between **2 minutes and 3 hours**. This range excludes trivial short clips (unnecessary for chaptering) as well as overly long videos (often unstructured like live streams).
>
> ==时长范围：2分钟 ~ 3小时（排除太短的和直播等无结构长视频）==

> **领域多样性：**
> We curate videos across a wide range of domains, including:
> - Educational lectures
> - DIY tutorials
> - Reviews & unboxings
> - Interviews & podcasts
> - Webinars & presentations
> - Gaming & music albums
> - Fitness & cooking
> - Documentaries
>
> ==覆盖 16 大类 100+ 子类，避免领域偏差==

---

## 3.2 Hierarchical Annotation

### 3.2.1 标注流程概览

> To generate high-quality video chaptering annotations, we design an automated annotation pipeline that leverages both multimodal content extraction and large language model (LLM)-based reasoning.
>
> ==自动化标注流程 = 多模态内容提取 + LLM 推理==

### 3.2.2 多模态信息提取

> Considering efficiency and cost, we avoid directly using multimodal large language models (MLLMs) for video annotation.
>
> ==设计考虑：避免直接用 MLLM（成本太高）==

**具体步骤：**

> **ASR 提取：**
> We use **Whisper-v3** to transcribe speech into text, segmented into sentences with the corresponding timestamps.
>
> ==ASR：Whisper-v3，句子级分段 + 时间戳==

> **视觉提取：**
> We uniformly sample video frames with a fixed sampling frame rate and employ **Qwen2.5-VL-7B** to extract visual captions and on-screen text (OCR).
>
> ==视觉：均匀采样帧 + Qwen2.5-VL-7B 提取视觉描述和 OCR==

> **时间对齐：**
> The visual captions and ASR transcripts are temporally aligned based on their respective timestamps. This process allows us to interleave the textual content from both modalities into a unified chronologically ordered sequence.
>
> ==融合：按时间戳对齐，交错成统一时序序列==

```
时间线 ────────────────────────────►
ASR:     [句子1] ... [句子2] ... [句子3] ...
Visual:  [描述A] ....... [描述B] .......
         ↓
融合:    [ASR1, Visual-A, ASR2, Visual-B, ASR3, ...]
```

### 3.2.3 LLM 推理和章节化

> The LLM is prompted to analyze the transcript and reorganize the content into a structured set of chapters.
>
> ==LLM 分析融合序列 → 生成结构化章节==

**每个章节包含：**
| 字段 | 说明 |
|------|------|
| Comprehensive Title | 全面的标题 |
| Abstract | 章节摘要 |
| Introduction | 详细介绍 |
| Temporal Boundaries | 精确时间边界 |

> Following this, we perform a **verification step** on the LLM's output to ensure that the generated chapter boundaries strictly adhere to the original timestamps.
>
> ==验证步骤：确保生成的边界与原始时间戳一致==

### 3.2.4 时间戳描述生成

> Building upon the verified structured chapter information, we further prompt the LLM to produce a comprehensive, timestamped narrative description for the entire video.
>
> ==基于验证后的章节，进一步生成带时间戳的完整视频描述==

---

## 3.3 Dataset Statistics

### 3.3.1 规模统计

| 统计项 | 数值 |
|--------|------|
| 视频数量 | **410k+** |
| 总时长 | **115k 小时** |
| 平均视频长度 | 16.8 分钟 |
| 平均章节数 | 5.5 个/视频 |
| 平均章节时长 | 182 秒 (~3分钟) |

### 3.3.2 时长分布

> Our dataset contains a wide spectrum of video and chapter lengths to ensure models are trained on diverse temporal structures.
>
> ==视频和章节时长分布广泛 → 模型接触多样时序结构==

> This comprehensive distribution makes the models exposed to:
> - Concise segments (短片段)
> - Hour-long narratives (小时级叙事)
> - Rapid topic shifts (快速话题转换)
> - Sustained thematic segments (持续主题片段)

### 3.3.3 领域分布

> VidAtlas covers a wide array of topics, including **16 primary categories** with **over 100 subcategories**.
>
> ==16 大类，100+ 子类==

**主要类别：**
Games, Knowledge, Technology, Music, Life, Animation, Sports, ...

> Videos in these categories are typically well-structured and information-dense, making them ideal for chaptering.
>
> ==这些类别的视频通常结构良好、信息密集，适合章节化==

---

## 💡 Key Takeaways

1. **数据来源**：视频平台上带用户章节标注的视频（人工验证 GT）
2. **筛选标准**：2分钟~3小时，16大类100+子类，避免偏差
3. **标注流程**：Whisper-v3 (ASR) + Qwen2.5-VL (视觉) → 时间对齐 → LLM 推理
4. **输出层级**：Short Title → Abstract → Introduction → Video Description
5. **验证机制**：确保生成边界与原始时间戳一致

---

## 📊 VidAtlas vs 其他数据集

| 数据集 | 视频数 | 时长 | 标注 |
|--------|--------|------|------|
| ActivityNet Captions | 20k | - | Dense captions |
| YouCook2 | 2k | - | Procedure steps |
| VidChapters-7M | 817k | - | 短标题 |
| **VidAtlas** | **410k+** | **115k h** | **层级标注** |

---

*[返回论文目录](../README.md)*
