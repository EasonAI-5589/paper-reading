[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
介绍视频时长增长趋势、现有方法的局限性（Vid2Seq 固定采样 100 帧），以及 Chapter-Llama 的三大贡献。

---

According to a study by [47], the video durations uploaded to the popular online video sharing platform YouTube have increased steadily over the years. Videos have become longer since the first video upload in 2005 [20, 48]. In 2020, 25% of videos were estimated to be longer than 15 minutes, 5% more than 3 hours [47]. Long-form videos such as news, sports, educational, and vlog streams can often span extensive durations and cover multiple topics [100]. Finding specific content within increased video duration and volume makes efficient content navigation more important than ever.

> 💡 **背景**: YouTube 视频越来越长，2020 年 25% 超过 15 分钟，5% 超过 3 小时。长视频需要高效的内容导航。

However, much of the traditional video analysis research has focused on processing short videos of a few seconds [4, 16, 35, 57, 65, 70, 77, 81, 88, 90, 101, 113]. At the same time, the definition of long videos has changed within the past decade. Early works claimed processing 100 frames (i.e., a few seconds) to be long [63, 96] as opposed to ingesting up to 16 frames [86, 95]. With the introduction of datasets containing 1-5 minute videos [30, 38, 45, 58, 85, 129], several minutes were considered very long. Studying hour-long videos has only recently seen an interest in the context of movie description [32], video captioning [41], or grounding [33, 87]. Very recently, the work of [112] collected the VidChapters-7M dataset with videos spanning from minutes to hours, along with their user-defined video chapters, and proposed the video chapter generation task, automatically dividing a video into thematic sections (i.e., chapters) with descriptive concise chapter titles. Video chaptering, if achieved successfully, can offer a compelling solution to long content indexing, bypassing the current need for time-consuming manual annotation by video owners [112].

> 💡 **"长视频"定义的演变**:
> - 早期：100 帧（几秒）就算"长"
> - 之后：1-5 分钟算"很长"
> - 现在：小时级才是真正的长视频
> - VidChapters-7M [112] 是第一个大规模 video chaptering 数据集

![Figure 1](../images/ed539e227f825da1a3ac983e2d3ec39e447afb0cd3ffa30854935206b5cc22ef.jpg)
*Figure 1. Chapter-Llama: Our method generates automatic video chapters for hour-long videos by training a large language model (LLM) to predict chapter boundaries and titles. The LLM processes transcribed speech (ASR) and descriptive captions of key frames, which are sampled based on ASR content. This text-based approach, equipped with speech-based frame selection, enables efficient processing of long-form content.*

> 💡 **Figure 1 批读**:
> - 整体流程：视频 → ASR + Speech-based Frame Selection → Caption → LLM → Chapter Boundaries + Titles
> - 关键点：不是所有帧都做 caption，而是根据 ASR 内容选择关键帧
> - LLM 的输入是纯文本（带时间戳的 ASR 和 caption 交错排列）

In this paper, we address the challenge of automatic video chaptering with a simple yet effective framework designed to handle hour-long videos. Existing work for chaptering [112] relies on a dense video captioning model Vid2Seq [113], which combines multimodal inputs from video frames and ASR-based speech transcriptions. However, Vid2Seq operates on a fixed number of equally sampled frames (i.e., 100 frames), potentially missing important visual information. Furthermore, their approach based on transformer architecture uses video frame features directly, which requires learning a mapping from the visual modality to the textual modality. In contrast, our method is designed to address these limitations by (i) dynamically sampling keyframes from the video based on the speech content, and (ii) designing a purely text-based model leveraging image captioning to convert RGB frames into text.

> 💡 **Vid2Seq 的两个局限**:
> 1. 固定采样 100 帧 → 可能遗漏重要视觉信息
> 2. 直接用视频帧特征 → 需要学习 visual-to-text mapping
>
> **Chapter-Llama 的解决方案**:
> 1. 动态采样关键帧（基于语音内容）
> 2. 纯文本模型（用 image captioning 把帧转为文本）

Our approach leverages a pretrained LLM, which we finetune specifically for the video chaptering task to predict jointly the chapter boundary timestamps and chapter titles, both in text form. The appeal of our model lies in processing only textual data as input, allowing us effectively leverage the long-context understanding capabilities of the LLM to scale to long videos. In particular, we incorporate speech transcriptions from automatic speech recognition (ASR) and automatic frame captions. Captioning has been used for video understanding as an intermediate representation in recent works, but in the context of retrieval or question answering (QA) for shorter videos (maximum 3 minutes) [60, 98, 119, 124]. In longer videos, since captioning every frame is computationally prohibitive, we employ a speech-based frame selection strategy that scales efficiently while preserving important content. Similar in spirit to [44], we primarily use audio to determine keyframes, specifically bootstrapping with an LLM trained only with the speech inputs. However, even when transforming a video into text, LLMs have a limited context window, allowing a maximum number of tokens as input in a single forward pass. To mitigate context window limitations for very long video inputs, we simply perform an iterative prediction, sequentially processing the video, where each iteration typically operates on a window length of about an hour duration. We evaluate our approach on 'short' (0-15 min), 'medium' (15-30 min), and 'long' (30-60 min) videos from the VidChapters-7M dataset [112], demonstrating significant improvements over the state of the art across multiple metrics, including temporal boundary accuracy and semantic relevance of chapter titles. Our experiments show that finetuning the LLM, our speech-based frame selection strategy, and the integration of modalities from both speech and captions are crucial for achieving high-quality video chaptering results.

> 💡 **方法要点**:
> - 纯文本输入 → 充分利用 LLM 长上下文能力
> - Caption 作为中间表示之前只用在短视频（≤3min），这里首次用在小时级视频
> - 对于超长视频，用 iterative prediction（滑动窗口处理）
> - 关键发现：finetuning + speech-based selection + 多模态融合缺一不可

Our contributions are the following: (i) We introduce Chapter-Llama: our framework leverages a pretrained LLM and finetunes for the underexplored task of video chaptering by transforming the video input into text form through ASR and captioning. (ii) We scale efficiently to hour-long videos by incorporating a speech-based frame sampling strategy, captioning only a subset of the video frames. (iii) Our simple and effective approach outperforms the state of the art on the recent VidChapters-7M benchmark by a large margin (e.g., 45.3 vs 26.7 F1 score). These results are complemented by a comprehensive set of experiments analyzing our components.

> 💡 **三大贡献**:
> 1. Chapter-Llama 框架：LLM + ASR + captioning 的纯文本方案
> 2. Speech-based frame sampling：高效扩展到小时级视频
> 3. 大幅超越 SOTA：45.3 vs 26.7 F1（+70%）

---

## 🔖 Section 总结

### 核心洞察
1. 现有方法（Vid2Seq）受限于固定帧采样和视觉特征映射
2. "视频→文本→LLM"是处理长视频的有效范式
3. Speech-based frame selection 同时解决了效率和信息保留两个问题
