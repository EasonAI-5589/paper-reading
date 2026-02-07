# 1. Introduction

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

According to a study by [47], the video durations uploaded to the popular online video sharing platform YouTube have increased steadily over the years. Videos have become longer since the first video upload in 2005 [20, 48]. In 2020, $2 5 \%$ of videos were estimated to be longer than 15 minutes, $5 \%$ more than 3 hours [47]. Long-form videos such as news, sports, educational, and vlog streams can often span extensive durations and cover multiple topics [100]. Finding specific content within increased video duration and volume makes efficient content navigation more important than ever.

However, much of the traditional video analysis research has focused on processing short videos of a few seconds [4, 16, 35, 57, 65, 70, 77, 81, 88, 90, 101, 113]. At the same time, the definition of long videos has changed within the past decade. Early works claimed processing 100 frames (i.e., a few seconds) to be long [63, 96] as opposed to ingesting up to 16 frames [86, 95]. With the introduction of datasets containing 1-5 minute videos [30, 38, 45, 58, 85, 129], several minutes were considered very long. Studying hour-long videos has only recently seen an interest in the context of movie description [32], video captioning [41], or grounding [33, 87]. Very recently, the work of [112] collected the VidChapters-7M dataset with videos spanning from minutes to hours, along with their userdefined video chapters, and proposed the video chapter generation task, automatically dividing a video into thematic sections (i.e., chapters) with descriptive concise chapter titles. Video chaptering, if achieved successfully, can offer a compelling solution to long content indexing, bypassing the current need for time-consuming manual annotation by video owners [112].

![](images/ed539e227f825da1a3ac983e2d3ec39e447afb0cd3ffa30854935206b5cc22ef.jpg)  
Figure 1. Chapter-Llama: Our method generates automatic video chapters for hour-long videos by training a large language model (LLM) to predict chapter boundaries and titles. The LLM processes transcribed speech (ASR) and descriptive captions of key frames, which are sampled based on ASR content. This text-based approach, equipped with speech-based frame selection, enables efficient processing of long-form content.

In this paper, we address the challenge of automatic video chaptering with a simple yet effective framework designed to handle hour-long videos. Existing work for chaptering [112] relies on a dense video captioning model Vid2Seq [113], which combines multimodal inputs from video frames and ASR-based speech transcriptions. However, Vid2Seq operates on a fixed number of equally sampled frames (i.e., 100 frames), potentially missing important visual information. Furthermore, their approach based on transformer architecture uses video frame features directly, which requires learning a mapping from the visual modality to the textual modality. In contrast, our method is designed to address these limitations by (i) dynamically sampling keyframes from the video based on the speech content, and (ii) designing a purely text-based model leveraging image captioning to convert RGB frames into text.

Our approach leverages a pretrained LLM, which we finetune specifically for the video chaptering task to predict jointly the chapter boundary timestamps and chapter titles, both in text form. The appeal of our model lies in processing only textual data as input, allowing us effectively leverage the long-context understanding capabilities of the LLM to scale to long videos. In particular, we incorporate speech transcriptions from automatic speech recognition (ASR) and automatic frame captions. Captioning has been used for video understanding as an intermediate representation in recent works, but in the context of retrieval or question answering (QA) for shorter videos (maximum 3 minutes) [60, 98, 119, 124]. In longer videos, since captioning every frame is computationally prohibitive, we employ a speech-based frame selection strategy that scales efficiently while preserving important content. Similar in spirit to [44], we primarily use audio to determine keyframes, specifically bootstrapping with an LLM trained only with the speech inputs. However, even when transforming a video into text, LLMs have a limited context window, allowing a maximum number of tokens as input in a single forward pass. To mitigate context window limitations for very long video inputs, we simply perform an iterative prediction, sequentially processing the video, where each iteration typically operates on a window length of about an hour duration. We evaluate our approach on ‘short’ $( 0 { - } 1 5 \operatorname * { m i n } )$ , ‘medium’ ( $1 5 { - } 3 0 \mathrm { m i n } )$ , and ‘long’ (30-60 min) videos from the VidChapters-7M dataset [112], demonstrating significant improvements over the state of the art across multiple metrics, including temporal boundary accuracy and semantic relevance of chapter titles. Our experiments show that finetuning the LLM, our speech-based frame selection strategy, and the integration of modalities from both speech and captions are crucial for achieving high-quality video chaptering results.

Our contributions are the following: (i) We introduce Chapter-Llama: our framework leverages a pretrained LLM and finetunes for the underexplored task of video chaptering by transforming the video input into text form through ASR and captioning. (ii) We scale efficiently to hour-long videos by incorporating a speech-based frame sampling strategy, captioning only a subset of the video frames. (iii) Our simple and effective approach outperforms the state of the art on the recent VidChapters-7M benchmark by a large margin (e.g., 45.3 vs 26.7 F1 score). These results are complemented by a comprehensive set of experiments analyzing our components.

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- Figure: ed539e227f825da1a3ac983e2d3ec39e447afb0cd3ffa30854935206b5cc22ef.jpg

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
