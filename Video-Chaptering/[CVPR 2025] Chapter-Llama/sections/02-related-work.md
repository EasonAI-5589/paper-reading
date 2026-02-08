[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
综述四个方向：时序视频分割、视频描述、长视频理解、LLM 在视频理解中的应用。

---

We provide an overview of video tasks related to video chaptering, such as temporal segmentation and captioning, along with a discussion on works focusing on long-form and LLM-based video understanding.

**Temporal video segmentation.** While video chaptering is a new task [112], there is a rich literature on methods focused on temporally segmenting a video in various forms. One task is shot detection [75, 79, 84], where any visual changes (e.g., shifting between two cameras) would require a temporal boundary, not necessarily modeling semantic shifts. Video scene segmentation, often studied on movies [39], is primarily focusing on grouping scenes with similar content [14, 15, 39, 40, 61, 68, 69, 74, 78, 80, 105, 114]. Another line of work considers boundary detection for temporal action segmentation [8, 24, 27, 49, 116], or localization [19, 56, 121, 123]. Unlike chaptering with free-form text, action segmentation assigns a label from a predefined set of categories, and typically defines short atomic actions as the unit. In contrast to these tasks, chapter boundaries can take various different forms depending on the type and the granularity of the video (e.g., each exercise within sports video, each slide within a lecture, each step in instructional video, each topic in a podcast video). Shot, scene, or action boundaries therefore may or may not correspond to complex chapter boundary definitions. Moreover, these tasks are mostly tackled with vision-only inputs [84, 116, 123], without leveraging speech. While text and audio segmentation have also been tackled separately [29, 76], video chaptering is based on both audio and vision inputs [112].

> 💡 **Chaptering vs 其他时序分割任务**:
> | 任务 | 粒度 | 标签类型 | 模态 |
> |------|------|---------|------|
> | Shot detection | 镜头切换 | 无语义 | 视觉 |
> | Scene segmentation | 场景 | 场景相似性 | 视觉 |
> | Action segmentation | 原子动作 | 预定义类别 | 视觉 |
> | **Video chaptering** | **语义章节** | **自由文本** | **视觉+语音** |
>
> Chapter 边界更复杂、更灵活，依赖于视频类型和粒度。

**Video captioning.** Generating chapter titles [112] is relevant to the task of captioning that seeks to describe the video content with text. There is a large literature on single video captioning [17, 52, 81, 83], often focusing on short video clips. Typical datasets for training such as MSR-VTT [110], WebVid [5], HowTo100M [59], Video-CC [62] include captions of videos spanning a few seconds (5-15sec on average). In generic event boundary captioning [103], event intervals are similarly short, in the order of 2 seconds. On the other hand, video summarization methods operate on longer videos; however, their goal is to reduce the entire video into a single summary description [1, 2, 34, 41, 53, 120, 126, 127], not necessarily with a temporal segmentation component. Dense video captioning [38, 45, 102, 113, 130, 131] is the closest to video chaptering in terms of problem formulation, aiming to both temporally localize and caption different events. Indeed, prior work on video chaptering trains the dense captioning method of Vid2Seq [113] on the VidChapters-7M dataset [112], but relies on a fixed number of equally sampled frames. In this paper, we leverage some of the annotations of this dataset to train an LLM-based chaptering model substantially outperforming previous methods [112, 113].

> 💡 **Dense video captioning 是最接近的任务**，同样需要时序定位+描述。但 chaptering 的章节更粗粒度、覆盖整个视频、且是连续不间断的。

**Long-form video understanding.** The definition of long videos has evolved with the release of various datasets spanning seconds [109, 111], a few minutes [23, 30, 58, 89], 10-30 minutes [2, 128], or one hour [25, 41, 87, 107, 112]. MLVU [128] introduces a benchmark for evaluating multiple long video understanding tasks such as summarization and QA; however, the data is not suitable for chaptering due to lack of annotations. Video-MME [25] also contains hour-long videos for QA. MAD [32, 87] provides audio description for long movies, but each description spans a few seconds and the sparse coverage over the video is different from contiguous chapters. Recently, Ego4D-HCap [41] was proposed for hierarchical video summarization. However, this dataset involves dense captioning with visual inputs only, while we focus on video chaptering with visual and speech inputs. To the best of our knowledge, VidChapters-7M [112] is the only open-sourced dataset for training and evaluating chapter generation, which we employ in this paper. Non-public related datasets include NewsNet [107] which includes hierarchical temporal segmentation annotations, the TV news chaptering dataset used in [31], and the ChapterGen dataset [11].

> 💡 **VidChapters-7M 是唯一公开的 chaptering 数据集**。其他长视频数据集要么没有 chaptering 标注（MLVU, Video-MME），要么只有视觉输入（Ego4D-HCap）。

Increased video lengths led to a range of works focusing on efficient temporal modeling strategies. A common technique to deal with longer videos is to use pre-extracted visual features [32, 87, 118]. For end-to-end learning with transformers, several works explored factorized spatio-temporal attention [3, 5, 9]. Others have looked at various ways to incorporate memory mechanisms [43, 106], blockwise attention [54, 55], or captioning frames to exploit LLMs [104, 124]. Given the redundancy in consecutive video frames, frame selection methods were explored in the context of short video captioning and action recognition [18, 108], as well as 'long' video QA in 3-minute durations [66, 91, 117]. Most common approach with current large video models is to perform sparse sampling with equal spacing [13, 46, 113]. SCSampler [44] exploits the low-dimensional audio modality to efficiently select salient video clips for action recognition. In our method, we also leverage audio, but in the form of ASR, and run the costly frame captioning step only on keyframes on locations predicted by a speech-based frame selection module.

> 💡 **长视频处理策略总结**:
> - 预提取视觉特征
> - 分解时空注意力
> - Memory 机制 / Blockwise attention
> - **Frame captioning + LLM**（本文属于此类）
> - Frame selection（SCSampler 用音频，本文用 ASR）

**LLM use in video understanding.** LLMs such as GPT [10, 71], Llama [21, 93, 94], and Gemini [28, 92], have been leveraged in different ways for improving video understanding. A popular approach is to train 'bridge' modules between pretrained visual backbones [72] and LLMs to build vision-language models (VLMs) that can ingest videos (e.g., Video-Llama [125], Video-LLaVa [50]). Other works have employed LLMs for automatic construction of video datasets [2, 41, 83, 99], tool use [60], storing memory in video QA [43], and temporal localization [37]. Similar to us, VideoTree [104] and VideoAgent [22] caption keyframes before passing them to an LLM together with a question for answer generation, addressing the limitations of [124] which performs a similar methodology without keyframe selection on shorter videos. In this study, we find that captioning alone is not sufficient, and needs to be complemented with ASR for competitive chaptering performance. Close to us, [2] exploits ASR on long videos and summarizes them with LLMs to generate pseudo-labels for video summarization training. In our work, we leverage LLMs, specifically finetuning a Llama model [21] for chaptering by prompting with speech transcription and frame captions. We show that finetuning is essential for adapting to the task so that the LLM picks up relevant content within the large context input [82].

> 💡 **与 VideoTree/VideoAgent 的对比**:
> - 它们也 caption 关键帧 → LLM，但只用于短视频 QA
> - **本文发现**: 只有 caption 不够，必须加 ASR 才能做好 chaptering
> - **Finetuning 是必须的**: 零样本 LLM 会被长输入中的冗余信息淹没 [82]

---

## 🔖 Section 总结

### 核心洞察
1. Video chaptering 是一个独特任务，与 shot/scene/action segmentation 都不同
2. VidChapters-7M 是目前唯一可用的公开 chaptering 数据集
3. "Caption + LLM" 范式之前只用在短视频，本文首次扩展到小时级
4. ASR + Visual caption 的多模态融合是 chaptering 的关键
