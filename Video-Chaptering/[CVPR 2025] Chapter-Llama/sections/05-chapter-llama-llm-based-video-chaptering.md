# 3. Chapter-Llama: LLM-based Video Chaptering

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

We provide an overview of our video chaptering framework, referred to as Chapter-Llama, in Fig. 2. Given video frames and speech transcripts, we aim at predicting relevant chapter boundaries and titles. For this, we first select video frames to process with a speech-based frame selection module. Then we use an off-the-shelf visual captioner to map the selected frames in the text space. We feed the resulting captions, along with speech transcripts, to the LLM which outputs the chapter boundaries and titles jointly as a single sequence of tokens. Finally, we devise an iterative prediction procedure in case the input text sequence is too long to handle for the LLM. We next describe in more detail each component.

Task formulation. Video chaptering [112] aims at segmenting a video into semantically meaningful chapters, and generating a title for each segment. The chapters are contiguous, with no gaps between them, and together span the entire video duration from start to end. Formally, given video frames $V = ( v _ { 1 } , v _ { 2 } , . . . , v _ { N } )$ and temporally-aligned speech transcripts $\boldsymbol { S } = ( s _ { 1 } , s _ { 2 } , . . . , s _ { M } )$ , where each speech transcript contains an utterance and its associated start and end timestamps, the task is to output a sequence of chapters $C = ( c _ { 1 } , c _ { 2 } , . . . , c _ { L } )$ , where each chapter $c _ { i }$ is a tuple $\left( \boldsymbol { b } _ { i } , t _ { i } \right)$ containing a start timestamp $b _ { i }$ and a descriptive title $t _ { i }$ . The end time of chapter $i$ is implicitly defined by the start time of the subsequent chapter $b _ { i + 1 }$ , or total video duration if $i = L$ .

Speech-based frame selection. Video chaptering involves processing hour-long videos. Therefore, densely sampling frames is computationally intractable due to numerous inference passes through a vision model (e.g., a visual captioner) and exceeding standard LLM context lengths. Upon inspection of our data, we found that while the speech transcription has 257 tokens per minute on average, a caption is 66 tokens long on average hence captions would take 3,960 tokens per minute when sampling a video at 1 FPS. To address these challenges, we employ a frame selection strategy.

Specifically, we use speech transcripts to guide which video frames to process for the vision model. This is done by first training a speech-only variant of our LLM to predict a sequence of chapter boundaries $\{ \hat { b } _ { 1 } , \hat { b } _ { 2 } , . . . , \hat { b } _ { K } \}$ from speech transcripts $S$ only. For each predicted boundary $\hat { b } _ { i }$ , we sample a frame $v _ { i }$ from the video at that timestamp. Note that this variant is cheaper compared to the full model as it only needs ASR transcription from the audio stream, without requiring any processing of the RGB stream (i.e., captioning). We then process the video frames only at the time locations predicted by this model. The visual information thus complements the previous ‘blind’ predictions from the narrations, and allows us to refine the predictions. This results in a video representation $V _ { s a m p l e d } = ( v _ { 1 } , v _ { 2 } , . . . , v _ { K } )$ where $K < < N$ . For the videos that lack speech entirely (e.g., about $3 \%$ of the videos in [112]), we sample frames at 10-second intervals, with an upper bound of 100 frames to maintain computational practicality.

![](images/4176601f4caa53493499480a88a0fb01172cf68041899dbdba76bba3c3ebec02.jpg)  
Figure 2. Method overview: Our Chapter-Llama framework first selects video frames to process using speech information. Then we use a visual captioner to map the selected frames in the text space. We feed the resulting captions, along with speech transcripts, to the LLM which outputs the chapter boundaries and titles jointly as a single sequence of tokens.

Mapping video to text with timestamps. To leverage the knowledge of a pretrained LLM, we map all our inputs to text. This includes: (1) speech transcriptions $\boldsymbol { S } = ( s _ { 1 } , s _ { 2 } , \dots , s _ { M } )$ from the audio modality, and (2) caption descriptions $V _ { c a p t i o n s } = ( d _ { 1 } , d _ { 2 } , . . . , d _ { K } )$ from the visual modality. In detail, for speech transcriptions, we use ASR outputs provided by [112], obtained using the Whisper-Large-V2 [73] model through the WhisperX [6] implementation. For captioning, we employ MiniCPM-V [115] as an image captioner, applied independently on the selected video frames, i.e., $d _ { i } { = } C a p t i o n e r ( v _ { i } )$ .

As we aim at predicting relevant chapter boundaries, we provide temporal information to the LLM. For both modalities, we prepend the timestamp information formatted as “HH:MM:SS” to encode the location at which the speech or caption is obtained.

Captions naturally come from a single point in time. Speech segments cover intervals, but their duration is typically very short (3-4 seconds). We therefore simply use the start time of each transcribed speech interval. We interleave the speech and caption inputs based on their timestamps in a sorted order. We add a modality-specific prefix to each timestamp to denote which modality the information is extracted from (i.e., ASR for speech transcripts, Caption for captions).

We prepend the text combining speech transcripts and captions with a fixed prompt that provides task instructions (see sup. mat. for the exact wording). This prompt occupies approximately 90 tokens and is independent of video length.

Language model. We derive our framework by making use of a powerful pretrained LLM. Specifically, we employ the recent Llama-3.1-8B-Instruct [21] model and further finetune on chapter annotations using the LoRA technique [36]. Given the input structure previously described, the LLM is trained to output chapters, where each chapter consists of a timestamp in HH:MM:SS format followed by a free-form chapter title. We treat both the timestamps and titles simply as text tokens and apply the standard cross-entropy loss over the original vocabulary of the pretrained LLM. We apply teacher forcing during training and decode tokens autoregressively at inference. Note that the final model (taking both speech and captions as input) is trained independently from the speech-only version of our model used for frame selection, but these two models share the same backbone, and only differ in their LoRA parameters (13MB each). Across all experiments, we finetune models for a single epoch and use the same hyperparameters. We provide these hyperparameters, along with implementation details in Appendix A, and provide experiments with several Llama variants in Appendix C.

Iterative prediction for long videos. The inputs may exceed the context window limitation of the LLM, especially in the case of long videos. For example, on an A6000 GPU, the Llama-3.1- 8B-Instruct [21] model can process videos up to around $1 5 \mathrm { k }$ tokens during training, which corresponds to 50 minutes of video content on average, and $2 5 \mathrm { k }$ tokens during inference, which corresponds to 80 minutes of video content on average. To address this issue, during training, we select videos that have less than 15k tokens. Since there are videos up to 1 hour long in the training set that satisfy this constraint, and since we do not need the entire training dataset to achieve good performance, this token limitation does not hinder our training. During evaluation, we predict chapters for each chunk sequentially, such that the start of a chunk is the end of the previous chunk. Finally, we merge the predictions from all chunks to obtain chapter boundaries for the complete video. We provide more details in Appendix A.4.

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- Figure: 4176601f4caa53493499480a88a0fb01172cf68041899dbdba76bba3c3ebec02.jpg

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
