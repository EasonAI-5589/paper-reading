# A.2. Prompt details

> 来源: [CVPR 2025] Chapter-Llama

---

## 📄 原文

The base prompt contains the instructions as follows:

Given the complete transcript of a video of duration {duration}, {task}. Identify the approximate start time of each chapter in the format   
‘hh:mm:ss - Title’.   
Ensure each chapter entry is on a new line.   
Focus on significant topic changes that would merit a new chapter in a video, but do not provide summaries of the chapters.   
{transcript}

where duration represents the length of the video in HH:MM:SS format (e.g., $0 0 : 0 9 : 5 2 )$ , while task and transcript are specific to the input modalities used.

For example, when utilizing both ASR and captions as input modalities, the task is defined as follows:

use the provided captions and ASR transcript to identify distinct chapters based on content shifts.

For the transcript, when training Chapter-Llama with both modalities, we prepend the modality names and interleave the outputs as illustrated below:

ASR $0 0 : 0 0 : 0 0$ : This place has blown our minds.   
Caption $0 0 : 0 0 : 0 1$ : The image features two individuals, a man and a woman, standing outdoors in a natural setting with rocky terrain and sparse vegetation in the background.   
ASR $0 0 : 0 0 : 0 4$ : Look at this.   
ASR 00:00:05: In this episode, we’re exploring Buckhorn Wash, Utah.

When training with only ASR (e.g., frame selector module), we simplify the input format by omitting the modality prefix, as there is only one source of information in the transcript.

We refer to Tab. A.4 for an experiment with/without these prefixes, where we observe slight gains by specifying the modalities. When using a single modality as input (e.g., ASR), there is no need to prepend the modality name to the transcript:

$0 0 : 0 0 : 0 0$ : This place has blown our minds. 00:00:04: Look at this. 00:00:05: In this episode, we’re exploring Buckhorn Wash, Utah.

---

## 💡 理解

### 核心要点
- [ ] 待填写

### 关键公式/概念
- 

### 图表解读
- 无图表

### 我的疑问
- [ ] 

---

*笔记生成时间: 自动生成，待完善*
