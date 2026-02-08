[← 返回 README](../README.md)

# Abstract

## 📌 预览
论文提出 Chapter-Llama，一个基于 LLM 的视频章节生成框架，通过将视频转为文本（ASR + 关键帧 caption）来实现小时级视频的高效章节划分。

---

We address the task of video chaptering, i.e., partitioning a long video timeline into semantic units and generating corresponding chapter titles. While relatively underexplored, automatic chaptering has the potential to enable efficient navigation and content retrieval in long-form videos. In this paper, we achieve strong chaptering performance on hour-long videos by efficiently addressing the problem in the text domain with our 'Chapter-Llama' framework. Specifically, we leverage a pretrained large language model (LLM) with large context window, and feed as input (i) speech transcripts and (ii) captions describing video frames, along with their respective timestamps. Given the inefficiency of exhaustively captioning all frames, we propose a lightweight speech-guided frame selection strategy based on speech transcript content, and experimentally demonstrate remarkable advantages. We train the LLM to output timestamps for the chapter boundaries, as well as free-form chapter titles. This simple yet powerful approach scales to processing one-hour long videos in a single forward pass. Our results demonstrate substantial improvements (e.g., 45.3 vs 26.7 F1 score) over the state of the art on the recent VidChapters-7M benchmark. To promote further research, we release our code and models at our project page.

> 💡 **Abstract 批读**:
> - **任务**: Video chaptering = 时间线分段 + 生成章节标题
> - **核心思路**: 把视频转成纯文本（ASR 语音转录 + 关键帧 caption），然后用 LLM 处理。纯文本方案的好处是能利用 LLM 的长上下文能力
> - **关键创新**: Speech-guided frame selection — 不是密集 caption 所有帧，而是先用语音预测关键位置，只 caption 那些帧
> - **核心数字**: F1 45.3 vs 26.7（SOTA Vid2Seq），提升近一倍
> - **可扩展性**: 单次 forward pass 处理 1 小时视频

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| F1 score | 45.3 vs 26.7 (SOTA) |
| 基准数据集 | VidChapters-7M |
| 处理能力 | 1 小时视频 / 单次 forward pass |

### 核心洞察
1. 将视频理解问题转化为纯文本问题是处理长视频的有效策略
2. Speech-guided frame selection 是效率的关键——避免了对所有帧做 captioning
3. LLM 同时输出时间戳和标题，端到端解决分段+描述
