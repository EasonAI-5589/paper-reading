[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 分两部分：(1) LVLM 架构概述（vision encoder + projector + LLM），强调 dynamic tiling 带来的 token 爆炸问题；(2) Token Compression 方法分类（vision encoder 阶段、LLM decoder 阶段、embedding space 阶段），定位 ToDRE 的独特之处。

---

## 2.1. Large Vision-Language Models

Large vision-language models (LVLMs) [5, 51, 66] have
demonstrated remarkable advancements by extending the
reasoning capabilities of pretrained LLMs [3, 52, 53] to
image and video comprehension tasks. Typically, LVLMs
employ a vision encoder to extract visual features, which are
subsequently projected into the LLM's embedding space via
a visual projector (e.g., Q-Former [26] or MLP [31, 37]). To
process real-world high-resolution images, previous LVLMs
[4, 36] resize input images to a fixed resolution, which introduces geometric distortion and degrades fine-grained local
details. To tackle this, subsequent studies adopt dynamic
tiling [11, 25, 37], which partitions images into regions and
encodes each region independently using a shared vision
encoder. However, dynamic tiling can yield thousands of
visual tokens, significantly increasing computational overhead. This issue becomes even more pressing in video-based
LVLMs [5, 33], since processing multiple video frames demands significantly more visual tokens. These challenges
highlight the urgent need for accelerating LVLM inference
in resource-constrained real-world environments.

> 💡 **批注**:
> - LVLM 标准三件套：Vision Encoder + Projector (Q-Former/MLP) + LLM
> - **Dynamic tiling** 是 token 爆炸的元凶：高分辨率图像被切成多个 region，每个独立编码 → 数千 token
> - Video 更严重：多帧 × 每帧数百 token
> - 这段为 token compression 的必要性做铺垫

---

## 2.2. Token Compression for LVLMs

Given that _spatially_ _redundant_ visual tokens outnumber
_information-dense_ text tokens by tens to hundreds of times
[43], one natural solution to optimize LVLM inference
is visual token compression. Several earliest attempts
[7, 28, 30, 59] modify model components and introduce
additional training costs. More recently, training-free token
compression methods have been widely adopted due to their
efficiency and effectiveness. These methods can be categorized into two main groups: (1) Token compression in the
vision encoder [6, 32, 46], the LLM decoder [10, 35, 63],
or both [19]: For example, ToMe [6] reduces tokens in the
vision encoding phase by merging redundant tokens via a
binary soft-matching algorithm. Other approaches prune
tokens during the LLM decoding stage by evaluating token
redundancy through criteria such as attention scores with
text tokens [10, 63] or observed divergence with LLM outputs [35, 60]. Subsequent studies [19, 38, 67] perform token
compression during both stages to further enhance inference efficiency. (2) Token compression in LLM embedding
space [2, 62]: A representative example is FasterVLM [62],
which measures the token redundancy more accurately by
the cross-attentions between the [CLS] token and visual
tokens. Unlike previous methods, our proposed ToDRE simultaneously reduces tokens in both the LLM embedding
space and the LLM decoder. Our two-stage approach effectively captures both visual token diversity and token-task
relevance—two orthogonal yet critical aspects previously
overlooked—achieving superior inference efficiency while
maintaining competitive performance.

> 💡 **批注 — Token Compression 方法分类**:
> 
> | 压缩位置 | 代表方法 | 策略 |
> |----------|----------|------|
> | Vision Encoder | ToMe [6], LLaVA-PruMerge [46] | Token merging (similarity) |
> | LLM Decoder | FastV [10], SparseVLM [63], VTW [35] | Attention-based pruning / output divergence |
> | Both | TrimVLM [19], FocusLLaVA [67] | 两阶段 |
> | LLM Embedding Space | FasterVLM [62], DivPrune [2] | [CLS] attention / diversity |
> 
> **ToDRE 的定位**: LLM Embedding Space + LLM Decoder（两阶段），同时考虑 diversity 和 relevance

---

## 🔖 Section 总结

### 核心洞察
1. Visual token 数量是 text token 的 10-100 倍，冗余度高
2. 现有方法按压缩位置可分为 encoder、decoder、embedding space 三类
3. ToDRE 是第一个在 embedding space 和 decoder 同时压缩，且同时考虑 diversity 和 relevance 的方法
