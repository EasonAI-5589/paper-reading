[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
综述 LVLM 架构演进（动态分片带来的 token 爆炸）和 token compression 方法分类（vision encoder / LLM decoder / embedding space），为 ToDRE 的两阶段定位提供上下文。

---

# 2.1. Large Vision-Language Models

Large vision-language models (LVLMs) [5, 51, 66] have demonstrated remarkable advancements by extending the reasoning capabilities of pretrained LLMs [3, 52, 53] to image and video comprehension tasks. Typically, LVLMs employ a vision encoder to extract visual features, which are subsequently projected into the LLM's embedding space via a visual projector (e.g., Q-Former [26] or MLP [31, 37]). To process real-world high-resolution images, previous LVLMs [4, 36] resize input images to a fixed resolution, which introduces geometric distortion and degrades fine-grained local details. To tackle this, subsequent studies adopt dynamic tiling [11, 25, 37], which partitions images into regions and encodes each region independently using a shared vision encoder. However, dynamic tiling can yield thousands of visual tokens, significantly increasing computational overhead. This issue becomes even more pressing in video-based LVLMs [5, 33], since processing multiple video frames demands significantly more visual tokens. These challenges highlight the urgent need for accelerating LVLM inference in resource-constrained real-world environments.

> 💡 **LVLM 架构回顾**:
> - Vision Encoder → Projector (Q-Former/MLP) → LLM
> - 固定分辨率 → 几何畸变；Dynamic tiling → token 爆炸（数千个）
> - 视频场景更严重：多帧 × 每帧数百 token
> - **ToDRE 的出发点**: 在 token 进入 LLM 前和 prefilling 期间做压缩

---

# 2.2. Token Compression for LVLMs

Given that spatially redundant visual tokens outnumber information-dense text tokens by tens to hundreds of times [43], one natural solution to optimize LVLM inference is visual token compression. Several earliest attempts [7, 28, 30, 59] modify model components and introduce additional training costs. More recently, training-free token compression methods have been widely adopted due to their efficiency and effectiveness. These methods can be categorized into two main groups: (1) Token compression in the vision encoder [6, 32, 46], the LLM decoder [10, 35, 63], or both [19]: For example, ToMe [6] reduces tokens in the vision encoding phase by merging redundant tokens via a binary soft-matching algorithm. Other approaches prune tokens during the LLM decoding stage by evaluating token redundancy through criteria such as attention scores with text tokens [10, 63] or observed divergence with LLM outputs [35, 60]. Subsequent studies [19, 38, 67] perform token compression during both stages to further enhance inference efficiency. (2) Token compression in LLM embedding space [2, 62]: A representative example is FasterVLM [62], which measures the token redundancy more accurately by the cross-attentions between the [CLS] token and visual tokens. Unlike previous methods, our proposed ToDRE simultaneously reduces tokens in both the LLM embedding space and the LLM decoder. Our two-stage approach effectively captures both visual token diversity and token-task relevance—two orthogonal yet critical aspects previously overlooked—achieving superior inference efficiency while maintaining competitive performance.

> 💡 **Token Compression 方法分类**:
>
> | 压缩位置 | 代表方法 | 策略 |
> |----------|----------|------|
> | Vision Encoder | ToMe [6], LLaVA-PruMerge [46] | Merge/prune in ViT |
> | LLM Decoder | FastV [10], SparseVLM [63], VTW [35] | Attention-based / divergence-based pruning |
> | 两阶段 | FocusLLaVA [67], Han et al. [19] | Encoder + decoder |
> | LLM Embedding Space | FasterVLM [62], DivPrune [2] | [CLS] attention / diversity |
>
> **ToDRE 的定位**: LLM Embedding Space + LLM Decoder，独特之处在于分离了 diversity 和 relevance 两个正交维度

---

## 🔖 Section 总结

### 核心洞察
1. Visual token 数量是 text token 的数十到上百倍 → 压缩潜力巨大
2. Training-free 方法因无需重训练而受青睐
3. ToDRE 的创新在于同时在 embedding space 和 decoder 内做压缩，且明确区分 intra-modal 和 cross-modal 冗余
