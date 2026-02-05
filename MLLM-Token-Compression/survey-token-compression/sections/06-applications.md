# 6. Application Scenarios

## 6.1 Image Understanding

### Medical Image Processing

> MLLMs must rapidly and accurately interpret clinical data, underscoring the need to balance **efficiency and accuracy**.
>
> ==医学影像：效率和准确性的平衡==

- 高分辨率医学影像需要高效处理
- Token compression 可提升效率同时保持诊断准确性

### Multi-page Document Understanding

> Models must process **long documents** and generate concise summaries.
>
> ==文档理解：处理长文档生成摘要==

- 类似高分辨率图像处理的加速技术可迁移
- 代表工作：mPLUG-DocOwl2, mPLUG-Owl3

### Satellite and Remote Sensing Imagery

> These images typically contain rich structural information at high resolutions, yet practical deployments face **computational resource constraints**.
>
> ==遥感图像：高分辨率 + 资源受限==

- 工业应用场景
- Token compression 可处理更高分辨率输入

---

## 6.2 Video Understanding

### Embodied AI / Robot Learning

> Embodied agents must respond in **real time** to visual input during continuous video perception.
>
> ==具身 AI：实时响应连续视频输入==

- 需要高效捕获时空信息
- 细粒度视频理解 + 计算效率的平衡

### Streaming Video Understanding

> Models must process **continuous video streams** and deliver real-time responses with minimal latency.
>
> ==流视频理解：连续视频流 + 实时响应==

- 高时序冗余（1-10 FPS）
- 需要内存机制存储紧凑的历史表示
- 推理时高效检索 query 相关的 KV cache
- 代表工作：TimeChat-Online, LiveVLM

### Instructional Video Summary

> Meeting summarization and lecture keypoint extraction.
>
> ==会议摘要、讲座要点提取==

- 保留细粒度细节的同时实现高效理解
- 选择性保留信息性 tokens

---

## 6.3 Other Applications

### Reducing Visual Hallucinations

> By filtering out background noise and irrelevant objects, models can allocate computational capacity to **critical visual information**.
>
> ==减少视觉幻觉：过滤背景噪声，聚焦关键信息==

- Token pruning 引导模型关注最相关的图像/视频区域
- 改善模型输出与实际视觉上下文的一致性

### Attention Guidance

> A key advantage is the ability to **guide model attention** toward the most relevant image or video regions.
>
> ==注意力引导：将模型注意力聚焦到最相关区域==

- 过滤背景噪声和无关物体
- 改善视觉推理的准确性
