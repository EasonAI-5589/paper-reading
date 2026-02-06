# 6. Application Scenarios

> ==Token Compression 的实际应用场景==

---

## 6.1 Image Understanding

### Medical Image Processing

> MLLMs must rapidly and accurately interpret clinical data. Current models remain limited in handling high-resolution medical imaging examination results.
>
> ==医学影像：需要快速准确解读临床数据，当前模型在高分辨率医学图像上有限制==

**Token Compression 的价值：** 在保持准确性的同时提升效率，处理高分辨率医学图像。

### Multi-page Document Understanding

> Models must process long documents and generate concise summaries or meaningful solutions.
>
> ==多页文档理解：处理长文档，生成摘要或解决方案==

**借鉴高分辨率图像处理的经验：**
- mPLUG-DocOwl2
- mPLUG-Owl3

### Satellite and Remote Sensing Imagery

> These images typically contain rich structural information at high resolutions, yet practical deployments face computational resource constraints.
>
> ==遥感图像：高分辨率 + 丰富结构信息 + 计算资源受限==

**应用价值：** 使模型能够更高效地处理高分辨率输入，对工业部署至关重要。

---

## 6.2 Video Understanding

### Embodied AI

> Embodied agents or robots must respond in real time to visual input during continuous video perception.
>
> ==具身智能：机器人需要实时响应连续视频输入==

**关键能力：**
- 高效捕获时空信息
- 细粒度视频理解
- 保持计算效率

### Streaming Video Understanding

> Models must process continuous video streams and deliver real-time responses with minimal latency.
>
> ==流式视频理解：处理连续视频流，实时响应，最小延迟==

**技术策略：**
- 处理密集视频流的高时间冗余（1-10 FPS）
- 通过 Memory 机制存储紧凑历史表示
- 推理时高效检索相关 KV cache

**代表作：** TimeChat-Online 等

### Instructional Video Summary

> Meeting summarization and lecture key-point extraction require efficient video understanding while preserving fine-grained details.
>
> ==教学视频摘要：会议总结、讲座要点提取==

**核心思想：** 选择性保留信息性 tokens，丢弃冗余 tokens。

---

## 6.3 Other Applications

### Attention Guidance

> A key advantage is its ability to guide model attention toward the most relevant image or video regions.
>
> ==注意力引导：引导模型关注最相关的图像/视频区域==

**效果：**
- 过滤背景噪声和无关物体
- 将计算能力分配给关键视觉信息
- 提高对 prompt 的响应准确性

### Mitigating Visual Hallucinations

> Prior studies have shown that improved focus can mitigate visual hallucinations, where models generate text inconsistent with visual input.
>
> ==减少视觉幻觉：通过选择性 token pruning，改善模型输出与视觉上下文的一致性==

---

## 💡 应用场景总结

| 场景 | 挑战 | Token Compression 价值 |
|------|------|------------------------|
| 医学影像 | 高分辨率 + 准确性要求 | 效率 + 准确性平衡 |
| 文档理解 | 长文档 + 上下文限制 | 更长输入 + 更高效率 |
| 遥感图像 | 高分辨率 + 资源受限 | 工业部署可行性 |
| 具身智能 | 实时响应 + 连续视频 | 时空信息高效捕获 |
| 流式视频 | 连续流 + 最小延迟 | 紧凑历史表示 + KV cache |
| 视频摘要 | 长视频 + 细节保留 | 选择性 token 保留 |
| 幻觉缓解 | 输出与视觉不一致 | 注意力聚焦关键区域 |

---

*[返回论文目录](../README.md)*
