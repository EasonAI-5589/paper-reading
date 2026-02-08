[← 返回 README](../README.md)

# 6. Application Scenarios

## 📌 预览
Token compression 的实际应用场景：图像理解（医学影像、文档、遥感）、视频理解（具身 AI、流式视频、教学视频）、其他应用（减少幻觉）。

---

## 6.1 Image Understanding

**Medical Image Processing.** MLLMs must rapidly and accurately interpret clinical data, balancing efficiency and accuracy. Current models remain limited in handling high-resolution medical imaging. Token compression presents a promising avenue to improve both efficiency and effectiveness.

> 💡 **医学影像**: 高分辨率是刚需（CT/MRI 细节不能丢），但计算资源有限。Token compression 需要在"不丢关键细节"的前提下压缩 — 这对现有方法是重大挑战。

**Multi-page Document Understanding.** Models must process long documents and generate summaries or solutions. Inspired by high-resolution image processing, similar techniques [124], [148] can be applied to accelerate document understanding while managing longer inputs within limited context lengths.

**Satellite and Remote Sensing Imagery.** Rich structural information at high resolutions, but deployment faces computational constraints. Recent studies [283], [284] have explored token compression strategies to handle higher-resolution inputs more efficiently.

> 💡 **图像应用共性**: 都是"高分辨率 + 资源受限"的场景。Token compression 的价值在于在不牺牲关键信息的前提下降低计算开销。

---

## 6.2 Video Understanding

**Embodied AI.** Embodied agents/robots must respond in real time to visual input during continuous video perception. Token compression [74] addresses this by efficiently capturing spatial and temporal information for fine-grained video understanding while maintaining computational efficiency.

**Streaming Video Understanding.** Models must process continuous video streams with real-time responses and minimal latency. Prior studies [57], [285]–[287] adopt token compression to address high temporal redundancy in dense video streams (1-10 FPS), store compact historical representations, and efficiently retrieve question-relevant KV caches.

**Instructional Video Summary.** Applications such as meeting summarization and lecture keypoint extraction require efficient video understanding while preserving fine-grained details. Central idea: selective retention of informative tokens while discarding redundant ones.

> 💡 **视频应用的核心需求**: 实时性（Embodied AI、Streaming）+ 长时记忆（长视频总结）。Token compression 在视频场景的价值远大于图像 — 因为时间维度带来了指数级更多的 tokens。

---

## 6.3 Other Applications

Token pruning demonstrates considerable potential across diverse applications beyond image/video acceleration. A key advantage is guiding model attention toward the most relevant regions [288]. By filtering out background noise and irrelevant objects, models can allocate computational capacity to critical visual information. Prior studies [289], [290] have shown that this improved focus can **mitigate visual hallucinations**, where models generate text inconsistent with visual input.

> 💡 **减少幻觉**: Token pruning 不仅是效率工具，还能提升质量！通过去除无关视觉信息，减少模型被噪声干扰产生幻觉的可能性。这是一个意外但重要的发现。

---

## 🔖 Section 总结

### 应用场景速查
| 场景 | 核心需求 | 关键挑战 |
|------|----------|----------|
| 医学影像 | 高分辨率 + 实时 | 不能丢失诊断关键细节 |
| 文档理解 | 长文档 + 多页 | 上下文长度限制 |
| 遥感影像 | 大面积高分辨率 | 计算资源部署受限 |
| 具身 AI | 实时响应 | 延迟极低要求 |
| 流式视频 | 连续处理 + 实时 | 内存管理 + 时序保持 |
| 减少幻觉 | 提升质量 | 选择"正确的"tokens |
