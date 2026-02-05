# 4. How to Select the Desirable Token Compression Strategy

> The proliferation of token compression designs necessitates **guidelines** to help practitioners select optimal strategies for specific deployment scenarios.
>
> ==本章目的：提供方法选择指南==

关键选择因素：
1. **Temporal-enhanced compression** (§4.1): 视频输入的时序压缩
2. **Text-guided vs Purely-visual** (§4.2): 是否利用文本引导
3. **Token Pruning vs Merging** (§4.3): 删除还是合并
4. **Plug-in vs Re-training** (§4.4): 即插即用还是重新训练
5. **Efficient Training vs Inference** (§4.5): 优化训练还是推理

---

## 4.1 Temporal-Enhanced Compression for Video

视频特有的三大挑战：
1. **Spatial-temporal compression** (§4.1.1): 联合空间+时间压缩
2. **Temporal structure preservation** (§4.1.2): 保持时间戳信息
3. **Scalability to extreme lengths** (§4.1.3): 小时级超长视频

### 4.1.1 Spatial-Temporal Compression

| 类型 | 方法 | 代表工作 |
|------|------|---------|
| **Fixed - Pooling** | 时间维度平均相邻 tokens | PLLaVA, Video-ChatGPT |
| **Fixed - Convolution** | 2D/3D 卷积时空下采样 | VideoLLaMA2, Qwen2-VL |
| **Fixed - Query-based** | 可学习 queries 聚合视频 tokens | Clapper, LinVT, CrossLMM |
| **Fixed - Sequential** | 时序处理 + 时间戳 embedding | BLIP-3-Video, STORM |
| **Dynamic - Merging** | 跨帧合并冗余 tokens | TESTA, AuroraCap, DyCoke |
| **Dynamic - Pruning** | 丢弃时序低显著性 tokens | LongVU, TimeChat-Online |
| **Hybrid - Global-Local** | 全局事件聚类 + 局部帧聚合 | Video-XL, HiCom, PruneVid |
| **Hybrid - Slow-Fast** | 双流：慢路径高分辨率 + 快路径高帧率 | SlowFast-LLaVA, LLaVA-Video |
| **Hybrid - Memory-bank** | 长期记忆 + 短期记忆 | MovieChat, Flash-VStream |

> **SlowFast 双流架构** (借鉴动作识别):
> - Slow pathway: 低帧率 + 高空间细节
> - Fast pathway: 高帧率 + 紧凑 tokens
>
> ==Keye-VL 1.5 进一步优化：动态路由显著帧到 slow 分支，静态帧到 fast 分支==

### 4.1.2 Temporal Structure Preservation

> During video compression, atomic operations like merging and pruning can blur or discard **spatiotemporal positional information**, impairing temporal localization tasks.
>
> ==问题：压缩操作可能破坏时间戳信息，影响时间定位任务==

**三种保持时间结构的方法：**

| 方法 | 原理 | 代表工作 |
|------|------|---------|
| **Temporal Positional Embeddings** | 给 visual tokens 加时间位置编码 | BLIP-3-Video, TimeChat-Online, PVC |
| **Temporal Encoding Modules** | 专门的时序建模模块 | STORM (Mamba), PVC (渐进编码) |
| **Special Timestamp Tokens** | 插入显式时间戳 tokens | Video-XL-2, Qwen3-VL |

> **Qwen3-VL** 用文本形式的时间戳 (e.g., `<3.0 seconds>`)，比 Video-RoPE 更精确
>
> ==时间戳的文本化表示是新趋势==

### 4.1.3 Extreme-Long Video (小时级)

> In hour-long video scenarios, MLLMs must process thousands of frames.
>
> ==挑战：处理数千帧，计算和内存都是瓶颈==

**发展脉络：**

| 阶段 | 方法 | 能力 |
|------|------|------|
| 早期 | MovieChat (Memory Bank) | 10,000+ 帧 (24GB GPU) |
| 进阶 | Video-XL (VSTs) | 2,048 帧, 16x 压缩近无损 |
| 当前 | Video-XL-Pro (ReCoT) | 8,000+ 帧, 99% 准确率 |
| 最新 | Video-XL-2 (Bi-level KV) | 推理时 KV Cache 压缩 |

> **ReTaKe** detects keyframes via inter-frame SSIM scores, retaining only structurally distinct frames to support **10-hour videos**.
>
> ==ReTaKe：通过帧间 SSIM 检测关键帧，支持 10 小时视频==

---

## 4.2 Purely-Visual vs Text-Guided Compression

| 维度 | Purely-Visual | Text-Guided |
|------|---------------|-------------|
| **原理** | 仅依赖视觉冗余（空间/时间相似性） | 利用文本语义聚焦相关区域 |
| **优点** | 多轮对话兼容，query 无关 | 单轮 QA 更精准，压缩率更高 |
| **缺点** | 可能保留与问题无关的细节 | 多轮场景需要重新压缩 |
| **代表** | VisionZip, HoloV, VideoLLaMA2 | FastV, PyramidDrop, LongVU |

> **Text-guided 的问题**: In multi-turn dialogues, different questions require **recomputing** relevant visual tokens, limiting reusability.
>
> ==Text-guided 在多轮对话中需要重新计算，复用性差==

> **混合策略**: First applying text-agnostic compression, then refining based on query relevance.
>
> ==最佳实践：先 purely-visual 粗压缩，再 text-guided 精压缩==

---

## 4.3 Token Pruning vs Token Merging

| 维度 | Pruning (删除) | Merging (合并) |
|------|---------------|----------------|
| **原理** | 直接丢弃低重要性 tokens | 聚合相似 tokens 为代表 |
| **优点** | 更激进的压缩率 | 保留全局语义，减少信息损失 |
| **缺点** | 可能丢失关键上下文 | 可能模糊局部细节 |
| **适用** | 稀疏显著语义保留 | 空间冗余较高的场景 |

> **Pruning** preserves **sparse salient semantics** while discarding background regions, but risks losing subtle contextual cues.
>
> ==Pruning：保留稀疏显著语义，风险是丢失微妙上下文==

> **Merging** maintains **global semantic coherence** by smoothing token representations but may blur spatial locality.
>
> ==Merging：保持全局语义连贯，风险是模糊空间局部性==

> **最佳实践**: Hybrid strategies — **先 merge 低层（保留全局），再 drop 高层（去冗余）**
>
> ==混合策略效果最佳==

---

## 4.4 Plug-in vs Re-training

| 维度 | Plug-in (即插即用) | Re-training (重新训练) |
|------|-------------------|----------------------|
| **优点** | 零成本部署，兼容现有模型 | 更高性能，模型可自适应 |
| **缺点** | 性能可能下降，泛化性受限 | 训练成本高，验证周期长 |
| **代表** | FastV, PyramidDrop, DART | Q-Former, TokenPacker, LLaVA-Mini |

> Most mainstream MLLMs adopt **simple projector strategies** (pooling, pixel shuffle) during training due to the high cost of validating novel approaches.
>
> ==观察：主流 MLLM 训练时用简单策略（pooling/pixel shuffle），因为验证新方法成本太高==

---

## 4.5 Efficient Training vs Efficient Inference

| 维度 | Training Efficiency | Inference Efficiency |
|------|---------------------|---------------------|
| **目标** | 减少训练时 GPU 内存和时间 | 减少推理延迟和部署成本 |
| **位置** | 通常在 Projector | 通常在 LLM |
| **方法** | Transformation-based (Pooling, PixelShuffle) | Attention-based pruning, KV Cache compression |

> **Training**: Early-stage compression (VE/Projector) reduces memory throughout training.
>
> **Inference**: LLM-stage compression and KV-cache optimization target deployment efficiency.
>
> ==训练优化在前端（VE/Projector），推理优化在后端（LLM/KV Cache）==

---

## 选择决策树

```
需要处理视频？
├── 是 → 需要时间定位？
│   ├── 是 → 使用 Temporal Structure Preservation 方法
│   └── 否 → 使用 Spatial-Temporal Compression
└── 否 → 继续

多轮对话场景？
├── 是 → Purely-Visual Compression（query 无关）
└── 否 → 单轮 QA → Text-Guided Compression（更高压缩率）

需要快速部署？
├── 是 → Plug-in 方法（FastV, PyramidDrop）
└── 否 → 可接受 Re-training → Query-based (Q-Former) 或 Importance-driven

细粒度任务（OCR/文档）？
├── 是 → 谨慎压缩，保留更多 tokens (144-576/image)
└── 否 → 自然场景可激进压缩 (~9 tokens/image)
```

---

## 总结

| 选择维度 | 推荐 |
|---------|------|
| 视频 + 时间定位 | Temporal Positional Embeddings + Timestamp Tokens |
| 单轮 QA | Text-Guided (FastV, PyramidDrop) |
| 多轮对话 | Purely-Visual (VisionZip, HoloV) |
| 快速部署 | Plug-in (FastV, DART) |
| 最高性能 | Re-training (TokenPacker, LLaVA-Mini) |
| 细粒度任务 | 低压缩率，保留更多 tokens |
