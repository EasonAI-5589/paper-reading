# 4. How to Select the Desirable Token Compression Strategy

> ==核心：5 个选择维度的决策指南==

---

## 选择维度概览

| # | 选择维度 | 选项 A | 选项 B |
|---|----------|--------|--------|
| 4.1 | 视频时序增强 | Fixed Compression | Dynamic Compression |
| 4.2 | 压缩依据 | Purely-Visual | Text-Guided |
| 4.3 | 压缩方式 | Token Merging | Token Dropping |
| 4.4 | 部署方式 | Plug-in | Re-training |
| 4.5 | 优化目标 | Efficient Training | Efficient Inference |

---

## 4.1 Temporal-Enhanced Compression for Video

> Compared with static images, video input introduces an additional temporal dimension that substantially increases computational demands.
>
> ==视频相比图像多了时间维度，计算量爆炸==

### 4.1.1 Spatial-Temporal Compression

**Fixed Temporal Compression:**
| 方法 | 思路 | 代表作 |
|------|------|--------|
| Pooling | 时间维度平均相邻 tokens | PLLaVA, Video-ChatGPT |
| Convolution | 2D/3D 卷积时空下采样 | VideoLLaMA2, Qwen2-VL |
| Query-based | 可学习 queries 聚合视频 tokens | Clapper, LinVT, CrossLMM |
| Sequential Models | 按时序处理 + 时间戳编码 | BLIP-3-Video, STORM |

**Dynamic Temporal Compression:**
| 方法 | 思路 | 代表作 |
|------|------|--------|
| Token Merging | 合并跨帧冗余 tokens | TESTA, AuroraCap, DyCoke |
| Token Dropping | 丢弃时间低显著性 tokens | LongVU, TimeChat-Online |

**Hybrid Strategies:**
| 方法 | 思路 | 代表作 |
|------|------|--------|
| Global-Local Fusion | 全局事件聚类 + 局部帧级聚合 | LongVLM, Video-XL, PruneVid |
| Slow-Fast Pathways | 慢路径高分辨率 + 快路径高帧率 | SlowFast-LLaVA, LLaVA-Video |
| Memory-bank | 长期记忆 + 短期记忆 | MovieChat, Flash-VStream |

### 4.1.2 Temporal Structure Preservation

> Token merging and pruning can blur or discard spatiotemporal positional information, disrupting temporal structure.
>
> ==合并/剪枝可能破坏时序结构，影响时间定位任务==

**保留时序结构的方法：**
| 方法 | 代表作 |
|------|--------|
| Temporal Positional Embeddings | BLIP-3-Video, TimeChat-Online, PVC |
| Temporal Encoding Modules | STORM (Mamba), PVC (渐进编码) |
| Special Timestamp Tokens | Video-XL-2, Qwen3-VL (文本时间戳) |

### 4.1.3 Extreme-Long Video Compression

> In hour-long video scenarios, MLLMs must process thousands of frames.
>
> ==小时级视频 = 数千帧，需要专门设计==

**演进路线：**
1. **Memory-bank** (MovieChat, Flash-VStream) — 长短期记忆
2. **Video-XL 系列**:
   - Video-XL: VSTs 动态分区，2048 帧
   - Video-XL-Pro: ReCoT 重建能力，8000+ 帧
   - Video-XL-2: KV Cache 稀疏化，10000+ 帧
3. **Query-aware** (LinVT) — 根据文本 query 筛选帧
4. **Hybrid Architecture** (TimeViper) — Mamba-Transformer 混合，O(n) 复杂度

---

## 4.2 Purely-Visual vs. Text-Guided Compression

| 维度 | Purely-Visual | Text-Guided |
|------|---------------|-------------|
| **依据** | 视觉内部冗余 | 文本-视觉相关性 |
| **适用场景** | 多轮对话、流式视频、图像描述 | 单轮 QA、长视频 QA、视觉定位 |
| **压缩率** | 中等 | 可以很高 |
| **多轮对话** | ✅ 一次压缩可复用 | ❌ 每轮需重新编码 |
| **代表作** | VisionZip, DART, HoloV, TimeChat-Online | FastV, SparseVLM, PyramidDrop, LLaVA-Mini |

**Takeaway:**
> Purely-visual and text-guided strategies are complementary. A practical design is to first derive compact visual representations via purely-visual compression and then apply text-guided selection within the language module.
>
> ==实践建议：先 purely-visual 初步压缩，再 text-guided 精细化==

---

## 4.3 Token Merging vs. Token Dropping

| 维度 | Token Merging | Token Dropping |
|------|---------------|----------------|
| **策略** | 软压缩：聚合相似 tokens | 硬压缩：直接丢弃 |
| **优点** | 保留整体语义，适合空间冗余 | 保留稀疏显著语义 |
| **缺点** | 可能模糊空间/时间局部性 | 可能丢失细粒度上下文 |
| **适合** | 低层特征、密集视觉输入 | 高层特征、稀疏语义场景 |
| **代表作** | ToMe, TESTA, HoliTom | VisPruner, DART, FlexSelect |

**⚠️ Attention-based 选择的问题：**
> DART and FEATHER reported that attention scores introduce a positional bias, favoring tokens located at the lower-right region of the image. HoloV highlighted that MLLMs often over-fit to "highlighted tokens" and overlook holistic context.
>
> ==Attention-based 有位置偏差（偏向右下角），且过度关注显著区域==

**解决方案：**
> Recent approaches increasingly adopt similarity-based token selection, where redundancy is measured via feature-level similarity rather than attention magnitude.
>
> ==用 similarity-based 替代 attention-based，更稳定==

**Takeaway:**
> Merging and dropping are complementary. Merging provides smooth aggregation for dense/temporally redundant inputs; dropping is preferable for sparse, high-level semantics. Future: adaptive hybrid designs.
>
> ==Merging + Dropping 互补，未来方向是自适应混合==

---

## 4.4 Plug-in Methods vs. Re-training Methods

| 维度 | Plug-in | Re-training |
|------|---------|-------------|
| **定义** | 无需训练，直接集成 | 需要额外训练 |
| **优点** | 训练免费、轻量、易部署 | 性能上限高、任务自适应 |
| **缺点** | 细粒度任务性能下降 | 训练成本高、跨模型迁移差 |
| **代表作** | FastV, PyramidDrop, DynTok, DyCoke | Q-Former, TokenPacker, HiCo |

**Plug-in 方法类型：**
1. **Pooling** — TC-LLaVA, PLLaVA, DeCo
2. **Pixel Shuffle** — NVLM, InternVL 1.5
3. **Similarity-based** — DynTok, LLaVA-Scissor, DivPrune
4. **KV Cache Compression** — DyCoke, MustDrop

**Re-training 方法类型：**
1. **Query-based** — Q-Former, MQT, C-/D-Abstractor
2. **Downsampled-as-query** — TokenPacker, HiRes-LLaVA
3. **Text-guided** — TG-LLaVA, QueCC, VCM
4. **Multi-stage** — CrossGET, MustDrop, PAR

**Takeaway:**
> Hybrid strategies: lightweight plug-in for early spatial reduction → re-trained modules for semantic refinement → KV cache pruning for decoding efficiency.
>
> ==混合策略：早期 plug-in 空间压缩 → 中期 re-training 语义精炼 → 后期 KV cache 剪枝==

---

## 4.5 Efficient Training vs. Efficient Inference

| 维度 | Efficient Training | Efficient Inference |
|------|-------------------|---------------------|
| **目标** | 减少训练时 token 数量 | 减少推理时 token 数量 |
| **验证成本** | 高（需要完整训练验证） | 低（快速测试） |
| **研究数量** | 较少 | 较多 |
| **代表作** | Q-Former, LLaVA-OneVision, InternVL3.5 | FastV, PyramidDrop, VisionZip |

**主流 MLLM 的训练压缩策略：**
| 年份 | 模型 | 策略 |
|------|------|------|
| 2022 | Flamingo | GATED XATTN-DENSE |
| 2023 | BLIP-2, mPLUG-Owl, Qwen-VL, MiniGPT-4 | Q-Former 及变体 |
| 2024 | PLLaVA, LongVLM, VideoLLaMA 2 | Temporal + Spatial Pooling |
| 2024 | InternVL, Qwen2VL | Pixel Shuffle |
| 2025 | Seed1.5-VL | Average Pooling |

**为什么训练时不用更多新方法？**
> Three main reasons: (1) compatibility with Flash Attention, (2) validation cost is far more expensive, (3) inductive bias may degrade performance on dense visual tasks.
>
> ==三个原因：Flash Attention 兼容性、验证成本高、归纳偏差可能降低密集视觉任务性能==

---

## 💡 Decision Flowchart

```
开始
  ↓
是视频输入吗？
  ├─ 是 → 需要时序增强压缩 (§4.1)
  │       ├─ 固定长度 → Fixed (Pooling/Conv/Query)
  │       └─ 动态长度 → Dynamic (Merging/Dropping)
  └─ 否
  ↓
单轮还是多轮对话？
  ├─ 单轮/长视频 QA → Text-Guided (§4.2)
  └─ 多轮/流式 → Purely-Visual (§4.2)
  ↓
冗余类型？
  ├─ 空间/时间密集 → Token Merging (§4.3)
  └─ 稀疏显著 → Token Dropping (§4.3)
  ↓
资源限制？
  ├─ 训练资源有限 → Plug-in (§4.4)
  └─ 追求高性能 → Re-training (§4.4)
  ↓
优化阶段？
  ├─ 训练阶段 → Efficient Training (§4.5)
  └─ 推理阶段 → Efficient Inference (§4.5)
```

---

*[返回论文目录](../README.md)*
