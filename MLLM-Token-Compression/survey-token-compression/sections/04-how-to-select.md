# 4. How to Select the Desirable Token Compression Strategy

> 本章从5个关键设计维度进行对比分析，为实践者提供策略选择指南。

## 决策分类图 (Figure 6)

```
How to Select Strategy
├── §4.1 Temporal-Enhanced Compression for Videos
├── §4.2 Purely-Visual vs. Text-guided Compression
├── §4.3 Token Merging vs. Token Dropping
├── §4.4 Plug-in Methods vs. Re-training Methods
└── §4.5 Efficient Training vs. Efficient Inference
```

---

## 4.1 Temporal-Enhanced Compression for Video Input

视频比静态图像多出**时序维度**，带来三大挑战：
1. **Spatial-temporal interaction**: 如何联合压缩空间 $(h,w)$ 和时间 $t$ 维度
2. **Temporal structure preservation**: 压缩后如何保留时空结构（运动估计、时序定位）
3. **Scalability to extreme lengths**: 如何扩展到包含数万帧的小时级视频

### 4.1.1 Spatial-Temporal Compression

| 类别 | 方法 | 代表工作 |
|------|------|---------|
| **Fixed - Pooling** | 时序维度上average pooling相邻token | PLLaVA, Video-ChatGPT |
| **Fixed - Convolution** | 2D/3D卷积联合时空下采样 | VideoLLaMA2 (STC Connector), Qwen2-VL |
| **Fixed - Query-based** | 可学习query通过attention聚合所有token | Clapper, LinVT, CrossLMM |
| **Fixed - Sequential** | 按时序处理+时间戳嵌入+循环记忆 | BLIP-3-Video (Grouped Sequential Model), STORM |
| **Dynamic - Merging** | 自适应合并跨帧相似/冗余token | TESTA, AuroraCap, DyCoke |
| **Dynamic - Pruning** | 丢弃时序低显著性/冗余token | LongVU, TimeChat-Online |
| **Hybrid - Global-Local** | 全局事件聚类 + 局部帧级聚合 | LongVLM, Video-XL, TempMe, PruneVid |
| **Hybrid - Slow-Fast** | 高分辨率慢通路(空间细节) + 低分辨率快通路(运动) | SlowFast-LLaVA, LLaVA-Video, Clapper |
| **Hybrid - Memory-bank** | 长期记忆 + 短期记忆互补 | MovieChat, Flash-VStream, VidCompress |

### 4.1.2 Temporal Structure Preservation

token合并/剪枝可能**模糊时空位置信息**，影响时序定位等任务。

三种保留时间信息的方法：

| 方法 | 策略 | 代表工作 |
|------|------|---------|
| **Temporal Positional Embeddings** | 为视觉token增加时间位置信息 | BLIP-3-Video, TimeChat-Online, PVC |
| **Temporal Encoding Modules** | 专用时序编码组件 | STORM (MambaMixer), PVC (渐进式编码) |
| **Special Timestamp Tokens** | 插入显式时间戳token到视觉序列 | Video-XL-2, Qwen3-VL (文本格式时间戳如`<3.0 seconds>`) |

### 4.1.3 Extreme-Long Video Compression

小时级视频（数千帧）的专用设计：

| 方向 | 代表工作 | 关键特点 |
|------|---------|---------|
| **Memory-bank** | MovieChat (sliding window + dual memory), Flash-VStream | 10,000帧 / 24GB GPU |
| **Video-XL系列** | Video-XL → Video-XL-Pro → Video-XL-2 | 2048帧 / 16×-32×压缩 / 8000帧99%准确率 |
| **Query-aware** | LinVT, Long-VMNet (固定5880 token memory bank) | <1GB内存 / 10小时视频 |
| **Keyframe-based** | ReTaKe (帧间距离检测关键帧 → 非关键帧KV剪枝) | 8×更长序列 |
| **Hybrid architecture** | TimeViper (Mamba-Transformer混合) | 线性复杂度 + 精确attention |

**Summary**: 极长视频理解需要多维度协同: 1) 自适应关键帧采样 2) 多模块协作编码 3) query-aware策略 4) KV-cache稀疏化

---

## 4.2 Purely-Visual vs. Text-guided Compression

### 对比表 (Table 3)

| 维度 | Purely-Visual | Text-guided |
|------|--------------|-------------|
| **方法** | 基于视觉内在冗余保留信息token | 基于文本语义选择对齐的视觉token |
| **适用场景** | 多轮对话、流式视频、视觉字幕、易部署 | 单轮对话、长视频QA、高压缩率、视觉定位 |
| **优点** | 文本无关、一次压缩可复用、低延迟 | 高压缩率、任务相关保留、准确率高 |
| **缺点** | 可能保留不必要token | 需要重新编码历史token、多轮效率低 |
| **代表工作** | DeCo, VisionZip, DART, HoloV, TimeChat-Online | FastV, SparseVLM, Q-Former, QueCC, PyramidDrop, LLaVA-Mini |

### Takeaway

> **两者是互补的**。实践设计：先purely-visual获得紧凑视觉表示 → 再text-guided在LLM内做query相关精炼。

---

## 4.3 Token Merging vs. Token Dropping

### 对比表 (Table 4)

| 维度 | Token Merging | Token Dropping |
|------|--------------|----------------|
| **本质** | **Soft**策略，聚合冗余token为代表性嵌入 | **Hard**策略，直接丢弃不重要/无关token |
| **优点** | 保留整体语义、细粒度信息、适合空间冗余 | 保留稀疏显著语义、适合高层视觉特征 |
| **缺点** | 可能模糊空间/时序局部性 | 可能丢失上下文中的细微线索 |
| **代表** | ToMe, TESTA, HoliTom, MustDrop | VisPruner, MADTP, DivPrune, DART, FlexSelect, CDPruner |

### 关键发现

- **LLMC+** 分析: 空间冗余场景 → drop-based更优于Vision Encoder和LLM
- **DART / FEATHER**: attention scores存在**位置偏差**（偏向图像右下角的token）
- **HoloV**: MLLMs过度关注"highlighted tokens"而忽略整体上下文

### 趋势

> 趋向**自适应混合策略**：根据模态特征和冗余类型动态切换soft聚合和hard剪枝。

---

## 4.4 Plug-in Methods vs. Re-training Methods

### 对比表 (Table 5)

| 维度 | Plug-in | Re-training |
|------|---------|-------------|
| **方法** | 无参数/少参数，直接集成到frozen模型 | 引入可学习模块，需要额外训练 |
| **特点** | 无需训练、轻量、易部署 | 性能上限更高、但训练成本高、跨模型迁移差 |
| **代表** | FastV, SparseVLM, PyramidDrop, MustDrop | Honeybee, DeCo, TokenPacker, HiCo |

### Plug-in的四种策略
1. **参数无关空间变换**: global/adaptive pooling (TC-LLaVA, PLLaVA, DeCo, AVG-LLaVA)
2. **像素重排**: pixel shuffle + space-to-depth (NVLM, InternVL 1.5)
3. **相似度压缩**: DynTok (分组合并), LLaVA-Scissor (SCC), DivPrune (最大多样性)
4. **推理时KV-cache压缩**: DyCoke (attention引导), MustDrop (output-aware KV policy)

### 趋势

> 混合策略: lightweight plug-in做早期空间降采样 → re-trained cross-attention/query-based做语义精炼 → KV-cache剪枝加速解码（如MustDrop的多阶段设计）

---

## 4.5 Efficient Training vs. Efficient Inference

### 对比表 (Table 6)

| 维度 | Efficient Training | Efficient Inference |
|------|-------------------|-------------------|
| **目标** | 减少预训练/SFT的token数 → 降低训练成本 | 在prefilling/decoding阶段减少token → 降低推理延迟 |
| **特点** | 方法简单、但验证成本巨大 | 方法多样、验证成本低 |
| **代表** | Flamingo, Q-Former, LLaVA-OneVision, Qwen2.5-VL, InternVL3.5 | FastV, SparseVLM, PyramidDrop, VisionZip, SparseMM |

### 主流MLLM的训练压缩策略 (Table 7)

| 年份 | 模型 | 训练压缩策略 |
|------|------|------------|
| 2022 | Flamingo | GATED XATTN-DENSE |
| 2023 | BLIP-2, mPLUG-Owl, Qwen-VL等 | Q-Former及变体 |
| 2024 | PLLaVA, LongVLM, VideoLLaMA 2 | Temporal + Spatial Pooling |
| 2024 | LLaVA-OneVision | Bilinear Interpolation |
| 2025 | InternVL系列, Qwen2系列 | Pixel Shuffle |
| 2025 | Seed1.5-VL | Average Pooling |

### 为什么训练压缩方法未被主流LVLM广泛采用？

三个原因：
1. **兼容性问题**: 很多prefilling加速方法与Flash Attention不兼容
2. **验证成本**: 训练验证比推理验证昂贵得多
3. **归纳偏置**: 现有方法基于特定任务/benchmark的观察设计，可能在分布外场景下降性能

---

## 个人笔记

<!-- 在此添加你对策略选择的思考 -->

### 对我的场景最合适的策略组合
- TODO

### 我倾向的设计选择
- TODO

