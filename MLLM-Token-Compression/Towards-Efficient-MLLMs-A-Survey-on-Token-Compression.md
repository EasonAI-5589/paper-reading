# 📄 Towards Efficient Multimodal Large Language Models: A Survey on Token Compression

> **Paper Reading #001** | 阅读日期: 2026-02-06

## 📌 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | Towards Efficient Multimodal Large Language Models: A Survey on Token Compression |
| **作者** | Linli Yao, Long Xing, Yang Shi 等 |
| **机构** | 北京大学、中科大、南洋理工等 |
| **发布** | 2025年12月 (TechRxiv v1.0) |
| **类型** | Survey |
| **论文链接** | [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176823010.07236701/v1) |
| **GitHub** | [yaolinli/MLLM-Token-Compression](https://github.com/yaolinli/MLLM-Token-Compression) |

---

## 🎯 一句话总结

这是一篇关于 **MLLM 视觉 Token 压缩** 的全面综述，系统梳理了 100+ 篇论文，按压缩位置（Vision Encoder / Projector / LLM）和压缩策略进行分类，并提供了方法选择指南。

---

## 📊 核心图表

### Fig 1: Token Compression 方法发展时间线

<img src="https://github.com/user-attachments/assets/b552ed71-902b-48a2-ae67-8323c335a793" width="800" alt="Timeline of Token Compression Methods"/>

> 2022-2025 年 Token Compression 方法快速发展，2024 年后爆发式增长

### Fig 2: 分类体系 (Taxonomy)

```
Where to Compress?
├── Vision Encoder
│   ├── Inside-Encoder (Token Dropping/Merging)
│   └── Outside-Encoder (Purely-Vision/Text-guided)
├── Projector
│   ├── Transformation-based (Pooling/PixelShuffle/Conv)
│   ├── Query-based (Q-Former)
│   └── Importance-driven
├── LLM
│   ├── Prefilling Stage
│   └── Decoding Stage (KV Cache)
└── Hybrid (Multi-Module)
```

### Fig 6: 方法选择决策树

```
How to Select?
├── Temporal-Enhanced? → Video专属压缩
├── Text-guided vs Purely-Visual? → 单轮QA vs 多轮对话
├── Merging vs Dropping? → 保留细节 vs 激进压缩
├── Plug-in vs Re-training? → 快速部署 vs 高性能
└── Training vs Inference? → 训练成本 vs 推理成本
```

---

## 核心问题

高分辨率图像和长视频会产生大量 visual tokens，导致：
- **计算复杂度爆炸**: Transformer 的 attention 是 O(n²)
- **GPU 内存消耗巨大**
- **推理延迟高**

**解决思路**: Token Compression（减少 token 数量，同时保持关键语义信息）

---

## 核心贡献

### 1. 分类体系 (Taxonomy)

按 **压缩位置** 分类：

```
Where to Compress?
├── Vision Encoder (§3.1)
│   ├── Inside-Encoder: 在 ViT 内部压缩
│   │   ├── Token Dropping (剪枝)
│   │   ├── Token Merging (合并)
│   │   └── Multi-Scale Compression
│   └── Outside-Encoder: ViT 输出后压缩
│       ├── Purely-Vision Compression
│       └── Text-guided Compression
│
├── Projector (§3.2)
│   ├── Transformation-based: Pooling, Pixel Shuffle, Conv
│   ├── Query-based: Q-Former 及其变体
│   └── Importance-driven: 基于重要性筛选
│
├── LLM (§3.3)
│   ├── Prefilling Stage: 首次前向时压缩
│   │   ├── Importance-based
│   │   ├── Learnable Module-based
│   │   ├── Token Merging-based
│   │   └── Fusion-based (Cross-Attention)
│   └── Decoding Stage: KV Cache 压缩
│
└── Hybrid (§3.4): 多模块协同/渐进压缩
```

### 2. 方法选择指南 (How to Select)

| 选择维度 | 选项A | 选项B | 适用场景 |
|---------|-------|-------|---------|
| 压缩依据 | Purely-Visual | Text-guided | 多轮对话 vs 单轮QA |
| 压缩方式 | Token Merging | Token Dropping | 保留细节 vs 激进压缩 |
| 部署方式 | Plug-in | Re-training | 快速部署 vs 高性能 |
| 优化目标 | Efficient Training | Efficient Inference | 训练成本 vs 推理成本 |

### 3. 视频专属挑战

- **时空交互压缩**: 联合空间和时间维度压缩
- **时序结构保留**: 保持时间戳信息用于定位
- **超长视频**: 小时级视频的处理策略（Memory Bank, KV Cache Sparsification）

---

## 关键方法速览

### Vision Encoder 阶段

| 方法 | 核心思想 | 亮点 |
|------|---------|------|
| **ToMe** | Bipartite soft matching 合并相似 tokens | 开山之作，简单有效 |
| **VisionZip** | 重要性估计 + 代表性约束 | 可达 16x 压缩 |
| **DART** | 发现 attention-based 选择有 positional bias | 用 similarity 替代 attention |
| **HoloV** | 保留全局视觉上下文，避免过度关注 salient regions | 平衡前景/背景 |

### Projector 阶段

| 方法 | 核心思想 | 亮点 |
|------|---------|------|
| **Q-Former** (BLIP-2) | Learnable queries + Cross-attention | 经典设计，广泛使用 |
| **TokenPacker** | Coarse-to-fine: 下采样特征作为 query | 保留高分辨率细节 |
| **LLaVA-Mini** | 额外 Modality Pre-Fusion 模块 | 缓解信息丢失 |

### LLM 阶段

| 方法 | 核心思想 | 亮点 |
|------|---------|------|
| **FastV** | 发现 vision tokens attention 极度稀疏 | 第2层剪掉 50% |
| **PyramidDrop** | 多阶段渐进剪枝 | 层越深冗余越高 |
| **DyCoke** | 动态 KV Cache 压缩 | 适合视频解码 |

### 视频理解

| 方法 | 核心思想 | 亮点 |
|------|---------|------|
| **TimeChat-Online** | 保留时间动态变化的 tokens | 80% tokens 自然冗余 |
| **Video-XL** | Visual Summarization Tokens (VSTs) | 支持 2048 帧 |
| **SlowFast-LLaVA** | 双流：慢路径高分辨率 + 快路径高帧率 | 借鉴动作识别 |

---

## 关键发现 (Insights)

### 1. Attention Bias Problem
- 基于 attention score 的剪枝存在 **位置偏差**（偏向序列末端的 tokens）
- 原因：RoPE 的长期衰减特性
- 解决：用 similarity-based 方法替代 attention-based

### 2. Visual Token 冗余极高
- 大部分 benchmark（COCO 等自然场景）只需 **9 tokens/image** 就能处理好
- 但 OCR、文档理解需要 **144-576 tokens/image**
- 启示：需要 task-aware 和 content-aware 的自适应压缩

### 3. Merging vs Dropping
- **Merging**: 保留全局语义，适合空间冗余，可能模糊局部性
- **Dropping**: 保留稀疏显著语义，可能丢失上下文
- 最佳实践：**混合策略**（先 merge 低层，再 drop 高层）

### 4. 训练 vs 推理压缩
- 主流 MLLMs 训练时用简单策略（Pooling, Pixel Shuffle）
- 推理压缩方法更丰富，但很多不兼容 Flash Attention
- 原因：训练验证成本高，不敢冒险用新方法

---

## Open Challenges

1. **缺乏理论基础**: 现有方法多是 heuristic，没有从因果/充分性角度论证
2. **缺乏任务/内容自适应**: 统一压缩率无法适应不同任务复杂度
3. **细粒度任务性能下降**: OCR、文档理解等任务压缩后掉点严重
4. **评估标准不统一**: 各论文选不同 benchmark，难以公平比较

---

## 写作借鉴

### 1. 分类框架设计
- **双维度分类**: Where (位置) + How (策略)
- **清晰的层次结构**: 大类 → 子类 → 具体方法
- **表格总结**: 每个维度用表格对比 (Table 3-6)

### 2. 图表设计
- **Fig 1**: Timeline 展示发展历程
- **Fig 2**: 完整 taxonomy 树状图
- **Fig 3-5**: 每个模块的压缩示意图
- **Table 1**: 代表性方法大表（50+ 篇论文）
- **Table 8**: Benchmark 汇总

### 3. 分析结构
- 先讲 **What** (问题定义)
- 再讲 **Where** (在哪压缩)
- 然后 **How** (如何选择策略)
- 最后 **Challenges** (开放问题)

### 4. 语言风格
- 用 "we identify three main reasons" 这类清晰的结构
- 每个小节开头给出 roadmap
- 善用 "In contrast", "However", "Furthermore" 等连接词

---

## 相关资源

- **GitHub Repo**: https://github.com/yaolinli/MLLM-Token-Compression
  - Paper Table (按年份排序)
  - Benchmark 说明
  - 持续更新

---

## 与我们研究的关联

这篇综述对以下研究方向有参考价值：
- Video LLM 效率优化
- 长视频理解
- 多模态模型压缩

可以重点关注：
- §4.1 Temporal-Enhanced Compression (视频专属)
- §3.4 Multi-Module Compression (多阶段协同)
- §7 Open Challenges (未来方向)

---

*阅读笔记 by 3号机 📚*
*2026-02-06*
