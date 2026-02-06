# Towards Efficient MLLMs: A Survey on Token Compression

**类型**: Survey 综述论文  
**来源**: TechRxiv 2025  
**GitHub**: https://github.com/yaolinli/MLLM-Token-Compression

---

## 一句话总结

系统性综述 MLLM 中的 **Token 压缩技术**，按压缩位置分类（Vision Encoder / Projector / LLM Backbone / Hybrid）。

---

## Survey 结构

### 1. MLLM 基础架构
- Vision Encoder → Projector → LLM Backbone
- 视觉 Token 数量远超文本，计算开销巨大
- 高分辨率/长视频场景下问题更突出

### 2. Token 压缩分类体系

| 压缩位置 | 代表方法 | 特点 |
|----------|----------|------|
| **Vision Encoder** | ToMe, EViT | 在 ViT 内部合并/剪枝 |
| **Projector** | Q-Former, Resampler | 通过 cross-attention 压缩 |
| **LLM Backbone** | FastV, SparseVLM | 在 LLM 层内剪枝 |
| **Hybrid** | 多阶段组合 | 分层级逐步压缩 |

### 3. 核心方法分析

#### Vision Encoder 内压缩
- **ToMe (Token Merging)**: 相似 token 合并
- **EViT**: 注意力引导的 token 选择

#### Projector 压缩  
- **Q-Former (BLIP-2)**: 固定数量 query tokens
- **Perceiver Resampler**: 可学习的压缩

#### LLM Backbone 压缩
- **FastV**: 第2层后固定比例剪枝
- **PyramidDrop**: 金字塔式渐进剪枝
- **SparseVLM**: 文本引导 + Token 回收

### 4. 评估指标
- **效率**: FLOPs、延迟、显存
- **性能**: 多种 benchmark (VQA, POPE, MME 等)
- **压缩率**: 保留 token 数量

---

## 关键 Insight

1. **视觉信息天然稀疏**: 图像中大量背景/冗余区域
2. **浅层聚合，深层可剪**: 信息在浅层聚合到 anchor tokens
3. **任务相关性**: 不同问题需要不同视觉区域
4. **Training-free 方法实用性强**: 无需额外训练开销

---

## 相关资源

- **GitHub Repo**: https://github.com/yaolinli/MLLM-Token-Compression
- **论文列表**: 收录 50+ 相关论文
- **方法对比表**: 详细的性能-效率对比

---

## 与本仓库其他论文的关系

| 论文 | 分类 | Survey 中的位置 |
|------|------|-----------------|
| FastV | LLM Backbone | §4.3 |
| PyramidDrop | LLM Backbone | §4.3 |
| SparseVLM | LLM Backbone | §4.3 |
| SwiftVLM | Projector | §4.2 |

---

*3号机整理 @ 2026-02-06*
