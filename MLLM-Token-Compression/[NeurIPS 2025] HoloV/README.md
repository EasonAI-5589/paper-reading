# Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention

**作者**: Xin Zou, Di Lu†, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Xu Zheng, Linfeng Zhang, Xuming Hu*  
**单位**: HKUST(GZ), HKUST, INSAIT (Sofia University), Shanghai Jiao Tong University  
**会议**: NeurIPS 2025  
**链接**: [arXiv](https://arxiv.org/abs/2510.02912) | [OpenReview](https://openreview.net/forum?id=zKoeRtye8o) | [GitHub](https://github.com/obananas/HoloV)

## 一句话总结

HoloV 提出**全局视觉上下文保留**的 token pruning 框架：通过 crop-wise 自适应分配剪枝预算 + 语义多样性评分，替代传统 attention-first 策略，在 88.9% 剪枝率下保留 95.8% 性能。

## 核心贡献

1. **发现问题**：揭示 attention-first pruning 的三大缺陷——信息冗余（保留语义相似 token）、位置偏置（序列首尾 token 被高估）、注意力分散（text-vision attention 过于分散）
2. **提出 HoloV**：将图像 token 划分为 crops，每个 crop 内计算语义多样性（variance）+ [CLS] attention 的混合评分，按 crop 重要性自适应分配剪枝配额
3. **Training-free + Plug-and-play**：兼容 LLaVA-1.5/NeXT、Qwen2.5-VL 等多种架构，支持 Flash-Attention
4. **Visual Context Refetching**：在 LLM 中间层重新注入被剪枝的 token 作为 KV memory，补偿信息损失

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1/2 |
| [01 - Introduction](sections/01-introduction.md) | 动机：attention-first 的三大问题 + HoloV 核心思想 |
| [02 - Related Work](sections/02-related-work.md) | MLLMs 挑战 / 视觉冗余识别 / Token 压缩与剪枝 |
| [03 - Preliminary & Motivation](sections/03-preliminary-motivation.md) | 预备知识 + 信息冗余分析 + "全局上下文优于局部重复" 的实验验证 |
| [04 - Methodology](sections/04-methodology.md) | HoloV 框架：crop 划分 → 多样性评分 → 自适应配额 → top-k 选择 |
| [05 - Experiments](sections/05-experiments.md) | 主实验 + 高分辨率 + 效率分析 + 消融 + 可视化 + Qwen 泛化 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 性能保留 (LLaVA-1.5, 64 tokens, ↓88.9%) | **95.8%** |
| 性能保留 (LLaVA-1.5, 128 tokens, ↓77.8%) | **98.0%** |
| 性能保留 (LLaVA-1.5, 192 tokens, ↓66.7%) | **99.2%** |
| 性能保留 (LLaVA-NeXT, 320 tokens, ↓88.9%) | **95.6%** |
| 性能保留 (Qwen2.5-VL, ↓88.9%) | **90.5%** |
| 推理时间减少 (90% pruning) | **42.7%** (49:41→27:36) |
| 延迟减少 (90% pruning) | **42.8%** (0.334s→0.176s) |
| GPU 显存减少 (90% pruning) | 19.0G→14.5G |
| POPE 准确率 (88.9% pruning) | 80.3% (次优 76.0%) |

## 方法对比

| 特性 | FastV | SparseVLM | HiRED | CDPruner | **HoloV** |
|------|-------|-----------|-------|----------|-----------|
| 评分依据 | Text-vision attn | Cross-modal attn | 层次式 attn | DPP diversity | **Variance + CLS attn** |
| 全局上下文 | ✗ | ✗ | 部分 | ✓ | **✓ (crop-wise)** |
| Training-free | ✓ | ✓ | ✓ | ✓ | **✓** |
| Flash-Attention | ✗ | ✗ | ✓ | ✓ | **✓** |
| 剪枝位置 | LLM 内部 | LLM 内部 | LLM 前 | LLM 前 | **LLM 前** |
| 高剪枝率鲁棒性 | 差 | 差 | 中 | 好 | **最好** |

---

## 📊 Citation Landscape

**Semantic Scholar**: [Link](https://www.semanticscholar.org/paper/f390d11e155df71ae24618c9ed80f83d0ec9b027)  
**Connected Papers**: [Link](https://www.connectedpapers.com/main/2510.02912)

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | ~100 |
| 被引次数 | 5 (截至 2026.02) |

### 参考文献分组 (Top 5 per group)

#### Visual Token Pruning/Compression (32 papers)
| 论文 | 年份 | 引用数 |
|------|------|--------|
| Greed is good: algorithmic results for sparse approximation | 2004 | 3809 |
| Token Merging (ToMe) | 2022 | 771 |
| LLaMA-VID | 2023 | 507 |
| FastV: An Image is Worth 1/2 Tokens After Layer 2 | 2024 | 376 |
| LLaVA-PruMerge | 2024 | 237 |

#### MLLMs & Architecture (32 papers)
| 论文 | 年份 | 引用数 |
|------|------|--------|
| LLaMA | 2023 | 18473 |
| Llama 2 | 2023 | 15792 |
| Llama 3 | 2024 | 12663 |
| BLIP | 2022 | 6008 |
| Qwen2.5-VL | 2025 | 3353 |

#### Attention & Efficiency (6 papers)
| 论文 | 年份 | 引用数 |
|------|------|--------|
| Attention is All You Need | 2017 | 166270 |
| FlashAttention | 2022 | 3603 |
| FlashAttention-2 | 2023 | 2252 |

#### Benchmarks & Evaluation (4 papers)
| 论文 | 年份 | 引用数 |
|------|------|--------|
| VQA V2: Making the V in VQA Matter | 2016 | 3904 |
| Survey on Hallucination in LLMs | 2023 | 2142 |
| TextVQA: Towards VQA Models That Can Read | 2019 | 1788 |

### 📌 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| FlashVLM: Text-Guided Visual Token Selection | 2025 | 4 |
| ConsensusDrop: Fusing Visual and Cross-Modal Saliency | 2026 | 0 |
| Focus-Scan-Refine: From Human Visual Perception to Efficient Pruning | 2026 | 0 |
| Vision Token Reduction via Attention-Driven Self-Compression | 2026 | 0 |
| VisionTrim: Unified Visual Token Compression | 2026 | 0 |
| ViTCoP: Visual and Textual Semantic Collaborative Pruning | 2026 | 0 |
| ViCA: Efficient MLLMs with Vision-Only Cross-Attention | 2026 | 0 |
| HIPPO: Holistic-aware Parallel Speculative Decoding | 2026 | 0 |
| SwiftVLM: Cross-Layer Token Bypass | 2026 | 0 |
| FlashVID: Tree-based Spatiotemporal Token Merging | 2026 | 0 |
