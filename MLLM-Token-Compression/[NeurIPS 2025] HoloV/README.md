# Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention (HoloV)

**作者**: Xin Zou, Di Lu, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Xu Zheng, Linfeng Zhang, Xuming Hu  
**机构**: HKUST(GZ), HKUST, INSAIT Sofia University, Shanghai Jiao Tong University  
**会议**: NeurIPS 2025  
**链接**: [arXiv](https://arxiv.org/abs/2510.02912) | [GitHub](https://github.com/obananas/HoloV)

## 一句话总结

HoloV 通过 crop-wise 自适应分配策略，在视觉 token 剪枝时保留全局语义上下文而非仅追逐高 attention token，实现了 88.9% 剪枝率下 95.8% 的性能保留。

## 核心贡献

1. **问题发现**: 揭示 attention-first 剪枝方法在高剪枝率下的三大缺陷——信息冗余、位置偏差、注意力弥散
2. **方差调制评分**: 提出 diversity variance + [CLS] attention 的 holistic score，衡量 token 的语义独特性而非仅靠 attention
3. **Crop-wise 自适应分配**: 将视觉 token 分为多个空间 crop，按 crop 信息量动态分配保留配额，防止 representational collapse
4. **Visual Context Refetching**: 高不确定性时通过 FFN key-value memory 重新注入被剪枝的视觉信息
5. **广泛验证**: 在 LLaVA-1.5、LLaVA-NeXT、Video-LLaVA、Qwen2.5-VL 等多种架构上一致最优

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、方法、核心结果 |
| [01 - Introduction](sections/01-introduction.md) | 动机：attention-first 方法的缺陷 + Figure 1-3 |
| [02 - Related Work](sections/02-related-work.md) | MLLMs、视觉冗余识别、token 压缩三大方向综述 |
| [03 - Preliminary & Motivation](sections/03-preliminary-motivation.md) | MLLM 架构 + 位置偏差/注意力弥散分析 + Random/Thumbnail 实验 |
| [04 - Methodology](sections/04-methodology.md) | HoloV 框架：crop 划分 → holistic score → 自适应分配 → top-k 选择 |
| [05 - Experiments](sections/05-experiments.md) | 10+ benchmark 全面实验 + 效率分析 + 消融 + 可视化 + 跨架构验证 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + 局限性 + 未来方向 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 最高剪枝率 | 88.9% (保留 64/576 tokens) |
| 性能保留 (LLaVA-1.5@88.9%) | 95.8% |
| 性能保留 (LLaVA-NeXT@88.9%) | 95.6% |
| 推理时间节省 (@90%) | 42.7% |
| 延迟降低 (@90%) | 42.8% |
| 显存节省 (@90%) | 19.0G → 14.5G |
| POPE 幻觉评估优势 (@88.9%) | 80.3% vs 77.0% (次优) |

## 📊 Citation Landscape

> 数据来源: [Semantic Scholar](https://www.semanticscholar.org/paper/f390d11e155df71ae24618c9ed80f83d0ec9b027) | [Connected Papers](https://www.connectedpapers.com/main/2510.02912)

**TLDR**: This work proposes HoloV, a simple yet effective, plug-and-play visual token pruning framework for efficient inference that rethinks token retention from a holistic perspective and achieves superior performance across various tasks, MLLM architectures, and pruning ratios compared to SOTA methods.

| 统计 | 数值 |
|------|------|
| 参考文献数 | 98 |
| 被引次数 | 9 |
| Influential Citations | 1 |

### 📚 参考文献分组

#### Visual Token Pruning / Compression

| 论文 | 年份 | 引用 | 会议 |
|------|------|------|------|
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 | 3603 | NeurIPS [arXiv](https://arxiv.org/abs/2205.14135) |
| An Image is Worth 1/2 Tokens After Layer 2 (FastV) | 2024 | 376 | ECCV [arXiv](https://arxiv.org/abs/2403.06764) |
| LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models | 2024 | 237 | arXiv [arXiv](https://arxiv.org/abs/2403.15388) |
| Token Merging: Your ViT But Faster (ToMe) | 2022 | 771 | ICLR [arXiv](https://arxiv.org/abs/2210.09461) |
| FlashAttention-2: Faster Attention with Better Parallelism | 2023 | 2252 | ICLR [arXiv](https://arxiv.org/abs/2307.08691) |

#### Multimodal LLMs

| 论文 | 年份 | 引用 | 会议 |
|------|------|------|------|
| GPT-4 Technical Report | 2023 | 21993 | — [arXiv](https://arxiv.org/abs/2303.08774) |
| Visual Instruction Tuning (LLaVA) | 2023 | 7914 | NeurIPS [arXiv](https://arxiv.org/abs/2304.08485) |
| BLIP: Bootstrapping Language-Image Pre-training | 2022 | 6008 | ICML [arXiv](https://arxiv.org/abs/2201.12086) |
| Flamingo: a Visual Language Model for Few-Shot Learning | 2022 | 5093 | NeurIPS [arXiv](https://arxiv.org/abs/2204.14198) |
| Improved Baselines with Visual Instruction Tuning (LLaVA-1.5) | 2023 | 4398 | CVPR [arXiv](https://arxiv.org/abs/2310.03744) |

#### Benchmarks & Evaluation

| 论文 | 年份 | 引用 | 会议 |
|------|------|------|------|
| Making the V in VQA Matter (VQA v2) | 2016 | 3904 | IJCV [arXiv](https://arxiv.org/abs/1612.00837) |
| GQA: A New Dataset for Real-World Visual Reasoning | 2019 | 2753 | CVPR |
| Towards VQA Models That Can Read (TextVQA) | 2019 | 1788 | CVPR [arXiv](https://arxiv.org/abs/1904.08920) |
| MMBench: Is Your Multi-modal Model an All-around Player? | 2023 | 1759 | ECCV [arXiv](https://arxiv.org/abs/2307.06281) |
| A Survey on Hallucination in LLMs | 2023 | 2142 | ACM TOIS [arXiv](https://arxiv.org/abs/2311.05232) |

#### LLMs (Foundation)

| 论文 | 年份 | 引用 | 会议 |
|------|------|------|------|
| Training language models to follow instructions with human feedback (InstructGPT) | 2022 | 18463 | NeurIPS [arXiv](https://arxiv.org/abs/2203.02155) |
| Llama 2: Open Foundation and Fine-Tuned Chat Models | 2023 | 15792 | arXiv [arXiv](https://arxiv.org/abs/2307.09288) |
| The Llama 3 Herd of Models | 2024 | 12663 | — [arXiv](https://arxiv.org/abs/2407.21783) |
| OPT: Open Pre-trained Transformer Language Models | 2022 | 4503 | arXiv [arXiv](https://arxiv.org/abs/2205.01068) |

### 🔮 推荐论文

| # | 论文 | 年份 | 引用 | 链接 |
|---|------|------|------|------|
| 1 | ConsensusDrop: Fusing Visual and Cross-Modal Saliency for Efficient Token Pruning | 2026 | 0 | [arXiv](https://arxiv.org/abs/2602.00946) |
| 2 | Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Compression | 2026 | 0 | [arXiv](https://arxiv.org/abs/2602.05809) |
| 3 | Vision Token Reduction via Attention-Driven Self-Compression | 2026 | 0 | [arXiv](https://arxiv.org/abs/2602.12618) |
| 4 | VisionTrim: Unified Vision Token Compression for Training-Free MLLMs | 2026 | 0 | [arXiv](https://arxiv.org/abs/2601.22674) |
| 5 | ViTCoP: Accelerating Large Vision-Language Models via Visual and Textual Token Compression | 2026 | 0 | [arXiv](https://arxiv.org/abs/2601.17818) |
| 6 | ViCA: Efficient Multimodal LLMs with Vision-Only Cross-Attention | 2026 | 0 | [arXiv](https://arxiv.org/abs/2602.07574) |
| 7 | HIPPO: Accelerating Video LLMs Inference via Holistic Token Pruning | 2026 | 0 | [arXiv](https://arxiv.org/abs/2601.08273) |
| 8 | SwiftVLM: Efficient Vision-Language Model Inference via Cross-Layer Token Compression | 2026 | 0 | [arXiv](https://arxiv.org/abs/2602.03134) |
| 9 | FlashVID: Efficient Video LLMs via Training-free Token Pruning | 2026 | 0 | [arXiv](https://arxiv.org/abs/2602.08024) |
| 10 | FastAV: Efficient Token Pruning for Audio-Visual LLMs | 2026 | 0 | [arXiv](https://arxiv.org/abs/2601.13143) |
