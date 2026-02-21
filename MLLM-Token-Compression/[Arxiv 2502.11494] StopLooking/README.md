# Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More (DART)

> **一句话总结**: 不要找"重要"的 token，而是删掉"重复"的 token——DART 通过 pivot token + 余弦相似度实现 training-free、FlashAttention 兼容的视觉 token 剪枝，88.9% 压缩下仍保持 93.7% 性能。

## 📋 论文元信息

| 项目 | 内容 |
|------|------|
| **标题** | Stop Looking for "Important Tokens" in Multimodal Language Models: Duplication Matters More |
| **作者** | Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He†, Linfeng Zhang† |
| **机构** | Shanghai Jiao Tong University, Shanghai AI Laboratory, Sun Yat-sen University, Peking University |
| **发表** | EMNLP 2025 (arXiv: 2502.11494) |
| **代码** | [GitHub](https://github.com/ZichenWen1/DART) |
| **引用数** | 55 (Semantic Scholar, 截至 2026-02-21) |

## 🎯 核心贡献

1. **挑战 Importance Paradigm**: 实证表明 attention-based importance score 作为 token pruning 指标存在四大缺陷（静态评分、FlashAttention 不兼容、位置偏差、不如 random）
2. **提出 Duplication Paradigm (DART)**: 选取少量 pivot token（≤2%），通过余弦相似度删除高重复 token，完全不需要 attention score → 天然兼容 FlashAttention
3. **理论保障**: 基于 Hausdorff 距离和 Lipschitz 连续性证明了输出误差上界
4. **全面实验验证**: 4 个 MLLM × 10+ benchmark，88.9% 压缩下超第二名 2.2%
5. **跨模态泛化**: 扩展到语音（ASR）和机器人操作（VLA），均展示优越性

## 📖 批读导航

| Section | 文件 | 要点 |
|---------|------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | DART 核心思想与关键数字 |
| 1 Introduction | [01-introduction.md](sections/01-introduction.md) | 四大问题 + DART 设计动机 |
| 2 Related Work | [02-related-work.md](sections/02-related-work.md) | MLLM 架构与 token compression 方法综述 |
| 3 Methodology | [03-methodology.md](sections/03-methodology.md) | Preliminary → Importance 缺陷 → DART 方法 → 理论分析 |
| 4 Experiments | [04-experiments.md](sections/04-experiments.md) | 图像/视频理解主实验 |
| 5 Analysis | [05-analysis.md](sections/05-analysis.md) | 效率分析、pivot 选取、层/数量影响、模态分析 |
| 6-7 Conclusion & Limitations | [06-conclusion.md](sections/06-conclusion.md) | 总结与局限 |
| Appendix | [07-appendix.md](sections/07-appendix.md) | 大模型验证、语音/VLA 扩展、可视化 |

## 🔢 关键数字速查

| 指标 | 数值 |
|------|------|
| LLaVA-1.5-7B 88.9% 压缩 (64 tokens) | 93.7% 原始性能 |
| LLaVA-Next-7B 88.9% 压缩 (320 tokens) | 93.9% 原始性能 |
| Qwen2-VL-72B 88.9% 压缩 | 92.2% 原始性能 |
| 总推理加速 | 1.99× |
| Prefill 加速 | 2.99× |
| DART 额外计算开销 | < 0.08s |
| Pivot token 数量 | 8 (4 visual + 4 text) |
| 默认剪枝层 | Layer 2 |
| 不同 pivot 策略最大性能差 | 2.3% (94.9%~97.2%) |
| 不同策略保留 token 重叠率 | < 50% |
| DART† (训练时应用) 192 tokens | 100.4% (超过原模型) |
| 深层删除所有 vision token 性能降 | 0.1~1.6% |

## 🗺️ Citation Landscape

**被引用 55 次** (Semantic Scholar, 2026-02-21)

### 代表性后续工作
- EntropyPrune: Matrix Entropy Guided Visual Token Pruning (2026)
- IDPruner: Harmonizing Importance and Diversity (2026)
- D2Pruner: Debiased Importance and Structural Diversity (2025)
- "All You Need Are Random Visual Tokens?" — 进一步验证 random > importance (2025)
- VLM-Pruner: Buffering for Spatial Sparsity (2025)
- Script: Graph-Structured Token Pruning (2025)

### 关键引用 (本文引用)
- **FastV** (Chen et al., ECCV 2024): Attention-based token pruning，本文主要对比对象
- **SparseVLM** (Zhang et al., ICML 2025): Text-guided pruning，不兼容 FlashAttention
- **ToMe** (Bolya et al., ICLR 2023): Token merging，破坏跨模态交互
- **FlashAttention** (Dao et al., NeurIPS 2022): DART 的兼容优势来源
- **Over-smoothing** (Nguyen et al., NeurIPS 2023): Token duplication 的理论基础

## 💡 个人评价

**优点**:
- 核心洞察优秀：importance < random 是非常有力的实验证据，duplication paradigm 逻辑自洽
- 方法极简但有效：仅需余弦相似度，0.08s 开销，兼容 FlashAttention
- 实验全面：4 模型 × 10+ benchmark × 3 压缩率 + 消融 + 跨模态
- Pivot 不敏感是关键发现，大大降低了方法的超参敏感性

**局限**:
- 理论分析的 Lipschitz 假设在实际 transformer 中可能很松
- OCRBench 等信息密度高的任务上 DART 优势缩小
- 缺少与 ToMe (merging) 的深度对比——merging 本质上也是处理 duplication

## BibTeX

```bibtex
@inproceedings{DBLP:conf/emnlp/WenGWZZLHZ25,
  author       = {Zichen Wen and
                  Yifeng Gao and
                  Shaobo Wang and
                  Junyuan Zhang and
                  Qintong Zhang and
                  Weijia Li and
                  Conghui He and
                  Linfeng Zhang},
  title        = {Stop Looking for "Important Tokens" in Multimodal Language Models:
                  Duplication Matters More},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2025, Suzhou, China, November 4-9, 2025},
  pages        = {9961--9980},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://doi.org/10.18653/v1/2025.emnlp-main.505},
  doi          = {10.18653/V1/2025.EMNLP-MAIN.505}
}
```
