# HiDivDrop: Vision Token Reduction in MLLMs via Late Injection and Differentiable Top-K

**作者**: Anonymous (under double-blind review)  
**会议**: ICLR 2026 (Under Review)  
**链接**: [OpenReview](https://openreview.net/forum?id=2baJBgfr9S) | [PDF](https://openreview.net/pdf?id=2baJBgfr9S)

## 一句话总结

将MLLM的层级划分为浅层（传播者）→中层（稀疏融合中心）→深层（语言推理），通过Late Injection跳过浅层、Concave Pyramid Pruning在中层积极剪枝、Early Exit在深层丢弃vision tokens，实现~90%压缩率几乎无损性能。

## 核心贡献

1. **诊断两个误解**: 浅层不是融合器而是传播者；pruning schedule不应该是均匀的
2. **Late Injection**: 首次提出"延迟注入"而非"提前剪枝"，vision tokens直到Layer 9才进入LLM
3. **Concave Pyramid Pruning**: 前快后慢的非均匀剪枝，配合ILVAS指标自动选择最佳剪枝层
4. **Early Exit**: Layer 25后完全丢弃vision tokens，深层纯做language reasoning
5. **Differentiable Top-K**: 可微分token选择，端到端训练，比Hard Top-K提升2%

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 核心创新总结 |
| [01 - Introduction](sections/01-introduction.md) | 两个误解 + HiDivDrop动机 + Figure 1对比 |
| [02 - Processing Dynamics](sections/02-processing-dynamics.md) | ⭐ MLLM三层结构分析（最重要的分析Section） |
| [03 - Method](sections/03-method.md) | Late Injection + Concave Pyramid + DTop-K + 工程细节 |
| [04 - Experiments](sections/04-experiments.md) | 11 benchmarks × 3 backbones + 详细ablation |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 与STAR-Pro对比 |
| [06 - Related Work](sections/06-related-work.md) | Pre-LLM / In-LLM / Joint分类 |

## 关键数字速查

| 指标 | 数值 |
|------|------|
| Visual token压缩率 | 88.9% (576→64) |
| 性能保持 (88.9%压缩) | 98.3% |
| 性能保持 (91.7%压缩) | 96.5% |
| 训练加速 (7B) | 1.69× (159→94 GPU hours) |
| 推理FLOPs减少 | 9.1× (3.82T→0.42T) |
| Prefill延迟降低 | 49% (63.6→32.6ms) |
| Late Injection layer | 9 (7B) / 15 (2.7B) |
| Early Exit layer | 25 (7B) / 28 (2.7B) |
| Filtering layers | {10, 14, 16, 18} (7B) |
| DTop-K vs Hard Top-K | +2.0% (PT+FT设置) |

## 🏗️ 方法架构

```
Input: 576 vision tokens + Nt text tokens
  │
  ├── Layer 1-8:  只处理text tokens ←── Late Injection
  │               (并行运行vision encoder)
  │
  ├── Layer 9:    注入全部576 vision tokens
  ├── Layer 10:   DTop-K → ~256 tokens ←── ILVAS filtering
  ├── Layer 14:   DTop-K → ~128 tokens     layers
  ├── Layer 16:   DTop-K → ~96 tokens
  ├── Layer 18:   DTop-K → ~64 tokens
  │
  ├── Layer 25:   丢弃所有vision tokens ←── Early Exit
  │
  └── Layer 26-32: 纯language reasoning
```

## 🔬 与STAR-Pro的对比

| 维度 | HiDivDrop | STAR-Pro |
|------|-----------|----------|
| 核心问题 | **WHERE** — 在哪些层做什么 | **WHAT** — 用什么indicator选token |
| 层级理解 | shallow/middle/deep三段式 | 未显式区分层级 |
| Token选择 | DTop-K (attention-based) | Star indicator (multi-criteria) |
| 创新点 | Late Injection, Concave Pyramid | Importance indicator设计 |
| 互补性 | 可在HiDivDrop框架中替换DTop-K为STAR indicator |

## 📊 Citation Landscape

> ⚠️ 论文处于ICLR 2026 double-blind review阶段，尚未公开作者信息，Semantic Scholar暂无收录。

### 参考文献分组

**Progressive Token Pruning**:
- FastV (Chen et al., 2024b) — 单次early pruning
- PDrop / PyramidDrop (Xing et al., 2024) — 均匀progressive pruning
- TwigVLM (Shao et al., 2025) — twig block辅助剪枝
- Multi-stage VTD (Liu et al., 2024c) — 多阶段token dropping

**Token Compression (Non-pruning)**:
- VoCo-LLaMA (Ye et al., 2024b) — 压缩到VoCo token
- LLaVA-PruMerge (Chen et al., 2024a) — 池化压缩
- TokenPacker (Li et al., 2024b) — compact projector
- Honeybee (Cha et al., 2024) — locality-enhanced projector

**Adaptive Pruning**:
- ATP-LLaVA (Ye et al., 2024a) — adaptive token pruning
- Dynamic-LLaVA (Huang et al., 2024) — soft gating
- FocusLLaVA (Zhu et al., 2024) — 粗到细

**Differentiable Selection**:
- Differentiable Top-K (Liu et al., 2024b) — HiDivDrop采用的基础operator

**MLLM Architectures**:
- LLaVA-1.5 (Liu et al., 2023a) — 基础架构
- Qwen-VL (Bai et al., 2023) / Qwen2.5-VL (Bai et al., 2025)
- GPT-4V (OpenAI, 2023) / GPT-4o (OpenAI, 2024)
