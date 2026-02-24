# IDPruner: Harmonizing Importance and Diversity in Visual Token Pruning for MLLMs

> **arXiv**: [2602.13315](https://arxiv.org/abs/2602.13315) | **日期**: 2026-02-10 | **代码**: [Tencent/AngelSlim](https://github.com/Tencent/AngelSlim)

## 📋 论文基本信息

| 项目 | 内容 |
|------|------|
| **标题** | IDPruner: Harmonizing Importance and Diversity in Visual Token Pruning for MLLMs |
| **作者** | Yifan Tan (Tsinghua), Yifu Sun, Shirui Huang, Hong Liu, Guanghua Yu, Jianchen Zhu (Tencent), Yangdong Deng |
| **机构** | 清华大学软件学院 + 腾讯 |
| **会议/期刊** | arXiv 预印本（2026-02-10） |
| **任务** | MLLM 视觉 Token 剪枝（推理加速） |

## 🎯 一句话总结

IDPruner 将视觉 token 剪枝重新表述为信息检索中的重排序问题，采用 **最大边际相关性（MMR）算法**，在 token 重要性与语义多样性之间实现 Pareto 最优平衡，无需注意力图，兼容 FlashAttention，在多架构多任务上达到 SOTA。

## 🔑 核心贡献

1. **系统性分析框架**：用 Hopkins Statistic 量化多样性，用 Importance Retention Ratio 量化重要性，首次用 Pareto 前沿可视化两者的权衡关系
2. **IDPruner 算法**：将 MMR 算法适配到视觉 token 剪枝，通过迭代选择使重要性和多样性联合最优
3. **工程友好**：无需注意力图 → 兼容 FlashAttention；一次性剪枝（one-shot）→ 易于集成 vLLM；O(KN) 复杂度开销可忽略
4. **强泛化性**：在 Qwen2.5-VL、LLaVA-1.5、LLaVA-OV、Qwen2.5-VL-3B 四种架构上均达 SOTA

## ⚠️ 关键局限

- **依赖 VisionSelector（需要训练）**：重要性估计模块来自 VisionSelector，不是完全 training-free 的方法
- 未在长视频理解 benchmark 上评测
- λ 超参数未做精细搜索（但默认 0.5 表现稳定）

## 🗺️ Section 导航

| Section | 内容 | 文件 |
|---------|------|------|
| Abstract | 摘要 | [00-abstract.md](sections/00-abstract.md) |
| 1. Introduction | 动机与贡献 | [01-introduction.md](sections/01-introduction.md) |
| 2. Related Work | 相关工作分类 | [02-related-work.md](sections/02-related-work.md) |
| 3. Empirical Analysis | 重要性-多样性权衡分析 + MMR机制 | [03-empirical-analysis.md](sections/03-empirical-analysis.md) |
| 4. Method | IDPruner 算法详细 | [04-method.md](sections/04-method.md) |
| 5. Experiments | 主实验结果 | [05-experiments.md](sections/05-experiments.md) |
| 6. Conclusion | 结论 | [06-conclusion.md](sections/06-conclusion.md) |
| Appendix | 更多实验 + 消融 + 可视化 | [07-appendix.md](sections/07-appendix.md) |

## 🔄 与 STAR-Pro 的关系（竞品分析）

IDPruner 是 **STAR-Pro 的直接竞品**，以下是关键对比：

| 维度 | IDPruner | STAR-Pro |
|------|----------|----------|
| **核心思路** | MMR 算法：重要性 - 冗余惩罚 | （待对比） |
| **是否 Training-Free** | ❌ **否** — 依赖 VisionSelector 训练模块 | ✅ Training-free |
| **重要性来源** | VisionSelector（可学习，DiffTopK） | （STAR-Pro 方案） |
| **多样性保证** | MMR 迭代选择，余弦相似度惩罚 | （STAR-Pro 方案） |
| **注意力图依赖** | ❌ 无需 | 待确认 |
| **FlashAttention** | ✅ 兼容 | 待确认 |
| **剪枝方式** | One-shot（前期一次性剪枝） | 待确认 |

**STAR-Pro 需在 Related Work → Hybrid Strategies 中引用 IDPruner**，定位为「使用 MMR 的 hybrid 方法，但依赖训练模块，而 STAR-Pro 无需训练」。

## 📊 Citation Landscape

**来源**: Semantic Scholar API（2026-02-24 查询）

| 指标 | 数值 |
|------|------|
| **arXiv ID** | 2602.13315 |
| **引用数** | 0（新论文，2026-02） |
| **参考文献数** | 37 |
| **TLDR** | Proposes IDPruner using MMR to achieve Pareto-optimal balance between token importance and semantic diversity, achieving SOTA performance with superior generalization |

### 重要参考文献（按引用量）

| 论文 | 年份 | 引用量 | 关联 |
|------|------|--------|------|
| PagedAttention (vLLM) | 2023 | 4687 | 工程集成基础 |
| LLaVA-1.5 | 2023 | 4402 | 测试架构 |
| FlashAttention | 2022 | 3606 | 兼容性设计目标 |
| Qwen2-VL | 2024 | 3275 | 主要测试架构 |
| ScienceQA | 2022 | 1971 | 评测基准 |
| TextVQA | 2019 | 1790 | 评测基准 |
| MMR (Carbonell 1998) | 1998 | 1521 | **IDPruner 核心算法来源** |
| ChartQA | 2022 | 1209 | 评测基准 |
| DocVQA | 2020 | 1169 | 评测基准 |
| LLaVA-OneVision | 2024 | 1911 | 测试架构 |
| Video-MME | 2024 | 925 | 视频评测基准 |
| PyramidDrop | 2024 | 155 | 竞品（importance-based） |
| SparseVLM | 2024 | 220 | 竞品 |
| [CLS] Attention Pruning | 2024 | 60 | 竞品基础 |
| DART (duplication matters) | 2025 | 55 | 竞品（diversity-based） |
| PACT | 2025 | 22 | 竞品（hybrid） |
| CDPruner (conditional diversity) | 2025 | 24 | 竞品（hybrid） |
| VisionSelector | 2025 | 1 | **IDPruner 的重要性估计来源** |
| SCOPE | 2025 | 1 | 竞品（hybrid） |

### 关键 Related Work 图谱

```
视觉 Token 剪枝
├── Importance-based
│   ├── FastV (2024) — LLM decoder attention
│   ├── VisionZip (2025) — CLS attention
│   ├── HiPrune (2025) — hierarchical attention
│   └── VisionSelector (2025) ← IDPruner 的重要性模块
├── Diversity-based
│   ├── DivPrune (2025) — Max-Min Diversity Problem
│   └── DART (2025) — pivot tokens + neighbor elimination
└── Hybrid
    ├── VisPruner (2024) — CLS + diversity
    ├── CDPruner (2025) — DPP
    ├── SCOPE (2025)
    └── IDPruner (2026) ← 本文，MMR
```
