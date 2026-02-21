# Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning

| 属性 | 值 |
|------|-----|
| **论文** | [arXiv:2602.02951](https://arxiv.org/abs/2602.02951) |
| **会议** | ICLR 2026 |
| **作者** | Yihong Huang, Fei Ma, Yihua Shao, Jingcai Guo, Zitong Yu, Laizhong Cui, Qi Tian |
| **机构** | 粤港澳大湾区数字经济实验室 (SZ), 西安电子科技大学, 香港理工大学, 大湾区大学, 深圳大学, 华为 |
| **代码** | [GitHub](https://github.com/Man-PaperRejected/Nuwa) |
| **核心任务** | Vision Token Pruning for VLMs |
| **关键词** | Token Pruning, Visual Grounding, Spatial Integrity, Position Embedding |

## 🎯 一句话总结

现有 token pruning 方法在 VG 任务上崩溃（保留率 1.88%~7.28%），根因是**全局空间参考系被破坏**；Nüwa 通过两阶段空间感知 pruning 将 VG 保留率提升至 47.2%，同时保持 VQA 95% 性能。

## 📊 核心数据

| 配置 | VQA 保留率 | VG 保留率 | Token 压缩率 | TFLOPs 减少 | Prefill 减少 |
|------|-----------|----------|-------------|------------|-------------|
| 192 tokens | 98.80% | 79.29% | 66.7% | - | - |
| 128 tokens | 97.87% | 75.20% | 77.8% | - | - |
| 64 tokens | 94.91% | 47.19% | 88.9% | 89% | 62% |

## 🔑 三个核心发现

1. **Finding 1**: 高级 pruning 方法在 VQA 上不比 random/pooling 好多少，在 VG 上全面崩溃，pooling 表现最好
2. **Finding 2**: VLM 视觉处理是多阶段流水线（global → fine-grained），VG 任务在中间层对视觉信息需求最高
3. **Finding 3**: VG 退化的根因是 Position Embedding 的全局空间参考系被破坏，可通过 RPME 修复

## 🏗️ 方法架构

```
Stage 1: Vision Encoder 输出端                 Stage 2: LLM 中间层
┌──────────────────────────────┐           ┌──────────────────────┐
│ 1. Separation (Grid分区)      │           │ Text-guided Pruning  │
│    N×N → M×M regions         │           │                      │
│ 2. Alignment (显著性选择)      │    →      │ cosine(proj(v'), q̄)  │
│    CLS_attn × ||k||₂         │           │ 保留 top-K_final     │
│ 3. Aggregation (空间近邻聚合)  │           │                      │
│    W = A(semantic) × P(spatial)│          └──────────────────────┘
│    Pillar(不改) + Collector(聚合)│
└──────────────────────────────┘
```

## 📁 批读文件

| 文件 | 内容 |
|------|------|
| [00-abstract.md](sections/00-abstract.md) | Abstract |
| [01-introduction.md](sections/01-introduction.md) | Introduction |
| [02-analysis.md](sections/02-analysis.md) | Sec 2: 视觉处理流水线分析（Finding 1-3） |
| [03-methodology.md](sections/03-methodology.md) | Sec 3: Nüwa 方法论 |
| [04-experiments.md](sections/04-experiments.md) | Sec 4: 实验结果 + 消融 |
| [05-conclusion.md](sections/05-conclusion.md) | Conclusion |
| [competitor-analysis.md](competitor-analysis.md) | Citation Landscape |

## 💡 关键启示

1. **空间信息 ≠ 语义信息**：pruning 研究长期只关注语义保留，忽视了空间结构
2. **Position Embedding 的隐式作用**：PE 策略（PERC/PESP/RPME）对 VG 有决定性影响
3. **Grid Partitioning 是核心**：消融实验显示，仅加 region 就能让 VG 从 6.83% 跳到 43.50%
4. **Pooling 基线被低估**：它隐式保留了空间拓扑，是很多"精心设计"方法的上限
5. **Boids 算法的启发**：群体智能中的 separation-alignment-cohesion 规则可以很自然地映射到 token pruning

## 🔗 与其他方法的关系

- **vs FastV**: FastV 在 layer 2 后用 attention score pruning（PESP），Nüwa 在 encoder + LLM 两阶段 pruning
- **vs VisionZip**: VisionZip 用 CLS attention + 全局语义相似度（PERC），Nüwa 加入空间约束
- **vs PruMerge**: PruMerge 用纯语义相似度合并（无空间约束），Nüwa 结合 semantic + spatial proximity
- **vs SparseVLM**: SparseVLM 多层渐进 pruning，Nüwa 两阶段但保持空间完整性
