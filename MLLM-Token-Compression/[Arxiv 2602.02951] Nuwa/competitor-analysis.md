# Citation Landscape: Nüwa

## 核心竞争方法

### Vision Encoder-Side Pruning
| 方法 | 会议 | 核心思路 | PE策略 | VQA↓64 | VG↓64 |
|------|------|----------|--------|--------|-------|
| **VisionZip** | CVPR'25 | CLS attn + 语义相似度 merge | PERC | 93.99% | 7.28% |
| **PruMerge(+)** | ICCV'25 | 语义相似度 token merging | PERC | 91.71% | - |
| **Nüwa (ours)** | ICLR'26 | Boids-inspired spatial pruning | RPME | **94.91%** | **47.19%** |

### LLM Single-Layer Pruning
| 方法 | 会议 | 核心思路 | PE策略 | VQA↓64 | VG↓64 |
|------|------|----------|--------|--------|-------|
| **FastV** | ECCV'24 | Attention score pruning at layer 2 | PESP | 79.36% | 3.81% |
| **FEATHER** | ICCV'25 | Revisit pruning + analysis | - | - | 48.38%@192 |

### LLM Multi-Layer Pruning
| 方法 | 会议 | 核心思路 | PE策略 | VQA↓64 | VG↓64 |
|------|------|----------|--------|--------|-------|
| **PyramidDrop** | CVPR'25 | 金字塔式逐层 drop | - | 71.56% | - |
| **SparseVLM** | ICML'25 | 多层动态稀疏化 | PESP | 89.93% | 1.88% |

### 相关分析/基础工作
| 方法 | 核心贡献 |
|------|----------|
| **Darcet et al. (2024)** | ViT register tokens — 高 L2-norm token 的特殊角色 |
| **Boids (Reynolds, 1998)** | 群体智能算法 — separation/alignment/cohesion |
| **ToMe (Bolya et al., 2023)** | Token merging 基础框架 |

## 关键对比维度

### 1. Pruning 位置
```
Vision Encoder ────────────── LLM ──────────────────
     ↑                    ↑         ↑
  VisionZip            FastV    SparseVLM
  PruMerge                      PyramidDrop
  Nüwa Stage1          Nüwa Stage2
```

### 2. 空间保持能力
| 方法 | 空间均匀性 | PE 完整性 | VG 保留率@64 |
|------|-----------|----------|-------------|
| VisionZip | ✘ | PERC (压缩) | 7.28% |
| FastV | ✘ | PESP (稀疏) | 3.81% |
| SparseVLM | ✘ | PESP (稀疏) | 1.88% |
| Pooling | ✔ | 隐式保留 | ~20% |
| **Nüwa** | **✔** | **RPME (扩展)** | **47.19%** |

### 3. 信息聚合方式
| 方法 | 聚合基准 | 约束 |
|------|----------|------|
| PruMerge | 纯语义相似度 | 无空间约束 |
| VisionZip | CLS attention | 无空间约束 |
| Pooling | 固定网格 | 纯空间，无语义 |
| **Nüwa** | **Semantic × Spatial** | **Pillar/Collector 角色分化** |

## 发展脉络

```
Token Merging (ToMe, 2023)
    ↓
Attention-based Pruning (FastV, ECCV 2024)
    ↓
Semantic Similarity Pruning (VisionZip, PruMerge, 2024-2025)
    ↓
多层渐进 Pruning (PyramidDrop, SparseVLM, CVPR/ICML 2025)
    ↓
质疑有效性 (FEATHER, Wen et al., 2024-2025)
    ↓
空间感知 Pruning (Nüwa, ICLR 2026) ← 回答了"为什么 VG 崩溃"
```
