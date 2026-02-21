# Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning

**作者**: Yihong Huang, Fei Ma*, Yihua Shao, Jingcai Guo, Zitong Yu, Laizhong Cui, Qi Tian  
**单位**: Guangdong Lab of AI & Digital Economy (SZ), Xidian University, HK PolyU, Great Bay University, Shenzhen University, Huawei  
**会议**: ICLR 2026 (Poster) ✅  
**链接**: [arXiv 2602.02951](https://arxiv.org/abs/2602.02951) | [GitHub](https://github.com/Man-PaperRejected/Nuwa) | [OpenReview](https://openreview.net/forum?id=C9yclwdquU)

## 一句话总结

首次系统分析 token pruning 对 visual grounding (VG) 的破坏性影响，发现根因是**全局空间参考系丢失**，提出两阶段空间感知框架 Nüwa（Boids 启发聚合 + text-guided 精调），在 88.9% token 压缩下 VQA 保持 95%、VG 从 7% 跃升至 47%。

## 核心贡献

1. **诊断问题**: 揭示所有现有 pruning 方法在 VG 任务上系统性退化（甚至不如 random），根因是全局空间参考系（Global Spatial Reference Frame）的破坏
2. **分析流水线**: 通过 VAE/OCC 指标和 attention flow 分析，揭示 VLM 多阶段视觉处理流水线（全局→精细→任务特定）
3. **Nüwa 框架**: 两阶段 training-free pruning——Stage 1 (ViT encoder): Boids-inspired Separation-Alignment-Aggregation；Stage 2 (LLM mid-layer): text-guided cosine similarity pruning
4. **SOTA 验证**: 13 个数据集 × 3 个模型（LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL），VG 性能 7×提升

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义、方法概述、关键数字 |
| [01 - Introduction](sections/01-introduction.md) | 动机、三个核心问题、Boids 灵感、三大贡献 |
| [02 - Dissecting Pipeline](sections/02-dissecting-pipeline.md) | **核心分析**: Finding 1-3, VAE/OCC 指标, PE 分类, 位置重建 |
| [03 - Methodology](sections/03-methodology.md) | 两阶段框架：Separation→Alignment→Aggregation + Text-guided S2 |
| [04 - Experiments](sections/04-experiments.md) | 主实验 (Tables 5-6) + 效率分析 (Table 4) + 消融 (Tables 7-8) |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 未来方向思考 |
| [06 - Related Work](sections/06-related-work.md) | Appendix A: 高效 VLM + Token Pruning 分类 |
| [07 - Appendix](sections/07-appendix.md) | Appendix B-E: 多模型结果、Attention Blocking、可视化、Case Study |

## 关键数字

| 指标 | 数值 |
|------|------|
| Token 压缩比 (最激进) | 88.9% (576→64) |
| VQA 保持率 (64 tokens) | 94.91% |
| VG 保持率 (64 tokens) | **47.19%** (前作最好 7.28%) |
| VG 保持率 (128 tokens) | **75.20%** (前作最好 18.55%) |
| TFLOPs 减少 | 89% |
| Prefill 时间减少 | 62% |
| Stage 1 额外开销 | +117.6 MFLOPs (+1ms) |
| 最优距离阈值 | 26% max distance |

## 方法流程图

```
输入: N² vision tokens (e.g., 576 = 24×24)
│
├─ Stage 1: Spatial Cohesion Pruning (Vision Encoder) ────────────────┐
│   1. Separation: 24×24 → M×M non-overlapping regions              │
│   2. Alignment: 每 region 选 top-n tokens (CLS_attn × L2_norm)     │
│   3. Aggregation:                                                   │
│      - Pillar Tokens (top 25% L2-norm): 保持不变 (register tokens)  │
│      - Collector Tokens: 聚合邻居 (ReLU(cos_sim) × spatial_prox)    │
│   → K benchmark tokens (e.g., 112)                                  │
├─────────────────────────────────────────────────────────────────────┘
│
├─ LLM Early Layers: Multimodal Alignment (几层后 text-vision 对齐)
│
├─ Stage 2: Text-Modulated Pruning (LLM Mid-Layer) ──────────────────┐
│   1. avg-pool text embeddings → query vector q̄                     │
│   2. cosine_sim(proj(v_i'), q̄) → relevance score R_i               │
│   3. 保留 top-K_final tokens (e.g., 16)                             │
├─────────────────────────────────────────────────────────────────────┘
│
└─ LLM Later Layers: Reasoning + Generation
```

## Citation Landscape

### 核心前作（Nüwa 直接对比的方法）
| 方法 | 会议 | 类别 | Nüwa 的评价 |
|------|------|------|------------|
| **FastV** (Chen et al., 2024) | ECCV'24 | LLM Single-Layer | PESP 策略，VG 崩溃 (3.81%) |
| **VisionZip** (Yang et al., 2024) | CVPR'25 | Encoder-Side | PERC 策略，VQA 强但 VG 崩 (7.28%) |
| **SparseVLM** (Zhang et al., 2025b) | ICML'25 | LLM Multi-Layer | VQA 表现好，VG 几乎为 0 (1.88%) |
| **PyramidDrop** (Xing et al., 2024) | CVPR'25 | LLM Multi-Layer | VQA 表现好，VG 未测 |
| **PruMerge** (Shang et al., 2024) | ICCV'25 | Encoder-Side | 有 merge 但无空间约束 |
| **ToME** (Bolya et al., 2023) | - | Encoder-Side Merge | Token merging 的奠基工作 |

### 关键分析工具/概念
| 概念 | 来源 | 在 Nüwa 中的作用 |
|------|------|-----------------|
| Boids 算法 | Reynolds, 1998 | Stage 1 设计灵感 (Separation-Alignment-Cohesion) |
| Register Tokens | Darcet et al., 2024 | Pillar Token 设计依据 |
| Multimodal Alignment | Shukor & Cord, 2024 | Stage 2 剪枝时机的理论支撑 |
| Grad-CAM | Selvaraju et al., 2016 | Gradient-weighted attention 分析 |

### Nüwa 的独特定位
```
                        Training-Free Token Pruning 方法谱系
                        
Encoder-Side ─────── LLM-Side ─────── Multi-Stage
  ToME                 FastV              MustDrop
  VisionZip            SparseVLM          LightVLM
  PruMerge             PyramidDrop        GlobalCom²
  DivPrune             HiPrune
      │                    │                  │
      └─── 全部在 VG 上崩溃 ←──────────────────┘
                    │
            ┌───────┴───────┐
            │   Nüwa (本文)   │
            │ Encoder + LLM  │
            │ 空间感知 + 文本引导│
            │ VG 47% (7x↑)   │
            └────────────────┘
```

---

## BibTeX

```bibtex
@inproceedings{huang2026nuwa,
  title     = {N{\"u}wa: Mending the Spatial Integrity Torn by {VLM} Token Pruning},
  author    = {Huang, Yihong and Ma, Fei and Shao, Yihua and Guo, Jingcai and Yu, Zitong and Cui, Laizhong and Tian, Qi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
