# ToDRE: Effective Visual Token Pruning via Token Diversity and Task Relevance

**作者**: Duo Li, Zuhao Yang, Xiaoqin Zhang, Ling Shao, Shijian Lu
**单位**: NTU Singapore, ZJUT China, Terminus AI Lab / UCAS China
**发表**: arXiv 2505.18757 (2025)
**链接**: [arXiv](https://arxiv.org/abs/2505.18757)

## 一句话总结

发现 visual token 冗余由两个正交因素构成——intra-modal diversity 和 cross-modal task relevance，提出两阶段 training-free 框架：Stage 1 用 greedy max-sum diversification 保留多样化子集，Stage 2 在 LLM decoder 深层全删 visual token，90% 剪枝率下保持 95.0% 性能、2.6× 加速。

## 核心贡献

1. **重新定义冗余**: 证明 token diversity 和 task relevance 是正交的，应分别处理
2. **两阶段 plug-and-play 框架**: Embedding space diversification + decoder-layer relevance pruning，兼容 FlashAttention
3. **全面验证**: 4 个 LVLM × 12 个 benchmark（图像+视频），一致 SOTA

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义、两阶段方法、核心结果 |
| [01 - Introduction](sections/01-introduction.md) | 动机、现有方法局限、Information Migration 发现、三大贡献 |
| [02 - Related Work](sections/02-related-work.md) | LVLM 架构 + Token Compression 方法分类 |
| [03 - Preliminary Analysis](sections/03-preliminary-analysis.md) | FLOPs 分布分析 + 冗余正交分解 |
| [04 - Method](sections/04-method.md) | Stage 1 Diversification + Stage 2 Relevance Reduction |
| [05 - Experiments](sections/05-experiments.md) | 全面实验 + 效率对比 + 消融实验 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + 未提及的局限性 |
| [07 - Supplementary](sections/07-supplementary.md) | 理论证明 + 超参数消融 + Case Study |

## 关键数字

| 指标 | 数值 |
|------|------|
| Visual token 剪枝率 | 90% (保留 288/2880) |
| 性能保持 (7B, 10%) | 95.0% |
| 性能保持 (7B, 25%) | 98.2% |
| FLOPs 减少 | 80.9% |
| 推理加速 | 1.9× throughput / 2.6× total |
| 内存节省 | 14.5% (15.9→13.6 GB) |
| Stage 2 alone 性能 | 100.0% (无损) |
| FLOPs 比例 encoding:prefilling:decoding | 1:63.6:0.4 (7B) |
| 默认阈值 τ | 0.10 |
| 默认检测层 | 7L/8 |

## 方法速览

```
Input Image/Video
       │
       ▼
  Vision Encoder (ViT)
       │
       ▼ Projector
  Visual Token Embeddings (n ≈ 2880)
       │
       ├── Stage 1: Diversity-Driven Selection
       │     1. [CLS] attention → select pivot token
       │     2. Greedy max-sum diversification → retain k tokens
       │     Output: k diverse tokens (e.g., k=288, 10%)
       │
       ▼
  LLM Decoder Prefilling
       │
       ├── Layer 1 ~ 7L/8: normal computation
       │
       ├── Stage 2: Relevance-Driven Reduction @ Layer 7L/8
       │     Check α_t→v < τ AND α_v→t < τ
       │     If yes: remove ALL visual tokens
       │
       ├── Layer 7L/8+1 ~ L: no visual tokens
       │
       ▼
  LLM Decoding (no visual KV cache)
       │
       ▼
  Output
```

---

## Citation Landscape

### 🔵 ToDRE 的核心前驱（直接对标/改进的工作）

| 论文 | 关系 | 要点 |
|------|------|------|
| **FastV** [10] (ECCV 2024) | 直接对标 | Attention-based pruning after layer 2；ToDRE 指出其 positional bias 缺陷 |
| **FasterVLM** [62] (2024) | 直接对标 | [CLS]-to-visual attention pruning；ToDRE 指出其 attention 分布不平衡 |
| **SparseVLM** [63] (2024) | 直接对标 | Cross-modal attention + recycling；ToDRE 指出其 13B 迁移性差 |
| **VTW** [35] (AAAI 2025) | Stage 2 灵感源 | Decoding 阶段 information migration + KL divergence 选层；ToDRE 改用 attention ratio |
| **DivPrune** [2] (2025) | 最近前辈 | Min-max diversity pruning；ToDRE 用 max-sum diversity + 增加 Stage 2 |
| **ToMe** [6] (ICLR 2023) | Token merging 前辈 | ViT 中 binary soft-matching merge；ToDRE 指出 merge < pruning |

### 🟢 关键背景/支撑工作

| 论文 | 角色 |
|------|------|
| **LLaVA-NeXT** [37] (2024) | 主要实验 backbone (7B/13B) |
| **Qwen2.5-VL** [5] (2025) | 跨模型验证 backbone |
| **InternVL2** [50] (2024) | 跨模型验证 backbone |
| **CLIP-ViT** [45] (ICML 2021) | Vision encoder，[CLS] attention 的来源 |
| **FlashAttention** [13] (NeurIPS 2022) | ToDRE 兼容的高效 attention 算子 |
| **Wen et al.** [55] (2025) | 发现 attention positional bias (attention shift) |

### 🔴 竞争方法

| 论文 | 策略 | ToDRE 的优势 |
|------|------|-------------|
| **GlobalCom2** [39] (2025) | Thumbnail 引导 crop 压缩 | 仅支持图像，不支持视频 |
| **FocusLLaVA** [67] (2024) | Coarse-to-fine 两阶段 | 需多次评估 |
| **FiP** [60] (AAAI 2025) | Training-free KL-based | 需校准集 |
| **Multi-Stage VTD** [38] (2024) | 多阶段 dropping | 复杂度高 |

### 📊 方法谱系图

```
Token Compression for LVLMs
├── Training-required
│   ├── TokenPacker [28], Matryoshka [7], DeCo [59]
│   └── LLaVA-KD [8]
│
└── Training-free (ToDRE 所在)
    ├── Attention-based
    │   ├── FastV [10] — avg attention pruning
    │   ├── FasterVLM [62] — [CLS] attention pruning
    │   ├── SparseVLM [63] — cross-modal attention ranking
    │   └── LLaVA-PruMerge [46] — attention-guided merge
    │
    ├── Similarity-based
    │   ├── ToMe [6] — bipartite soft-matching merge
    │   └── GraphPrune [23] — graph-based token pruning
    │
    ├── Divergence-based
    │   ├── VTW [35] — KL divergence layer selection
    │   └── FiP [60] — fast KL-based pruning
    │
    ├── Diversity-based
    │   ├── DivPrune [2] — min-max diversity
    │   └── ★ ToDRE — max-sum diversity + relevance reduction
    │
    └── Hybrid / Multi-stage
        ├── GlobalCom2 [39] — thumbnail-guided
        ├── FocusLLaVA [67] — coarse-to-fine
        └── Han et al. [19] — unified paradigm
```

---

## BibTeX

```bibtex
@article{li2025todre,
  title={ToDRE: Effective Visual Token Pruning via Token Diversity and Task Relevance},
  author={Li, Duo and Yang, Zuhao and Zhang, Xiaoqin and Shao, Ling and Lu, Shijian},
  journal={arXiv preprint arXiv:2505.18757},
  year={2025}
}
```
