# Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning (FSR)

**作者**: Enwei Tong, Yuanchao Bai*, Yao Zhu, Junjun Jiang, Xianming Liu*  
**单位**: Harbin Institute of Technology, Zhejiang University  
**状态**: arXiv 2602.05809 (2025.02)  
**链接**: [arXiv](https://arxiv.org/abs/2602.05809) | [GitHub](https://github.com/ILOT-code/FSR)

## 一句话总结

仿人类"聚焦→扫描→精炼"的认知过程，提出三阶段 training-free visual token pruning 框架 FSR：Focus 用双通道评分选局部关键 token，Scan 用条件最远点采样补全局上下文，Refine 将丢弃 token 加权聚合到 Scan 锚点，在多个 VLM/benchmark 上取得 SOTA 精度-效率权衡。

## 核心贡献

1. **动态分配框架**: 用累计信息阈值 ρ 自动确定 Focus/Scan 的 budget 分配，task-dependent 而非 static
2. **三阶段 pipeline**: Focus（saliency × relevance）→ Scan（CCS, 2-近似覆盖保证）→ Refine（top-M 加权合并）
3. **广泛验证**: 6 个 VLM backbone、13 个 benchmark、图像+视频，一致 SOTA

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、三阶段方法、效果 |
| [01 - Introduction](sections/01-introduction.md) | 动机、三类方法局限、FSR 总览 + Figures 1-2 |
| [02 - Related Work](sections/02-related-work.md) | Attention/Similarity/Joint 三类 pruning 方法梳理 |
| [03 - Method](sections/03-method.md) | **核心**: 认知启发 + Focus/Scan/Refine 详解 + 理论保证 |
| [04 - Experiments](sections/04-experiments.md) | Tables 1-7 + 效率分析 + 消融实验 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 未提及的局限性分析 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LLaVA-1.5-7B, 64 tokens (↓88.9%) | 96.1% avg retained |
| LLaVA-NeXT-7B, 960 tokens (↓66.7%) | 100.0% (无损) |
| LLaVA-NeXT-13B, 960 tokens (↓66.7%) | 102.1% (超越baseline) |
| Qwen2.5-VL, ↓90% | 84.0% (vs HoloV 82.1%) |
| Video (LLaVA-Video), ↓60% | 99.6% |
| FLOPs reduction (64 tokens) | ~75% |
| KV cache compression | 9× |
| Prefill speedup | 3.9× |
| 默认超参 | α=3, β=1, ρ=0.9, κ=1 |

## 方法速览

```
Input: V (N visual tokens), q (query), K (budget)

Stage 1 - Focus:
  s_i = avg [CLS] attention (vision encoder)
  r_i = cos(v_i, CLIP_text(q))
  φ_i = r̂^α · ŝ^β         → α=3, β=1
  K_F = min k s.t. Σφ ≥ ρZ  → ρ=0.9, 动态确定
  F = top-K_F by φ

Stage 2 - Scan:
  K_S = K - K_F
  CCS: Farthest Point Sampling from F as seeds
  S = K_S complementary anchors

Stage 3 - Refine:
  Assign discarded D → nearest s∈S
  Select top-M (M=κ|S|) by similarity
  Weighted merge into S anchors (w=φ)

Output: V̄ = F ∪ S (K tokens, F intact, S refined)
```

## Citation Landscape

### FSR 在 Token Compression 领域的定位

```
Training-free Visual Token Pruning
├── Attention-based (偏局部)
│   ├── FastV [ECCV 2024] — cross-attn in LLM shallow layers
│   ├── LLaVA-PruMerge [2024] — attn pruning + token merging
│   ├── SparseVLM [ICML 2025] — text-guided attn + recycling
│   ├── PyramidDrop [2024] — layer-wise progressive dropping
│   ├── TopV [CVPR 2025] — FlashAttention-compatible
│   ├── FitPrune [AAAI 2025] — attn distribution divergence
│   ├── FasterVLM [2024] — [CLS]-based early pruning
│   ├── HiRED [AAAI 2025] — region-aware [CLS] pruning
│   └── SparseVILA [ICCV 2025] — decoupled prefill/decode
│
├── Similarity-based (偏全局)
│   ├── DivPrune [CVPR 2025] — max-min diversity selection
│   └── DART [EMNLP 2025] — duplication-based pivot pruning
│
├── Joint Attention-Similarity (静态混合)
│   ├── VisionZip [CVPR 2025] — attn + redundancy reduction
│   ├── VisPruner [ICCV 2025] — visual cues + redundancy
│   ├── CDPruner [NeurIPS 2025] — DPP conditional diversity
│   └── HoloV [NeurIPS 2025] — partition-wise + connectivity
│
└── Human-Inspired Dynamic (FSR 的定位)
    └── ★ FSR [arXiv 2025] — Focus-Scan-Refine, 动态分配
        ├── Focus: dual-pathway (saliency × relevance)
        ├── Scan: CCS (conditioned FPS, 2-approx guarantee)
        └── Refine: top-M weighted aggregation to scan anchors
```

### 与同类方法的关键差异

| 维度 | CDPruner | HoloV | VisPruner | **FSR** |
|------|---------|-------|-----------|---------|
| 局部/全局分配 | 静态 DPP | Partition-wise | 静态混合 | **动态阈值 ρ** |
| 全局采样 | DPP sampling | Connectivity | Redundancy | **CCS (conditioned FPS)** |
| 信息恢复 | ✗ | ✗ | ✗ | **✓ (Refine stage)** |
| 理论保证 | ✗ | ✗ | ✗ | **✓ (2-approx covering)** |
| 需 CLIP text | ✓ | ✗ | ✗ | ✓ (可选) |

---

## BibTeX

```bibtex
@article{tong2026fsr,
  title={Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning},
  author={Tong, Enwei and Bai, Yuanchao and Zhu, Yao and Jiang, Junjun and Liu, Xianming},
  journal={arXiv preprint arXiv:2602.05809},
  year={2026}
}
```
