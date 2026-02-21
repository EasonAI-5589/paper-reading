# [Arxiv 2602.05809] Focus-Scan-Refine (FSR)

> **Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning**
> Yuanchao Bai et al. | Harbin Institute of Technology & Zhejiang University
> arXiv: 2602.05809 | 2025.02
> Code: https://github.com/ILOT-code/FSR

## 📊 一句话总结

FSR 是一个受人类视觉认知启发的 training-free visual token pruning 框架，通过三阶段 Focus（局部证据）→ Scan（全局上下文）→ Refine（聚合精炼）动态分配 token budget，在多个 VLM 和 benchmark 上实现 SOTA 的精度-效率权衡。

## 🏗️ 核心方法

```
Input: N visual tokens + query → Output: K tokens (K << N)

Stage 1: FOCUS (local evidence)
├── Dual-pathway scoring: φ_i = r̂_i^α · ŝ_i^β
│   ├── s_i: CLS attention saliency (vision encoder)
│   └── r_i: CLIP cosine similarity (instruction relevance)
└── Dynamic budget: K_F = min{k | Σ φ_{π(j)} ≥ ρZ}  (ρ=0.9)

Stage 2: SCAN (global context)
├── Conditional Context Sampling (CCS) = Farthest Point Sampling
│   ├── Initialize with Focus set as anchors
│   └── Greedily select K_S = K - K_F most different tokens
└── 2-approximation coverage guarantee (k-center theory)

Stage 3: REFINE (aggregation)
├── Only merge into Scan anchors (Focus tokens unchanged!)
├── Top-M = κ|S| most similar discarded tokens (κ=1)
└── Score-weighted averaging: v_j ← (w_j·v_j + w_i·v_i)/(w_j+w_i)
```

## 📈 关键实验结果

| Model | Tokens | Reduction | Avg. Performance |
|-------|--------|-----------|-----------------|
| LLaVA-1.5-7B | 64/576 | 88.9% | **96.1%** (vs CDPruner 95.7%) |
| LLaVA-NeXT-7B | 320/2880 | 88.9% | **97.6%** (vs CDPruner 97.3%) |
| LLaVA-NeXT-13B | 640/2880 | 77.8% | **101.7%** (超过 baseline!) |
| Qwen2.5-VL-7B | ↓80% | 80% | **91.9%** (vs HoloV 88.6%) |

**效率**: 64 tokens → FLOPs ↓75%, KV cache ↓9×, prefill 3.9× speedup

## 📑 批读章节

| 章节 | 链接 |
|------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) |
| Introduction | [01-introduction.md](sections/01-introduction.md) |
| Related Work | [02-related-work.md](sections/02-related-work.md) |
| Method | [03-method.md](sections/03-method.md) |
| Experiments | [04-experiments.md](sections/04-experiments.md) |
| Conclusion | [05-conclusion.md](sections/05-conclusion.md) |

## 🗺️ Citation Landscape

### FSR 在 Token Pruning 领域的定位

```
                    Training-free Visual Token Pruning
                    ┌─────────────────────────────────┐
                    │                                 │
    ┌───────────────┼─────────────┬───────────────────┤
    │               │             │                   │
Attention-based  Similarity-based  Joint           FSR (Ours)
    │               │             │                   │
├── FastV           ├── DivPrune  ├── VisionZip      ├── Focus: dual-pathway
├── PruMerge        └── DART     ├── VisPruner       ├── Scan: CCS (FPS)
├── SparseVLM                    ├── CDPruner        └── Refine: aggregation
├── PyramidDrop                  └── HoloV
├── TopV
├── FitPrune
├── FasterVLM
└── HiRED
```

### 关键对比

| 方法 | 信号 | 局部/全局 | 动态分配 | Merge |
|------|------|----------|---------|-------|
| FastV | LLM cross-attn | Local | ✗ | ✗ |
| DivPrune | Token similarity | Global | ✗ | ✗ |
| CDPruner | Attn + DPP diversity | Joint (一步) | ✗ | ✗ |
| HoloV | Partition + connectivity | Joint | 部分 | ✗ |
| **FSR** | **Saliency + Relevance** | **阶段性 L→G** | **✓ (ρ threshold)** | **✓ (Scan only)** |

### FSR 的创新点 vs 最强竞争者 CDPruner

| 维度 | CDPruner | FSR |
|------|----------|-----|
| token 选择 | DPP 一步到位 | Focus → Scan 两阶段 |
| Local/Global | 隐式平衡 | 显式动态分配 |
| Budget 分配 | 固定 | 自适应 (ρ threshold) |
| Merge | 无 | Refine 阶段聚合 |
| 理论保证 | 无 | 2-近似覆盖 |

## 🔖 BibTeX

```bibtex
@article{bai2025fsr,
  title={Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning},
  author={Bai, Yuanchao and others},
  journal={arXiv preprint arXiv:2602.05809},
  year={2025}
}
```
