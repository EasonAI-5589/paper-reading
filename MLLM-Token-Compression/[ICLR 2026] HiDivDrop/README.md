# HiDivDrop: Vision Token Reduction in MLLMs via Late Injection and Differentiable Top-K

**会议**: **ICLR 2026 (Accepted)** ✅
**链接**: [OpenReview](https://openreview.net/forum?id=2baJBgfr9S)

## 一句话总结

挑战"浅层对融合至关重要"的流行假设，揭示 MLLM 的三阶段层级结构（传声筒→融合→推理），提出 Late Injection + Concave Pyramid Pruning + Early Exit 三板斧，在 ~90% 视觉 token 压缩率下保持原始性能并加速训练 1.72×。

## 核心贡献

1. **诊断两个误区**: 浅层是传声筒而非融合器；固定剪枝调度不匹配非均匀信息流
2. **提出 HiDivDrop**: Late Injection（跳过浅层）+ Concave Pyramid Pruning（中间层自适应剪枝）+ Early Exit（深层全丢）+ ILVAS（数据驱动选剪枝层）+ DTop-K（可微 token 选择）
3. **SOTA 效率-精度 trade-off**: 88.9% 压缩 → 98.3% 性能，训练 1.72× 加速，推理 FLOPs 降 9.1×

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题诊断、三板斧、效果 |
| [01 - Introduction](sections/01-introduction.md) | 动机、两大误区、三大贡献 + Figure 1 |
| [02 - Analysis](sections/02-analysis.md) | 三阶段层级结构分析 + Figures 2-4（全文基石） |
| [03 - Method](sections/03-method.md) | HiDivDrop 详解：Late Injection、ILVAS、DTop-K + Figure 5-6 |
| [04 - Experiments](sections/04-experiments.md) | 主实验 + 效率对比 + 充分消融 (Tables 1-6, Figures 7-8) |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性分析 |
| [06 - Related Work](sections/06-related-work.md) | Pre-LLM / In-LLM / Joint 三类方法对比 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 视觉 token 压缩率 | ~88.9% (576→64) |
| 性能保持 (7B, 64 tokens) | 98.3% |
| 训练时间减少 (7B) | 40.7% (159.3→94.4 GPU hrs) |
| 推理 FLOPs 减少 (7B) | 88.9% (3.82T→0.42T) |
| Prefill 延迟 (7B) | 63.6→28.8 ms |
| Late Injection 层 (7B) | Layer 9 |
| Early Exit 层 (7B) | Layer 25 |
| Filtering layers (7B) | {10, 14, 16, 18} |
| vs PDrop 压缩激进倍数 | 4.8× (仅多降 1.6%) |
| DTop-K vs Hard Top-K | +2.0% (97.7→99.7) |

## Citation Landscape

```
视觉 Token 压缩方法谱系（In-LLM 方向）:

                    Training-free                    Training-based
                    ─────────────                    ──────────────
单次剪枝:           FastV [ECCV'24]
                    SparseVLM [2024]

渐进剪枝:                                           PDrop [2024] ──→ HiDivDrop [ICLR'26]
                                                     TwigVLM [2025]

表示压缩:                                           VoCo-LLaMA [2024]
                                                     LLaVA-PruMerge [NeurIPS'24]

自适应/可微:                                         Dynamic-LLaVA [2024] (近似梯度)
                                                     ATP-LLaVA [2024]
                                                     HiDivDrop [ICLR'26] (精确可微) ★

HiDivDrop 的核心创新链:
  FastV 的 attention 观察 → PDrop 的渐进剪枝 → HiDivDrop 的层级感知 + 可微选择

关键差异化:
  1. Late Injection（首创延迟注入，而非提前剪枝）
  2. ILVAS（数据驱动选剪枝层，而非手工等间距）
  3. DTop-K（可微 token 选择，而非硬性 top-k）
  4. 三阶段统一框架（浅跳-中剪-深丢）
```

---

## BibTeX

```bibtex
@inproceedings{hidivdrop2026iclr,
  title     = {{HiDivDrop}: Vision Token Reduction in {MLLMs} via Late Injection and Differentiable Top-K},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Under review}
}
```
