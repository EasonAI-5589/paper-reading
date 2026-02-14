# VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration

**作者**: Hanxun Yu, Wentong Li, Xuan Qu, Song Wang, Junbo Chen, Jianke Zhu  
**机构**: Zhejiang University, Nanjing University of Aeronautics and Astronautics, Udeer.ai  
**arXiv**: [2601.22674](https://arxiv.org/abs/2601.22674) | **日期**: 2026-01-30  
**代码**: [GitHub](https://github.com/hanxunyu/VisionTrim)

---

## 一句话总结

Training-free 的 MLLM 加速框架，通过 DVTS（global-local 双视角选 dominant token）+ TGVC（text-guided 聚类合并 complement token）在 vision encoder 和 LLM decoder 两个阶段统一压缩视觉 token，88.9% 压缩率下保持 98.8% 性能。

## 核心贡献

1. **统一两阶段压缩**: 首个在 ViT + LLM 两个阶段都做 training-free token 压缩的方法
2. **DVTS (Dominant Vision Token Selection)**: global [CLS] attention + local LTAM (dual-kernel affinity) + adaptive variance-based weighting
3. **TGVC (Text-Guided Vision Complement)**: 利用 CLIP text encoder 对被丢弃 token 做 text-guided 聚类合并，补回与文本相关的视觉信息
4. **全面验证**: 标准分辨率/高分辨率/视频，5 个 MLLM backbone，14 个 benchmark

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义 + 方案概览 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 现有方法不足 + 三点贡献 + Figure 1-2 |
| [02 - Related Work](sections/02-related-work.md) | MLLM 背景 + Token 压缩方法综述 |
| [03 - Methodology](sections/03-methodology.md) | **核心**: DVTS + TGVC + Multi-Stage Pruning |
| [04 - Experiment](sections/04-experiment.md) | 主实验 + Ablation + 效率分析 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性 |
| [06 - Appendix](sections/06-appendix.md) | 极端压缩(1 token)、OCR-heavy、更多模型、可视化 |

## 关键数字

| 设置 | 压缩率 | 性能保持 | 加速比 |
|------|--------|---------|--------|
| LLaVA-1.5, 64 tokens | 88.9% | 98.8% | 1.90× |
| LLaVA-NeXT, 320 tokens | 88.9% | 97.0% | 2.48× |
| Video-LLaVA, 136 tokens | 93.4% | 98.0% | — |
| Qwen2-VL, ~1/3 tokens | ~66.7% | ~99.9% | — |
| 极端: 1 token | 99.8% | 82.8% | — |

## 与同类方法对比

| 方法 | 压缩阶段 | 文本引导 | 核心差异 |
|------|----------|---------|---------|
| **VisionTrim** | ViT + LLM | ✅ TGVC | global-local 选 + text-guided 补 |
| VScan | ViT + LLM | ❌ | 无 text-guided merging |
| VisionZip | ViT | ❌ | [CLS] attention + similarity merging |
| FastV | LLM Layer 2 | ❌ | 只在 LLM 端做一次 pruning |
| SparseVLM | LLM | 部分 | Text-guided pruning，无 complement |
| PyramidDrop | LLM 渐进 | ❌ | 金字塔式渐进 pruning |

## BibTeX

```bibtex
@article{yu2026visiontrim,
  title={VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration},
  author={Yu, Hanxun and Li, Wentong and Qu, Xuan and Wang, Song and Chen, Junbo and Zhu, Jianke},
  journal={arXiv preprint arXiv:2601.22674},
  year={2026}
}
```
