# SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

**作者**: Jinhong Deng, Wen Li, Joey Tianyi Zhou, Yang He  
**机构**: UESTC, A*STAR Singapore  
**会议**: NeurIPS 2025  
**链接**: [arXiv 2510.24214](https://arxiv.org/abs/2510.24214) | [GitHub](https://github.com/kinredon/SCOPE)

## 一句话总结

SCOPE 提出联合建模 saliency（显著性）和 coverage（语义覆盖度）的 visual token pruning 策略，通过贪心迭代选择 SCOPE score = Coverage Gain × Saliency^α 最高的 token，实现 9× token 压缩仅损失 4% 性能。

## 核心贡献

1. **揭示问题**: Saliency-only pruning 导致语义不完整（θ-coverage 甚至低于 random）且受注意力偏斜分布影响
2. **SCOPE 方法**: 定义 set-coverage → token-coverage gain → 整合 saliency score，贪心迭代选择（submodular optimization 思想）
3. **Training-free**: 即插即用，适用于 LLaVA-1.5、LLaVA-Next、Video-LLaVA、Qwen2-VL 等多种 MLLM
4. **SOTA 性能**: 在所有配置下一致超越 FastV、SparseVLM、VisionZip、PDrop

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义、方法概述、核心结果 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + Figure 1（可视化 + 注意力偏斜 + 性能对比）+ 三点贡献 |
| [02 - Related Work](sections/02-related-work.md) | MLLM 范式 + 现有 token pruning 方法分类 |
| [03 - Method](sections/03-method.md) | Preliminary → θ-Coverage 分析 → SCOPE 算法（核心公式 + Algorithm 1） |
| [04 - Experiment](sections/04-experiment.md) | LLaVA-1.5/Next 主实验 + 视频 + 消融 + 效率 + 可视化 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 |
| [06 - Appendix](sections/06-appendix.md) | 13B 结果 + Qwen2-VL + OCR benchmarks + 超参数 + 更多可视化 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LLaVA-1.5 7B, 64 tokens (9× 压缩) | 保留 **96.0%** 性能 |
| LLaVA-1.5 7B, 192 tokens (3× 压缩) | 保留 **99.5%** 性能 |
| LLaVA-Next 7B, 160 tokens (18× 压缩) | 保留 **95.1%** 性能 |
| Video-LLaVA, 136 tokens (15× 压缩) | 保留 **100.5%** 性能 |
| 推理加速 (LLaVA-Next 7B) | **3.2×** |
| 最优超参数 α | 1.0 |

## 方法核心

```
SCOPE Score = Coverage_Gain(v) × Saliency(v)^α

Coverage_Gain(v) = Σ_u [max(sim(u,v), C(u,S)) - C(u,S)]
Saliency(v) = CLS attention weight (layer -2)

贪心迭代：每步选 SCOPE score 最大的 token → 更新覆盖 → 重复 K 次
```

## 与其他方法对比

| 方法 | Saliency | Coverage | Training-free | 64 tokens (LLaVA-1.5 7B) |
|------|----------|----------|--------------|--------------------------|
| FastV | ✅ (LLM attention) | ❌ | ✅ | 74.9% |
| SparseVLM | ✅ (text-guided) | ❌ | ✅ | 85.1% |
| VisionZip | ✅ (CLS attention) | ❌ | ✅ | 93.5% |
| DivPrune | ❌ | ✅ (diversity) | ✅ | - |
| **SCOPE** | ✅ (CLS attention) | ✅ (set-coverage) | ✅ | **96.0%** |
