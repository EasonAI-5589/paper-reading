# VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models

**作者**: Ce Zhang, Kaixin Ma, Tianqing Fang, Wenhao Yu, Hongming Zhang, Zhisong Zhang, Haitao Mi, Dong Yu  
**机构**: Carnegie Mellon University, Tencent AI Lab  
**期刊**: TMLR 2026 | **arXiv**: 2505.22654  
**链接**: [arXiv](https://arxiv.org/abs/2505.22654) | [OpenReview](https://openreview.net/forum?id=KZYhyilFnt) | [Code](https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan)

## 一句话总结

Training-free 两阶段视觉 token 压缩：Stage 1 在 visual encoder 用 global+local scan 互补选 token 并 merge，Stage 2 在 LLM 中间层做 text-aware pruning，在 LLaVA-NeXT-7B 上用 11.1% token 保留 95.4% 性能，2.91× prefill 加速。

## 核心贡献

1. **系统性实证分析**：揭示 visual encoder 从浅层局部到深层全局的注意力演变，以及 LLM 早期层存在位置偏差、中间层才开始有效整合视觉信息的规律
2. **两阶段 training-free 压缩框架 VScan**：
   - Stage 1 (Visual Encoding): Global scan (深层 [CLS] attention) + Local scan (浅层窗口内 [CLS] attention) 互补选 token，再 similarity-based merging
   - Stage 2 (LLM Decoding): 在中间层（而非早期层）按 text attention 剪枝，避免位置偏差
3. **广泛验证**：4 个 LVLM（LLaVA-1.5/NeXT, Qwen-2.5-VL, Video-LLaVA）× 16 个 benchmark，全面超越 SOTA

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 + Figure 1 |
| [02 - Related Work](sections/02-related-work.md) | 高效 LVLM + Token Reduction 两类方法 |
| [03 - Empirical Analysis](sections/03-empirical-analysis.md) | 三个关键实验发现（核心洞察） |
| [04 - Method](sections/04-method.md) | Global/Local Scan + Merging + Middle Layer Pruning |
| [05 - Experiments](sections/05-experiments.md) | 4 模型 × 16 benchmark 结果 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与局限 |
| [07 - Appendix](sections/07-appendix.md) | 额外实验 + 实现细节 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LLaVA-1.5-7B 保留 11.1% token | 96.7% 性能，1.77× prefill 加速 |
| LLaVA-NeXT-7B 保留 11.1% token | 95.4% 性能，2.91× prefill 加速，10× FLOPs 降低 |
| Qwen-2.5-VL RefCOCO 50% token | 96.1% 性能（grounding 任务） |
| Video-LLaVA 25% token | 近乎无损 |
| 默认 R₁/R₂ | 16.7% / 33.3%（平均 11.1%） |
| Local scan 层 | l=6 (LLaVA), l=8 (Qwen) |
| LLM pruning 层 | k=16 (LLaVA), k=14 (Qwen-7B) |

## 与同类方法对比

| 方法 | 压缩位置 | Text-aware | Training-free | 核心机制 |
|------|----------|------------|---------------|----------|
| FastV | LLM 早期层 | ✅ | ✅ | 第 2 层 attention 剪枝 |
| VisionZip | Visual Encoder 输出层 | ❌ | ✅ | [CLS] attention + merging |
| PyramidDrop | LLM 多层 | ✅ | ✅ | 逐层递减 |
| SparseVLM | LLM | ✅ | ✅ | 文本 token 引导评分 |
| **VScan** | **Encoder + LLM 中间层** | **✅** | **✅** | **Global+Local scan + middle pruning** |

## BibTeX

```bibtex
@article{DBLP:journals/tmlr/ZhangMFYZZMY26,
  author       = {Ce Zhang and Kaixin Ma and Tianqing Fang and Wenhao Yu and Hongming Zhang and Zhisong Zhang and Haitao Mi and Dong Yu},
  title        = {VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models},
  journal      = {Trans. Mach. Learn. Res.},
  volume       = {2026},
  year         = {2026},
  url          = {https://openreview.net/forum?id=KZYhyilFnt},
}
```
