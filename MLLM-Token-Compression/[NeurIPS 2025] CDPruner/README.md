# Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs

**作者**: Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu†, Yuan Zhang, Junwen Pan, Qi She†, Shanghang Zhang‡  
**单位**: Peking University, ByteDance  
**会议**: NeurIPS 2025  
**链接**: [GitHub](https://github.com/Theia-4869/CDPruner)

## 一句话总结
CDPruner 通过**行列式点过程（DPP）** 在指令条件下最大化视觉 token 的条件多样性，实现 training-free、model-agnostic 的视觉 token 剪枝，在多种 MLLM 上达到 SOTA。

## 核心贡献
1. 提出 CDPruner：用条件多样性统一 diversity + instruction relevance
2. 用 DPP 重新建模 token pruning 问题，核心公式：$\tilde{L} = \text{diag}(\tilde{r}) \cdot L \cdot \text{diag}(\tilde{r})$
3. Training-free + model-agnostic，兼容 FlashAttention
4. 在 LLaVA / LLaVA-NeXT / LLaVA-Video / Qwen2.5-VL 上全面 SOTA

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、方法、关键数字 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + Figure 1 对比 + 三点贡献 |
| [02 - Related Work](sections/02-related-work.md) | MLLM / Token reduction 三类方法 / DPP 背景 |
| [03 - Method](sections/03-method.md) | 核心方法：DPP + 指令相关性 → 条件 kernel |
| [04 - Experiments](sections/04-experiments.md) | 4 种模型 × 18 个 benchmark + 效率分析 + 消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性 + 未来方向 |

## 关键数字

| 指标 | 数值 |
|------|------|
| FLOPs 减少 | 95% (LLaVA-NeXT) |
| CUDA 延迟减少 | 78% |
| GPU 显存减少 | 17% |
| 性能保留 (LLaVA-1.5, 32 tokens) | 94.3% |
| 性能保留 (LLaVA-NeXT, 640 tokens) | 100.1% |
| 额外计算开销 | < 10ms/样本 |

## 方法对比

| 方法类型 | 代表 | 考虑多样性 | 考虑指令 | 兼容 FlashAttention |
|----------|------|:---------:|:-------:|:------------------:|
| Attention-based | FastV, PDrop | ❌ | ✅ | ❌ |
| Similarity-based | DART, DivPrune | ✅ | ❌ | ✅ |
| **CDPruner** | **本文** | **✅** | **✅** | **✅** |

---

## BibTeX

```bibtex
@inproceedings{DBLP:journals/corr/abs-2506-10967,
  author       = {Qizhe Zhang and
                  Mengzhen Liu and
                  Lichen Li and
                  Ming Lu and
                  Yuan Zhang and
                  Junwen Pan and
                  Qi She and
                  Shanghang Zhang},
  title        = {Beyond Attention or Similarity: Maximizing Conditional Diversity for
                  Token Pruning in MLLMs},
  booktitle    = {Advances in Neural Information Processing Systems ({NeurIPS})},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2506.10967},
  doi          = {10.48550/ARXIV.2506.10967},
  eprinttype    = {arXiv},
  eprint       = {2506.10967}
}
```
