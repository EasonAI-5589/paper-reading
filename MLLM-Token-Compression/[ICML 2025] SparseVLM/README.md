# SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

**作者**: Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, Shanghang Zhang  
**会议**: ICML 2025  
**链接**: [arXiv](https://arxiv.org/abs/2410.04417) | [GitHub](https://github.com/Gumpest/SparseVLMs)

## 一句话总结

SparseVLM 提出了首个 **text-guided + training-free** 的视觉 token 稀疏化框架，通过复用自注意力矩阵评估视觉 token 重要性，结合 rank-based 自适应裁剪和 token recycling，在 LLaVA 上实现 4.5× 压缩率、仅降 0.9% 精度。

## 核心贡献

1. **Training-free + Text-aware**: 首个不需要额外训练、利用文本引导的 VLM 推理加速方法
2. **Text Rater 选择**: 筛选与视觉相关的文本 token 作为评判者，避免无关词干扰
3. **Rank-based 自适应裁剪**: 用注意力矩阵的 rank 自动决定每层裁剪比例
4. **Token Recycling**: 被裁剪的 token 通过密度峰值聚类压缩重构，减少信息损失
5. **广泛验证**: 在 LLaVA、MGM、Qwen2-VL、VideoLLaVA 上一致超越 FastV、ToMe、PDrop

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、方法、关键词 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + Figure 1 对比 + 贡献列表 |
| [02 - Related Work](sections/02-related-work.md) | VLM 视觉 token 增长 + 两大压缩方向 |
| [03 - Method](sections/03-method.md) | 核心方法：重要性评估 + Rater 选择 + 自适应裁剪 + Token Recycling |
| [04 - Experiments](sections/04-experiments.md) | 图像理解 (8 benchmarks) + 视频理解 (4 benchmarks) |
| [05 - Analysis](sections/05-analysis.md) | 消融实验 + 效率分析 + 可视化 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 |
| [07 - Appendix](sections/07-appendix.md) | FlashAttention 兼容 + 计算分析 + 更多可视化 |

## 关键数字

| 指标 | 数值 |
|------|------|
| LLaVA 压缩率 | 4.5× (576→128 tokens) |
| 精度保持 (192 tokens) | 99.1% |
| CUDA Latency 减少 | 37-43% |
| FLOPs 减少 | 62.8% |
| KV Cache 减少 | 67% |
| vs FastV (LLaVA 64 tokens) | +17.3% |
| vs FastV (VideoLLaVA) | +14.7% |

## 方法概览

```
输入图像 + 问题
    ↓
[Stage a] Text Rater 选择
    H_v × H_q^T → 筛选视觉相关文本 token
    ↓
[Stage b] 逐层自适应稀疏化
    ├─ 提取文本→视觉注意力子矩阵 P
    ├─ 用 raters 计算视觉 token 重要性
    ├─ rank(P) → 决定裁剪数量 N
    ├─ 裁剪最不重要的 N 个视觉 token
    └─ Token Recycling: 聚类 → 求和重构 → 重新加入
    ↓
输出（更少 token，更快推理）
```

---

## BibTeX

```bibtex
@inproceedings{DBLP:conf/icml/0020FMZ0CGONKZ25,
  author       = {Yuan Zhang and
                  Chun{-}Kai Fan and
                  Junpeng Ma and
                  Wenzhao Zheng and
                  Tao Huang and
                  Kuan Cheng and
                  Denis A. Gudovskiy and
                  Tomoyuki Okuno and
                  Yohei Nakata and
                  Kurt Keutzer and
                  Shanghang Zhang},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {SparseVLM: Visual Token Sparsification for Efficient Vision-Language
                  Model Inference},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  volume       = {267},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/zhang25s.html}
}
```
