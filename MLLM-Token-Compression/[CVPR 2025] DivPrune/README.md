# DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models

**作者**: Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, Yong Zhang (Huawei Technologies Canada)
**会议**: CVPR 2025
**链接**: [arXiv](https://arxiv.org/abs/2412.11408)

## 一句话总结

将视觉 token 剪枝建模为 **Max-Min Diversity Problem (MMDP)**，通过最大化所选 token 的多样性来减少冗余，在不需要微调/校准的前提下实现 90% 剪枝率下的 SOTA 性能。

## 核心贡献

1. **新颖的问题建模**: 首次将 token pruning 建模为 MMDP，从「选重要的」转向「选多样的」
2. **Plug-and-play**: 无需训练、无需校准数据，兼容任意 LMM 架构和视觉编码器
3. **极端压缩下的鲁棒性**: 在 ~90% pruning ratio 下仍保持接近原模型性能，远超同类方法
4. **全面评测**: 16 个数据集（11 image + 5 video），4 个 LMM，5 个 baseline

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、方法、结果概览 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 现有方法缺陷 + 贡献（含 Figure 1 性能对比图） |
| [02 - Related Work](sections/02-related-work.md) | LMM 概述 + 高效 LMM + Token Pruning 方法分类 |
| [03 - Method](sections/03-method.md) | LMM 背景 + MMDP 建模 + 贪心算法（含 Figure 2 架构图 + Algorithm 1） |
| [04 - Experiments](sections/04-experiments.md) | 设置 + t-SNE 可视化 + 图像/视频对比 + 效率分析 + 消融实验 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性分析 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 默认 pruning ratio | 90.2% |
| TFLOP ratio (image) | ~15% |
| TFLOP ratio (video) | ~14% |
| 显存节省 | ~400MB |
| E2E 延迟减少 | ~22% |
| 测试数据集数 | 16 |
| 测试 LMM 数 | 4 |

## 方法核心

```
Token Pruning as MMDP:
  输入: M 个视觉 token
  目标: 选 M̃ 个 token，使 min pairwise distance 最大化
  算法: 贪心 —— 每步选离已选集合最远的 token
  距离: Cosine distance
  开销: 一次矩阵乘法（距离矩阵），可忽略
```

---

## BibTeX

```bibtex
@inproceedings{alvar2025divprune,
  title={DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models},
  author={Saeed Ranjbar Alvar and Gursimran Singh and Mohammad Akbari and Yong Zhang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9392--9401},
  year={2025}
}
```
