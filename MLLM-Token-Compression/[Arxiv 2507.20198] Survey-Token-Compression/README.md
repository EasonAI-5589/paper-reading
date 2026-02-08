# A Survey of Token Compression for Efficient Multimodal Large Language Models

**作者**: Kele Shao*, Keda Tao*, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, Huan Wang†  
**机构**: Zhejiang University, Westlake University, Xiamen University, NUS, UW-Madison, UCF, Columbia, Salesforce, Rice  
**年份**: 2025 | **平台**: TechRxiv / OpenReview  
**链接**: [arXiv 2507.20198](https://arxiv.org/abs/2507.20198) | [OpenReview](https://openreview.net/forum?id=G2od9JVHkE)

## 一句话总结

首篇系统综述多模态大语言模型（MLLM）中的 token 压缩技术，按**模态**（图像/视频/音频）和**机制**（变换/相似度/注意力/查询）双维度建立分类体系。

## 核心贡献

1. **首篇 MLLM Token 压缩系统综述**：覆盖图像、视频、音频三大模态
2. **双维度 Taxonomy**：模态（3类）× 机制（4类）= 12 个分类格
3. **全面方法对比**：提供 training-free 方法在图像/视频 benchmark 上的定量对比
4. **深度讨论**：与权重压缩的关系、方法组合策略、当前挑战、评估问题、未来方向

## 📖 批读导航

| Section | 文件 | 内容 |
|---------|------|------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 摘要 — 问题定义、分类框架 |
| §1 Introduction | [01-introduction.md](sections/01-introduction.md) | 动机、token 爆炸问题、综述结构 |
| §2 Background | [02-preliminaries.md](sections/02-preliminaries.md) | MLLM 架构、LLM/ViT token 压缩、问题定义 |
| §3 Image-centric | [03-methods.md](sections/03-methods.md) | 🔑 **核心章节** — 图像 token 压缩四大类方法 |
| §4 Video-centric | [04-how-to-select.md](sections/04-how-to-select.md) | 视频 token 压缩 — 时空冗余处理 |
| §5 Audio-centric | [05-experiments.md](sections/05-experiments.md) | 音频 token 压缩 — 频谱/时间冗余 |
| §6 Discussions | [06-applications.md](sections/06-applications.md) | 与权重压缩关系、挑战、未来方向 |
| §7 Applications | [07-challenges.md](sections/07-challenges.md) | GUI Agent、医疗、机器人、高效推理 |
| §8 Conclusion | [08-conclusion.md](sections/08-conclusion.md) | 总结 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 90 分钟视频 token 数 | **54M** (5400 万) |
| 典型 MLLM 中多模态 token 占比 | **>80%** |
| 低注意力 visual token 比例 | **>50%** |
| FastV: LLaVA-1.5 视觉 token attention 比例 | **0.21%** (第2层后) |
| Training-free 图像压缩甜蜜点 | **~128 token** (从 576, 保留 ~22%) |
| 视频 25% 保留率性能 | VideoMME **58.9** (vs 基线 58.6) |
| LLaVA-Mini: 每张图 token 数 | **1** |
| Transformation 方法典型压缩率 | **25%** (4倍) |

## 四大机制速查

| 机制 | 核心思路 | 优势 | 劣势 | 代表方法 |
|------|---------|------|------|---------|
| **Transformation** | 池化/卷积/像素重排 | 保结构、无参数 | 压缩率固定 | InternVL, Qwen2, LLaVA-OV |
| **Similarity** | 合并相似 token | 灵活 | 丢空间信息 | ToMe, DivPrune, HoliTom |
| **Attention** | 注意力稀疏性剪枝 | 动态、可解释 | FlashAttention 不兼容 | FastV, PyramidDrop, VisionZip |
| **Query** | 查询引导压缩 | 精准 task-aware | 不适合多轮对话 | Q-Former, LLaMA-VID, LLaVA-Mini |

---

## BibTeX

```bibtex
@article{shao2025survey,
  title={A Survey of Token Compression for Efficient Multimodal Large Language Models},
  author={Kele Shao and Keda Tao and Kejia Zhang and Sicheng Feng and Mu Cai and Yuzhang Shang and Haoxuan You and Can Qin and Yang Sui and Huan Wang},
  journal={arXiv preprint arXiv:2507.20198},
  year={2025}
}
```
