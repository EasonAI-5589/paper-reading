# VidChapters-7M: Video Chapters at Scale

**作者**: Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid  
**会议**: NeurIPS 2023 (Datasets and Benchmarks Track)  
**机构**: Inria Paris, University of Oxford, Czech Technical University  
**链接**: [项目主页](https://antoyang.github.io/vidchapters.html) | [arXiv](https://arxiv.org/abs/2309.13952)

## 一句话总结

提出 VidChapters-7M——一个包含 817K 视频和 7M 用户标注章节的大规模数据集，定义了三个视频章节化任务，并证明了其作为视频-语言预训练资源的巨大价值（在 YouCook2 上 CIDEr 提升 +18.9）。

## 核心贡献

1. **大规模数据集**: 从 YouTube 自动爬取 817K 视频的 7M 用户章节标注，零额外标注成本
2. **三个任务定义**: Chapter Generation、Chapter Generation (GT Boundaries)、Chapter Grounding
3. **全面 Benchmark**: 评测了 zero-shot baseline 和 SOTA 模型（PDVC, Vid2Seq）
4. **预训练价值**: 在 VidChapters-7M 上预训练后，大幅提升 dense video captioning SOTA
5. **Scaling 特性**: 下游性能随预训练数据量增长持续提升

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1（数据集示例） |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 + Figure 2（任务定义）+ Table 1（数据集对比） |
| [02 - Related Work](sections/02-related-work.md) | 大规模视觉-语言数据集 + 相关视频任务 |
| [03 - Dataset](sections/03-dataset.md) | 数据收集、处理、分析 + Figure 3（统计）+ Table 2（质量评估） |
| [04 - Experiments](sections/04-experiments.md) | 三个任务实验 + 迁移学习 + Tables 3-8 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性 + 社会影响 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 总视频数 | 817K |
| 总章节数 | 7M |
| 平均视频时长 | 23 min |
| 平均章节数/视频 | 8.3 |
| 含 ASR 比例 | 97.3% |
| YouCook2 CIDEr SOTA | 67.2 (+18.9) |
| ViTT CIDEr SOTA | 50.0 (+6.5) |
| 最佳 Chapter Gen SODA_c | 11.4 |

---

## BibTeX

```bibtex
@inproceedings{DBLP:conf/nips/YangNLSS23,
  author       = {Antoine Yang and
                  Arsha Nagrani and
                  Ivan Laptev and
                  Josef Sivic and
                  Cordelia Schmid},
  editor       = {Alice Oh and
                  Tristan Naumann and
                  Amir Globerson and
                  Kate Saenko and
                  Moritz Hardt and
                  Sergey Levine},
  title        = {VidChapters-7M: Video Chapters at Scale},
  booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                  LA, USA, December 10 - 16, 2023},
  year         = {2023},
  url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/9b5c3e00d6ed30aad7adac9e7a664de1-Abstract-Datasets\_and\_Benchmarks.html}
}
```
