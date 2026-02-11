# MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning

**作者**: Anonymous (double-blind review)
**会议**: ICLR 2026 (Rejected, scores: 8-6-2)
**链接**: OpenReview (under review)

## 一句话总结

从医学教育 YouTube 视频中提取时序连贯帧序列，构建 2,851 个多图 VQA，测试 MLLM 跨图临床推理能力——所有模型准确率低于 55%，暴露出 MLLM 在多图比较推理上的关键短板。

## 核心贡献

1. **首个多图医学 VQA benchmark**：每个问题绑定 2-5 张时序连贯帧，要求跨图综合推理，而非孤立单图分析
2. **自动化数据构建 pipeline**：从 YouTube 医学视频 → 关键帧提取 → 帧-字幕配对 → 多帧合并 → VQA 生成，全流程 GPT-4o 驱动
3. **全面 benchmark 11 个 MLLM**：揭示所有模型准确率 < 55%，推理模型优于非推理模型 ~9%，但仍远不够
4. **错误分析**：模型忽略关键帧信息、跨图证据聚合失败、早期错误在推理链中传播

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：benchmark 动机、构建方法、核心发现 |
| [01 - Introduction](sections/01-introduction.md) | 动机：单图→多图的 gap + Figure 1 对比 + Table 1 |
| [02 - Related Work](sections/02-related-work.md) | 推理 MLLM、医学 benchmark、视频数据 |
| [03 - Benchmark](sections/03-benchmark.md) | 核心：数据构建 pipeline 四阶段 + 过滤策略 |
| [04 - Experiments](sections/04-experiments.md) | 11 模型评测 + 错误案例分析 + Table 2/3 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结与局限 |
| [06 - Appendix](sections/06-appendix.md) | 数据分布、Prompt 模板、VQA 示例 |

## 关键数字

| 指标 | 数值 |
|------|------|
| VQA 对数 | 2,851 |
| 高质量帧 | 9,237 |
| 源视频数 | 3,420 |
| 人体系统 | 9 |
| 器官数 | 43 |
| 每题帧数 | 2-5（均值 3.24） |
| 最佳模型准确率 | Gemini-2.5-Flash 54.75% |
| 最差模型准确率 | GPT-4o-mini 34.55% |
| 推理 vs 非推理提升 | +9.08%（Gemini-2.5-Flash vs GPT-4o） |

## 🔗 Agent-Memory 课题关联

作为 Agent-Memory 课题下的 benchmark 论文，关注点：
- **数据构建 pipeline**：视频→帧→字幕配对→多帧合并→VQA，可借鉴其从视频构建多步推理数据的思路
- **跨帧推理定义**：时序连贯帧序列中的信息综合，非简单多图拼接
- **模型失败模式**：忽略中间帧、错误传播——这些也是 Agent 长程记忆面临的挑战

## BibTeX

```bibtex
@article{yu2025medframeqa,
  title={MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning},
  author={Yu, Suhao and Wang, Haojin and Wu, Juncheng and Luo, Luyang and Xie, Cihang and Rajpurkar, Pranav and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2505.16964},
  year={2025}
}
```
