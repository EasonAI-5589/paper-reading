# ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries

**作者**: Junfu Pu*, Teng Wang*, Yixiao Ge†, Yuying Ge, Chen Li, Ying Shan (ARC Lab, Tencent PCG)  
**来源**: arXiv 2025 (Technical Report) | **日期**: 2025-11-18  
**链接**: [arXiv](https://arxiv.org/abs/2025.xxxxx) | [GitHub](https://github.com/TencentARC/ARC-Chapter)

## 一句话总结

首个百万级长视频 chaptering 模型，配套双语层级标注数据集 VidAtlas 和 many-to-one 评估指标 GRACE，在 VidChapters-7M 上 F1 +14.0%、SODA +11.3%。

## 核心贡献

1. **VidAtlas 数据集**：41 万+视频（11.5 万小时），双语（EN/ZH），层级标注（Short Title → Structural Chapter → Timestamp-Aligned Description），比之前大 50 倍
2. **半自动标注管线**：利用用户 chapter markers + Whisper ASR + Qwen2.5-VL caption → LLM 推理生成层级标注
3. **GRACE 指标**：many-to-one 匹配 + DTW 寻优 + BERTscore，解决 chaptering 的粒度模糊问题
4. **GRPO 强化学习**：用 temporal reward 直接优化边界预测精度，且跨模态迁移
5. **Scaling Law**：首次证明 video chaptering 性能随数据量持续提升，不饱和

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 关键数字 |
| [01 - Introduction](sections/01-introduction.md) | 三大挑战 + 三大贡献 + Figure 1（模型能力展示）|
| [02 - Related Works](sections/02-related-works.md) | Global → Temporal → Long-Form 三条线索 + Figure 2（标注管线）|
| [03 - Data Collection](sections/03-data-collection.md) | VidAtlas 数据集：来源、筛选、标注管线、统计 + Figure 3 |
| [04 - Method](sections/04-method.md) | 模型架构 + 训练策略 + GRACE 指标 + GRPO RL + Figure 4/5 + 5 个公式 |
| [05 - Experiments](sections/05-experiments.md) | SOTA 对比 + 迁移性 + Scaling/层级标注/GRPO 消融 + Table 1-6 + Figure 6-8 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + 局限性分析 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 训练数据 | 41 万+ 视频, 11.5 万小时 |
| 基座模型 | Qwen2.5-VL-7B |
| VidChapters7M F1 | 45.3 → **59.3** (+14.0) |
| VidChapters7M SODA | 19.3 → **30.6** (+11.3) |
| VidChapters7M CIDEr | 100.9 → **186.6** (+85.7) |
| VidAtlas F1 | 48.7 → **66.2** (+17.5) |
| VidAtlas GRACE | 19.8 → **34.1** (+14.3) |
| YouCook2 F1 | 33.5 → **37.9** (+4.4) |

## 方法概览

```
输入: 视频帧(≤768帧) + ASR转录(Whisper-v3) + Task Prompt(18种模板)
      ↓
模型: Qwen2.5-VL-7B (frozen vision encoder + trainable LLM)
      ↓
训练: SFT (VidAtlas + VidChapters-7M) → GRPO RL (temporal reward)
      ↓
输出: Short Title / Structural Chapter / Video Description
```

## 我的评价

**优点**:
- 数据驱动的范式非常实用：利用平台已有的 chapter markers 作为种子，大幅降低标注成本
- GRACE 指标设计合理，many-to-one 匹配确实更适合 chaptering 场景
- Scaling law 的发现有重要启示：之前认为 ~20k 就饱和是因为数据不够好
- GRPO 的跨模态迁移是个有趣的发现

**局限**:
- 数据源有 selection bias（只选有 chapter markers 的视频）
- 只用了 7B 模型，scaling model size 的实验缺失
- 标注管线的 LLM hallucination 问题没有讨论
- GRACE 指标需要更多验证（与人类判断的相关性分析较少）

---

## BibTeX

```bibtex
@misc{pu2025arcchapter,
  title={ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries},
  author={Junfu Pu and Teng Wang and Yixiao Ge and Yuying Ge and Chen Li and Ying Shan},
  year={2025},
  eprint={2511.14349},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
