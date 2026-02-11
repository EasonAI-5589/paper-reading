# Mirage: Machine Mental Imagery — Empower Multimodal Reasoning with Latent Visual Tokens

**作者**: Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, Chuang Gan  
**机构**: UMass Amherst, MIT  
**链接**: [arXiv 2506.17218](https://arxiv.org/abs/2506.17218) | [Code](https://github.com/UMass-Embodied-AGI/Mirage) | [Project Page](https://vlm-mirage.github.io)

## 一句话总结

VLM 推理时通过特殊 token 触发 **latent visual token** 生成——将当前 hidden state 重铸为视觉表示插入上下文，实现 interleaved 多模态推理，无需生成像素级图片。

## 核心贡献

1. **Mirage 框架**: 提出 latent visual token 机制，VLM 产生 `<latent>` token 时直接复用 hidden state 作为视觉 embedding，跳过 language head，实现文本-视觉交织推理
2. **两阶段训练**: Stage 1 用 ground-truth image embedding 做 cosine distillation 锚定视觉子空间；Stage 2 去掉视觉 loss，仅用 text loss，梯度回传让 latent token 自由适应任务
3. **RL 增强**: 在两阶段 SFT 后用 GRPO 进一步提升，latent token 的可微性使 RL 梯度能流经视觉 embedding
4. **一致性提升**: 在 VSP、Jigsaw、SAT、COMT 四个空间推理 benchmark 上超越 text-only baseline 和统一模型 (Anole, MVoT)

## 与 VisMem 的关键区别

| | **Mirage** | **VisMem** |
|---|---|---|
| 视觉记忆来源 | 重铸当前 hidden state | 独立 Memory Former 模块 |
| Token 生成 | 复用 LLM 最后一层 hidden state | 专用模块生成 latent memory token |
| 训练信号 | Stage 1 cosine distill + Stage 2 text-only | 两阶段 GRPO |
| 设计哲学 | Mental imagery (内部草图) | 显式短期+长期记忆系统 |
| 作为 baseline | VisMem 论文中唯一 latent space baseline | — |

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：框架概览 + 两阶段训练 |
| [01 - Introduction](sections/01-introduction.md) | 动机：mental imagery 启发 + Figure 1 示例 |
| [02 - Related Work](sections/02-related-work.md) | Multimodal CoT + Latent Reasoning in LLMs |
| [03 - Method](sections/03-method.md) | 核心方法：数据生成 + 两阶段训练 + RL |
| [04 - Experiments](sections/04-experiments.md) | 四个 benchmark 实验结果 + 消融 |
| [05 - Analysis](sections/05-analysis.md) | 小模型泛化 + 数据质量 + t-SNE 可视化 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与局限 |
| [07 - Appendix](sections/07-appendix.md) | 数据生成细节 + 实现细节 + 效率分析 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Base model | Qwen2.5-VL 7B |
| Latent token 数量 $k$ | 4 (默认) |
| 训练数据量 | 1k SFT + 2k RL per task |
| VSP Spatial Reasoning 提升 | +3% over Direct SFT, +4% over CoT SFT+GRPO |
| VSP Spatial Planning 提升 | +11% over Direct SFT |
| 最优 $k$ 范围 | 2-6 (k=8 下降 13%) |
| 训练耗时 | Stage 1: 3.5h + Stage 2: 7.2h (1×H100) |

## BibTeX

```bibtex
@article{yang2025mirage,
  author       = {Zeyuan Yang and
                  Xueyang Yu and
                  Delin Chen and
                  Maohao Shen and
                  Chuang Gan},
  title        = {Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual
                  Tokens},
  journal      = {CoRR},
  volume       = {abs/2506.17218},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2506.17218},
  doi          = {10.48550/ARXIV.2506.17218},
  eprinttype   = {arXiv},
  eprint       = {2506.17218}
}
```
