# Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs

**作者**: Lucas Ventura, Antoine Yang, Cordelia Schmid, Gül Varol  
**会议**: CVPR 2025  
**机构**: LIGM (Ecole des Ponts), Inria, Google DeepMind  
**链接**: [Project Page](https://imagine.enpc.fr/~lucas.ventura/chapter-llama/)

## 一句话总结

用 LLM 处理纯文本输入（ASR + speech-guided 关键帧 caption）实现小时级视频的自动章节划分，以极少训练数据（2.5%）大幅超越 SOTA。

## 核心贡献

1. **Chapter-Llama 框架**: 将视频转为纯文本（ASR + Caption），用 finetuned Llama-3.1-8B 预测章节边界和标题
2. **Speech-based frame selection**: 先用 speech-only LLM 预测边界，只在预测位置 caption（~10 帧 vs 100 帧），效率极高且效果更好
3. **大幅超越 SOTA**: F1 45.3 vs 26.7（Vid2Seq），只用 20k 训练视频（2.5% 数据）

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：任务定义、方法概览、核心数字 |
| [01 - Introduction](sections/01-introduction.md) | 背景动机、Vid2Seq 局限、三大贡献 |
| [02 - Related Work](sections/02-related-work.md) | 时序分割、视频描述、长视频理解、LLM 应用 |
| [03 - Method](sections/03-method.md) | Speech-based frame selection、文本映射、LLM 训练、迭代预测 |
| [04 - Experiments](sections/04-experiments.md) | SOTA 对比、模态消融、帧采样策略、数据量、迭代预测 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结、局限性、未来方向 |
| [06 - Appendix](sections/06-appendix.md) | 实现细节、数据分析、补充实验精选 |

## 关键数字

| 指标 | Chapter-Llama | Vid2Seq (SOTA) | 提升 |
|------|:---:|:---:|:---:|
| F1 | 45.3 | 26.7 | +70% |
| tIoU | 71.8 | 58.6 | +23% |
| SODA | 19.3 | 11.6 | +66% |
| CIDEr | 100.9 | 55.8 | +81% |

| 效率指标 | 数值 |
|---------|------|
| 训练数据 | 20k 视频 (2.5%) |
| 平均采样帧数 | ~10.3 帧/视频 |
| LoRA 参数量 | 13MB/模型 |
| 训练耗时 | 40min on 4×H100 |

## 方法示意

```
视频 → ASR (Whisper) → Speech-only LLM → 预测边界位置
                                              ↓
                                     在边界位置采样帧
                                              ↓
                                     MiniCPM-V Caption
                                              ↓
                              ASR + Caption 交错排列 (带时间戳)
                                              ↓
                                   Llama-3.1-8B (LoRA finetuned)
                                              ↓
                                  章节边界 + 标题 (文本输出)
```
