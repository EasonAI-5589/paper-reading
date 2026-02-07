# Abstract

Dense Video Captioning (DVC) 是一个具有挑战性的任务：定位视频中的所有事件并用自然语言描述它们。DVC 的主要目标是**视频故事描述**——生成简洁的视频故事，帮助人类理解视频内容而无需观看。

## 现有评测问题

ActivityNet Challenge 的官方评测框架存在问题：
- 不考虑视频的**故事性**
- 不考虑 caption 的**顺序**
- 对生成几百个冗余 caption 的系统给出高分

## SODA 解决方案

1. **时序最优匹配**: 用动态规划找到生成 caption 和参考 caption 之间的最优匹配，考虑时序顺序
2. **F-measure**: 用 F 值惩罚过多或过少的 caption

## 实验结论

- SODA 对不合适的 caption（太多/太少）给出低分
- SODA 对顺序错误的 caption 给出更低分
- 比现有评测框架更符合人工评估

---

**关键词**: Automatic Evaluation, Dense Video Captioning, Video Story Description
