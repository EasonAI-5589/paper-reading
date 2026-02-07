# 1. Introduction

## Dense Video Captioning 任务

DVC 主要包含两个子任务：
1. **Event Detection**: 识别视频中的所有事件
2. **Caption Generation**: 用自然语言描述这些事件

DVC 是 ActivityNet Challenge 的重要任务之一（自 2017 年起）。

## 主要目标

生成**简洁的 caption** 来描述视频的故事，帮助人类理解视频。
- 人类平均用 **3-4 个 caption** 描述一个视频
- 生成的 caption 用于快速了解视频内容，无需完整观看

## 现有评测框架的问题

### 问题 1: 忽略故事性和顺序

框架首先匹配生成和参考 caption（IoU > 阈值），然后计算 METEOR 分数并平均。
- 不考虑 caption 的顺序
- 不考虑视频的故事性

### 问题 2: 冗余 caption 得高分

- 分数只依赖匹配对的数量
- 生成几百个 caption 反而能获得高分
- 现有系统平均生成 **几百个 caption**，而参考只有 **3-4 个**

## SODA 的解决方案

1. **动态规划最优匹配**: 找到最大化 IoU 之和的一对一匹配，同时考虑时序顺序
2. **F-measure**: 基于 METEOR 分数计算 Precision 和 Recall，最终用 F 值评估

## 贡献

1. 证明 ActivityNet Challenge 的评测框架不适合评估视频故事描述
2. 提出 SODA 评测框架，考虑 caption 顺序
3. 引入 F-measure 防止冗余 caption 获得高分
4. 开源代码: https://github.com/fujiso/SODA
