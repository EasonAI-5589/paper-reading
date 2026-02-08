# SODA: Story Oriented Dense Video Captioning Evaluation Framework

**作者**: Soichiro Fujita, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, Masaaki Nagata  
**机构**: Tokyo Institute of Technology, NTT Communication Science Laboratories  
**会议**: ECCV 2020  
**链接**: [GitHub](https://github.com/fujiso/SODA)

## 一句话总结

现有 Dense Video Captioning 评估框架（ActivityNet Challenge 官方 scorer）忽略字幕顺序且不惩罚冗余，SODA 通过**动态规划时序最优匹配 + F-measure** 解决这两个问题。

## 核心贡献

1. **揭示问题**: 现有框架的松散匹配 + 平均 METEOR 导致冗余字幕（几百条 vs 参考 3-4 条）得高分
2. **提出 SODA**: 用 DP 求时序最优一对一匹配（LCS 变体），用 F-measure 同时惩罚冗余和不足
3. **IoU 加权**: SODA(c) 让时间定位质量直接参与评分（cost = IoU × METEOR）
4. **实验验证**: SODA 对字幕数量和顺序都更敏感，与人类判断一致性更高（0.94 vs 0.72）

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 问题定义 + SODA 核心思想 |
| [01 - Introduction](sections/01-introduction.md) | 详细背景 + 三点贡献 |
| [02 - Related Work](sections/02-related-work.md) | DVC 方法/数据集 + 评估方法现状 |
| [03 - Current Framework](sections/03-current-framework.md) | 现有框架公式详解（Eq.1-3）+ IoU 匹配例子 |
| [04 - Problems](sections/04-problems.md) | 松散匹配 + 平均方式的两大问题 |
| [05 - SODA Method](sections/05-soda-method.md) | DP 最优匹配 + F-measure + IoU 加权（核心方法） |
| [06 - Experiments](sections/06-experiments.md) | 字幕数量/顺序实验 + 人工评估 |
| [07 - Conclusion](sections/07-conclusion.md) | 总结 + 对 Video Chaptering 的启示 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 数据集 | ActivityNet Captions, 4,915 验证视频 |
| 平均参考字幕数 | 3.52 |
| E2E Transformer 平均生成数 | 228.21 |
| LSTM 平均生成数 | 97.10 |
| Current 框架分数波动 (E2E) | 3.78 → 4.19 (m=0.1 → all) |
| SODA(c) F1 波动 (E2E) | 1.47 → 0.63 (m=0.1 → all), 峰值 4.02 (m=1.0) |
| 人工评估准确率 (顺序) | SODA 0.94 vs Current 0.72 |
| 人工评估准确率 (系统比较) | SODA 0.76 vs Current 0.66 |

## 核心方法图示

```
现有框架:  IoU > τ → 松散匹配（一对多）→ METEOR 平均（按配对数）
                     ↓ 问题：不考虑顺序，不惩罚冗余

SODA:      按时间排序 → DP 最优匹配（一对一，保序）→ F-measure（除以 |P| 和 |G|）
                     ↓ 优势：考虑顺序，惩罚冗余/不足
```
