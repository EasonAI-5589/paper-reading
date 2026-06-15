[← 返回 ODA-Fin-RL-8B](./README.md)

# ODA-Fin-RL Benchmark Coverage

论文使用 9 个 benchmark，分为一般金融理解、情感分析和数值推理三组。仓库原先只独立收录了 FinQA、TAT-QA 和 ConvFinQA，本轮补齐其余 6 项。

| 类别 | Benchmark | 指标 | Qwen3-8B | ODA-Fin-SFT-8B | ODA-Fin-RL-8B | 收录状态 |
|------|-----------|------|----------|----------------|---------------|----------|
| 一般金融理解 | [FinEval](../../Financial-Benchmarks/%5BarXiv%202308.09975%5D%20FinEval/) | Acc / zh | 77.8 | 76.0 | 77.0 | 本轮补充 |
| 一般金融理解 | [Finova](../../Financial-Benchmarks/%5BarXiv%202507.16802%5D%20Finova/) | Acc / zh | 44.9 | 47.8 | 54.6 | 本轮补充 |
| 一般金融理解 | [FinanceIQ](../../Financial-Benchmarks/%5BDataset%202023%5D%20FinanceIQ/) | Acc / zh | 72.5 | 72.1 | 74.2 | 本轮补充 |
| 情感分析 | [FOMC](../../Financial-Benchmarks/%5BACL%202023%5D%20FOMC-Hawkish-Dovish/) | Weighted F1 / en | 57.5 | 63.9 | 61.0 | 本轮补充 |
| 情感分析 | [Financial PhraseBank](../../Financial-Benchmarks/%5BJASIST%202014%5D%20Financial-PhraseBank/) | Weighted F1 / en | 76.8 | 75.6 | 83.4 | 本轮补充 |
| 情感分析 | [Commodity News Headlines](../../Financial-Benchmarks/%5BarXiv%202009.04202%5D%20Commodity-News-Headlines/) | Weighted F1 / en | 76.0 | 78.2 | 78.5 | 本轮补充 |
| 数值推理 | [FinQA](../../Financial-Benchmarks/%5BEMNLP%202021%5D%20FinQA/) | Acc / en | 72.2 | 69.8 | 73.3 | 已收录 |
| 数值推理 | [TAT-QA](../../Financial-Benchmarks/%5BACL%202021%5D%20TAT-QA/) | Acc / en | 87.1 | 87.0 | 89.3 | 已收录 |
| 数值推理 | [ConvFinQA](../../Financial-Benchmarks/%5BEMNLP%202022%5D%20ConvFinQA/) | Acc / en | 78.8 | 78.3 | 80.4 | 已收录 |

## 结果解读

- RL 的最大提升来自 Finova：相对 SFT 增加 6.8 分，说明困难且可验证的 RL 数据对复杂金融任务有效。
- 数值推理三项均提升，其中 TAT-QA 达到 89.3，符合论文强调的 hard-but-verifiable 定位。
- FOMC 从 SFT 的 63.9 降到 RL 的 61.0，说明 outcome RL 可能损伤部分细粒度语义分类能力。
- ODA-Fin-RL 的九项平均分为 74.6，高于 Qwen3-8B 的 71.5，并接近 Qwen3-32B 的 74.7。

数据来自论文 Table 1；表中的 `FinIQ`、`HL` 和 `CFQA` 分别对应 FinanceIQ、Headlines 和 ConvFinQA。
