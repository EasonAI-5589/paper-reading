# Financial Models

金融领域预训练模型、指令模型与推理增强模型汇总。

| 模型 | 基础模型 / 规模 | 主要方向 | 与 FinChain 的关系 |
|------|-----------------|----------|--------------------|
| [FinBERT](./%5BIJCAI%202020%5D%20FinBERT/) | BERT | 金融文本预训练与分类 | 金融 NLP 早期基础模型 |
| [BloombergGPT](./%5BarXiv%202303.17564%5D%20BloombergGPT/) | 50B decoder LLM | 金融与通用混合预训练 | 金融大模型背景工作 |
| [FinGPT](./%5BarXiv%202307.10485%5D%20FinGPT/) | 多种开源基础模型 | 数据中心化、低成本金融适配 | 开源金融模型背景工作 |
| [FinMA](./%5BNeurIPS%202023%5D%20FinMA-PIXIU/) | LLaMA | 金融多任务指令微调 | PIXIU 中提出的金融模型 |
| [Fino1](./%5BarXiv%202502.08127%5D%20Fino1/) | Llama-3.1-8B | 金融 CoT 微调与强化学习 | FinChain 主实验金融模型 |
| [FinR1](./%5BarXiv%202503.16252%5D%20FinR1/) | 7B | 金融推理强化学习 | FinChain 主实验金融模型 |
| [DianJin-R1](./%5BarXiv%202504.15716%5D%20DianJin-R1/) | Qwen2.5 7B/32B | 结构化推理监督 + GRPO | FinChain 主实验金融模型 |
| [Qwen-Open-Finance-R-8B](./%5BarXiv%202511.08621%5D%20Qwen-Open-Finance-R-8B/) | Qwen3-8B | 多语言金融指令微调 | 候选继续训练基线 |

## 模型谱系

```text
通用基础模型
├── BERT → FinBERT
├── LLaMA → FinMA
├── Llama-3.1-8B → Fino1
├── Qwen2.5 → Fin-R1 / DianJin-R1
└── Qwen3-8B → Qwen-Open-Finance-R-8B

金融语料预训练
└── BloombergGPT

开源数据与适配框架
└── FinGPT
```

> 当前目录用于基础收录和模型谱系整理，尚未进行分章节批读。
