# Financial Models

金融领域模型与 FinChain 主实验对比模型汇总。目前包含 17 个独立模型条目；FinChain 评测的全部 26 个模型及成绩见 [FinChain Model Comparison](../Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/MODEL-COMPARISON.md)。

## 金融专项模型

| 模型 | 基础模型 / 规模 | FinChain ChainEval | 定位 |
|------|-----------------|--------------------|------|
| [Fin-o1](./%5BarXiv%202502.08127%5D%20Fino1/) | Llama-3.1-8B | 41.50 | 金融 CoT 微调与强化学习 |
| [Fin-R1](./%5BarXiv%202503.16252%5D%20FinR1/) | Qwen2.5-7B-Instruct | **58.14** | 金融推理 SFT + GRPO |
| [DianJin-R1](./%5BarXiv%202504.15716%5D%20DianJin-R1/) | Qwen2.5-7B-Instruct | 51.95 | 结构化监督 + GRPO |
| [Finance-LLaMA](./%5BModel%202025%5D%20Finance-LLaMA/) | DeepSeek-R1-Distill-Llama-8B | 41.35 | 金融指令 LoRA SFT |
| [Finance-Qwen](./%5BModel%202025%5D%20Finance-Qwen/) | Qwen2.5-7B | 34.57 | 金融指令 LoRA SFT |
| [Qwen-Open-Finance-R-8B](./%5BarXiv%202511.08621%5D%20Qwen-Open-Finance-R-8B/) | Qwen3-8B | 未评测 | 候选继续训练基线 |

## 数学增强模型

| 模型 | 基础模型 / 规模 | FinChain ChainEval |
|------|-----------------|--------------------|
| [Mathstral](./%5BModel%202024%5D%20Mathstral/) | Mistral-7B / 7B | **59.87** |
| [Qwen2.5-Math](./%5BModel%202024%5D%20Qwen2.5-Math/) | Qwen2.5 / 7B | 55.35 |
| [WizardMath](./%5BModel%202023%5D%20WizardMath/) | Mistral-7B / 7B | 24.33 |
| [MetaMath](./%5BModel%202023%5D%20MetaMath/) | Llemma / 7B | 7.93 |

## 通用开源基线

| 模型 | 规模 | FinChain ChainEval | 关系 |
|------|------|--------------------|------|
| [Qwen2.5-7B-Instruct](./%5BModel%202024%5D%20Qwen2.5-7B-Instruct/) | 7B | **60.35** | Fin-R1、DianJin-R1 基座 |
| [Llama-3.1-8B-Instruct](./%5BModel%202024%5D%20Llama-3.1-8B-Instruct/) | 8B | 53.99 | Fin-o1 基座 |
| [Qwen3-8B](./%5BModel%202025%5D%20Qwen3-8B/) | 8B | 43.32 | Qwen-Open-Finance-R-8B 基座 |

## 金融模型背景工作

| 模型 | 规模 | 主要方向 |
|------|------|----------|
| [FinBERT](./%5BIJCAI%202020%5D%20FinBERT/) | BERT | 金融文本预训练与分类 |
| [BloombergGPT](./%5BarXiv%202303.17564%5D%20BloombergGPT/) | 50B | 金融与通用混合预训练 |
| [FinGPT](./%5BarXiv%202307.10485%5D%20FinGPT/) | 多种基座 | 数据中心化、低成本金融适配 |
| [FinMA](./%5BNeurIPS%202023%5D%20FinMA-PIXIU/) | LLaMA | 金融多任务指令微调 |

## 当前基线判断

- **最强开源直接基线**：Qwen2.5-7B-Instruct，ChainEval 60.35。
- **最强金融专项模型**：Fin-R1，ChainEval 58.14，代码和权重公开。
- **最强数学专项模型**：Mathstral，ChainEval 59.87。
- **适合继续改造的新基座**：Qwen-Open-Finance-R-8B，但需要先复现其在 FinChain 上的结果。

> 当前目录用于基础收录、模型谱系和实验对比，尚未对全部模型论文进行分章节批读。
