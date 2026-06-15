# FinChain Model Comparison

本页整理 FinChain v4 的主实验模型、模型来源和核心结果。论文共列出 26 个模型，其中 25 个模型在完整 benchmark 上评测；Grok 4 Heavy 因推理成本较高，仅在随机抽取的 200 个样本上评测。

## 完整 Benchmark 结果

| 类别 | 模型 | 规模 | ChainEval | FAC |
|------|------|------|-----------|-----|
| Frontier | GPT-5-mini | 未公开 | **67.17** | 80.28 |
| Frontier | GPT-5 | 未公开 | 66.57 | 82.03 |
| Frontier | Claude Sonnet 4.5 | 未公开 | 66.33 | 83.34 |
| Frontier | Claude Sonnet 4 | 未公开 | 66.20 | 82.62 |
| Frontier | Gemini 2.5 Pro | 未公开 | 66.04 | 84.34 |
| Frontier | Gemini 2.5 Flash | 未公开 | 65.96 | 83.90 |
| Frontier | Claude Sonnet 3.7 | 未公开 | 65.51 | 83.14 |
| Frontier | GPT-4.1 | 未公开 | 65.34 | **84.66** |
| Frontier | DeepSeek-V3.1 | 未公开 | 65.29 | 84.34 |
| Frontier | DeepSeek-V3.2 | 未公开 | 65.23 | 84.17 |
| Frontier | GPT-4.1-mini | 未公开 | 65.06 | 84.59 |
| Frontier | Grok-4 Fast | 未公开 | 60.69 | 66.54 |
| Frontier | DeepSeek-R1 | 未公开 | 51.22 | 28.97 |
| Finance-specific | Fin-R1 | 7B | **58.14** | 52.76 |
| Finance-specific | DianJin-R1 | 7B | 51.95 | 37.69 |
| Finance-specific | Fin-o1 | 8B | 41.50 | **52.79** |
| Finance-specific | Finance-LLaMA | 8B | 41.35 | 25.21 |
| Finance-specific | Finance-Qwen | 7B | 34.57 | 31.62 |
| Math-enhanced | Mathstral | 7B | **59.87** | 54.03 |
| Math-enhanced | Qwen2.5-Math | 7B | 55.35 | **62.62** |
| Math-enhanced | WizardMath | 7B | 24.33 | 41.28 |
| Math-enhanced | MetaMath | 7B | 7.93 | 23.97 |
| General open-weight | Qwen2.5-Instruct | 7B | **60.35** | **65.41** |
| General open-weight | Llama-3.1-Instruct | 8B | 53.99 | 32.72 |
| General open-weight | Qwen3 | 8B | 43.32 | 32.28 |

## Grok 4 Heavy 子集结果

Grok 4 Heavy 只在随机抽取的 200 个样本上评测，不能与上表完整 benchmark 的结果直接比较。

| 模型 | ChainEval | FAC |
|------|-----------|-----|
| GPT-5 | 68.42 | 69.50 |
| Gemini 2.5 Pro | 67.92 | 73.00 |
| Grok 4 Heavy | 65.64 | 81.00 |
| Mathstral | 64.02 | 51.00 |
| Fin-R1 | 56.44 | 34.00 |

## 开源模型来源

| 模型 | Backbone | 权重 |
|------|----------|------|
| Fin-o1 | Llama-3.1-8B | [TheFinAI/Fin-o1-8B](https://huggingface.co/TheFinAI/Fin-o1-8B) |
| Fin-R1 | Qwen2.5-7B-Instruct | [SUFE-AIFLM-Lab/Fin-R1](https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1) |
| DianJin-R1 | Qwen2.5-7B-Instruct | [DianJin/DianJin-R1-7B](https://huggingface.co/DianJin/DianJin-R1-7B) |
| Finance-LLaMA | DeepSeek-R1-Distill-Llama-8B | [WiroAI/WiroAI-Finance-Llama-8B](https://huggingface.co/WiroAI/WiroAI-Finance-Llama-8B) |
| Finance-Qwen | Qwen2.5-7B | [WiroAI/WiroAI-Finance-Qwen-7B](https://huggingface.co/WiroAI/WiroAI-Finance-Qwen-7B) |
| WizardMath | Mistral-7B-v0.1 | [WizardMath-7B-V1.1](https://huggingface.co/WizardLMTeam/WizardMath-7B-V1.1) |
| MetaMath | Llemma-7B | [MetaMath-7B-V1.0](https://huggingface.co/meta-math/MetaMath-7B-V1.0) |
| Mathstral | Mistral-7B-v0.1 | [Mathstral-7B-v0.1](https://huggingface.co/mistralai/Mathstral-7B-v0.1) |
| Qwen2.5-Math | Qwen2.5-7B | [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) |
| Llama 3.1 | - | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Qwen2.5 | - | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Qwen3 | - | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |

## 主要观察

- 完整 benchmark 的最高 ChainEval 是 GPT-5-mini 的 67.17，而最高 FAC 是 GPT-4.1 的 84.66。
- 开源模型中 Qwen2.5-Instruct 的 ChainEval 最高，为 60.35，超过全部金融专项模型。
- 金融模型中 Fin-R1 最强，ChainEval 为 58.14，但仍略低于通用 Qwen2.5-Instruct 和 Mathstral。
- 数学增强并不必然有效：Mathstral 达到 59.87，而 MetaMath 只有 7.93，说明训练方式和输出格式对 ChainEval 影响很大。
- Qwen3-8B 的表现低于 Qwen2.5-7B-Instruct，更新的基座并不自动意味着更好的可验证金融推理。

**来源**: FinChain v4 Table 2、Table 4 与 Table 9。
