# Qwen-Open-Finance-R-8B

**论文**: The LLM Pro Finance Suite: Multilingual Large Language Models for Financial Applications
**发表**: arXiv 预印本
**arXiv**: [2511.08621](https://arxiv.org/abs/2511.08621)
**权重**: [DragonLLM/Qwen-Open-Finance-R-8B](https://huggingface.co/DragonLLM/Qwen-Open-Finance-R-8B)
**基础模型**: Qwen3-8B
**规模**: 8B
**许可**: Apache-2.0
**语言**: English / French / German

LLM Pro Finance Suite 中公开的金融模型之一。训练数据包含约 54.4% 金融数据、19.8% 翻译数据、15.6% 通用数据、8% RAG 数据，以及 2.2% 推理、数学和代码数据。

## 可用性

- 模型权重已公开，但 Hugging Face 当前要求登录并同意访问条件。
- 使用 Safetensors，可通过 Transformers 加载。
- 基于 Qwen3-8B，适合作为金融继续训练和可验证推理增强的基线。

## 可改进方向

当前训练数据中推理、数学和代码占比仅为 2.2%。可以使用 FinChain、FinanceMATH 与 FinanceReasoning 增加步骤级 SFT，并通过可执行答案或 ChainEval 构建强化学习奖励。

> 当前仅完成基础收录。
