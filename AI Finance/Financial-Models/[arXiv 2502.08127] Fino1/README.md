# Fino1

**论文**: Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance
**发表**: arXiv 预印本
**arXiv**: [2502.08127](https://arxiv.org/abs/2502.08127)
**原论文基础模型**: Llama-3.1-8B-Instruct / Qwen2.5-14B-Instruct
**后续公开权重**: [Fin-o1-8B](https://huggingface.co/TheFinAI/Fin-o1-8B) / [Fin-o1-14B](https://huggingface.co/TheFinAI/Fin-o1-14B)
**后续权重基础模型**: Qwen3-8B / Qwen3-14B
**数据**: [TheFinAI/FinCoT](https://huggingface.co/datasets/TheFinAI/FinCoT)

通过金融领域 CoT 微调与强化学习增强金融推理，并系统研究通用推理能力向金融文本、表格和公式任务的迁移。Hugging Face 后续发布了基于 Qwen3 的 Apache-2.0 版本，因此实验时必须明确区分论文原始 checkpoint 与后续同名权重。

**FinChain**: 主实验中的 Finance Specific LLM，规模 8B。

> 当前仅完成基础收录。
