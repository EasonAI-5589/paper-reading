# FinR1

**论文**: FinR1: A Large Language Model for Financial Reasoning through Reinforcement Learning
**发表**: arXiv 预印本
**arXiv**: [2503.16252](https://arxiv.org/abs/2503.16252)
**最新版本**: v5, 2026-03-19
**基础模型**: Qwen2.5-7B-Instruct
**规模**: 7B
**代码**: [SUFE-AIFLM-Lab/Fin-R1](https://github.com/SUFE-AIFLM-Lab/Fin-R1)
**权重**: [SUFE-AIFLM-Lab/Fin-R1](https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1)

面向金融推理的强化学习模型。使用 DeepSeek-R1 蒸馏并经过双轮质量筛选构建 60,091 条 Fin-R1-Data，再对 Qwen2.5-7B-Instruct 依次执行 SFT 与 GRPO。

## 开源情况

- 模型权重：已公开，BF16 Safetensors，可直接从 Hugging Face 下载。
- 代码仓库：已公开，包含数据构建、训练方法和 vLLM 推理说明。
- 推理服务：官方示例支持 `vllm serve`，最大上下文配置为 16,384。
- 训练数据：公开说明了组成和数量；需要进一步确认仓库是否提供全部原始/蒸馏数据文件。

```bash
git clone https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1
```

**FinChain**: 主实验金融模型中表现最佳，ChainEval 为 58.14，接近部分通用前沿模型。

> 当前仅完成基础收录。
