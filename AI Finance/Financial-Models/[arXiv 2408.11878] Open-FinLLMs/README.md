# Open-FinLLMs

**论文**: Open-FinLLMs: Open Multimodal Large Language Models for Financial Applications<br>
**发表**: arXiv 预印本<br>
**arXiv**: [2408.11878](https://arxiv.org/abs/2408.11878)<br>
**组织**: [TheFinAI](https://huggingface.co/TheFinAI)<br>
**模型入口**: [FinLLaMA](https://huggingface.co/TheFinAI/FinLLaMA) / [FinLLaMA-instruct](https://huggingface.co/TheFinAI/FinLLaMA-instruct)

金融多模态模型族：FinLLaMA 在 52B 金融 token 上继续预训练，语料覆盖文本、表格和时间序列；FinLLaMA-instruct 使用 573K 金融指令；FinLLaVA 使用 1.43M 图文指令处理金融表格和图表。论文报告其在文本任务、多模态任务和交易模拟中均有提升。

**开放性说明**: TheFinAI 存在同名 Hugging Face 仓库，但 FinLLaMA-instruct 当前为 gated，模型元数据与论文描述也存在不一致；截至 2026-06-15 未核实到完整 FinLLaVA 官方权重。因此本条更适合作为多模态训练谱系与数据规模参照，使用权重前需单独验证。

> 当前仅完成基础收录。
