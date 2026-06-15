# StockLLM + FinSeer

**论文**: Enhancing Financial Time-Series Forecasting with Retrieval-Augmented Large Language Models<br>
**发表**: arXiv 预印本<br>
**arXiv**: [2502.05878](https://arxiv.org/abs/2502.05878)<br>
**生成模型**: [TheFinAI/StockLLM](https://huggingface.co/TheFinAI/StockLLM)<br>
**检索模型**: [TheFinAI/FinSeer](https://huggingface.co/TheFinAI/FinSeer)<br>
**许可**: Hugging Face 模型卡标注 MIT

首个面向金融时间序列预测定制的 RAG 框架。StockLLM 是约 1.2B 参数的 Llama 架构生成骨干，FinSeer 是约 109M 参数的 BERT 检索器；系统利用 LLM 反馈选择候选历史序列，并训练检索器寻找与当前查询相关且对预测真正有影响的历史片段。

论文报告完整 RAG 框架优于 StockLLM 单模型与随机检索，FinSeer 在 BIGDATA22 上比已有检索方法提高 8% 准确率。两套官方权重均已公开，是当前生态中少数可以直接复现“金融检索器 + 小型生成模型”的工作。

**与 FinChain 的关系**: 可把 FinChain 的题目、公式或执行轨迹作为检索对象，检验专用金融 retriever 是否能改善长上下文推理，而不必先训练更大的生成模型。

> 当前仅完成基础收录。
