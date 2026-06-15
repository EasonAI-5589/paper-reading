# Fin-PRM

**论文**: Fin-PRM: A Domain-Specialized Process Reward Model for Financial Reasoning in Large Language Models<br>
**发表**: arXiv 预印本<br>
**arXiv**: [2508.15202](https://arxiv.org/abs/2508.15202)<br>
**模型**: [DianJin/DianJin-Fin-PRM](https://huggingface.co/DianJin/DianJin-Fin-PRM)<br>
**数据**: [DianJin-Fin-PRM-Data](https://huggingface.co/datasets/DianJin/DianJin-Fin-PRM-Data)<br>
**代码**: [aliyun/qwen-dianjin](https://github.com/aliyun/qwen-dianjin)

面向金融推理的过程奖励模型，同时使用 step-level 与 trajectory-level 监督。它支持三类用途：筛选 CoT 轨迹做 SFT、为 RL 提供稠密过程奖励，以及在推理阶段执行 reward-guided Best-of-N。

论文报告相对基线在监督学习、强化学习和测试时推理中分别获得最高 12.9%、5.2% 和 5.1% 的提升。Hugging Face 数据集包含 4,969 条中文金融考试问题、逐步推理轨迹和多维质量标注。

**与 FinChain 的关系**: FinChain 已提供 gold steps 和可执行 trace，可用于构造比人工/模型打分更可验证的金融 PRM 监督。

> 当前仅完成基础收录。
