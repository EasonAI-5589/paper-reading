# FEVO

**论文**: FEVO: Financial Knowledge Expansion and Reasoning Evolution for Large Language Models<br>
**发表**: arXiv 预印本<br>
**arXiv**: [2507.06057](https://arxiv.org/abs/2507.06057)<br>
**基础模型**: Qwen2.5-32B<br>
**模型系列**: FEVO-C32B / S32B / R32B

FEVO 将金融推理训练拆成三个阶段：继续预训练扩充金融知识，SFT 蒸馏结构化推理，再用 RL 融合领域知识与推理能力。论文在七个 benchmark 上评估，报告 FEVO-R32B 在五个金融 benchmark 上达到当时的最佳结果，并显著优于直接从 Qwen2.5-32B-Instruct 进行 RL 的 FEVO-R32B-0。

**开放情况**: 截至 2026-06-15，未在论文入口或 Hugging Face 检索到官方 FEVO 权重与 FEVO-Train 数据。当前适合作为训练配方和消融设计参照，而不是可直接复现的首选基线。

> 当前仅完成基础收录。
