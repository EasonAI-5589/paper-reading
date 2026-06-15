# ODA-Fin-RL-8B

**论文**: Unlocking Data Value in Finance: A Study on Distillation and Difficulty-Aware Training<br>
**发表**: arXiv 预印本<br>
**arXiv**: [2603.07223](https://arxiv.org/abs/2603.07223)<br>
**基础模型**: Qwen3-8B<br>
**权重**: [ODA-Fin-SFT-8B](https://huggingface.co/OpenDataArena/ODA-Fin-SFT-8B) / [ODA-Fin-RL-8B](https://huggingface.co/OpenDataArena/ODA-Fin-RL-8B)<br>
**数据**: [ODA-Fin-SFT-318k](https://huggingface.co/datasets/OpenDataArena/ODA-Fin-SFT-318k) / [ODA-Fin-RL-12k](https://huggingface.co/datasets/OpenDataArena/ODA-Fin-RL-12k)<br>
**许可**: Apache-2.0

以数据质量为核心的金融推理模型。先用 318K 中英金融 CoT 数据进行 SFT，再从 hard-but-verifiable 样本中选择 12K 数据进行 GRPO。论文在九个金融任务、情感和数值推理 benchmark 上评估，报告其超过同规模开源金融模型。

**评测明细**: [9 个 benchmark、成绩与仓库覆盖情况](./BENCHMARKS.md)

**研究价值**: 模型、SFT/RL 数据和中间 SFT checkpoint 均公开，是目前开放资产最完整的金融推理训练基线之一。尚未核实独立训练代码是否完整公开。适合与 FinChain 的可执行 trace 结合，研究难度和可验证性感知的数据课程。

> 当前仅完成基础收录。
