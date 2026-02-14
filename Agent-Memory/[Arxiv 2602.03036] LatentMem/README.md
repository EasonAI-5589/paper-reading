# LatentMem: Customizing Latent Memory for Multi-Agent Systems

**作者**: Muxin Fu, Guibin Zhang, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, Yang Yang  
**机构**: Tongji University, Shanghai AI Laboratory, NUS, CUHK, Nanjing University, SJTU  
**链接**: [arXiv 2602.03036](https://arxiv.org/abs/2602.03036) | [GitHub](https://github.com/KANABOON1/LatentMem)

## 一句话总结

提出 LatentMem，一个可学习的多智能体 latent memory 框架：用 memory composer 将历史轨迹压缩为角色感知的固定长度 latent token，并通过 LMPO（基于 GRPO 的强化学习）端到端优化 memory 质量，在 6 个 benchmark、4 个 MAS 框架上取得最高 19.36% 的提升。

## 核心贡献

1. **LatentMem 框架**：experience bank + memory composer，将原始轨迹压缩为角色感知、固定长度的 latent memory，注入 agent 推理过程
2. **LMPO 优化算法**：基于 GRPO 的 token-level 策略优化，通过 latent memory 的可微性将任务反馈反传到 memory composer
3. **高效 + 泛化**：token 用量减少 50%，推理时间降至 2/3；在 unseen MAS 框架和 OOD 数据集上仍有显著提升
4. **即插即用**：不修改底层 MAS 框架架构，直接通过 latent token 注入增强 agent

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义 + 方法概览 + 主要结果 |
| [01 - Introduction](sections/01-introduction.md) | 动机：memory homogenization + information overload |
| [02 - Related Works](sections/02-related-works.md) | LLM-based MAS + Multi-agent memory 相关工作 |
| [03 - Preliminary](sections/03-preliminary.md) | 符号定义 + 问题形式化 |
| [04 - Methodology](sections/04-methodology.md) | LatentMem 框架：Experience Bank + Memory Composer + LMPO |
| [05 - Experiments](sections/05-experiments.md) | 6 benchmarks × 4 MAS 框架，主实验 + 消融 + Case Study |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + Impact Statement |
| [07 - Appendix](sections/07-appendix.md) | 实验细节、额外结果、Prompt 模板、示例轨迹 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 最大性能提升 (vs vanilla) | 19.36% (PopQA, MacNet) |
| 最大性能提升 (vs SOTA memory) | 16.20% (TriviaQA, AutoGen) |
| Token 节省 | ~50% fewer tokens |
| 推理时间 | ~2/3 of mainstream memory |
| Latent memory 长度 | L'=8 tokens |
| OOD 提升 (PDDL) | +7.10% |
| Unseen MAS 提升 (CAMEL) | +7.90% |
| vs MARTI (multi-agent finetuning) | +11.73% on TriviaQA |
