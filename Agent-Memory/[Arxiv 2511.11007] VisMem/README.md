# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models

**作者**: Xinlei Yu, Chengming Xu, Guibin Zhang, Zhangquan Chen, Yudong Zhang, Yongbo He, Peng-Tao Jiang, Jiangning Zhang, Xiaobin Hu, Shuicheng Yan  
**机构**: NUS (Shuicheng Yan 组), Fudan, THU, ZJU, USTC, vivo  
**链接**: [arXiv 2511.11007](https://arxiv.org/abs/2511.11007) | [GitHub](https://github.com/YU-deep/VisMem)

## 一句话总结

受认知心理学 Dennis Norris Theory 启发，VisMem 为 VLM 引入双路 latent vision memory（短期视觉主导 + 长期语义主导），通过特殊 token 按需触发、两阶段 GRPO 训练，在 12 个 benchmark 上平均提升 11%，且不破坏原始能力。

## 核心贡献

1. **Latent Vision Memory 范式**: 填补了 latent space paradigm 中视觉记忆的空白，区别于纯语言的 Coconut/MemGen
2. **双路记忆设计**: 短期记忆挂 vision encoder（LoRA）提供视觉细节，长期记忆挂 language model（LoRA）提供语义知识——与认知心理学完美对应
3. **按需调用机制**: 4 个特殊 token 触发记忆，模型自己学会何时、用哪种记忆
4. **两阶段 GRPO 训练**: Stage I 训记忆内容（冻结 policy），Stage II 训调用策略（冻结 memory former）+ type/negative penalty
5. **全面验证**: 9 个 base model（3B~38B），12 个 benchmark，跨域泛化 + 抗遗忘 + 低延迟

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 问题定义 + 框架概览 |
| [01 - Introduction](sections/01-introduction.md) | Visual processing bottleneck + 四范式对比 + Dennis Norris Theory + VisMem vs MemGen |
| [02 - Related Work](sections/02-related-work.md) | 视觉能力增强四范式 + 记忆增强方法谱系 |
| [03 - Methodology](sections/03-methodology.md) | ⭐ **核心**: Memory Invocation (特殊 token) + Memory Formation (Query Builder + 双路 Memory Former) + 两阶段 GRPO 训练 |
| [04 - Experiments](sections/04-experiments.md) | 12 benchmark 主实验 + 跨域泛化 + 持续学习 + 9 模型兼容 + 消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性讨论 |
| [06 - Appendix](sections/06-appendix.md) | 理论基础 + GRPO 完整公式 + 超参数 + 补充实验 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 平均提升 (vs vanilla Qwen2.5-VL-7B) | **+11.0%** |
| 理解 / 推理 / 生成 | +8.9% / +14.4% / +10.6% |
| vs 最强 baseline (Vision-R1) | +3.0% |
| 跨域泛化 (2 数据集训练→4 unseen) | +6.9~20.2% |
| 持续学习遗忘 (MMVet 4 stage) | 72.1% (最佳) |
| 兼容 base model | 9 个 (3B~38B, 3 系列) |
| 推理延迟增加 | 8.2%~43.8% |
| Memory query 长度 K | 8 |
| 短期/长期 memory token 数 | 8 / 16 |

## 与 MemGen 的关系

VisMem 和 MemGen 是 NUS Shuicheng Yan 组的**姊妹工作**，解决不同维度的记忆问题：

| | VisMem | MemGen |
|--|--------|--------|
| 目标 | VLM 单次推理中的视觉遗忘 | Agent 跨 episode 的经验记忆 |
| 模态 | 视觉 + 语义 dual latent memory | 纯文本 latent memory |
| 触发 | 特殊 token (<m_I^s>, <m_I^l>) | <MEM> token |
| 挂载 | Vision encoder + Language model (LoRA) | Language model |
| 训练 | 两阶段 GRPO | 端到端 GRPO |
| 互补性 | ✓ 理论上可联合使用 | ✓ |
