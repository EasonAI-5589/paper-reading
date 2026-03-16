# ParamMem: Augmenting Language Agents with Parametric Reflective Memory

**作者**: Tianjun Yao, Yongqiang Chen, Yujia Zheng, Pan Li, Zhiqiang Shen, Kun Zhang
**来源**: arXiv 2602.23320 | **年份**: 2026
**链接**: [arXiv](https://arxiv.org/abs/2602.23320) | [GitHub](https://github.com/tianyao-aka/ParamAgent)

## 一句话总结

提出 **ParamMem**，一个参数化记忆模块，通过 LoRA 微调将跨样本的 reflection 模式编码到模型参数中，在推理时通过 temperature 控制采样生成多样化的 reflection，显著提升 Agent 的推理性能。

## 核心贡献

1. **ParamMem**: 用 LoRA 微调轻量模块，将跨样本 reflection 模式编码为参数，推理时通过 temperature 采样生成多样 reflection
2. **ParamAgent/ParamAgent-plus**: 统一 episodic memory + cross-sample memory + parametric memory 的 agent 框架
3. **Reflective Diversity ↔ Performance**: 实证发现 reflection 多样性与任务成功率强正相关（Pearson r=0.76）
4. **Sample Efficiency**: 仅需 ~500 训练样本
5. **Weak-to-Strong Transfer**: 弱模型训练的 ParamMem 也能提升强模型

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction & Preliminaries](sections/01-introduction.md) | Reflexion 框架 + Diversity-Performance 关联 |
| [02 - Method](sections/02-method.md) | ParamMem 构建 + ParamAgent 框架 |
| [03 - Experiments](sections/03-experiments.md) | 代码生成、数学推理、多跳 QA |

## 关键数字

| 指标 | 数值 |
|------|------|
| 训练样本 | ~500 即有效 |
| Diversity-Performance 相关性 | Pearson r = 0.76 |
| 任务 | HumanEval, MBPP, MATH, GSM8K, HotPotQA |
| Base LLM | LLaMA-3.1-8B |
