# MEMORYLLM: Towards Self-Updatable Large Language Models

**作者**: Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, Julian McAuley
**会议**: ICML 2024 | **年份**: 2024
**链接**: [arXiv](https://arxiv.org/abs/2402.04624) | [GitHub](https://github.com/wangyu-ustc/MemoryLLM)

## 一句话总结

在 Transformer 每一层嵌入固定大小的 **memory pool**（1B 参数的隐向量），通过 self-update 机制实现知识的持续注入和指数衰减遗忘，无需反向传播即可更新，近百万次更新后仍保持功能完整。

## 核心贡献

1. **Memory Pool 设计**: 在 Llama2-7B 每层嵌入 7,680 个 memory tokens（共 1B 参数），作为可自更新的知识存储
2. **Self-Update 机制**: 用 Transformer 本身处理新知识，生成新 memory tokens 替换旧的，无需额外模块
3. **指数遗忘保证**: Random dropping 实现类 Ebbinghaus 遗忘曲线，保留率趋近 $1/e$
4. **三阶段训练**: 新知识注入 + 连续上下文理解 + 遗忘缓解
5. **鲁棒性**: 近百万次 memory 更新后模型功能完好

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 三类现有方法的不足 |
| [02 - Preliminaries & Method](sections/02-method.md) | 问题定义 + Memory Pool 结构 + Self-Update + 遗忘分析 + 训练策略 |
| [03 - Experiments](sections/03-experiments.md) | 模型编辑、长上下文、知识保留、鲁棒性 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 基座模型 | Llama2-7B |
| Memory Pool | 32层 × 7,680 tokens × 4,096 dim = 1.066B 参数 |
| 每次更新 tokens | K 个（K << N）|
| 保留率极限 | $(1-K/N)^{N/K} \to 1/e \approx 36.8\%$ |
| 更新次数测试 | ~1M 次，无退化 |
