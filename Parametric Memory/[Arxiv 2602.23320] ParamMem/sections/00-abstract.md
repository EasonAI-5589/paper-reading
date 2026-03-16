[← 返回 README](../README.md)

# Abstract

## 📌 预览
ParamMem: 参数化 reflection 记忆，通过编码跨样本模式来增加 reflection 多样性，提升 Agent 推理。

---

Self-reflection enables language agents to iteratively refine solutions, yet often produces repetitive outputs that limit reasoning performance. Recent studies have attempted to address this limitation through various approaches, among which increasing reflective diversity has shown promise. Our empirical analysis reveals a strong positive correlation between reflective diversity and task success, further motivating the need for diverse reflection signals. We introduce ParamMem, a parametric memory module that encodes cross-sample reflection patterns into model parameters, enabling diverse reflection generation through temperature-controlled sampling. Building on this module, we propose ParamAgent, a reflection-based agent framework that integrates parametric memory with episodic and cross-sample memory. Extensive experiments on code generation, mathematical reasoning, and multi-hop question answering demonstrate consistent improvements over state-of-the-art baselines. Further analysis reveals that ParamMem is sample-efficient, enables weak-to-strong transfer across model scales, and supports self-improvement without reliance on stronger external model.

> 💡 **Abstract 批读**:
> - **核心问题**: Self-reflection 容易产生重复输出 → 推理效果受限
> - **关键发现**: Reflection diversity ↔ task success 强正相关 (r=0.76)
> - **ParamMem 的 idea**: 不靠 prompt 或检索来增加多样性，而是把跨样本的 reflection 模式**编码到参数中**，推理时用 temperature 采样生成多样化的 reflection
> - **与其他 Parametric Memory 的区别**: Titans/MemoryLLM 的 parametric memory 存储的是「事实知识」，ParamMem 存储的是「推理模式/策略」

---

## 🔖 Section 总结
ParamMem 是 parametric memory 在 Agent 层面的应用——不是存知识，而是存反思策略。
