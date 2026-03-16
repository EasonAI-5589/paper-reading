[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览
代码生成、数学推理、多跳 QA 三类任务上 ParamAgent 全面超越基线。

---

## 4.1 Setup

- **Tasks**: HumanEval, MBPP (code), MATH, GSM8K (math), HotPotQA (multi-hop QA)
- **Base LLM**: LLaMA-3.1-8B
- **Baselines**: Reflexion, DoT, DoT-bank
- **Max iterations**: T_max (varies by task)

## 4.2 Experimental Results

ParamAgent and ParamAgent-plus consistently outperform all baselines across all five datasets.

Key findings:
1. **ParamAgent > DoT-bank** on all tasks, despite ParamMem being much simpler
2. **ParamAgent-plus** achieves the best overall performance (combining all three memory types)
3. **Sample efficiency**: ~500 training samples suffice for strong performance
4. **Self-improvement**: ParamMem trained with base LLM's own data still improves performance
5. **Weak-to-strong transfer**: ParamMem trained on LLaMA-8B helps LLaMA-70B

> 💡 **关键实验发现**:
> - ParamMem 用 500 个样本训练就够了 → 非常 sample-efficient
> - 不需要强模型生成训练数据（self-improvement）→ self-contained
> - 弱模型的 reflection 模式能迁移到强模型 → 说明反思策略是跨规模通用的

---

## 🔖 Section 总结

### 核心洞察
1. Parametric memory 在 Agent 领域的应用价值：不是存知识，而是存「如何思考」
2. 三种记忆（episodic + cross-sample + parametric）互补效果显著
3. Sample efficiency 和 weak-to-strong transfer 使 ParamMem 非常实用
