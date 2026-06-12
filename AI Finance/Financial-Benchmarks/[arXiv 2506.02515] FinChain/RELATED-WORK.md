# FinChain Related Work Map

本页依据 FinChain v4 正文和参考文献表整理，区分论文直接引用、实验对比模型和领域补充。

## 金融推理 Benchmark

| 工作 | FinChain 中的定位 | FinChain 的主要区别 |
|------|------------------|---------------------|
| FinQA | 金融数值推理前作 | 程序监督较弱，不系统评估自然语言中间步骤 |
| ConvFinQA | 对话式金融数值推理 | 关注对话依赖与最终答案，不做严格 step alignment |
| TAT-QA | 表格 + 文本金融 QA | 证据形式复杂，但不专注可执行 CoT 验证 |
| MultiHiertt | 多层级、多表格推理 | 强调跨表格证据，不提供统一符号模板 |
| FinTextQA | 长篇解释型金融 QA | 重点是检索与长答案，而非可执行计算链 |
| FinanceMATH | 金融知识密集型数学推理 | 有 Python 解答，但覆盖和步骤验证方式不同 |
| DocMath-Eval | 长文档数学推理 | 更偏通用专业文档，不完全金融专属 |
| FinanceReasoning | 金融数值推理与 Python 解答 | 提升答案可靠性，但不系统验证预测步骤对齐 |
| BizBench | 商业与金融定量推理 | 综合任务集合，不专注金融公式的细粒度诊断 |
| PIXIU / FinBen / FLUE | 综合金融能力评测 | 覆盖广，FinChain 则聚焦可验证多步符号推理 |

## FinChain 主实验模型

| 类别 | 模型 |
|------|------|
| Frontier proprietary | GPT-5/4.1、Claude Sonnet、Gemini 2.5、DeepSeek V3/R1、Grok 4 |
| Finance-specific | Fin-o1、Fin-R1、DianJin-R1、Finance-LLaMA、Finance-Qwen |
| Math-enhanced | WizardMath、MetaMath、Mathstral、Qwen2.5-Math |
| General open-weight | LLaMA-3.1、Qwen2.5、Qwen3 |

主实验结论不是“金融模型一定更强”：Qwen2.5 Instruct 的 ChainEval 为 60.35，略高于 Mathstral 59.87 和 Fin-R1 58.14；Finance-LLaMA 与 Finance-Qwen 则明显较弱。论文据此认为模型规模和窄领域微调都不足以保证可靠的步骤级金融推理。

## ChainEval 方法来源

| 工作 | 作用 |
|------|------|
| GSM-Symbolic | 参数化符号模板和可控扰动的直接灵感来源 |
| ROSCOE | 逐步推理质量指标体系 |
| Faithful Chain-of-Thought Reasoning | 推理忠实性与中间步骤一致性背景 |
| ROUGE / BERTScore | ChainEval 的文本相似度对照指标 |

## 非直接引用的领域补充

FinanceBench 与 FinDER 已保留在总目录中，但未出现在 FinChain v4 的参考文献表里，应视为金融文档 QA/RAG 的领域补充，而不是 FinChain 直接前作。

## 下一篇向外扩展建议

1. **FinanceMATH**：与 FinChain 的任务最接近，适合继续追踪金融数学推理数据和模型。
2. **FinanceReasoning**：直接承接可执行 Python solution，可继续扩展到 reasoning model。
3. **FinR1**：FinChain 中表现最强的金融专项模型，可向训练数据、RL 和基座模型方向继续扒。
4. **FinBen**：覆盖面最广，适合建立完整的金融任务分类树。
