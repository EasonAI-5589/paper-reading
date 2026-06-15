# Financial Benchmarks

金融问答、文档理解、数值推理、检索与可验证推理 benchmark 汇总。

统一评测框架、现有工具覆盖和推荐实现见 [Unified Financial Evaluation](./UNIFIED-EVALUATION.md)。

| Benchmark | 论文 / 发表 | 主要任务 |
|-----------|-------------|----------|
| [TAT-QA](./%5BACL%202021%5D%20TAT-QA/) | ACL 2021 / arXiv 2105.07624 | 财报表格与文本混合数值问答 |
| [FinQA](./%5BEMNLP%202021%5D%20FinQA/) | EMNLP 2021 / arXiv 2109.00122 | 带可执行推理程序的金融问答 |
| [MultiHiertt](./%5BACL%202022%5D%20MultiHiertt/) | ACL 2022 / arXiv 2206.01347 | 多层级、多表格金融数值推理 |
| [ConvFinQA](./%5BEMNLP%202022%5D%20ConvFinQA/) | EMNLP 2022 / arXiv 2210.03849 | 多轮对话金融数值推理 |
| [BizBench](./%5BarXiv%202311.06602%5D%20BizBench/) | arXiv 2311.06602 | 商业与金融程序生成、定量推理 |
| [DocMath-Eval](./%5BarXiv%202311.09805%5D%20DocMath-Eval/) | arXiv 2311.09805 | 长篇专业文档数学推理 |
| [FinanceBench](./%5BarXiv%202311.11944%5D%20FinanceBench/) | arXiv 2311.11944 | 上市公司财务文档开放式问答 |
| [FinDER](./%5BarXiv%202504.15800%5D%20FinDER/) | arXiv 2504.15800 | 金融 RAG 检索与问答 |
| [FinChain](./%5BarXiv%202506.02515%5D%20FinChain/) | arXiv 2506.02515 | 可验证金融思维链推理 |
| [FinTextQA](./%5BACL%202024%5D%20FinTextQA/) | ACL 2024 | 长篇、有来源的金融问答 |
| [FinanceMATH](./%5BACL%202024%5D%20FinanceMATH/) | ACL 2024 | 金融知识密集型数学推理 |
| [FinanceReasoning](./%5BACL%202025%5D%20FinanceReasoning/) | ACL 2025 | Python 解答驱动的金融数值推理 |
| [PIXIU](./%5BNeurIPS%202023%5D%20PIXIU/) | NeurIPS 2023 | 金融模型、指令数据与综合评测 |
| [FinBen](./%5BNeurIPS%202024%5D%20FinBen/) | NeurIPS 2024 | 36 数据集、24 任务综合评测 |
| [FLUE / FLANG](./%5BEMNLP%202022%5D%20FLUE-FLANG/) | EMNLP 2022 | 金融语言理解评测与预训练模型 |
| [FinMTEB](./%5BEMNLP%202025%5D%20FinMTEB/) | EMNLP 2025 | 金融 embedding 与检索评测 |
| [RealFin](./%5BarXiv%202602.07096%5D%20RealFin/) | arXiv 2602.07096 | 缺失前提识别与拒答能力 |
| [FinMMEval](./%5BCLEF%202026%5D%20FinMMEval/) | CLEF 2026 | 多语言、多模态金融理解与决策 |
| [FinEval](./%5BarXiv%202308.09975%5D%20FinEval/) | arXiv 2308.09975 | 中文金融知识、行业、安全与 Agent 综合评测 |
| [Finova](./%5BarXiv%202507.16802%5D%20Finova/) | arXiv 2507.16802 | 金融 Agent、复杂推理、安全与合规 |
| [FinanceIQ](./%5BDataset%202023%5D%20FinanceIQ/) | Dataset 2023 | CPA、CFA 等中文金融职业考试 |
| [FOMC Hawkish-Dovish](./%5BACL%202023%5D%20FOMC-Hawkish-Dovish/) | ACL 2023 | 货币政策鹰派/鸽派立场分类 |
| [Financial PhraseBank](./%5BJASIST%202014%5D%20Financial-PhraseBank/) | JASIST 2014 | 专家标注金融句子情感分类 |
| [Commodity News Headlines](./%5BarXiv%202009.04202%5D%20Commodity-News-Headlines/) | arXiv 2009.04202 | 黄金商品新闻标题与市场信号分类 |

## 能力分类

综合 benchmark 会横跨多个类别，表中按其主要用途放置，不能据此认为它只测试单一能力。

| 能力类别 | Benchmark | 主要指标或输出 |
|----------|-----------|----------------|
| 金融知识与职业考试 | FinEval、FinanceIQ | Accuracy |
| 金融情感与政策立场 | FOMC Hawkish-Dovish、Financial PhraseBank、Commodity News Headlines | Weighted F1 |
| 表格、文本与对话数值推理 | FinQA、TAT-QA、MultiHiertt、ConvFinQA | Answer/program accuracy、EM/F1 |
| 金融数学与程序推理 | FinanceMATH、FinanceReasoning、BizBench、DocMath-Eval | Accuracy、可执行程序或 Python 解答 |
| 可验证过程推理 | FinChain | Answer accuracy + ChainEval + execution trace |
| 开放式问答与长文生成 | FinanceBench、FinTextQA | 答案正确性、证据与生成质量 |
| RAG、检索与向量表示 | FinDER、FinMTEB | Retrieval、QA、embedding metrics |
| Agent、安全与合规 | Finova | 工具规划、槽位、推理与合规指标 |
| 可靠性、缺失信息与拒答 | RealFin | 正常回答、信息不足识别与拒答 |
| 多语言与多模态决策 | FinMMEval | QA、跨语言理解与金融决策 |
| 综合金融能力套件 | FLUE/FLANG、PIXIU、FinBen | 分类、抽取、QA、预测、风险、Agent、RAG 等多指标 |

## 按用途选择

- **快速训练回归**：FinanceIQ、FPB、FinQA、TAT-QA，可较快判断知识、分类和数值推理是否退化。
- **ODA-Fin-RL 同款对比**：FinEval、Finova、FinanceIQ、FOMC、FPB、Headlines、FinQA、TAT-QA、ConvFinQA。
- **方法核心评测**：FinChain，用于验证最终答案之外的步骤正确性和执行一致性。
- **最终完整评测**：增加 Finova、RealFin、FinanceReasoning、FinMMEval，测试业务操作、可靠性、代码执行和多模态泛化。

FinChain 的直接引用关系、主实验模型和 ChainEval 方法来源见 [FinChain Related Work Map](./%5BarXiv%202506.02515%5D%20FinChain/RELATED-WORK.md)。

ODA-Fin-RL 的九项评测、模型成绩和收录状态见 [ODA-Fin Benchmark Coverage](../Financial-Models/%5BarXiv%202603.07223%5D%20ODA-Fin-RL-8B/BENCHMARKS.md)。

> 当前目录用于基础收录与横向索引，尚未进行分章节批读。
