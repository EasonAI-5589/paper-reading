# AI Finance

本专题以 [FinChain](./Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/) 为起点，整理金融模型、benchmark、开放训练资产、统一评测方法和候选研究方向。

最初的文献范围来自 FinChain 的参考文献、主实验模型与评测设置。在此基础上，我们进一步扩展了较新的金融推理、检索、可靠性、多语言和多模态评测工作，并补充可直接复现或继续训练的开源金融模型。

## 文档导航

| 想了解什么 | 入口文档 | 内容 |
|------------|----------|------|
| 专题里收了什么 | [AI Finance 首页](./README.md) | 总体范围、目录结构和推荐阅读路线 |
| 有哪些金融模型 | [Financial Models](./Financial-Models/) | 27 个金融专项、通用基线、PRM、RAG 和多模态模型 |
| 哪些模型能直接训练或复现 | [Model Ecosystem](./Financial-Models/MODEL-ECOSYSTEM.md) | 权重、数据、代码、训练框架、开放程度和研究方向 |
| FinChain 对比了哪些模型 | [FinChain Model Comparison](./Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/MODEL-COMPARISON.md) | 26 个主实验模型、规模、类别和 ChainEval 成绩 |
| 有哪些金融 benchmark | [Financial Benchmarks](./Financial-Benchmarks/) | 24 个评测集及能力分类 |
| 模型训练完以后怎么评测 | [Unified Financial Evaluation](./Financial-Benchmarks/UNIFIED-EVALUATION.md) | vLLM、FinBen/lm-eval、FinEval、Finova、FinChain 的组合方案 |
| ODA-Fin-RL 跑了什么 | [ODA-Fin Benchmark Coverage](./Financial-Models/%5BarXiv%202603.07223%5D%20ODA-Fin-RL-8B/BENCHMARKS.md) | 9 项 benchmark、指标、成绩和仓库覆盖情况 |
| FinChain 延伸出了哪些论文 | [FinChain Related Work](./Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/RELATED-WORK.md) | 引用、前置 benchmark、ChainEval 和后续工作 |

## 当前范围

| 模块 | 内容 | 数量 |
|------|------|------|
| [Financial Benchmarks](./Financial-Benchmarks/) | 金融问答、数值推理、RAG、金融知识、Agent、情感与多模态评测 | 24 |
| [Financial Models](./Financial-Models/) | 金融专项、过程奖励、检索增强、多模态、交易决策与通用基线 | 27 |
| Financial Agents | [FinMem](./%5BarXiv%202311.13743%5D%20FinMem/)、[FinAgent](./%5BarXiv%202402.18485%5D%20FinAgent/)，作为金融决策与交易 Agent 背景 | 2 |

## 目录结构

```text
AI Finance/
├── README.md                         # 专题总入口
├── Financial-Models/
│   ├── README.md                     # 模型分类索引
│   ├── MODEL-ECOSYSTEM.md            # 权重、数据、代码、训练框架和研究路线
│   └── [Paper or Model]/README.md    # 单个模型卡片
├── Financial-Benchmarks/
│   ├── README.md                     # Benchmark 分类索引
│   ├── UNIFIED-EVALUATION.md         # 统一跑分方案
│   ├── [FinChain]/
│   │   ├── MODEL-COMPARISON.md       # FinChain 主实验模型比较
│   │   └── RELATED-WORK.md           # FinChain 相关文献地图
│   └── [Benchmark]/README.md         # 单个 benchmark 卡片
├── [FinMem]/README.md                # 交易 Agent 背景工作
└── [FinAgent]/README.md              # 多模态交易 Agent 背景工作
```

## 推荐阅读路线

### 准备做模型训练

1. 先读 [Model Ecosystem](./Financial-Models/MODEL-ECOSYSTEM.md)，判断模型、数据和代码是否真的开放。
2. 以 ODA-Fin-SFT/RL checkpoint 和数据作为开放基线。
3. SFT 使用 LLaMA-Factory，GRPO 使用 VeRL；过程奖励可参考 DianJin-PRM 的 TRL + VeRL 实现。
4. 训练后按 [Unified Financial Evaluation](./Financial-Benchmarks/UNIFIED-EVALUATION.md) 启动 vLLM，并调用各 benchmark runner。

### 准备写论文

1. 从 [FinChain Related Work](./Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/RELATED-WORK.md) 理解可验证金融推理脉络。
2. 用 [Model Ecosystem](./Financial-Models/MODEL-ECOSYSTEM.md) 选择 ODA-Fin-RL、Fin-PRM、Fin-R1 和 DianJin-R1 等对照。
3. 从 [Benchmark Index](./Financial-Benchmarks/) 组合金融知识、情感、数值推理、Agent 和可靠性评测。
4. 当前优先方向是 `FinChain executable trace + process reward + difficulty/verifiability-aware data selection`。

## FinChain 扩展路线

1. **从 FinChain 还原研究脉络**：整理其直接引用的 benchmark、ChainEval 方法来源和主实验模型。
2. **扩展 benchmark**：加入 FinTextQA、FinanceMATH、FinanceReasoning、FinEval、Finova、FinanceIQ、RealFin、FinMMEval 等工作，覆盖检索、推理、业务 Agent、拒答、多语言和多模态能力。
3. **整理金融模型谱系**：从领域预训练和 FinChain 式确定性推理，扩展到 Fin-PRM、StockLLM/FinSeer、Trading-R1 与 Trade-R1 的检索和随机决策。
4. **寻找可训练基线**：优先复现完整开放的 ODA-Fin-RL-8B，并研究 FinChain 与金融过程奖励模型的结合。

FinChain 的引用关系、实验模型和评测方法见 [FinChain Related Work Map](./Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/RELATED-WORK.md)。
FinChain 之后的模型、训练数据、开放状态和候选研究方向见 [Financial Reasoning Model Ecosystem](./Financial-Models/MODEL-ECOSYSTEM.md)。

## 背景工作

| 论文 | arXiv | 定位 |
|------|-------|------|
| [FinMem](./%5BarXiv%202311.13743%5D%20FinMem/) | 2311.13743 | 分层记忆与角色设计的 LLM 交易 Agent |
| [FinAgent](./%5BarXiv%202402.18485%5D%20FinAgent/) | 2402.18485 | 多模态、工具增强的金融交易 Agent |

> 当前以文献收录和研究地图为主，尚未对全部论文进行分章节批读。
