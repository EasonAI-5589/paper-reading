# AI Finance

本专题以 [FinChain](./Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/) 为起点，整理可验证金融推理相关的 benchmark、金融模型和训练基线。

最初的文献范围来自 FinChain 的参考文献、主实验模型与评测设置。在此基础上，我们进一步扩展了较新的金融推理、检索、可靠性、多语言和多模态评测工作，并补充可直接复现或继续训练的开源金融模型。

## 当前范围

| 模块 | 内容 | 数量 |
|------|------|------|
| [Financial Benchmarks](./Financial-Benchmarks/) | 金融问答、数值推理、RAG、可验证 CoT、可靠性与多模态评测 | 18 |
| [Financial Models](./Financial-Models/) | 金融专项、数学增强、通用基线与金融背景模型 | 17 |
| Financial Agents | FinMem、FinAgent，作为金融决策与交易 Agent 的背景工作 | 2 |

## FinChain 扩展路线

1. **从 FinChain 还原研究脉络**：整理其直接引用的 benchmark、ChainEval 方法来源和主实验模型。
2. **扩展 benchmark**：加入 FinTextQA、FinanceMATH、FinanceReasoning、FinMTEB、RealFin、FinMMEval 等工作，覆盖检索、推理、拒答、多语言和多模态能力。
3. **整理金融模型谱系**：从 FinBERT、BloombergGPT、FinGPT 扩展到 Fin-R1、DianJin-R1 和 Qwen-Open-Finance-R-8B。
4. **寻找可训练基线**：优先关注具有公开代码和权重的 Fin-R1，以及可继续进行金融推理 SFT/RL 的 Qwen-Open-Finance-R-8B。

FinChain 的引用关系、实验模型和评测方法见 [FinChain Related Work Map](./Financial-Benchmarks/%5BarXiv%202506.02515%5D%20FinChain/RELATED-WORK.md)。

## 背景工作

| 论文 | arXiv | 定位 |
|------|-------|------|
| [FinMem](./%5BarXiv%202311.13743%5D%20FinMem/) | 2311.13743 | 分层记忆与角色设计的 LLM 交易 Agent |
| [FinAgent](./%5BarXiv%202402.18485%5D%20FinAgent/) | 2402.18485 | 多模态、工具增强的金融交易 Agent |

> 当前以文献收录和研究地图为主，尚未对全部论文进行分章节批读。
