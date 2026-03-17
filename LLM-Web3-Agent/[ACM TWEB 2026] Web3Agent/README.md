# Web3Agent: Automating On-Chain Operations via Natural Language Interfaces

**作者**: Sizheng Fan, Tian Min  
**单位**: University of International Business and Economics (对外经济贸易大学), Keio University (庆应义塾大学)  
**期刊**: ACM Transactions on the Web, Volume 20, Issue 1 (February 2026)  
**链接**: [ACM DL](https://doi.org/10.1145/3777446)

## 一句话总结

提出 Web3Agent，首个 LLM 驱动的端到端 Web3 智能代理系统，通过自然语言分解用户指令为结构化链上操作流程，集成 RAG 增强推理 + 可控校准机制，在模拟环境中实现 94.1% 查询任务成功率和 80.3% 链上操作任务成功率。

## 核心贡献

1. **首个 Web3 AI Agent 框架**: 区别于 Web2 Agent（模拟浏览器点击），Web3Agent 直接与智能合约/区块链 API 交互，支持转账、Swap、Staking 等多步链上操作
2. **六模块流水线架构**: Intent Extraction → Instruction Chains → Previous Action Description → Action Prediction → Controllable Calibration → Executor，模块间通过 JSON 通信
3. **领域专用 RAG**: 三类语义分区 chunk（Operation / API / Error），按模块路由检索，显著提升参数恢复和错误处理能力
4. **可控校准层**: 执行前进行逻辑一致性 + 上下文可行性双重验证，防止 LLM 幻觉导致的链上误操作

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义、系统概述、评估概要 |
| [01 - Introduction](sections/01-introduction.md) | Web3 三大障碍 + AI Agent 解法 + Web3Agent 定位 |
| [02 - Related Work](sections/02-related-work.md) | Web3 架构演进、Web2 vs Web3 Agent、RAG 技术 |
| [03 - Operations on Web3](sections/03-operations.md) | 链上信息查询 (3层) + 链上操作分类 (3类) |
| [04 - System Overview](sections/04-system-overview.md) | 六模块架构详解 + 形式化定义 + RAG 集成 |
| [05 - User Interface](sections/05-user-interface.md) | Vue.js 前端：Chatbot + Operation Log + Operation Flow |
| [06 - Experiments](sections/06-experiments.md) | 数据集构建 + Intent/Parameter 评估 + 消融实验 |
| [07 - Discussion](sections/07-discussion.md) | 模块化设计反思 + 高风险领域推理 + 部署考量 |
| [08 - Limitations & Conclusion](sections/08-conclusion.md) | 可扩展性、安全性、未来方向 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Intent Parsing Accuracy (IPA) | 93.9% |
| Parameter Retrieve Accuracy (PRA) | 89.6% |
| Query Task SR (Step / Task) | 96.8% / 94.1% |
| Operation Task SR (Step / Task) | 91.2% / 80.3% |
| 评估任务数 | 35 个 Web3 任务 |
| 每任务变体 | 5 条自然语言指令 |
| 核心 LLM | GPT-4 |
| w/o Instruction Chain (Op Task SR) | 15.2% / 48.3% ⬇️ |
| w/o Operation Chunks (Op Task SR) | 10.1% / 25.6% ⬇️ |

---

## BibTeX

```bibtex
@article{fan2026web3agent,
  author    = {Sizheng Fan and Tian Min},
  title     = {Web3Agent: Automating On-Chain Operations via Natural Language Interfaces},
  journal   = {ACM Transactions on the Web},
  volume    = {20},
  number    = {1},
  article   = {9},
  pages     = {1--27},
  year      = {2026},
  doi       = {10.1145/3777446}
}
```
