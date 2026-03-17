[← 返回 README](../README.md)

# Abstract

## 📌 预览
Web3Agent 的核心：用 LLM Agent 自动化复杂的多步链上操作，解决 Web3 对普通用户的可访问性问题。

---

Recent advances in large language models (LLMs) have enabled the emergence of intelligent agents capable of performing complex multi-step tasks across various domains. In parallel, the growth of Web3 has introduced a decentralized web infrastructure, yet remains largely inaccessible to non-technical users due to operational complexity, fragmented information, and security risks. In this article, we present Web3Agent, an AI agent system that integrates LLM-based interaction with blockchain environments to enable language-driven on-chain operations. Web3Agent automatically decomposes user instructions into structured workflows, dynamically queries blockchain data and APIs, and performs multi-step operations such as asset transfers, token swaps, and smart contract execution. Web3Agent incorporates real-time inspection, error handling, and interaction transparency across its operation log, and flow visualization components. We evaluate the system and perform ablation study with customized dataset in a simulated environment, demonstrating its feasibility in orchestrating complex Web3 tasks and highlighting implications for agent-based abstraction in decentralized systems.

> 💡 **Abstract 批读**:
> - **背景**: LLM Agent 已在多领域展现多步任务执行能力；Web3 增长迅速但对非技术用户门槛极高
> - **问题**: 操作复杂性 + 信息碎片化 + 安全风险，三大障碍阻碍 Web3 大规模采用
> - **方案**: Web3Agent — LLM 驱动的 AI Agent 系统，自然语言 → 结构化工作流 → 链上执行
> - **能力**: 资产转账、Token Swap、智能合约执行等多步操作
> - **特色**: 实时检查、错误处理、交互透明性（操作日志 + 流程可视化）
> - **评估**: 模拟环境 + 自定义数据集 + 消融实验

---

## 🔖 Section 总结

### 核心洞察
1. Web3 的可访问性问题本质上是一个 HCI 问题，LLM Agent 提供了自然语言抽象层
2. 不同于简单对话助手，Web3Agent 强调多步、有状态的端到端执行
3. 系统设计同时关注功能性（自动化执行）和安全性（透明性 + 错误处理）
