[← 返回 README](../README.md)

# 2 Related Works

## 📌 预览
从 Web 演进、Web2 vs Web3 Agent 差异、到 RAG 技术，建立 Web3Agent 的技术背景。

---

## 2.1 Web3 架构演进

| 阶段 | 时间 | 特点 |
|------|------|------|
| Web 1.0 | 1990-2004 | 静态网页，只读，用户是消费者 |
| Web 2.0 | 2004- | 用户生成内容，社交平台，数据中心化 (Facebook, YouTube) |
| Web3 | 现在 | 去中心化，区块链，智能合约，DAO 治理，用户拥有数据和资产 |

## 2.2 Web2 Agent vs Web3 Agent

| 维度 | Web2 Agent | Web3 Agent |
|------|-----------|------------|
| **操作对象** | HTML 元素 / 视觉界面 | 链上数据 / 智能合约 API |
| **执行方式** | 模拟用户点击（Web Navigation） | 推理区块链状态 + 编程式 API 调用 |
| **环境** | 中心化应用 | 去中心化、无信任生态 |
| **代表工作** | Mind2Web (HTML LLM), WebGum (多模态) | Web3Agent (本文) |
| **关键挑战** | 页面噪声、结构动态变化 | 交易感知、上下文敏感、安全设计 |

> 💡 **批读**: Web2 → Web3 Agent 的范式转变很关键：不再是"模拟点击"而是"推理链上状态 + 调用 API"。这意味着 Agent 需要理解区块链语义（合约地址、Gas、Nonce 等），而不仅仅是 DOM 结构。

### 与现有框架对比

| 系统 | 局限性 | Web3Agent 优势 |
|------|--------|---------------|
| Gorilla | 单步 API 代码合成，无多步规划 | 多步规划 + 动态环境适应 |
| API-Bank | 静态 Web2 任务分解，固定 API 集 | 实时状态变化 + Web3 API |
| Nguyen et al. | 简单任务，无跨链推理 | 复杂多步任务 + 鲁棒性机制 |

## 2.3 RAG 技术

- **经典 RAG**: 文档数据库 + 检索器 + 上下文注入
- **进阶**: A priori prompting（检索前引导）、A posteriori prompting（检索后验证）、Active RAG（不确定时重新查询）
- **检索方法**: BM25（稀疏）→ Dense retrieval（text-embedding-ada-002）
- **Web3Agent 的 RAG**: 模块化、指令驱动、领域分区语义 chunk、按模块路由检索

> 💡 **批读**: RAG 的领域分区设计（Operation / API / Error 三类 chunk）是本文的一个亮点，避免了通用 RAG 中不相关信息的干扰。

---

## 🔖 Section 总结

### 核心洞察
1. Web3 Agent 需要的能力与 Web2 Agent 根本不同：从界面操作 → 链上推理
2. 现有 LLM Agent 框架（Gorilla, API-Bank）不适用于动态、有状态的 Web3 场景
3. RAG 的领域定制化是提升 Agent 专业能力的关键手段
