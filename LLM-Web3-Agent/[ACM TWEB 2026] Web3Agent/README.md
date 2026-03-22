# Web3Agent: Automating On-Chain Operations via Natural Language Interfaces

**作者**: Sizheng Fan, Tian Min  
**单位**: University of International Business and Economics (对外经济贸易大学), Keio University (庆应义塾大学)  
**期刊**: ACM Transactions on the Web, Volume 20, Issue 1 (February 2026)  
**链接**: [ACM DL](https://doi.org/10.1145/3777446)

---

## 一句话总结

提出 Web3Agent，首个 LLM 驱动的端到端 Web3 智能代理系统，通过自然语言分解用户指令为结构化链上操作流程，集成 RAG 增强推理 + 可控校准机制，在模拟环境中实现 94.1% 查询任务成功率和 80.3% 链上操作任务成功率。

---

## 核心贡献

1. **首个 Web3 AI Agent 框架**: 区别于 Web2 Agent（模拟浏览器点击），Web3Agent 直接与智能合约/区块链 API 交互，支持转账、Swap、Staking 等多步链上操作
2. **六模块流水线架构**: Intent Extraction → Instruction Chains → Previous Action Description → Action Prediction → Controllable Calibration → Executor，模块间通过 JSON 通信
3. **领域专用 RAG**: 三类语义分区 chunk（Operation / API / Error），按模块路由检索，显著提升参数恢复和错误处理能力
4. **可控校准层**: 执行前进行逻辑一致性 + 上下文可行性双重验证，防止 LLM 幻觉导致的链上误操作
5. **Mediator 设计哲学**: Agent 负责推理和规划，私钥管理和交易签名委托给 MetaMask 等可信钱包，降低攻击面

---

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
| [08 - Limitations & Conclusion](sections/08-limitations-conclusion.md) | 可扩展性、信任/人因、比较评估、结论 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| Intent Parsing Accuracy (IPA) | 93.9% |
| Parameter Retrieve Accuracy (PRA) | 89.6% |
| Query Task SR (Step / Task) | 96.8% / 94.1% |
| Operation Task SR (Step / Task) | 91.2% / 80.3% |
| 评估任务数 | 35 tasks × 5 utterances = 175 |
| 核心推理引擎 | GPT-4 |
| RAG Chunk 类型 | 3 (Operation / API / Error) |
| 支持链 | Ethereum, BNB, Arbitrum, Optimism, zkSync, Base, Solana, Sui 等 |

---

## 方法概览

```
用户: "Swap 1000 USDC to ETH on Ethereum"
          ↓
┌─────────────────────────────────┐
│  1. Chatbot & Intent Extraction │ ← Operation Chunk (RAG)
│     解析意图 + 提取参数          │
├─────────────────────────────────┤
│  2. Instruction Chains Generator│ ← Operation Chunk (RAG)  
│     分解为原子操作序列           │
├─────────────────────────────────┤
│  3. Previous Action Description │
│     历史动作的自然语言摘要       │
├─────────────────────────────────┤
│  4. Action Prediction           │ ← API + Error Chunk (RAG)
│     预测下一个链上动作           │
├─────────────────────────────────┤
│  5. Controllable Calibration    │
│     逻辑一致性 + 可行性验证      │
├─────────────────────────────────┤
│  6. Executor                    │
│     执行 API 调用 → 区块链       │
└─────────────────────────────────┘
          ↓
  区块链返回结果 → 反馈循环
```

---

## 批读总评

**优点**:
- 首个系统性的 Web3 AI Agent 框架论文，填补了领域空白
- 模块化设计清晰，每个模块的职责明确
- 领域专用 RAG 设计精巧（按模块路由检索，减少噪声）
- 论文写作诚实，不回避局限性（"exploratory attempt"、"feasibility-oriented"）
- Mediator 设计哲学（Agent 不碰私钥）在安全性上很明智

**不足**:
- 实验规模小（175 条数据），缺少和其他系统的端到端对比
- 模拟环境无法完全代表真实链上场景
- 6.1 System Implementation 过于简略，影响可复现性
- 安全性方面的贡献更多是展望而非实际实现
- 未进行用户研究（信任、可用性、认知负担）

**评分**: ⭐⭐⭐⭐ (4/5) — 作为开拓性工作值得认可，为 Web3 AI Agent 领域建立了 baseline 框架和评估范式
