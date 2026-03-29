# Eliza: A Web3 Friendly AI Agent Operating System

**作者**: Shaw Walters\*, Sam Gao\*, Shakker Nerd, Feng Da, Warren Williams, Ting-Chien Meng, Amie Chow, Hunter Han, Frank He, Allen Zhang, Ming Wu, Timothy Shen, Maxwell Hu, Jerry Yan  
**单位**: Eliza Labs, AI3 Labs, Heurist AI, GoPlus, Zero Gravity Labs, PipLabs, TownSquareLabs, MIT  
**arXiv**: [2501.06781](https://arxiv.org/abs/2501.06781) | cs.AI  
**发表时间**: 2025年1月（v1: Jan 12, v2: Jan 24）  
**代码**: [elizaOS/eliza](https://github.com/elizaOS/eliza)

---

## 一句话总结

Eliza 是第一个开源 Web3 友好型 AI Agent 操作系统，基于 TypeScript 构建，通过插件化模块设计无缝集成区块链操作（读写链上数据、与智能合约交互）和社交媒体，目标是让 Web3 应用的部署变得毫不费力。

---

## 核心贡献

1. **首个 Web3 原生 AI Agent OS**: 填补了 AI 与 Web3 交叉领域缺乏完整 Agent 框架的空白，支持 Ethereum/Solana/TON 等多条链
2. **可插拔模块化架构**: Core Runtime + 四大组件（Adapter/Character/Client/Plugin），开发者可自由扩展
3. **Character File 系统**: JSON 格式的 Agent 人格配置，定义行为风格、模型提供商、链上能力——像打造 J.A.R.V.I.S. 一样创建专属 Agent
4. **多层意图识别**: 结合符号化动作定义 + 上下文感知 + 向量记忆检索，实现跨平台一致的意图理解
5. **Web3 "图灵测试"框架**: 提出 Basic/Intermediate/Advanced 三级 Web3 Agent 能力评估标准

---

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                  ElizaOS Runtime                 │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │Character │  │Providers │  │  Evaluators   │  │
│  │  (人格)  │  │(上下文)  │  │  (记忆/目标)  │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │              Actions (动作层)             │    │
│  │  Buy/Sell | NFT Mint | Smart Contract    │    │
│  │  Image Gen | Transcription | Web Search  │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │              Plugins (插件层)             │    │
│  │  Solana | EVM | Coinbase | TEE | GOAT    │    │
│  │  Twitter | Discord | Telegram | Farcaster│    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
         ↕ Database Adapter（SQLite/PostgreSQL）
```

---

## 五大核心概念

| 概念 | 职责 |
|------|------|
| **Agents** | 自主交互的核心载体，管理消息/记忆/状态/动作 |
| **Character Files** | JSON 人格配置，定义 Agent 身份、能力、风格 |
| **Providers** | 实时数据注入（市场数据、钱包、情绪分析、时间） |
| **Actions** | 具体执行单元（下单、铸造 NFT、生成图片、转账） |
| **Evaluators** | 对话评估器，构建长期记忆、追踪目标、提取事实 |

---

## 支持范围

**链兼容性**: Ethereum, Solana, BNB, Arbitrum, Optimism, Aptos, TON, Near, Sui, zkSync Era, ICP, MultiversX 等

**模型提供商**: OpenAI, Anthropic (Claude), Meta Llama, Qwen 等

**社交平台**: Twitter, Discord, Telegram, Farcaster

**插件类型**:
- 媒体生成：Image/Video/3D 生成、NFT 铸造
- Web3 集成：Coinbase Suite、多链支持、GOAT 跨链工具包
- 基础设施：Browser/PDF/Speech/Transcription/Video Service、TEE 安全执行

---

## 评测结果

**GAIA Benchmark（通用 AI Agent 评测）**:
- 使用 3 个同质 Agent + 多数投票机制
- 达到中等水平（相比 GPT 系列 + 插件的 baseline）

**Web3 三级能力标准（论文提出）**:
- **Basic**: 创建钱包、转账/收款、与智能合约交互、接入社交平台
- **Intermediate**: Text2Video/3D、RAG 支持、音频转文字、隐私/安全插件
- **Advanced**: 自主规划和推理、从无序 API 自动生成执行流水线
- 当前 Eliza 处于 Basic → Intermediate 过渡阶段

---

## 批读总评

**优点**:
- 工程实用性极强，TypeScript 原生 + 模块化，Web3 开发者上手门槛低
- 开源社区驱动（GitHub Stars 极高），真实项目落地验证
- 插件生态丰富，多链/多模型/多平台支持覆盖面广
- 提出了 Web3 Agent "图灵测试" 评估框架，有一定学术价值

**不足**:
- 学术深度有限，更偏向系统报告而非研究论文
- 缺乏严格的 Web3 场景定量评测（GAIA 是通用测试，不专门针对 Web3）
- 安全性（私钥管理、链上误操作防护）讨论不够深入
- 与其他 Web3 Agent 框架（RIG、ZerePy 等）的对比主要依赖开发者主观打分

**定位**: 这是一篇**系统论文 + 开源框架介绍**，适合了解 Web3 AI Agent 生态工具，学术贡献相对有限，工程参考价值更高。

**评分**: ⭐⭐⭐ (3/5) — 作为开源项目的学术宣传文章，框架设计清晰，但学术严谨性不足

---

## 与同类工作对比

| 框架 | 特点 |
|------|------|
| **Eliza** | 开源，Web3 原生，TypeScript，社区活跃 |
| Web3Agent (ACM TWEB 2026) | 学术严谨，六模块流水线，侧重链上操作自动化 |
| RIG | Rust 实现，性能导向 |
| ZerePy | Python，简单易用 |
| Virtual | 闭源，商业产品 |
