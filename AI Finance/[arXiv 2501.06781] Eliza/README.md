# Eliza: A Web3 Friendly AI Agent Operating System

**作者**: Shaw Walters\*, Sam Gao\*, Shakker Nerd, Feng Da, Warren Williams, Ting-Chien Meng, Amie Chow, Hunter Han, Frank He, Allen Zhang, Ming Wu, Timothy Shen, Maxwell Hu, Jerry Yan  
**单位**: Eliza Labs, AI3 Labs, Heurist AI, GoPlus, Zero Gravity Labs, PipLabs, TownSquareLabs, MIT  
**arXiv**: [2501.06781](https://arxiv.org/abs/2501.06781) | cs.AI  
**发表时间**: 2025年1月（v1: Jan 12, v2: Jan 24）  
**代码**: [elizaOS/eliza](https://github.com/elizaOS/eliza)

---

## 一句话总结

Eliza 是第一个开源 Web3 友好型 AI Agent 操作系统，基于 TypeScript 构建，通过插件化模块设计无缝集成区块链操作与社交媒体，让 Web3 应用的 Agent 化部署门槛大幅降低。

---

## 核心贡献

1. **首个 Web3 原生 AI Agent OS**: 填补 AI × Web3 交叉领域缺乏完整 Agent 框架的空白，支持 Ethereum/Solana/TON 等多条链
2. **可插拔模块化架构**: Core Runtime + 四大组件（Adapter/Character/Client/Plugin），开发者可自由扩展，不触碰核心
3. **Character File 系统**: JSON 格式 Agent 人格配置，定义行为风格、模型提供商、链上能力
4. **多层意图识别**: 符号化动作定义 + 上下文感知 + 向量记忆检索，跨平台一致意图理解
5. **Web3 "图灵测试"框架**: 提出 Basic/Intermediate/Advanced 三级 Web3 Agent 能力评估标准

---

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题定义、框架概述、开源声明 |
| [01 - Introduction](sections/01-introduction.md) | Web3 三大需求背景 + Eliza 的定位与设计哲学 |
| [02 - Background](sections/02-background.md) | 去中心化交易机器人、链上商业洞察、社交媒体交互三大场景 |
| [03 - Design Principles](sections/03-design-principles.md) | Web3 开发者优先 + 插件化模块设计 + "粗糙胜于复杂"哲学 |
| [04 - Related Works](sections/04-related-works.md) | Plugins（内部/外部增强）+ 框架对比（RIG/ZerePy/G.A.M.E 等） |
| [05 - ElizaOS](sections/05-elizaos.md) | 五大核心概念：Agent / Character / Providers / Actions / Evaluators + 意图识别 + 插件架构 |
| [06 - Benchmarks](sections/06-benchmarks.md) | GAIA 通用评测 + Web3 图灵测试三级框架 |
| [07 - Use Cases](sections/07-use-cases.md) | Solana 插件实现 + 图像生成插件示例 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| GAIA Benchmark（3 Agent 多数投票）| 中等水平（相比 GPT 系列 baseline）|
| 支持链 | ETH, SOL, BNB, Arbitrum, Optimism, TON, Near, Sui, zkSync Era, ICP 等 |
| 支持模型 | OpenAI, Anthropic, Meta Llama, Qwen 等 |
| 支持社交平台 | Twitter, Discord, Telegram, Farcaster |
| 代码语言 | TypeScript |
| 论文页数 | 20 pages, 5 figures |

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

## 批读总评

**优点**:
- 工程实用性强，TypeScript 原生，Web3 开发者上手门槛低
- 开源社区驱动，真实项目落地验证，插件生态丰富
- 提出 Web3 Agent "图灵测试"评估框架，有学术参考价值
- 模块化设计清晰，核心与扩展解耦良好

**不足**:
- 更偏系统报告而非研究论文，学术深度有限
- 缺乏严格的 Web3 场景定量评测（GAIA 是通用基准）
- 安全性讨论（私钥管理、链上误操作防护）不够深入
- 与其他 Web3 框架对比主要依赖开发者主观打分，客观性存疑

**定位**: 系统论文 + 开源框架介绍，工程参考价值 > 学术贡献

**评分**: ⭐⭐⭐ (3/5) — 框架设计清晰，开源影响力大，但学术严谨性不足
