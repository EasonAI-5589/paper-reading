[← 返回 README](../README.md)

# 7 Use Cases

## 📌 预览
两个具体实现示例：Solana 插件（链上操作的完整实现范例）和图像生成插件（多 Provider 验证的代码示例）。这一节是给开发者看的 —— 展示如何扩展 Eliza。

---

### 7.1 Solana Plugin Example

The Solana plugin provides functionality for interacting with the Solana blockchain, including token management, swapping, and trust score evaluation.

**Core Features**:
- Token Management (TokenProvider)
- Wallet Integration (WalletProvider)
- Trust Score Evaluation (TrustScoreManager)
- Token Swapping
- FOMO and PumpFun Integration

**Key Components**:

`TokenProvider` — 处理 token 相关操作（计算购买金额、获取 DexScreener 数据、查询价格）

`WalletProvider` — 管理钱包交互（获取投资组合总值，包含 USD 和 SOL 计价）

`TrustScoreManager` — 评估 token 和推荐者的信任评分（综合 riskScore + consistencyScore）

**Plugin Actions（注意 similes 字段）**:
```typescript
export const executeSwap: Action = {
  name: "EXECUTE_SWAP",
  similes: ["SWAP_TOKENS", "TOKEN_SWAP", "TRADE_TOKENS"],
  handler: async (...) => {
    const trustScore = await runtime.getProvider('trustScore').evaluateSwap(params);
    if (trustScore < runtime.getMinimumTrustThreshold()) {
      return false;  // 信任评分不足，拒绝执行
    }
    return true;
  }
};
```

> 💡 **TrustScore 机制的设计亮点**: Solana 插件引入了 Trust Score 作为执行 Action 的前置条件——这是 Web3 场景下特有的安全设计。对于一个 swap 操作，在真正执行前先评估 token 的信任度（防止 rug pull）和推荐者的历史可信度，是非常务实的风控机制。但论文没有详细说明 TrustScore 的计算逻辑（什么数据、什么权重、阈值如何设定）——这些细节对实际使用至关重要。FOMO 和 PumpFun 集成暗示 Eliza 对 Solana 生态的 meme coin 交易场景有专门支持，这是一个非常具体的 use case。

**Configuration**（环境变量要求）:
- `WALLET_SECRET_SALT`, `SOL_ADDRESS`, `SLIPPAGE`, `RPC_URL`, `HELIUS_API_KEY`, `BIRDEYE_API_KEY`

> 💡 **配置要求揭示的依赖**: Helius（Solana RPC 服务）和 Birdeye（链上数据 API）是 Solana 生态两个重要的基础设施服务。Eliza 对它们的依赖意味着生产部署需要购买这些 API 的使用权，不是完全的免费使用。这是实际部署时需要考虑的成本因素。

---

### 7.2 Advanced Implementation Example: Image Generation

The image generation plugin demonstrates how to implement multi-provider validation:

```typescript
const imageGeneration: Action = {
  name: "GENERATE_IMAGE",
  similes: ["IMAGE_GENERATION", ..., "MAKE_A"],
  validate: async (runtime: IAgentRuntime, _message: Memory) => {
    const anthropicApiKeyOk = !!runtime.getSetting("ANTHROPIC_API_KEY");
    const falApiKeyOk = !!runtime.getSetting("FAL_API_KEY");
    return anthropicApiKeyOk || ... || falApiKeyOk;
  }
};
```

> 💡 **插件设计模式的示范价值**: 图像生成插件展示了 Eliza 插件的标准模式：`name`（唯一标识）→ `similes`（意图触发词）→ `validate`（前置条件检查）→ `handler`（执行逻辑）。`validate` 用 OR 逻辑检查多个 Provider 的 API Key，只要有一个可用就允许执行——这是"降级策略"的体现，提高了系统可用性。
>
> 这一节对开发者的参考价值很高，但对学术读者来说更像是文档，而非研究贡献。整个 Section 7 是给想要基于 Eliza 开发的工程师看的，而不是为了支撑论文的学术主张。

---

## 总结性批读

Eliza 的 Use Cases 部分清楚地展示了框架的工程质量：接口设计统一、代码结构清晰、扩展机制合理。但从学术角度看，这两个 case 都没有量化评测（执行成功率、延迟、错误处理覆盖率等），只是展示了"能做什么"而非"做得多好"。这与整篇论文的风格一致：工程文档水准 > 学术论文水准。
