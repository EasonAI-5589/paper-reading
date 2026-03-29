[← 返回 README](../README.md)

# 4 Related Works

## 📌 预览
从两个维度梳理相关工作：插件（内部推理增强 X-of-T 系列 + 外部知识集成 RAG/AIGC）和框架（工业级 vs 学术级，Web3 专用框架对比）。Eliza 在 Web3 友好度、模型支持、社交平台集成上声称优于其他开源框架。

---

As an AI Agent operating system focusing on web3 and social media, we aim to define our position and differentiate ourselves from both industrial AI Agent frameworks (i.e. Bedrock (AWS), Swarm (OpenAI), and smolagent (Huggingface)) and academic-oriented projects.

### 4.1 Plugins

Along with the rapid growth of off-the-shelf plugins, the agent's enhancement can be categorized into two principle forms: Internal and External.

**Internal enhancement** — tapping into the full potential of the LLM itself: Representative works include Chain-of-Thoughts (CoT), Zero-shot CoT, Tree-of-Thoughts (ToT), Graph-of-Thoughts (GoT), and Layer-of-Thoughts (LoT). CoT introduced step-by-step explanations, ToT allowed branching to explore multiple solutions, and GoT connected reasoning pathways in a network. LoT (Oct 2024) is a hierarchical reasoning AI that organizes thoughts into layers for structured problem-solving.

> 💡 **内部增强 - X-of-T 系列**: CoT→ToT→GoT→LoT 的演进轨迹是 LLM 推理增强的主线。这些技术与 Eliza 的直接关联并不强——Eliza 不是一个推理增强框架，这部分更多是背景综述。值得注意的是 LoT 是 2024 年 10 月的新工作，说明作者在写作时比较关注最新进展。

**External enhancement** — integrating knowledge from various sources: This includes Retrieval Augmented Generations (RAGs), vector databases, and web searches. Furthermore, as AI-Generated Content (AIGC) matures, the ability to convert text into images, videos, and 3D models opens up new possibilities for AI agents.

Eliza offers robust support for a variety of blockchain plugins, encompassing everything from on-chain transactions to Trusted Execution Environments (TEEs). The comprehensive web3 toolkit is designed to be user-friendly and easily extensible. Additionally, the integration of social media support broadens the range of application scenarios.

> 💡 **外部增强 - RAG + AIGC**: Eliza 的 Plugin 系统在这两个方向都有覆盖——RAG 通过 Provider 机制实现外部知识注入，图像/视频/3D 生成通过 Media Generation Plugin 支持。TEE（可信执行环境）的支持是 Web3 场景特有的安全需求，值得关注。

### 4.2 Frameworks

AI agent frameworks flourished at the emergence of ChatGPT in 2023, where AutoGPT, LangGraph (LangChain) and Camel released their first versions. For web3 industry, a series of web3-oriented AI Agent frameworks start to emerge:

- **Open source**: RIG, G.A.M.E, ZerePy, Heurist, REI
- **Close source**: Virtual

As shown in the developer survey (50+ AI researchers and senior blockchain developers), Eliza outperforms other frameworks in terms of: model providers, chain compatibility, functionality and social media.

> 💡 **框架对比的局限性**: 论文的 Web3 框架对比（Figure 2）完全依赖"50+ 开发者主观打分"，没有客观基准测试。在同行评审的学术论文中，这种评估方式是很弱的——主观打分受到认知偏差、框架熟悉程度、以及样本选择偏差的影响。RIG（Rust 实现，性能导向）、ZerePy（Python，简单易用）和 Eliza（TypeScript，Web3 原生）各自服务不同的用户群体，"哪个更好"本身就不是一个有普遍答案的问题。这部分更应该被理解为"Eliza 的开发者认为 Eliza 在某些维度更强"，而非客观事实。
