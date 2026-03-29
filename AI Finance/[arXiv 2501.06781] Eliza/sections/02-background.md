[← 返回 README](../README.md)

# 2 Background

## 📌 预览
这一节通过三个具体使用场景来论证 Web3 原生 AI Agent 的必要性：去中心化交易、链上商业洞察、社交媒体信息处理。Eliza 的设计正是围绕这三类需求展开的。

---

**Decentralized Trading Bots**: At the heart of the crypto or web3 world lies the functionality of trading, such as transferring tokens and participating in Token Generation Events (TGEs), minting NFTs, and swapping tokens through decentralized exchanges (DEXs). With the proliferation of blockchain public chains like ETH, SOL, BASE and others, managing and operating one's investment portfolio over fragmented blockchains has become increasingly challenging. Individual investors are in dire need of a system to help manage their portfolios and conduct intelligent operations and trades. Platforms like GMGN, Dexscreener, and Bull X have filled this gap to a great extent, but for intermediate to advanced users with customized needs, the basic functionalities of these platforms may fall short.

> 💡 **场景一 - 去中心化交易**: GMGN、Dexscreener、Bull X 是现有的链上数据分析和交易工具，作者用它们来说明"已有方案对高级用户不够用"。这是一个合理的需求分析：现有平台是 UI 驱动的，缺乏可编程、可组合的 Agent 层。Eliza 的 Solana 插件（Section 7）正是对应这一场景的具体实现。

**Business Insights**: Secondly, blockchain data itself contains a wealth of crucial information for traders to make decisions. From simple metrics like changes in token holder counts, token prices, market capitalization, and Total Value Locked (TVL), to more advanced indicators such as the proportion of whale accounts, market-maker styles, and candlestick patterns, all can provide effective assistance to different types of cryptocurrency investors. The emergence of AI agents has brought hope for structuring the complex data on blockchains into high-quality insights to aid investors in making wiser decisions. However, extracting data intelligence is a challenging task, and using a general AI Agent framework for this purpose demands a high level of expertise from users. Therefore, there is an urgent need for a Web3-native AI Agent framework to achieve this.

> 💡 **场景二 - 商业洞察**: 链上数据本质上是结构化的公开数据，非常适合 RAG + LLM 分析。从持有人数量变化到鲸鱼账户占比，这些"链上信号"对 DeFi 交易者至关重要但难以实时处理。这是 Eliza Providers 模块的核心应用场景——Provider 可以实时注入这类动态数据到 Agent 上下文。

**Interaction**: Finally, for the Web3 industry, social media platforms like Twitter, Discord, and Farcaster are essential for connecting with users, obtaining cutting-edge information, and making trading decisions. As an increasing number of Key Opinion Leaders (KOLs) flock to these platforms, the information they disseminate becomes more complex and fragmented. Navigating this landscape to acquire organic insights and critically assess the credibility of KOLs is a universal challenge for traders. An exemplary Agent would enable users to sift through the vast information pool, distilling valuable intelligence without succumbing to information overload, and serving as a genuine intermediary in social media interactions with other users or agents.

> 💡 **场景三 - 社交媒体交互**: Web3 生态对 Twitter/Discord/Farcaster 的依赖程度远超传统行业——项目公告、KOL 推荐、社区情绪很大程度上影响价格走势。Eliza 把 Client（消息接入层）作为独立模块，天然支持这些平台。这是 Eliza 相比学术系统（如 Web3Agent）的一个实际优势：Web3Agent 没有涉及社交媒体集成，而 Eliza 把它作为一类核心 Client。

In consideration of the needs above, Eliza emerges as the premier open-source, web3-friendly AI Agent Operating System, boasting a modular design that empowers developers and users to tailor solutions to their specific requirements. By harnessing the robust capabilities of AI models and a variety of add-ons, Eliza democratizes access to advanced AI functionalities, significantly reducing the barrier to entry for the general public without the need for extensive coding expertise.

> 💡 **小结**: 三个场景（交易 / 洞察 / 社交）正好对应了 Eliza 的三个核心能力层（Actions / Providers / Clients）。这种需求驱动的结构设计逻辑是清晰的，但论文没有明确说出这种对应关系，需要读者自己归纳。
