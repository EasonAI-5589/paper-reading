[← 返回 README](../README.md)

# 3 Design Principles

## 📌 预览
三条设计哲学定义了 Eliza 的风格：Web3 开发者优先（TypeScript 原生）、插件化模块解耦、以及"粗糙但可用胜于精巧但复杂"的工程实用主义。

---

Eliza is a powerful multi-agent simulation framework designed for creating, deploying, and managing autonomous AI agents. It is built using TypeScript and is capable of interacting across multiple platforms. Numerous projects have been developed based on our framework.

Eliza's success is attributed to its integration of the strong demands of web3 into a design that balances utility and ease of use. There are three main principles behind our choices:

**Put Web3 Developers First**: Since web3 primarily utilizes JavaScript/TypeScript, which is the dominant language for web development, Eliza allows developers to easily integrate blockchain functionality into existing web applications and build decentralized applications (dApps) by leveraging familiar tools and frameworks. Eliza should be a first-class member of that ecosystem. It adheres to the commonly established design goals of keeping interfaces simple and consistent, ideally with one idiomatic way of doing things.

> 💡 **原则一 - Web3 开发者优先**: 这是 Eliza 差异化的核心。Web3 生态天然是 JS/TS 的地盘（前端、钱包、dApp 全是 JS），而 AI 框架（LangChain、AutoGPT）大多是 Python。Eliza 选择 TypeScript 意味着 Web3 开发者不需要学新语言、不需要跨语言调用，可以直接在熟悉的工具链里构建 Agent。"one idiomatic way of doing things" 是 Go 语言哲学的影子，强调接口统一性。

**Pluggable Modular Design**: Eliza decouples its structure into a core Runtime along with four key components: Adapter (data), Character (agent personality), Client (message interaction), and Plugin (universal functionality). This design allows developers or users to freely add their own plugins, clients, characters, and adapters as they wish, without worrying about the details within the core Runtime. It makes extension incredibly easy and paves the way for Eliza to support the most model providers, platform integrations, chain compatibilities, and highly equipped functions.

> 💡 **原则二 - 插件化模块**: Core Runtime + 四大组件的解耦设计是 Eliza 可扩展性的基础。Adapter（数据层）/ Character（人格层）/ Client（接入层）/ Plugin（功能层）四个维度各自独立，开发者可以只扩展其中一个维度而不影响其他部分。这在工程上是合理的——社区贡献者可以独立维护某个链的插件（如 Solana Plugin）而不需要了解核心 Runtime 的细节。

**Roughness is better**: Given limited engineering resources and all else being equal, keeping Eliza's internal implementation simple saves time for adding features, adapting to new situations, and keeping pace with advancements in AI and Web3. Therefore, it is better to have a simple but slightly incomplete solution than a comprehensive yet complex and hard-to-maintain design.

> 💡 **原则三 - "粗糙胜于复杂"**: 这是一个工程实用主义的哲学，和学术论文追求完备性的倾向相反。这条原则很诚实——它承认 Eliza 不追求完美，而是追求可维护性和迭代速度。在 AI 和 Web3 双双快速演进的背景下，这个选择是务实的：一个简单但能跑的系统，比一个设计精美但难以修改的系统更有生命力。这也解释了为什么论文本身有些地方较为粗糙（缺乏严格评测）。
