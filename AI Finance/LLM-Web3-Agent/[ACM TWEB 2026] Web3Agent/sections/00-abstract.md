[← 返回 README](../README.md)

# Web3Agent: Automating On-Chain Operations via Natural Language Interfaces

SIZHENG FAN, University of International Business and Economics, Beijing, China
TIAN MIN, Faculty of Science and Technology, Keio University, Yokohama, Japan

Recent advances in large language models (LLMs) have enabled the emergence of intelligent agents capable of performing complex multi-step tasks across various domains. In parallel, the growth of Web3 has introduced a decentralized web infrastructure, yet remains largely inaccessible to non-technical users due to operational complexity, fragmented information, and security risks. In this article, we present Web3Agent, an AI agent system that integrates LLM-based interaction with blockchain environments to enable language-driven on-chain operations. Web3Agent automatically decomposes user instructions into structured workflows, dynamically queries blockchain data and APIs, and performs multi-step operations such as asset transfers, token swaps, and smart contract execution. Web3Agent incorporates real-time inspection, error handling, and interaction transparency across its operation log, and flow visualization components. We evaluate the system and perform ablation study with customized dataset in a simulated environment, demonstrating its feasibility in orchestrating complex Web3 tasks and highlighting implications for agent-based abstraction in decentralized systems.

> 💡 **摘要批读**: 这篇摘要的结构非常标准——背景（LLM + Web3 两条线）→ 问题（非技术用户的准入门槛）→ 方案（Web3Agent）→ 评估。值得注意的是，作者用了 "feasibility" 而非 "effectiveness" 来定位贡献，这是一个很诚实的选择：系统是在模拟环境中评估的，而不是真正上链跑的端到端测试。摘要中提到的三个核心能力——指令分解、动态查询、多步操作——恰好对应了后文的 Instruction Chains Generator、RAG 检索、和 Executor 模块。另外，"interaction transparency" 的强调暗示了作者对 Web3 场景下 AI Agent 安全性和可审计性的重视，这在后文 Discussion 部分会展开讨论。

CCS Concepts: • Human-centered computing → Human computer interaction (HCI); • Computing methodologies → Natural language processing; • Information systems → World Wide Web;

Additional Key Words and Phrases: Web3, blockchain, intelligent virtual assistants, large language models, process automation

ACM Reference Format:
Sizheng Fan and Tian Min. 2026. Web3Agent: Automating On-Chain Operations via Natural Language Interfaces. ACM Trans. Web 20, 1, Article 9 (February 2026), 27 pages. https://doi.org/10.1145/3777446
