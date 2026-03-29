[← 返回 README](../README.md)

# Eliza: A Web3 Friendly AI Agent Operating System

Shaw Walters\*, Sam Gao\*, Shakker Nerd, Feng Da, Warren Williams, Ting-Chien Meng, Amie Chow, Hunter Han, Frank He, Allen Zhang, Ming Wu, Timothy Shen, Maxwell Hu, Jerry Yan

Eliza Labs, AI3 Labs, Heurist AI, GoPlus, Zero Gravity Labs, PipLabs, TownSquareLabs, MIT

AI Agent, powered by large language models (LLMs) as its cognitive core, is an intelligent agentic system capable of autonomously controlling and determining the execution paths under user's instructions. With the burst of capabilities of LLMs and various plugins: i.e. RAG, text-to-image/video/3D and etc, the potential of AI Agents has been vastly expanded, with their capabilities growing stronger by the day. However, at the intersection between AI and web3, there is currently no ideal agentic framework that can seamlessly integrate web3 applications into AI agent functionalities. In this paper, we propose Eliza, the first open-source web3-friendly Agentic frameworks that make the deployment of web3 applications effortless. We emphasize that every aspect of Eliza is a regular Typescript program under the full control of its user, and it seamlessly integrates with web3 (i.e. reading and writing blockchain data, interacting with smart contracts and etc). Furthermore, we show how stable performance is achieved through the pragmatic implementation of the key components of Eliza's runtime. Our code is publicly available at [elizaOS/eliza](https://github.com/elizaOS/eliza).

> 💡 **摘要批读**: 摘要结构标准：背景（LLM Agent 能力爆发）→ 问题（AI × Web3 缺乏理想框架）→ 方案（Eliza）→ 亮点（TypeScript 全控、无缝 Web3 集成、稳定性实现）→ 开源声明。值得注意的是，作者用了 "effortless" 来描述部署体验，这是一个偏工程营销的措辞，而非学术论文常见的精准表达。摘要没有提具体指标（如评测结果数字），侧面印证这篇论文更偏系统介绍而非性能导向。"every aspect of Eliza is a regular Typescript program under the full control of its user" 这句话隐含了对其他框架（如 Python 系的 LangChain、AutoGPT）的对比定位：Eliza 面向的是 Web3 原生的 TypeScript 开发者生态。
