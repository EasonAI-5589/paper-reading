[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Web3 与 AI 的交叉地带存在明显空白——没有一个框架能让开发者既享受 LLM 的认知能力，又轻松接入区块链生态。Eliza 定位为填补这个空白的开源 Agent OS，以 TypeScript 为基础，设计上优先考虑 Web3 开发者的使用习惯。

---

In the rapidly evolving landscape of AI, the advent of AI Agent, a system driven by large language models (LLMs) as its cognitive foundation, marks a significant milestone. This intelligent agentic system is not only capable of autonomously controlling and determining the execution paths under user instructions but also possesses the adaptability to navigate complex tasks with ease. The surge in capabilities of LLMs, coupled with the integration of diverse plugins such as RAG, text-to-image/video/3D tools, and more, has exponentially expanded the potential of AI Agents (i.e. AutoGPT, LangGraph, Camel, OpenAI Swarm and MiniChain). Their capabilities are advancing at a pace that is nothing short of remarkable, with new functionalities being added and refined on a daily basis.

> 💡 **开场背景**: 列举了 AutoGPT、LangGraph、Camel、OpenAI Swarm、MiniChain 五个框架作为现有 AI Agent 生态的代表。这些都是通用型框架，为后文提出 Web3 专用框架的必要性做铺垫。注意这段没有提具体的技术局限，只是在渲染"能力快速增长"的背景感。

However, despite the significant advancements in AI technology, a conspicuous gap persists at the confluence of AI and web3. The web3 domain is notably lacking an ideal agentic framework capable of seamlessly integrating web3 applications into its ecosystem, thereby fully unleashing the transformative potential of decentralized AI. This represents a critical void, as the successful integration of AI Agents with web3 technologies has the potential to revolutionize our engagement with decentralized applications and blockchain networks. By doing so, it could pave the way for a more equitable world where the benefits of technological progress are more broadly and fairly distributed among humanity.

> 💡 **问题定位**: "a conspicuous gap" 和 "critical void" 是论文的核心 claim。作者用了较为宏大的叙事（"更公平的世界"），在工程/系统论文中有些罕见。问题本身是真实存在的：通用 Agent 框架（LangChain 等）并非为链上操作设计，需要大量胶水代码才能接入 Web3。

In this paper, we introduce Eliza, a pioneering open-source web3-friendly agentic operating system designed to bridge this gap. Eliza is the first of its kind, offering a platform that makes the deployment of web3 applications not only possible but also effortless. We emphasize that every aspect of Eliza is crafted as a regular Typescript program, ensuring that it remains under the full control of its users while also providing seamless integration with web3 functionalities. This includes, but is not limited to, reading and writing blockchain data, interacting with smart contracts, and much more functionality.

Furthermore, we delve into how the key components of Eliza's runtime are implemented. We explain how these components are designed to work in harmony, enabling the framework to achieve stable performance while maintaining the flexibility required to adapt to the ever-changing demands of web3 applications. By solving the challenges of integrating AI with web3, Eliza stands at the forefront of a new era in technology, where the possibilities are as boundless as the imagination of its users.

> 💡 **Eliza 定位**: "the first of its kind" 是强 claim，但论文并没有提供系统性的相关工作对比来支撑（Section 4 的对比也主要依赖开发者打分）。TypeScript 原生这一点是真正的差异化：Web3 开发者本就在 JS/TS 生态里工作，不需要跨语言；相比之下 Python 系的框架（LangChain/AutoGPT）对他们来说有天然摩擦。
