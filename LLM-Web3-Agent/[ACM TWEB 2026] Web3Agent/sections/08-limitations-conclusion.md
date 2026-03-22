[← 返回 README](../README.md)

# 8 Future Vision and Limitations

## 📌 预览
作者从可扩展性、信任/人因、比较评估三个维度讨论了局限性和未来方向，态度诚恳，没有回避核心问题。

---

## 8.1 Scalability and Generalization

The current single-LLM, modular architecture of Web3Agent offers functional separation and empirical reliability under controlled evaluation, but its scalability and ability to generalize across heterogeneous Web3 environments remain open challenges. On the scalability side, executing long dependency chains or servicing multiple concurrent requests may push the limits of context length, latency, and API throughput. On the generalization side, our chunk-based retrieval design, while effective for supported dApps, implicitly assumes that tasks follow recognizable patterns. This assumption can limit performance when facing previously unseen dApp structures, non-standard wallet flows, or protocols with idiosyncratic interaction models. More broadly, LLM-based agents in most domains require an architecture that can ingest and adapt to new knowledge—whether through RAG [19], fine-tuning [16], or other adaptive mechanisms, and many of these techniques could be transferred and adapted to better serve Web3-specific agents.

> 💡 **可扩展性的核心矛盾**: 当前 Web3Agent 的 RAG 依赖于预构建的 chunk store（Operation/API/Error），这意味着支持新的 dApp 或跨链协议需要手动添加新的 chunk。在 Web3 生态系统快速演化的背景下（新 DEX、新跨链桥、新 DeFi 协议每周都在出现），这种静态的知识库难以跟上。作者提到的"adaptive retrieval strategies"和"schema abstraction"是正确的方向，但实现难度不小。

Addressing these challenges is a non-trivial system design problem, and multiple directions may be worth exploring. One possible avenue is a more specialized multi-agent paradigm, where distinct components focus on dedicated roles such as risk assessment, bridge selection, and chain-aware planning. Another is to evolve the interface layer with schema abstraction and self-describing API adapters, enabling automated alignment with novel contract interfaces and cross-chain protocols. However, the viability and tradeoffs of these approaches depend on factors such as coordination overhead, fault isolation, and maintainability, which require careful empirical investigation.

Future research on improving scalability and generalization in Web3 agents could proceed along several complementary directions. First, designing adaptive retrieval strategies that dynamically adjust to new dApp schemas and protocol changes could help reduce brittleness. Second, exploring hybrid architectures that combine symbolic reasoning for transaction constraints with neural planning for high-level intent understanding may improve robustness in multi-chain settings. Third, understanding the human factors in scaling, such as how interface design supports oversight when an agent interacts with unfamiliar protocols, remains an open question.

> 💡 **Symbolic + Neural 混合架构**: 这个方向特别有潜力。区块链交易有很强的形式化约束（如"转账金额 ≤ 余额"、"必须先 approve 再 swap"），这些约束可以用符号推理硬编码，而高层的意图理解和多步规划则交给 LLM。这种混合方式可以同时获得符号系统的可靠性和神经网络的灵活性。

## 8.2 Trust and Human Factors

This work primarily focuses on the system structure and execution logic of Web3Agent. While we have implemented a user interface to facilitate interaction, it was not designed or evaluated as a core contribution of this article. As such, we have not systematically examined the human factors that influence how users perceive, trust, and supervise an AI agent operating in high-stakes Web3 environments.

In security-critical domains like blockchain, human supervision is a central safeguard. Even the advanced reasoning modules and pre-execution checks cannot fully eliminate the possibility of LLM errors, adversarial inputs, or unforeseen protocol changes. A human-in-the-loop architecture, where the agent produces transparent intermediate outputs and seeks explicit confirmation before committing to irreversible actions, remains the most reliable final validation barrier. This design principle aligns with both security best practices and established HCI research on trust calibration in automation [17].

> 💡 **用户信任研究的缺失**: 作者承认没有进行用户研究。对于一个 HCI 导向的会议论文来说这可能是硬伤，但 ACM TWEB 更偏技术系统，所以这个缺失是可以接受的。不过，未来如果要证明 Web3Agent 的实际价值，用户信任和可用性研究是必须的——用户是否真的理解 Agent 在做什么？他们是否会盲目信任 Agent 的建议？过度信任（automation complacency）在高风险金融场景中是一个严重的隐患。

Future research could explore how Web3 users form and adjust trust in such agents, and how interface design can support effective oversight without creating excessive cognitive burden. Controlled user studies could assess dimensions such as perceived safety, task efficiency, and mental workload when using an AI-assisted Web3 interface. Participatory design methods may help uncover user-specific needs, risk tolerances, and preferred verification mechanisms, enabling the co-creation of interaction patterns that balance automation with user control. Other promising directions include longitudinal field deployments to observe trust dynamics over time [24], scenario-based evaluations to probe decision-making under varying risk levels, and experimental manipulations of feedback granularity to identify the most effective cues for prompting human intervention.

## 8.3 Comparative Evaluation and Baselines

This work has not yet performed a systematic end-to-end comparison against alternative Web3 automation approaches, such as general-purpose LLM pipelines, scripted rule-based frameworks (e.g., Web3.py), or other LLM-agent toolkits. Our decision to focus on internal ablation studies was motivated in part by the absence of an agreed-upon benchmarking protocol for this emerging area, and by the difficulty of ensuring consistent execution conditions across heterogeneous systems. In practice, differences in prompt engineering, retrieval infrastructure, transaction execution environments, or even hardware of running the LLM make it hard to fully replicate the conditions of other systems [16], and thus to conduct fair head-to-head comparisons.

> 💡 **缺少 baseline 对比**: 这是本文最大的实验局限——没有和其他系统做端到端对比。虽然作者给出了合理的解释（没有标准化的 benchmark、不同系统的执行条件难以统一），但缺少和 Web3.py 脚本、LangChain Agent、AutoGPT 等工具的对比，使得读者难以判断 Web3Agent 的相对优势。即使不能做完美公平的对比，至少可以在相同任务集上跑一个 LangChain + Web3.py 的 baseline 来提供参考点。

Nevertheless, comparative evaluation remains an important next step. Future work could develop a unified evaluation harness with a standardized task set, shared interface definitions, debugging [10], and deterministic task replays. This would allow multi-tiered benchmarking: offline simulation to measure reasoning accuracy (e.g., step success rate, task success rate, failure attribution), testnet deployments to evaluate operational robustness, and, where safe, controlled mainnet trials to assess performance under live network conditions.

---

# 9 Conclusion

In this article, we presented Web3Agent, a feasibility-oriented system that explores the integration of LLM-based agents with Web3 infrastructures for end-to-end on-chain task automation. By bridging natural language understanding and structured blockchain execution, Web3Agent enables users to initiate complex Web3 operations — such as asset transfers, smart contract interactions, and DeFi transactions — through intuitive conversational interfaces. The modularized system architecture collectively forming a closed feedback loop between user intent and decentralized execution. We demonstrated the potential of LLM agents to abstract away operational complexity, reduce cognitive overhead, and support structured control over decentralized workflows. We believe that Web3Agent serves as a stepping stone toward more transparent, user-aligned, and intelligent interaction paradigms for the next generation of decentralized systems.

> 💡 **结论批读**: 结论的措辞很谨慎——"feasibility-oriented"、"explores"、"stepping stone"。这和很多论文里夸大贡献的倾向形成鲜明对比。Web3Agent 的核心价值确实是在于**证明了 LLM Agent 在 Web3 场景中的可行性**，而非提供一个生产就绪的解决方案。作为首篇在 ACM 顶级期刊上发表的 Web3 AI Agent 系统论文，它的定位是开拓性的，为后续工作（更好的 benchmark、更强的安全保障、更复杂的跨链操作）铺设了基础。
