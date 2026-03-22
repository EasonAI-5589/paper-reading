[← 返回 README](../README.md)

# 7 Discussion

## 📌 预览
Discussion 从三个角度展开：系统模块化设计的合理性与演进方向、高风险领域中的 Agent 推理安全问题、以及实际部署的用户体验考量。核心观点是：当前单 Agent + 模块化设计是探索性的，未来可能演进为多 Agent 协作；人类监督仍然不可或缺。

---

## 7.1 System Modularity

As described in Section 4, to address the procedural complexity and safety requirements of end-to-end Web3 automation, we adopt a six-module architecture, each responsible for a distinct and non-trivial stage in the agent execution lifecycle. This decomposition follows two design principles: (i) functional isolation: each module addresses a distinct reasoning or execution responsibility, thereby reducing coupling and facilitating independent optimization; and (ii) interface clarity: structured intermediate representations are exchanged between modules, making the data flow explicit and enabling safe integration with heterogeneous Web3 APIs.

From an empirical perspective, our ablation results suggest that this modular split is effective: removing key components such as the Instruction Chains Generator or the Previous Action Description Generator leads to notable declines in multi-step operation success, indicating that both explicit planning and contextual summarization are critical for reliable task execution. While the Controllable Calibration module yields smaller performance gains in a simulated setting, its role as a safeguard becomes increasingly important in irreversible, high-stakes blockchain contexts.

> 💡 **模块化 vs 端到端**: 作者正确地指出了模块化设计的两个核心优势——功能隔离和接口清晰。但也存在一个未讨论的权衡：模块间的串行依赖意味着**错误会传播**——如果 Intent Extraction 阶段解析错误，后续所有模块都会基于错误的输入运行。端到端系统（如直接让 LLM 一次性生成完整的 API 调用序列）虽然更容易出错，但错误不会被放大。模块化的优势在于可以在每个阶段进行检查和纠错，但代价是系统复杂度和延迟增加。

It is important to emphasize that this design is an exploratory attempt rather than a definitive solution [35]. The current split reflects a baseline architecture centered around a single-agent paradigm. As LLM capabilities evolve and multi-agent coordination becomes more viable, certain functional roles, currently realized as dedicated modules, may be replaced or restructured to better suit collaborative or task-specific workflows. For example, context maintenance could be merged into reasoning components, or calibration logic could be embedded within a more autonomous execution engine. Such evolutions may enable more optimized, domain-tailored designs while preserving the core principle of maintaining explicit and verifiable execution steps.

> 💡 **对未来演进的诚实态度**: "exploratory attempt rather than a definitive solution" 这句话在学术论文中难能可贵。作者没有过度声称自己的架构是最优的，而是承认这是一个 baseline 设计，未来可能被多 Agent 系统取代。这种谦逊使得论文更可信。参考 [35]（Shen et al. 2024 的 AI for Web3 综述）提供了更广阔的背景视角。

## 7.2 Agent Reasoning in High-stakes Domains

Blockchain-based financial operations represent a high-stakes application domain for AI-driven agents: transactions are irreversible, and a wide range of parameters, from slippage tolerance to gas configuration, can cause unintended outcomes if misconfigured [31]. Notably, such misconfigurations are not unique to AI; even human users regularly make errors in these environments. The operational context is further complicated by rapid market shifts, evolving smart contracts, and fluctuating transaction fees. These characteristics make the domain uniquely challenging for automated reasoning, but also present an opportunity: an AI agent, if properly designed, can systematically enforce procedural checks and prompt users for confirmation, potentially reducing the frequency of costly mistakes [8].

> 💡 **AI vs 人类的公平比较**: 作者指出即使人类用户也经常在 Web3 操作中犯错——这是一个重要的论据。AI Agent 不需要完美，只需要比人类用户犯更少的错误。[31]（Patlan et al. 2025）提出了对 Web3 Agent 的上下文操纵攻击，这是一个值得关注的安全威胁——恶意者可能通过注入虚假的链上数据来误导 Agent 做出错误决策。

In this article, Web3Agent follows a step-by-step execution paradigm common to other goal-oriented agents [7], but we designed it to aim for the goal of effectiveness in high-stakes contexts, and putting LLM reasoning under the human oversight. Rather than expecting an LLM to autonomously handle all contingencies, the system is designed to keep the human in the loop through explicit intermediate outputs, structured execution plans, and pre-execution validations. This approach aligns with the reality that, at present, LLMs alone cannot be fully trusted to execute irreversible Web3 operations without external checks. Looking ahead, improving reasoning reliability in such domains may require multi-agent collaboration, e.g., combining a dedicated risk assessment agent with a planning agent, alongside real-time transaction simulations and policy-guided confirmations. Producing verifiable reasoning traces, potentially anchored on-chain. Ultimately, however, human auditing and confirmation remain essential safeguards until AI systems demonstrate the robustness needed to operate independently.

It is important to note that Web3Agent deliberately avoids directly managing private keys or performing raw transaction signing. Instead, these critical functions are delegated to established and trusted wallet browser extensions (e.g., MetaMask, OKX), which handle authentication and key custody in accordance with existing user trust models. This boundary both reduces the attack surface of the agent and aligns the system with prevailing practices in Web3 security, though it also limits the scope of our present contribution. Beyond financial transactions, however, the same architectural principle, treating the agent as a mediator rather than a custodian, could naturally extend to emerging areas such as decentralized identifiers and verifiable credentials, where issues of security and delegation are equally central.

> 💡 **Agent 作为中介者而非保管者**: "treating the agent as a mediator rather than a custodian" 是一个精辟的设计哲学总结。这意味着 Web3Agent 只负责"思考"和"规划"，不负责"签名"和"持有"。这种分离确保了即使 Agent 被攻破，攻击者也无法直接窃取资金——因为资金的最终控制权仍在用户的钱包中。但这也意味着 Web3Agent 无法实现完全自动化的投资策略（如 DeFi yield farming bot），因为每笔交易都需要用户手动确认。向 DID（去中心化身份）和 VC（可验证凭证）的扩展方向很有远见。

## 7.3 Practical Deployment and User Experience

Beyond system design, deploying Web3 agents in practice also requires considering the broader user experience and developer integration workflows. For non-technical users, natural language interfaces can lower the entry barrier to Web3, but also risk over-simplifying critical parameters that determine financial safety. Thus, in our preliminary interface design as shown in Section 5, we ensured that key details, such as transaction costs or contract addresses, remain visible and interpretable. For developers, such agents may serve as higher-level abstractions that streamline testing or integration across multiple dApps, but adoption depends on whether the system integrates smoothly with existing wallets, APIs, and toolchains.

> 💡 **简化 vs 透明的张力**: 这段揭示了一个根本性的设计张力——自然语言界面在降低门槛的同时，可能隐藏了用户需要了解的关键信息（如 gas 费用、滑点、合约地址）。作者的解决方案是在 UI 中保持这些信息可见——但如何平衡"简单"和"完整"是一个 HCI 问题，需要用户研究来验证。7.3 整体比较简短，对开发者集成工作流的讨论也停留在表面。这一节可能是审稿过程中新增的，用于回应审稿人关于实际部署的提问。
