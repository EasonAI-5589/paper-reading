[← 返回 README](../README.md)

# 5 User Interface

## 📌 预览
Web3Agent 的 Web 前端由三部分组成：Chatbot（自然语言交互入口）、Operation Log（实时结构化日志）、Operation Flow（可视化操作流程图）。系统通过浏览器扩展连接加密钱包，不直接管理私钥。

---

To ensure secure and user-controllable interactions in the context of financial Web3 operations, we designed a user interface that foregrounds real-time transparency and allows for immediate user intervention when needed. On-chain activities often entail non-trivial risks due to their irreversible nature and sensitivity to parameter configuration. Consequently, providing users with detailed inspection and traceability throughout the interaction is essential for maintaining trust, reducing friction, and supporting safe execution.

![Figure 8](../images/001bf9bf429fcbe13142de90cfd39c1a91f9cffd6675a3769e6e5b6c5fdcf6df.jpg)

*Fig. 8. The web-based user interface for Web3Agent and the connection with crypto wallet.*

As shown in Figure 8, to address these needs, we developed a Web-based application powered by Vue.js and crypto-js, which interfaces with the user's crypto wallet via a third-party browser extension, without directly operate user's private key or signature. The interface is composed of three primary components: the Chatbot, the Operation Log, and the Operation Flow.

> 💡 **不管理私钥的设计决策**: 这是一个非常重要的安全边界——Web3Agent 不直接接触用户的私钥，而是通过 MetaMask 等浏览器扩展来处理签名。这意味着每次链上操作最终都需要用户在钱包中确认，保证了 human-in-the-loop。但这也限制了系统的自动化程度——用户仍然需要手动点击"确认交易"，无法实现完全无人值守的批量操作。这是安全性和便利性之间的权衡。

At the center of the UI lies the Chatbot component, which serves as the primary entry point for users to interact with the Web3Agent through natural language. Beyond processing on-chain operations, we further enhance the LLM with prompt engineering techniques to maintain task focus, discourage off-topic dialogue, and reduce the risk of instruction misalignment that could compromise transaction safety. To the left, the Operation Log offers a real-time stream of structured outputs from the LLM, including the inferred user intent, the parsed operation sequence, and any relevant parameters. This log is updated with each LLM response, enabling users to trace decision-making and inspect intermediate parsing results in a transparent and verifiable manner. On the right, the Operation Flow provides a visual and interactive representation of the operation sequence. Once the sequence is generated, each step is rendered as a node within a directed graph. Users can inspect the order of execution, review individual parameters, and modify the structure if needed. Upon confirmation, the user can trigger execution with the orange "play" button; the system will then highlight each node during runtime to reflect ongoing progress and provide immediate feedback.

> 💡 **三面板布局的可用性**: Chatbot + Operation Log + Operation Flow 的三面板布局提供了三种不同粒度的信息视图：自然语言（用户友好）、结构化日志（可审计）、可视化流程图（直观）。Operation Flow 中的有向图展示和"play button"交互设计特别有价值——用户可以在执行前检查和修改操作流程，类似于 CI/CD pipeline 的可视化。但作者在 Section 8.2 承认 UI 不是本文的核心贡献，也没有做过用户研究来验证其可用性。这在 ACM TWEB（一个关注 Web 技术的期刊）上发表时可能会被审稿人质疑。

Together, these interface components form a cohesive interaction loop between natural language commands, structured representations, and visual feedback. This design ensures that users maintain oversight and agency during high-stakes Web3 operations, while also enabling more intuitive and trustworthy interactions with LLM-based agents.

> 💡 **缺失的评估**: UI section 完全是描述性的——没有可用性测试、用户研究或 A/B 实验。对于一个号称解决"操作复杂性"问题的系统来说，仅展示界面截图而不验证其确实降低了用户的操作难度，论证上是不完整的。不过考虑到本文的核心贡献是后端的 Agent 架构而非前端，这种取舍可以理解。未来工作中加入用户研究（如 SUS 量表、任务完成时间对比）会显著增强论文的说服力。
