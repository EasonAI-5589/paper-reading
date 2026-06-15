# Trade-R1

**论文**: Trade-R1: Bridging Verifiable Rewards to Stochastic Environments via Process-Level Reasoning Verification<br>
**发表**: arXiv 预印本<br>
**arXiv**: [2601.03948](https://arxiv.org/abs/2601.03948)<br>
**权重/代码**: 截至 2026-06-15 未检索到作者官方公开资产

面向随机金融决策的过程验证框架。Trade-R1 将长金融文档上的推理验证重构为结构化 RAG，检查检索证据、推理链和最终决策之间的三角一致性，再用该语义信号过滤噪声市场收益。论文比较固定效应语义奖励 FSR 与动态效应语义奖励 DSR，并报告 DSR 在跨市场泛化和推理一致性上更优。

**与 FinChain 的关系**: FinChain 的执行 trace 适合确定性数值推理，Trade-R1 则处理无法仅凭最终收益判定推理质量的随机市场环境。两者可组合成“可执行步骤验证 + 证据一致性 + 市场结果”的混合过程奖励。

> 当前仅完成基础收录。
