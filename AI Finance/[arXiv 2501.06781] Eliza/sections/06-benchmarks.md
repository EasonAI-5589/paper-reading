[← 返回 README](../README.md)

# 6 Benchmarks

## 📌 预览
评测分两部分：通用 AI Agent 能力（GAIA Benchmark）和论文自己提出的 Web3 "图灵测试"三级框架。GAIA 结果仅达到"中等水平"，Web3 评测更多是能力框架定义而非实测数据。

---

### 6.1 General AI Agent Benchmark

GAIA is a benchmark designed to evaluate the general capabilities of AI agents in solving real-world problems. Successfully answering GAIA questions requires multiple skills: logical reasoning, multi-modal processing, web browsing, and tool utilization.

**Eliza's implementation**: 3 homogeneous agents + self-consistency (majority voting) for final decisions.

**Results**: Eliza achieves **moderate performance** compared to GPT-series with plugins, GPTSwarm, and other top-ranked methods.

> 💡 **GAIA 结果的局限性**: "moderate performance" 是一个非常保守的说法，意味着 Eliza 并没有在通用 Agent 评测上超越现有方案。这在意料之中——Eliza 是一个 Web3 专用框架，GAIA 是通用任务。这个评测更像是证明 Eliza 在通用任务上"不差"，而不是展示 Web3 特定优势。3-Agent 多数投票是一个简单但有效的 ensemble 策略，但论文没有分析各难度级别（Level 1/2/3）的分布，也没有 ablation study。

---

### 6.2 Web3 Benchmark

Given that current web3-oriented AI systems are not yet perfected, quantitative comparisons are time-consuming and complicated. Instead, the paper establishes a **foundational standard** for Web3 AI agents — a three-level "Turing Test":

**Basic Level（基础）**:
- 创建钱包
- 转账/收款 token
- 与智能合约交互
- 接入主流社交媒体平台
- 支持基础交易 API

**Intermediate Level（进阶）**:
- Text-to-Video/3D 生成
- RAG 支持
- 音频转文字
- Web3 隐私与安全插件

**Advanced Level（高级）**:
- 自主规划和推理
- 从无序 API 池自动生成执行流水线
- 无需人工干预的端到端自动化

**当前 Eliza 的位置**: Basic → Intermediate 过渡阶段。

> 💡 **Web3 图灵测试框架的价值与局限**: 这个三级框架是论文最有学术贡献的部分之一——它试图为一个没有标准评测的新领域建立基准。框架的分级逻辑清晰：Basic 是能力存在性验证，Intermediate 是能力丰富性验证，Advanced 是自主性验证。
>
> 但问题在于：论文没有对这些 Level 进行量化评测，没有测试用例，没有成功率数据。"Eliza 处于 Basic → Intermediate 过渡阶段"是作者的主观判断，不是实验结论。这是本文最大的实验缺陷——提出了一个好框架，却没有用它来严格评测自己。
>
> 另外，"图灵测试"这个比喻有些夸大——真正的图灵测试关注的是人机区分，而这里的"测试"本质上是功能清单检查（checklist），更准确的叫法应该是"能力成熟度模型"（Capability Maturity Model）。

The next phase involves rapidly integrating the latest advancements in AI (text-to-video/3D, RAG, audio-to-text). The ultimate goal: an agent capable of **autonomous planning and reasoning** based on user instructions, automatically devising suitable execution pipelines without human intervention.

> 💡 **对未来的展望**: 作者引用了 Dario Amodei（Anthropic CEO）的"datacenter of geniuses"愿景，暗示多 Agent 协作是 Eliza 的长期方向。这是一个有雄心的愿景，但当前 Eliza 距离"自主规划 + 无序 API 自动组合"还有相当距离。诚实地说，这些愿景目前在整个 AI 领域都还是研究前沿，不只是 Eliza 的待完成事项。
