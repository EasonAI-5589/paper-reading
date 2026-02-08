[← 返回 README](../README.md)

# 6.1 Benchmarks and Datasets

## 📌 预览
30+ benchmarks 分两大类：(1) 显式面向 memory/lifelong/self-evolving 评测，(2) 其他 agent benchmark 但隐式依赖 memory 能力。

---

## Memory/Lifelong/Self-Evolving Benchmarks

### Memory-oriented（显式评测记忆构建、维护、利用）

| Benchmark | 规模 | 核心评测 | 环境 |
|-----------|------|---------|------|
| **MemBench** | 53,000 samples | 模拟交互场景中的记忆 | simulated |
| **LoCoMo** | 300 samples | 对话记忆（**最常用基准**） | real |
| **LongMemEval** | 5 tasks / 500 samples | 交互式记忆 | simulated |
| **PersonaMem** | 15 tasks / 180 samples | 动态用户画像 | simulated |
| **HaluMem** | 3,467 samples | 记忆幻觉 | simulated |
| **MPR** | 108,000 samples | 用户个性化 | simulated |

### Lifelong-learning（跨 episode 知识积累与更新）

| Benchmark | 规模 | 核心评测 |
|-----------|------|---------|
| **MemoryBench** | ~20,000 samples | 持续学习 |
| **StreamBench** | 9,702 samples | 在线持续学习 |
| **LifelongAgentBench** | 1,396 samples | 终身学习 |

### Self-evolving（agent 能否自主改进）

| Benchmark | 规模 | 核心评测 |
|-----------|------|---------|
| **MemoryAgentBench** | 4 tasks | 多轮交互中的记忆管理 |
| **Evo-Memory** | 10 tasks / ~3,700 samples | test-time learning |

> 💡 **批注**: **LoCoMo 是事实上的标准 benchmark**，几乎所有开源框架都用它报告结果。但它只覆盖对话记忆场景，缺少 embodied/multimodal/multi-agent 评测。当前缺少统一 benchmark 评测 memory 的全生命周期（formation + evolution + retrieval）。

---

## Other Related Benchmarks

虽非专为 memory 设计，但长程/多步/多任务特性隐式依赖记忆能力：

| 类别 | 代表 | 与记忆的关系 |
|------|------|------------|
| **Embodied** | ALFWorld, ScienceWorld | 需记住观察、中间目标、环境动态 |
| **Web** | WebArena, WebShop, MMInA | 长上下文轨迹需回忆早期动作和约束 |
| **Code** | SWE-Bench Verified | 推理长文件和演化代码状态 |
| **Deep Research** | GAIA, xBench | 综合多步骤多来源信息 |
| **Tool Use** | ToolBench | 记忆工具输出和使用经验 |
| **Multi-task** | AgentGym, AgentBoard | 跨任务保留特定知识和策略 |

---

## 🔖 Section 总结

### 核心洞察
1. LoCoMo 是事实标准但覆盖面有限
2. 缺少全生命周期（F+E+R）的统一 benchmark
3. Embodied/Web/Code 类 benchmark 隐式但重度依赖 memory
4. Self-evolving benchmark 是最新且最稀缺的类别
