[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
从现有 memory 系统的三大局限出发，提出 memory skill 的概念，介绍 MemSkill 的三组件架构和闭环优化流程。

---

As Large Language Model (LLM) agents engage in longer, open-ended interactions, they must handle growing histories that are essential yet challenging to leverage, motivating memory for retaining experience and maintaining coherence (Hu et al., 2025). This need has driven rapid progress in agent memory, including approaches that summarize and retrieve past interactions or manage external memory stores (Kang et al., 2025; Chhikara et al., 2025; Packer et al., 2023; Xu et al., 2025). However, most methods still rely on static, hand-designed memory mechanisms, including fixed operation primitives (e.g., add/update/delete/skip) (Wang et al., 2025a; Yan et al., 2025) and heuristic modules that govern what to store, how to revise it (Kang et al., 2025; Fang et al., 2025), and when to prune it. Such designs bake in strong human assumptions and often suffer under diverse interaction patterns, scaling poorly as histories grow.

> 💡 **背景铺垫**: 点出核心问题——现有 memory 系统依赖 static, hand-designed 操作原语（add/update/delete/skip），这些 hard-code 了人的先验。这里引用的 Wang et al., 2025a 就是 Mem-α（RL 优化 memory），Yan et al., 2025 是 Memory-R1。

---

We argue that this formulation fundamentally limits the adaptability of agent memory. Rather than treating memory as the output of fixed operations or hand-designed modules, we propose to elevate memory extraction itself into a learnable abstraction. Concretely, we view memory construction as the outcome of applying a small set of generic, reusable memory skills: structured behaviors that specify when and how interaction traces should be transformed into memory and revised over time. This perspective reveals a key bottleneck of prior pipelines: they hard-code memory behaviors into fixed procedural workflows that interleave heuristics with LLM-mediated extraction and revision, making them brittle under distribution shift (Fang et al., 2025).

> 💡 **核心论点**: "elevate memory extraction itself into a learnable abstraction"——把 memory 提取本身变成可学习的抽象。这个 framing 很巧妙：不是优化某个具体操作，而是让操作本身变成可学习的对象。
> 
> 类比：之前的方法就像写死了 if-else 规则（遇到新信息就 INSERT，矛盾就 DELETE），MemSkill 则是让系统自己学习 "什么时候该怎么操作"。

---

Under this view, an ideal agent memory system should satisfy three properties. (i) Minimal reliance on human priors. Instead of manually encoding what is worth remembering for a domain (Zhong et al., 2024), memory behaviors should be shaped by interaction data and updated as task demands evolve. (ii) Support for larger extraction granularity. Many approaches are tuned to a fixed unit, such as per-turn processing (Fang et al., 2025), and can weaken when applied to longer spans. A practical system should be able to operate at larger extraction granularity when needed. (iii) Skill-conditioned, compositional memory construction. Existing systems often decompose memory construction into specialized modules (Kang et al., 2025). In contrast, we prefer to select and compose a small set of relevant skills for the current context and apply them in one generation step, enabling flexible reuse and evolution of memory behaviors.

> 💡 **三大设计原则**:
> 1. **Minimal human priors** — 不要手动编码 "什么值得记"，让数据说了算
> 2. **Larger extraction granularity** — 支持 span-level（而非 turn-level）处理，对长历史更友好
> 3. **Skill-conditioned compositional** — 选几个 skill 组合，一次 LLM call 完成，而非串行 pipeline
> 
> 第 (ii) 点特别值得注意：per-turn processing 是很多方法（如 MemoryBank、Mem0）的标配，但对长对话来说 LLM 调用次数太多。MemSkill 改成 span-level，大幅减少调用量。

---

Based on the above observations, we introduce MemSkill, which reframes memory operations as a learnable and evolvable set of memory skills. MemSkill maintains a shared skill bank, where each skill captures a reusable way to extract, consolidate, or revise memories from interaction text (Figure 1 shows the structured template of a memory skill). Given the current context, a controller learns to select a small set of relevant skills, and an LLM-based executor conditions on these skills to generate skill-guided memories in one pass. This skill-conditioned formulation is not tied to a fixed extraction unit and can be applied to different span lengths when processing long interaction histories.

> 💡 **MemSkill 介绍**: 三个核心概念：
> - **Skill Bank**（共享）：存放所有可复用的 memory skill
> - **Controller**（可学习）：根据 context 选 Top-K 个 skill
> - **Executor**（固定 LLM）：根据选中的 skill 生成 memory
> 
> "in one pass" 是关键——不需要逐 turn 调 LLM，一次搞定一个 span。

---

![Figure 1](../images/7d6d0f90de2fecde56b75c881cdfefa8f361cddedca73cf05a20d45a48437342.jpg)
*Figure 1. Comparison between (a) prior turn-level, handcrafted operations and (b) MemSkill's span-level, skill-conditioned generation. Prior methods interleave handcrafted operations with LLM calls to incrementally extract and revise memory turn by turn, while MemSkill selects a small set of skills from a shared skill bank and applies them in one pass to produce skill-guided memories.*

> 💡 **Figure 1 批读**:
> 左右对比非常清晰：
> - **(a) 传统方法**: 每个 turn 都要走一遍 pipeline（提取 → 判断操作类型 → 执行），操作原语是固定的 {INSERT, UPDATE, DELETE, SKIP}
> - **(b) MemSkill**: 把多个 turn 打包成一个 span，从 skill bank 里选一组 skill，一次性生成 memory
> 
> 右侧展示了 skill 的结构化模板：包含 description（用于选择）和 content specification（指导 executor 具体怎么做）。这个模板设计很像 agent tool 的定义方式。

---

Crucially, MemSkill goes beyond learning how to use a fixed set of skills. We introduce a closed-loop evolution process that alternates between learning to use the current skill bank and evolving the skill bank itself. Specifically, we train the controller with reinforcement learning (RL) using downstream task signals as feedback for skill selection. Periodically, a designer aggregates the hardest cases produced during training, selects representative failures, and uses an LLM to refine existing skills and propose new ones. After each evolution step, the controller continues training on the evolved skill bank, with additional exploration to facilitate adopting newly introduced skills. Overall, this process gradually strengthens both the skill selection policy and the evolving skill bank, moving toward a more adaptive memory management system driven by interaction data.

> 💡 **闭环进化**:
> 核心流程：
> ```
> Controller RL 训练 (学 skill 选择)
>        ↓ 收集 hard cases
> Designer LLM 分析 (进化 skill bank)
>        ↓ 更新 skill bank
> Controller 继续训练 (+ 探索新 skill)
>        ↓ ...循环
> ```
> 
> 两个层面的优化：
> 1. **Skill 使用策略**（Controller，RL 优化）
> 2. **Skill 集合本身**（Designer，LLM-guided 进化）
> 
> 这比 Mem-α 和 Memory-R1 更进一步：它们只优化 "选哪个固定操作"，MemSkill 还优化 "操作本身是什么"。

---

Experiments on LoCoMo, LongMemEval, HotpotQA, and ALFWorld show that MemSkill consistently improves task performance and generalizes well. Further analyses validate key components and showcase representative evolved skills, offering insights toward more adaptive, self-evolving memory management for LLM agents.

Our contributions can be summarized as follows.

• We propose MemSkill, an agent memory method that represents memory operations as an evolving skill bank, and constructs skill-guided memories by conditioning an LLM on a selected set of skills.

• We introduce a closed-loop optimization recipe that combines reinforcement learning for skill selection with LLM-guided skill evolution from hard cases, enabling continual refinement of the skill bank and taking a step toward self-evolving agent memory systems.

• We evaluate MemSkill on LoCoMo, LongMemEval, HotpotQA, and ALFWorld, showing consistent gains over baselines and strong generalization, offering insights toward self-evolving memory for LLM agents.

> 💡 **三大贡献**:
> 1. Skill bank 表示 + skill-conditioned memory 生成
> 2. RL 选 skill + LLM 进化 skill 的闭环优化
> 3. 四个 benchmark 上的验证 + 跨模型/跨数据集泛化
> 
> 个人认为贡献 2 是最核心的——闭环进化让系统真正 "self-evolving"。

---

## 🔖 Section 总结

### 核心洞察
1. 现有方法的根本问题：hard-coded memory behaviors → brittle under distribution shift
2. MemSkill 的三大设计原则：minimal human priors + span-level + skill-conditioned compositional
3. 双层优化：RL 优化 skill 选择策略 + LLM 进化 skill bank 本身
4. 跟前人的关键区别：Mem-α/Memory-R1 只优化固定操作的选择，MemEvolve 搜架构，MemSkill 搜操作策略
