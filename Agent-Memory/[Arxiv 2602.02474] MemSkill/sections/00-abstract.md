[← 返回 README](../README.md)

# Abstract

## 📌 预览
MemSkill 提出把 memory 操作从固定规则变成可学习、可进化的 skill，由 Controller 选 skill + Executor 执行 + Designer 进化 skill bank，形成闭环优化。

---

Most Large Language Model (LLM) agent memory systems rely on a small set of static, handdesigned operations for extracting memory. These fixed procedures hard-code human priors about what to store and how to revise memory, making them rigid under diverse interaction patterns and inefficient on long histories. To this end, we present MemSkill, which reframes these operations as learnable and evolvable memory skills, structured and reusable routines for extracting, consolidating, and pruning information from interaction traces. Inspired by the design philosophy of agent skills, MemSkill employs a controller that learns to select a small set of relevant skills, paired with an LLM-based executor that produces skillguided memories. Beyond learning skill selection, MemSkill introduces a designer that periodically reviews hard cases where selected skills yield incorrect or incomplete memories, and evolves the skill set by proposing refinements and new skills. Together, MemSkill forms a closed-loop procedure that improves both the skill-selection policy and the skill set itself. Experiments on LoCoMo, LongMemEval, HotpotQA, and ALFWorld demonstrate that MemSkill improves task performance over strong baselines and generalizes well across settings. Further analyses shed light on how skills evolve, offering insights toward more adaptive, self-evolving memory management for LLM agents. Code is available at https://github.com/ViktorAxelsen/MemSkill

> 💡 **Abstract 批读**:
> 
> 这篇是王文雅组的工作，核心 idea 非常清晰：**把 memory 管理从 hand-crafted operations 变成 learnable skills**。
> 
> 三个关键创新点：
> 1. **Skill Bank**：将 memory 操作（提取/整合/剪枝）抽象为结构化、可复用的 skill
> 2. **Controller + Executor**：Controller 学习选 skill，Executor 条件生成 memory（一次 LLM call 完成）
> 3. **Designer**：从 hard cases 出发进化 skill bank，形成 use → evolve 的闭环
> 
> 跟 MemEvolve 的区别：MemEvolve 搜的是 memory **架构**（模块组合），MemSkill 搜的是 memory **操作策略**（更细粒度）。这是一个从 "搜架构" 到 "搜行为" 的转变。
> 
> 实验覆盖面很广：对话（LoCoMo, LongMemEval）+ 多跳问答（HotpotQA）+ 具身任务（ALFWorld），说明方法的通用性。

---

## 🔖 Section 总结

### 核心洞察
1. 现有 memory 系统的根本问题：hard-code human priors → rigid + inefficient
2. MemSkill 的解法：memory operations as learnable, evolvable skills
3. 闭环：Controller RL 训练 + Designer LLM 进化 skill bank
