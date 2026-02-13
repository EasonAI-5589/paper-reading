[← 返回 README](../README.md)

# 5. Conclusion

## 📌 预览
总结 MemSkill 的贡献，展望 self-improving agent memory 的未来方向。

---

We present MemSkill, an agent memory method that reframes memory operations as an evolving skill bank. MemSkill learns to select a small set of relevant skills for each context span and conditions an LLM executor on these skills to construct memories in a skill-guided manner. Beyond learning how to use a fixed operation set, MemSkill introduces a designer that improves the skill bank itself by refining existing skills and proposing new ones from challenging cases, forming a closed-loop training procedure. Experiments on LoCoMo, LongMemEval, HotpotQA, and ALFWorld demonstrate consistent improvements over strong baselines, and qualitative analyses illustrate how evolving skills can yield more adaptive memory management behaviors. We hope MemSkill encourages future work on self-improving agent memory systems that learn not only to use memory, but also to continually improve how memory is constructed and maintained.

> 💡 **Conclusion 批读**:
> 
> 总结很精炼。最后一句话点出了 MemSkill 的 vision："learn not only to **use** memory, but also to continually improve **how** memory is constructed and maintained"。
> 
> 这区分了两个层次的 "学习"：
> 1. **学会用 memory**（Controller 选 skill）
> 2. **学会改进 memory 的方式**（Designer 进化 skill bank）
> 
> **我的思考 — MemSkill 的局限和未来方向**:
> 
> 1. **Retriever 是固定的**: 当前用 Contriever 检索，如果检索质量差，再好的 skill 也没用。能否把 retriever 也纳入进化？
> 2. **Designer 依赖 LLM 能力**: skill 进化的质量受限于 Designer LLM 的推理能力。如果 LLM 不够强，可能进化不出好 skill
> 3. **没有 skill 之间的组合约束**: 当前 Top-K 选择是独立的，没有考虑 skill 之间的互补性或冲突
> 4. **评估的局限**: LLM Judge (openai/gpt-oss-120b) 的可靠性？不同 judge 可能给出不同结论
> 5. **跟 Mem-α/Memory-R1 的直接对比缺失**: 这两个是最相关的 RL-based memory 方法

---

# Acknowledgements

This research/project is supported by the NTU Start-Up Grant (#023284-00001), Singapore, and the MOE AcRF Tier 1 Seed Grant (RS37/24, #025041-00001), Singapore.

> 💡 **基金信息**: NTU 启动基金 + 新加坡教育部 Tier 1 种子基金。这是王文雅老师到 NTU 后的早期项目。

---

# Impact Statement

MemSkill advances the design of agent memory by shifting emphasis from static, hand-crafted procedures to learnable and evolvable memory skills. This perspective can make long-running LLM agents more practical in settings where interaction histories grow and the information that matters changes over time. By improving how memories are extracted, consolidated, and revised, MemSkill can support more consistent assistance in applications such as multisession personal assistants, educational tutors, long-form customer support, and interactive research tools, where agents must preserve relevant context while avoiding redundant or stale information.

Beyond immediate applications, MemSkill also offers a reusable methodology for studying how memory behaviors should be specified and improved. The explicit skill bank provides a concrete interface for inspection and analysis, which may encourage more interpretable and controllable memory systems. More broadly, the idea of iteratively improving memory management behaviors from hard cases can inspire similar self-improvement mechanisms in other agent subsystems, such as tool use or planning, where fixed heuristics remain common.

As with any memory-enabled agent, responsible use benefits from basic safeguards. For example, deployments should avoid storing unnecessary sensitive information and should provide user-facing controls for memory inspection and removal. These considerations are standard for memoryaugmented systems and are not unique to MemSkill, but they become increasingly important as agent memory becomes more effective and widely adopted.

> 💡 **Impact Statement 批读**:
> 
> 应用场景列举很全面：多会话助手、教育辅导、长期客服、交互式研究工具。
> 
> 最有洞察力的一点：**skill bank 作为可检查的接口**。相比 black-box memory 系统，MemSkill 的 skill 是可读的文本，用户可以检查、理解甚至手动编辑。这在安全性和可控性上有天然优势。
> 
> Safety 考虑也到位：提到了敏感信息存储和用户控制的问题。

---

## 🔖 Section 总结

### 核心洞察
1. MemSkill 的核心 vision: 从 "学会用 memory" 到 "学会改进 memory 的方式"
2. 可解释性优势：skill bank 是可读文本，提供了 inspection 接口
3. 局限：Retriever 固定、Designer 依赖 LLM 能力、缺少 Mem-α/Memory-R1 对比
