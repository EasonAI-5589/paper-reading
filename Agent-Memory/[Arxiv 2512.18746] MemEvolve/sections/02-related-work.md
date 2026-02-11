[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
两个方向的文献综述：LLM Agent 系统的发展 + Agent Memory 架构的演进（按存储模态分类）。

---

## LLM Agent Systems

The past two years have witnessed rapid advances in LLM-based agent systems across multiple dimensions (Tran et al., 2025; Fang et al., 2025a). In terms of system complexity, development has progressed from early single-agent setups with manually defined workflows and limited tool configurations (Wu et al., 2023; Significant-Gravitas, 2023) to sophisticated multi-agent architectures featuring diverse MCP integrations and automated orchestration (Zhang et al., 2024a, 2025a; Wang et al., 2025b; Zhang et al., 2025c). From the perspective of task domains, capabilities have expanded from relatively constrained areas such as coding and mathematical reasoning (Hong et al., 2024; Yin et al., 2023) to more challenging domains, including deep research and scientific discovery (Du et al., 2025; Ghareeb et al., 2025). Today, numerous open-source multi-agent systems demonstrate competitive performance on demanding benchmarks such as GAIA (Mialon et al., 2023), HLE (Phan et al., 2025), BrowseComp (Wei et al., 2025a), and xBench (Chen et al., 2025), including CAMEL's OWL (Hu et al., 2025a), Tencent's CK-Pro (Fang et al., 2025c), Skywork's AgentOrchestra (Zhang et al., 2025f), and ByteDance's AIME (Shi et al., 2025b), among others.

> 💡 **Agent 系统演进**:
> - 复杂度：单 Agent → 多 Agent + MCP + 自动编排
> - 任务域：coding/math → deep research / 科学发现
> - 代表系统：OWL, CK-Pro, AgentOrchestra, AIME
> - 这些系统都是 MemEvolve 的潜在"宿主"框架

## Agent Memory Architectures

Agent memory systems can be broadly divided by objective into personalized memory and self-improving memory (Zhang et al., 2024b; Hu et al., 2025c). The former enables agent chatbots to dynamically capture user-specific information and preferences, while the latter focuses on distilling knowledge and skills from continual interactions with the environment to enhance performance, a focus adopted in this work.

> 💡 **记忆系统二分法**: 个性化记忆 vs 自进化记忆。MemEvolve 聚焦后者。

Self-improving memories are primarily differentiated by their storage modality. Early systems stored raw agent trajectories as few-shot examples (Wang et al., 2023; Zhong et al., 2024; Packer et al., 2023); subsequent designs abstracted these experiences into higher-level lessons, insights (Yang et al., 2025; Sun and Zeng, 2025; Wu et al., 2025b), procedural tips (Wang et al., 2025c; Zheng et al., 2025; Fang et al., 2025b), and more recently, reusable tools and structured repositories (Zhao et al., 2025; Qiu et al., 2025a,b; Zhang et al., 2025e). Despite their differences in representation, there approaches share the same ambition, i.e., to enable agents to learn, adapt, and improve in a human-esque manner.

> 💡 **自进化记忆的谱系（按存储模态）**:
> 1. **原始轨迹**: Voyager, MemoryBank, MemGPT → few-shot exemplars
> 2. **抽象经验**: ExpeL (insights), Reflexion (reflections), H2R (templates)
> 3. **程序化知识**: Mobile-Agent-E (tips/shortcuts), Cheatsheet (tips)
> 4. **工具/代码**: SkillWeaver (APIs), Alita (MCP), DGM (code repos)
>
> MemEvolve 的位置：不在这个谱系中的某一点，而是在**谱系之上** — 自动搜索最适合当前任务的点。

---

## 🔖 Section 总结

### 核心洞察
1. Agent 系统越来越复杂，对记忆系统的适应性要求更高
2. 自进化记忆的存储模态不断演进，但每种都有适用边界
3. MemEvolve 是第一个试图自动搜索最优记忆架构的工作
