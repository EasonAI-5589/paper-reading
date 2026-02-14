[← 返回 README](../README.md)

# Abstract

## 📌 预览

论文提出 LatentMem 框架，解决多智能体记忆的两大瓶颈：memory homogenization（角色无关的同质化记忆）和 information overload（细粒度记忆导致的信息过载）。核心思路是用可学习的 memory composer 将历史轨迹压缩为角色感知的 latent token，并用 LMPO 端到端优化。

---

Abstract Large language model (LLM)-powered multi-agent systems (MAS) demonstrate remarkable collective intelligence, wherein multi-agent memory serves as a pivotal mechanism for continual adaptation. However, existing multi-agent memory designs remain constrained by two fundamental bottlenecks: (i) memory homogenization arising from the absence of role-aware customization, and (ii) information overload induced by excessively fine-grained memory entries. To address these limitations, we propose LatentMem, a learnable multi-agent memory framework designed to customize agent-specific memories in a token-efficient manner. Specifically, LatentMem comprises an experience bank that stores raw interaction trajectories in a lightweight form, and a memory composer that synthesizes compact latent memories conditioned on retrieved experience and agent-specific contexts. Further, we introduce Latent Memory Policy Optimization (LMPO), which propagates task-level optimization signals through latent memories to the composer, encouraging it to produce compact and high-utility representations. Extensive experiments across diverse benchmarks and mainstream MAS frameworks show that LatentMem achieves a performance gain of up to $19.36\%$ over vanilla settings and consistently outperforms existing memory architectures, without requiring any modifications to the underlying frameworks.

> 💡 **Abstract 批读**:
> - **问题**：现有多智能体记忆的两大瓶颈——(i) Memory Homogenization：所有 agent 共享同一套记忆，忽略角色差异；(ii) Information Overload：多粒度记忆条目太多，淹没关键信息
> - **方法**：LatentMem = Experience Bank（存原始轨迹）+ Memory Composer（将轨迹 + agent profile → 固定长度 latent memory）
> - **训练**：LMPO（Latent Memory Policy Optimization），通过 latent memory 的可微性把任务 reward 反传到 composer
> - **亮点数字**：最高 19.36% 提升，不需要修改底层 MAS 框架
> - **关键词**：latent memory、role-aware、token-efficient、GRPO variant

---

## 🔖 Section 总结

### 核心洞察
1. 把记忆从 "离散文本" 变成 "连续 latent token" 是关键创新——既压缩了 token 数，又打开了端到端优化的通路
2. "不修改底层框架" 是很强的工程优势，意味着可以即插即用到 AutoGen、CAMEL 等现有系统
