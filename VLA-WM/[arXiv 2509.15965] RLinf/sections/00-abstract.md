[← 返回 README](../README.md)

# Abstract

## 📌 预览
系统论文，核心是 M2Flow——将高层 RL 工作流自动变换为优化的执行计划。

---

Reinforcement learning (RL) has demonstrated immense potential in advancing artificial general intelligence, agentic intelligence, and embodied intelligence. However, the inherent heterogeneity and dynamicity of RL workflows often lead to low hardware utilization and slow training on existing systems. In this paper, we present RLinf, a high-performance RL training system based on our key observation that the major roadblock to efficient RL training lies in system flexibility. To maximize flexibility and efficiency, RLinf is built atop a novel RL system design paradigm called macro-to-micro flow transformation (M2Flow), which automatically breaks down high-level, easy-to-compose RL workflows at both the temporal and spatial dimensions, and recomposes them into optimized execution flows. Supported by RLinf worker's adaptive communication capability, we devise context switching and elastic pipelining to realize M2Flow transformation, and a profiling-guided scheduling policy to generate optimal execution plans. Extensive evaluations on both reasoning RL and embodied RL tasks demonstrate that RLinf consistently outperforms state-of-the-art systems, achieving 1.07x ~ 2.43x speedup in end-to-end training throughput.

> 💡 **核心洞察**:
> - **问题**: RL 工作流天然异构（generation/inference/training/simulator 资源需求完全不同）+ 动态性（response 长度变化大）→ 现有系统硬件利用率低
> - **关键观察**: 效率瓶颈在于**系统灵活性不足**，而非算法本身
> - **方案**: M2Flow = 用户写宏观流程（macro logical flow）→ 系统自动变换为微观执行（micro execution flow）
> - **三大机制**: 弹性流水线（空间）+ 上下文切换（时间）+ profiling 引导调度
