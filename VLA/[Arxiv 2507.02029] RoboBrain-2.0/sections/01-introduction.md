# 1 Introduction

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 📄 原文

In recent years, large language models (LLMs) and vision-language models (VLMs) have emerged as key driving forces in the advancement of general artificial intelligence (AGI). Within digital environments, these models have demonstrated remarkable capabilities in perception [5, 16, 83], understanding [22, 73], and reasoning [2, 17, 18, 45, 65], and have been widely applied in tasks such as multimodal question answering [35, 60], image generation and editing [24, 57], GUI control [37, 71], and video understanding [7, 63, 72]. They have also seen early adoption in practical domains such as education, healthcare, search, and intelligent assistants [11, 21, 82].

> 💡 **背景铺垫**: 标准开场——LLM/VLM 在数字世界很强，但物理世界还不行。引出 embodied AI 的需求。

However, bridging the gap between "digital intelligence" and "physical intelligence"—enabling models to perceive their surroundings, understand embodied tasks, and interact with the real world—remains a critical challenge on the path toward AGI. Embodied foundation models [4, 64, 74] represent a promising research direction toward physical intelligence. Several recent efforts have extended the capabilities of LLMs and VLMs to embodied scenarios, advancing multimodal fusion, perception, and action execution. While these models have achieved encouraging progress, they still face three fundamental capability bottlenecks when deployed in complex and open-ended real-world environments: (1) Limited spatial understanding: Current models struggle to accurately model relative and absolute spatial relationships and identify affordances in physical environments, which hinders real-world applicability; (2) Weak temporal modeling: The lack of understanding of multi-stage, cross-agent temporal dependencies and feedback mechanisms limits long-horizon planning and closed-loop control; (3) Insufficient reasoning chains: Existing models are often incapable of extracting causal logic from complex human instructions and aligning it with dynamic environmental states, restricting their generalization to open-ended embodied tasks.

> 💡 **三大瓶颈 (核心 Motivation)**:
> ```
> ① Limited Spatial Understanding
> │   问题: 空间关系建模不准
> │   影响: affordance 识别、物体定位困难
> │
> ② Weak Temporal Modeling
> │   问题: 多阶段时序依赖缺失
> │   影响: long-horizon planning、closed-loop control 受限
> │
> ③ Insufficient Reasoning Chains
>     问题: 无法从复杂指令中提取因果逻辑
>     影响: open-ended 任务泛化差
> ```
> **评价**: 这三个瓶颈的划分很清晰，也正好对应了 2.0 的三个训练阶段（Stage 1→spatial, Stage 2→temporal, Stage 3→CoT reasoning）

To address these challenges, we present RoboBrain 2.0, our latest generation of embodied vision-language foundation models, tailored to bridge perception, reasoning, and planning in physically environments. RoboBrain 2.0 processes visual observations and language instructions in a unified architecture, enabling holistic understanding of the environment, goal-directed reasoning, and long-horizon planning. We release two variants of the model: the lightweight RoboBrain 2.0–7B and the full-scale RoboBrain 2.0–32B, designed to meet different deployment needs under varying resource constraints. On both spatial reasoning and temporal reasoning benchmarks, the 32B variant mostly achieves state-of-the-art performance, outperforming prior open-source and proprietary models, as shown in Figure 1. Model capabilities are summarized in Figure 2.

> 💡 **解决方案概览**: 两个规模 7B/32B，统一 perception + reasoning + planning。

This report provides a systematic overview of the design principles, core components and key innovations. In particular, we highlight the extensive data contributions that support spatial understanding, temporal reasoning, and causal inference, which form the foundation of RoboBrain 2.0's capabilities. To address the scarcity of spatial data, we develop a spatial data synthesis pipeline that constructs large-scale, high-quality datasets spanning tasks such as pointing, affordance prediction, and trajectory generation. To improve temporal reasoning and feedback modeling, we design multi-robot coordination templates across common scenarios via RoboOS [61], generate cross-agent long-horizon planning trajectories using external models [31], and simulate randomized failure events to collect closed-loop feedback data that enhances model robustness. To further enrich reasoning data, we extract step-by-step thought traces from powerful reasoning VLMs [22], conditioned on spatiotemporal task contexts. These traces serve as supervision signals for learning causal chains across vision, language, and action.

> 💡 **数据贡献是核心亮点**:
> ```
> Spatial 数据:
> ├── pointing (Pixmo-Points + GPT-4o 过滤)
> ├── affordance (PACO-LVIS + GPT-4o 生成 QA)
> ├── trajectory (RefSpatial pipeline)
> └── spatial understanding (826K samples, 31 spatial concepts)
>
> Temporal 数据:
> ├── RoboOS multi-robot templates
> ├── DeepSeek-V3 生成 planning trajectories
> └── 模拟 failure events → closed-loop feedback
>
> Reasoning 数据:
> └── GPT-4o 提取 CoT thought traces
> ```
> **和 1.0 的关键区别**: 1.0 的数据贡献是 ShareRobot (51K→1M QA)；2.0 的数据更多元，spatial/temporal/reasoning 都有专门的 pipeline

RoboBrain 2.0 adopts a high-efficiency heterogeneous architecture and a progressive multi-stage training strategy to support spatial understanding, temporal modeling, and long-chain causal reasoning in embodied settings. The model comprises a lightweight vision encoder with approximately 689M parameters and a decoder-only language model with 7B/32B parameters. It is trained using a three-stage curriculum—covering foundational spatiotemporal learning, embodied spatiotemporal enhancement, and chain-of-thought reasoning—on large-scale multimodal and embodied datasets. Training is conducted using our open-source framework FlagScale, which integrates hybrid parallelism, pre-allocated memory optimization, high-throughput I/O pipelines, and robust fault tolerance. These infrastructure innovations significantly reduce training and deployment costs while ensuring scalability for large-scale multimodal models. We evaluate RoboBrain 2.0 on over 12 public benchmarks covering spatial understanding, temporal modeling and multimodal reasoning, achieving state-of-the-art results on 6 of them despite its compact size. We release code, checkpoints, and benchmarks as open-source resources to benefit the research community. These materials facilitate reproducible research, accelerate embodied AI development, and enable practical deployment in robotic systems.

> 💡 **技术规格速览**:
> | 组件 | 规格 |
> |------|------|
> | Vision Encoder | ~689M params |
> | LLM Decoder | 7B / 32B (Qwen2.5-VL) |
> | 训练阶段 | 3 stages |
> | 训练框架 | FlagScale (开源) |
> | Benchmarks | 12+, 6 个 SOTA |

![](../images/f466c7e5c8dd007b4314b4c72f1fc1eceb9e3a6e941419b684e552bc3036f09b.jpg)
*Figure 2: The overview of RoboBrain 2.0's Capabilities. 支持 interactive reasoning (long-horizon planning + closed-loop feedback), spatial perception (point/bbox), temporal perception (trajectory), scene reasoning (scene graph construction + updating).*

> 💡 **Figure 2 批读**:
> ```
> 四大能力:
> ├── Interactive Reasoning: 多轮对话式规划 + closed-loop 反馈
> ├── Spatial Perception: 点坐标 + bounding box 预测
> ├── Temporal Perception: 未来轨迹估计
> └── Scene Reasoning: 场景图构建 + 实时更新
> ```
> 对比 1.0 的三大能力 (Planning/Affordance/Trajectory)，2.0 增加了 Interactive Reasoning 和 Scene Reasoning

To provide a comprehensive view of RoboBrain 2.0's architecture, training methodology, and capabilities, this report is organized as follows: Section 2 introduces the overall model design, including the coordination between the vision encoder and language model, as well as image and video input strategies. Section 3 describes the data curation and construction process, covering three major categories: general multimodal understanding, spatial reasoning, and temporal modeling. Section 4 presents our multi-stage training strategies, including foundational spatiotemporal learning, embodied enhancement, and chain-of-thought reasoning. Section 5 outlines the infrastructure stack supporting scalable training and inference, including hybrid parallelization, memory optimization, data loading, and failure recovery. Section 6 reports extensive evaluation results on public benchmarks, highlights RoboBrain 2.0's capabilities in spatial reasoning, temporal feedback, and embodied planning. Finally, Section 7 discusses current limitations, and outlines future research directions.

> 💡 **论文结构**: 标准 tech report 结构：Architecture → Data → Training → Infrastructure → Evaluation → Conclusion

---

## 💡 Section 总结

### 核心贡献
1. **统一 spatial + temporal reasoning** 的 embodied VLM
2. **大规模数据工程**: spatial/temporal/reasoning 三类数据 pipeline
3. **三阶段训练**: foundation → embodied → CoT+RLVR
4. **基础设施**: FlagScale 分布式训练框架

### 与 RoboBrain 1.0 的定位差异
| | 1.0 | 2.0 |
|---|---|---|
| 核心贡献 | ShareRobot 数据集 | 统一架构 + 多源数据 |
| 架构 | LLaVA + LoRA | Qwen2.5-VL (full fine-tune) |
| 能力 | Spatial only | Spatial + Temporal + CoT |
| 规模 | 7B | 7B + 32B |
