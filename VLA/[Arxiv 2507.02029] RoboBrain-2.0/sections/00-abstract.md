# Abstract

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029, BAAI RoboBrain Team)

---

## 📄 原文

We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent longhorizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.

> 💡 **vs RoboBrain 1.0 的核心升级**:
> ```
> RoboBrain 1.0 (CVPR 2025):
> ├── 单一 7B (Qwen2.5-7B base, LLaVA 架构)
> ├── Planning + Affordance + Trajectory
> ├── 2D 空间理解
> └── 无时序推理，无 CoT
>
> RoboBrain 2.0:
> ├── 7B + 32B 两个版本 (Qwen2.5-VL base)
> ├── 空间: pointing, affordance, trajectory, spatial referring, placement
> ├── 时序: closed-loop, multi-agent planning, scene graph updating
> ├── 3D 空间推理 (depth, 3D bbox)
> └── Chain-of-Thought + RLVR (GRPO)
> ```

> 💡 **关键能力分类**:
> - **Spatial**: affordance prediction, spatial referring, trajectory forecasting
> - **Temporal**: closed-loop interaction, multi-agent long-horizon planning, scene graph updating
> - 注意：这两类能力在 1.0 中是分开的（只有 spatial），2.0 才统一

![](../images/a808da69c76bee61e7c520fa20705382a73db1fa534e82b58996e4ca135aa768.jpg)
*Figure 1: Benchmark comparison across spatial and temporal reasoning. RoboBrain2.0-32B achieves best performance on both spatial and temporal reasoning benchmarks.*

> 💡 **Figure 1 批读**:
> ```
> Spatial benchmarks (雷达图左):
> ├── BLINK-Spatial: 83.95 (7B SOTA)
> ├── RoboSpatial: 72.43 (32B SOTA, 大幅领先)
> ├── RefSpatial-Bench: 54.00 (32B SOTA, 碾压所有 baseline)
> ├── Where2Place: 73.59 (32B SOTA)
> └── ShareRobot-Bench: Afford 35.28, Traj DFD 0.2368
>
> Temporal benchmarks (雷达图右):
> ├── EgoPlan2: 57.23 (32B SOTA)
> ├── Multi-Robot-Plan: 80.33~81.50
> └── RoboBench: 72.16 (7B SOTA)
> ```
> **关键发现**: 7B 在 BLINK/CV-Bench/RoboBench 上反而比 32B 好，说明小模型在某些任务上训练更充分

---

## 💡 Section 总结

### 核心定位
RoboBrain 2.0 = **Embodied VLM**（不是 VLA），统一 spatial + temporal reasoning

### 关键卖点
1. **7B + 32B** 两个规模，覆盖不同部署需求
2. **Spatial + Temporal 统一**：1.0 只有 spatial，2.0 加入 temporal
3. **CoT + RLVR**：引入推理链和强化学习
4. **12+ benchmarks SOTA**：6 个 SOTA，其余 near-SOTA
