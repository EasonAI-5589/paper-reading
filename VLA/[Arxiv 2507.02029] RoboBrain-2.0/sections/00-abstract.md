# Abstract

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029, BAAI RoboBrain Team)

---

## 📄 原文

We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model.

> 💡 **vs RoboBrain 1.0 的核心升级**:
> ```
> RoboBrain 1.0 (CVPR 2025):
> ├── 单一 7B 模型 (Qwen2.5-7B)
> ├── 三个能力: Planning + Affordance + Trajectory
> ├── 2D 空间理解
> └── 无时序推理
> 
> RoboBrain 2.0:
> ├── 7B + 32B 两个版本
> ├── 空间能力: pointing, affordance, trajectory, spatial referring, placement
> ├── 时序能力: closed-loop interaction, multi-agent planning, scene graph updating
> ├── 3D 空间推理 (depth, 3D bbox)
> └── Chain-of-Thought reasoning + RLVR 强化学习
> ```

Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models.

> 💡 **关键结果速览**:
> - **Spatial**: BLINK SOTA (83.95), RefSpatial-Bench SOTA (54.00), Where2Place SOTA (73.59), RoboSpatial SOTA (72.43)
> - **Temporal**: Multi-Robot Planning SOTA (81.50), EgoPlan2 SOTA (57.23), RoboBench SOTA (72.16)
> - 超越 GPT-4o、Gemini-2.5-Pro、Claude-Sonnet-4 等闭源模型

---

![Figure 1](../images/a808da69c76bee61e7c520fa20705382a73db1fa534e82b58996e4ca135aa768.jpg)
*Figure 1: Benchmark comparison — RoboBrain 2.0-32B 在空间和时序推理上均为最佳*

> 💡 **Figure 1 批读**:
> ```
> 空间推理 benchmark:
> ├── BLINK-Spatial: 87.41 (32B) ← 超越 GPT-o4-mini
> ├── RoboSpatial: 72.43 (32B) ← 大幅领先
> ├── RefSpatial-Bench: 54.00 (32B) ← 其他模型基本 < 20
> └── Where2Place: 73.59 (32B) ← 遥遥领先
> 
> 时序推理 benchmark:
> ├── EgoPlan2: 57.23 (32B) ← 超越 Qwen2.5-VL-32B
> └── Multi-Robot-Plan: 81.50 (7B) ← 甚至 7B 就超过所有对手
> ```

---

## 💡 Section 总结

### 一句话总结
RoboBrain 2.0 是一个 7B/32B 的具身视觉语言基础模型，统一空间理解（pointing/affordance/trajectory/referring/placement）和时序推理（closed-loop/multi-agent/scene graph），在 12 个 benchmark 上取得 6 个 SOTA。

### 核心升级 (vs v1)
| 维度 | v1 (CVPR 2025) | v2 (Tech Report) |
|------|----------------|-------------------|
| 模型规模 | 7B only | 7B + 32B |
| 基座 | LLaVA + Qwen2.5-7B | Qwen2.5-VL (7B/32B) |
| 空间能力 | bbox affordance + 2D traj | pointing + spatial referring + 3D + placement |
| 时序能力 | ❌ | closed-loop + multi-agent + scene graph |
| 推理 | ❌ | CoT + RLVR (Reason-RFT) |
| 训练数据 | ~3M | ~5M+ |
| 训练阶段 | 6 stages | 3 stages |
