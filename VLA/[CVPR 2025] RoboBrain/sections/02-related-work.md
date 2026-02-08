# 2. Related Work

> 来源: RoboBrain (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: 两条线：(1) MLLM 用于机器人规划，(2) 机器人操作数据集。

### MLLM for Robotic Manipulation Planning

Existing studies mostly utilize MLLMs primarily focus on understanding natural language and visual observation tasks, with fewer addressing the decomposition of high-level task instructions into actionable steps. PaLM-E generates multimodal inputs by mapping real-world observations into the language embedding space. RT-H and RoboMamba generate reasoning results along with robot actions obtained from an additional policy head.

> 💡 **现有方法的问题**:
> ```
> 方法谱系:
> ├── PaLM-E: 把观测映射到 language space → 但不做细粒度 affordance/trajectory
> ├── RT-H: 推理 + 动作 policy head → 但动作是隐式的
> └── RoboMamba: 类似 RT-H → 同样缺乏显式 affordance 和 trajectory
> 
> 共同问题: 都不做 affordance perception + trajectory prediction
> ```

### Datasets for Manipulation Planning

Early datasets mainly comprise annotated images and videos that highlight fundamental hand-object interactions. Recent advancements emphasize multi-modal and cross-embodiment datasets for enhanced generalization. Notably, RT-X compiles data from 60 datasets across 22 embodiments into the Open X-Embodiment (OXE) repository.

> 💡 **数据集演进**:
> ```
> 早期: DexYCB, HO3D → 单一手-物交互
>   ↓
> 中期: RH20T, BridgeDataV2, DROID → 多场景多样性
>   ↓
> RT-X / Open X-Embodiment → 60 个数据集、22 种机器人 ← ShareRobot 的数据来源
>   ↓
> ShareRobot → 从 OXE 精选 + 标注 planning/affordance/trajectory
> ```
> **关键区别**: OXE 只有高层描述（"pick up the cup"），ShareRobot 标注了低层指令（"move gripper to cup handle at position X"）

---

## 💡 Section 总结

| 对比维度 | PaLM-E | RT-H | RoboMamba | RoboBrain |
|----------|--------|------|-----------|-----------|
| Planning | ✅ | ✅ | ✅ | ✅ |
| Affordance | ❌ | ❌ | ❌ | ✅ |
| Trajectory | ❌ | ❌ | ❌ | ✅ |
| 数据来源 | 自有 | 自有 | 自有 | OXE 精选 + 自标注 |

### 核心洞察
- RoboBrain 的定位是 "三合一"：planning + affordance + trajectory，这是区别于前人的核心差异
- 数据集方面，站在 Open X-Embodiment 的肩膀上，通过精选 + 重标注来提升质量
