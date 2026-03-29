[← 返回 README](../README.md)

# II. Related Work

## 📌 预览
Related work 覆盖三个方向：VLA 模型、机器人视觉 prompting/sketches/traces，以及 think-before-act 推理。

---

## VLA 模型谱系

| 类别 | 代表方法 | 特点 | 局限 |
|------|----------|------|------|
| 端到端 VLA | RT-2, OpenVLA, DP, Octo | 直接观测→动作映射 | 计划意图隐藏在隐层，缺乏任务分解和因果解释 |
| 层级 VLA | RoboBrain, Helix, GR-RL | planner-controller 架构 | 推理瞬时，缺乏全局意图的持久建模，空间引用消歧弱 |
| Think-before-act | EO-1, OneTwoVLA, ThinkAct | 在统一 backbone 中集成显式推理 | 中间表达是纯文本或不可编辑的 latent，spatial grounding 仍是隐式的 |
| **Action-Sketcher** | 本文 | **显式可编辑 Visual Sketch** | — |

> 💡 **批注**:
> ThinkAct 最接近本文，也用了视觉计划 latent，但它压缩成了不可编辑的隐向量。
> Action-Sketcher 的核心差异：Sketch 是**持久的、人可编辑的、逐子任务生成的**。

---

## 机器人视觉 Prompting / Sketches / Traces

| 方法 | 描述 | 与本文的差异 |
|------|------|-------------|
| RT-Trajectory | 条件于粗糙轨迹 sketch | 静态输入，不逐步生成 |
| RT-Sketch | 利用手绘 goal sketch | sketch 是外部输入而非模型生成 |
| TraceVLA | 注入视觉 trace prompting | trace 是固定的，无在线修正 |
| RoVI | 对象中心符号（箭头/圆圈/颜色/数字）| 系统化了符号类型但仍是静态的 |
| LLARVA | 预测 2D traces 对齐视觉-动作空间 | trace 非人类可验证 |
| RoboBrain | 显式视觉引导（指向/affordance/轨迹）| 下游 policy 使用，非 unified model |
| MolmoAct | depth-aware tokens + mid-level spatial plans | trajectory traces 作为可编辑 sketch | 最接近；本文 Sketch 更丰富（含旋转箭头等）且逐子任务生成 |
| **Action-Sketcher** | **持久、人可编辑、逐子任务生成的 Sketch** | **表达更丰富的动作原语（接触关键点、旋转箭头、放置提示）** |

> 💡 **批注**:
> 现有方法的共同问题：sketch/trace 要么是**静态外部输入**，要么**压缩成不可编辑的 latent**。
> Action-Sketcher 的 Visual Sketch 是**模型实时生成 + 人可在线修改**的——这是本质区别。
