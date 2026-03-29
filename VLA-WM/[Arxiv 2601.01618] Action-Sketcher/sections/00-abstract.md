[← 返回 README](../README.md)

# Abstract

## 📌 预览
长时域操作面临两大瓶颈：**空间模糊**（语言指令难以精确定位目标）和**时序脆弱**（计划意图隐藏在隐层，人无法干预）。论文通过引入显式 Visual Sketch 同时解决两者。

---

Long-horizon robotic manipulation is increasingly important for real-world deployment, requiring spatial disambiguation in complex layouts and temporal resilience under dynamic interaction. However, existing end-to-end and hierarchical Vision–Language–Action (VLA) policies often rely on text-only cues while keeping plan intent latent, which undermines referential grounding in cluttered or underspecified scenes, impedes effective task decomposition of long-horizon goals with close-loop interaction, and limits causal explanation by obscuring the rationale behind action choices.

> 💡 **批注**:
> 现有 VLA 的三大问题：
> 1. **参考消歧弱**：text-only 指令在杂乱场景中无法定位具体目标
> 2. **任务分解差**：计划意图是隐层表示，无法有效拆解长时域任务
> 3. **可解释性低**：决策原因不透明，无法支持因果解释

---

To address these issues, we first introduce **Visual Sketch**, an explicit visual intermediate that renders points, boxes, arrows, and typed relations in the robot's current views to externalize spatial intent, connect language to scene geometry, and provide a human-verifiable bridge between high-level reasoning and low-level control.

> 💡 **批注**:
> Visual Sketch 的三个关键属性：
> - **外显空间意图**：把"在哪里/如何操作"渲染到图像平面
> - **语言-几何对接**：语言指令 → 具体坐标/箭头
> - **人可验证**：人可在执行前查看和修改，实现 Human-in-Loop

---

Building on Visual Sketch, we present **Action-Sketcher**, a VLA framework that operates in a cyclic **See → Think → Sketch → Act** workflow coordinated by adaptive token-gated strategy for reasoning triggers, sketch revision, and action issuance, thereby supporting reactive corrections and human interaction while preserving real-time action prediction.

> 💡 **批注**:
> token-gate 机制是关键：
> - `<BOR>` (begin-of-reasoning) → 触发推理模式
> - `<BOA>` (begin-of-action) → 触发动作模式
> 使得系统可以在保持实时性的同时支持按需推理

---

To enable scalable training, we curate diverse corpus with interleaved images, text, Visual Sketch supervision, and action sequences, and train with a **multi-stage curriculum** combining interleaved sequence alignment, language-to-sketch consistency, and imitation learning augmented with sketch-to-action reinforcement for robustness.

> 💡 **批注**:
> 三阶段课程的核心逻辑：
> - Stage 1：建立空间/时序基础能力
> - Stage 2：学会把语言推理变成精确 Sketch
> - Stage 3：从 Sketch 生成动作 + 学会何时切换模式

---

**结果**: Extensive experiments on cluttered scenes and multi-object tasks, in simulation and on real-world tasks, show improved long-horizon success, stronger robustness to dynamic scene changes, and enhanced interpretability via editable sketches and step-wise plans.
