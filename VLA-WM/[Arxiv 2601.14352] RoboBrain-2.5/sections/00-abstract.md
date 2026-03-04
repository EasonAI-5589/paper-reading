[← 返回 README](../README.md)

# Abstract

## 📌 预览
RoboBrain 2.5 是 BAAI 推出的下一代具身 AI 基础模型，在 RoboBrain 2.0 基础上新增两大核心能力：精确 3D 空间推理 和 密集时间价值估计。

---

We introduce RoboBrain 2.5, a next-generation embodied AI foundation model that advances general perception, spatial reasoning, and temporal modeling through extensive training on high-quality spatiotemporal supervision. Building upon its predecessor, RoboBrain 2.5 introduces two major capability upgrades. Specifically, it unlocks Precise 3D Spatial Reasoning by shifting from 2D pixel-relative grounding to depth-aware coordinate prediction and absolute metric constraint comprehension, generating complete 3D manipulation traces as ordered keypoint sequences under physical constraints. Complementing this spatial precision, the model establishes Dense Temporal Value Estimation that provides dense, step-aware progress prediction and execution state understanding across varying viewpoints, producing stable feedback signals for downstream learning. Together, these upgrades extend the framework toward more physically grounded and execution-aware embodied intelligence for complex, fine-grained manipulation. The code and checkpoints are available at project website: https://superrobobrain.github.io.

> 💡 **Abstract 批读**:
> - **核心定位**: 具身 AI 基础模型，从"语义推理器"进化为"物理接地的智能体"
> - **两大新能力**:
>   1. **Precise 3D Spatial Reasoning** ("Depth in Sight"): 从 2D 像素坐标 → 深度感知的 3D 坐标预测 + 操作轨迹生成
>   2. **Dense Temporal Value Estimation** ("Time in Mind"): 提供逐步感知的执行进度预测，支持闭环控制和 RL
> - **关键词**: spatiotemporal supervision, keypoint sequences, physical constraints, feedback signals
> - **开源**: 代码和权重公开

---

## 🔖 Section 总结

### 核心洞察
1. 当前具身模型的两大短板：空间上的"度量盲"（只有 2D）和时间上的"开环预测"（无中间反馈）
2. RoboBrain 2.5 通过 3D 空间推理 + 密集时间估计，实现从语义到物理的范式转变
3. 使用解耦的 $(u, v, d)$ 表示和 hop-based 标注策略
