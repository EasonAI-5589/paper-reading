# VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference

> **VLASH** — 未来状态感知的异步推理框架

| 属性 | 值 |
|------|-----|
| arXiv | [2512.01031](https://arxiv.org/abs/2512.01031) |
| 日期 | 2025-11 |
| 作者 | Jiaming Tang et al. (MIT Han Lab) |
| 课题 | VLA 实时推理加速 |
| 代码 | [github.com/mit-han-lab/vlash](https://github.com/mit-han-lab/vlash) |

## 核心问题

VLA 异步推理中，推理期间机器人和环境持续变化，导致**预测区间和执行区间之间的时间错位**，造成动作不稳定。

## 方法

- **Future-State-Aware**：用上一轮 action chunk 将 robot state 前向滚动，估计未来执行时的状态
- 桥接预测和执行之间的 gap
- **无需额外开销或架构修改**，通用异步推理框架

## 关键结果

| 指标 | 提升 |
|------|------|
| 推理加速 | **2.03x** vs 同步推理 |
| 反应延迟降低 | **17.4x** |
| 精度 | 完全保持原始精度 |

- 支持高反应速度任务：**打乒乓球、打地鼠**（同步推理完全失败的场景）

---

*待深度阅读* 📖
