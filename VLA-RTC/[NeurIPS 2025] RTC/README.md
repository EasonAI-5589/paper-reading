# Real-Time Execution of Action Chunking Flow Policies (RTC)

**作者**: Kevin Black, Manuel Y. Galliker, Sergey Levine
**机构**: Physical Intelligence, UC Berkeley
**会议**: **NeurIPS 2025**
**链接**: [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) | [Project Page](https://pi.website/research/real_time_chunking)

---

## 一句话总结

RTC 是一种纯推理时算法，通过将异步 action chunking 建模为 **inpainting 问题**（冻住已执行 action + soft masking guidance），让任何 diffusion/flow-based VLA 实现平滑实时执行，无需重新训练。

---

## 核心贡献

1. **Inpainting 框架**: 首次将 guidance-based inpainting 应用于实时 action chunking，用 ΠGDM 梯度引导保证 chunk 间连续性
2. **Soft Masking**: 不只冻住前 $d$ 个 action，而是对所有重叠区域指数衰减引导，大幅改善低延迟下的平滑性
3. **Kinetix 动态 Benchmark**: 12 个高动态力控仿真任务，填补了现有 quasi-static benchmark 的空白
4. **大规模真实验证**: 6 个双臂操作任务 × 480 episodes × 28 小时，用 π₀.₅ 3B VLA 验证
5. **延迟鲁棒性**: 即使注入 200ms 额外延迟（总延迟 >300ms），RTC 性能几乎不退化

---

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1（点火柴 + 轨迹对比） |
| [01 - Introduction](sections/01-introduction.md) | 动机：物理世界不等你 + action chunking 的不足 |
| [02 - Preliminaries](sections/02-preliminaries.md) | 数学定义 + 延迟数据 + naive 异步的问题 |
| [03 - Method](sections/03-method.md) | **核心**: ΠGDM inpainting + soft masking + 双线程系统 |
| [04 - Experiments](sections/04-experiments.md) | Kinetix 仿真 + π₀.₅ 真实世界 6 任务 |
| [05 - Related Work](sections/05-related-work.md) | 定位：与加速方法、MPC、BID、Hierarchical VLA 的关系 |
| [06 - Discussion & Appendix](sections/06-discussion.md) | 局限性 + β 消融 + 延迟测量 + 衰减函数对比 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| Model latency (RTC, π₀.₅) | 97ms (vs vanilla 76ms) |
| BID latency (batch=16, full) | 223ms (2.3× RTC) |
| 控制频率 | 50Hz (Δt = 20ms) |
| Prediction horizon $H$ | 50 (real), 8 (sim) |
| Denoising steps $n$ | 5 |
| β (guidance clipping) | 5 |
| 真实实验规模 | 480 episodes, 28 小时 |
| 速度提升 | 比同步推理快 ~20% |

---

## 方法概要

```
┌──────────────────────────────────────────────────────┐
│                  RTC 系统架构                         │
│                                                      │
│  Thread 1: Controller                                │
│  ┌─────────────────────────────────────┐             │
│  │ 每 20ms: 取 action → 发给机器人     │             │
│  │         收 observation → 传给推理线程 │             │
│  └─────────────────────────────────────┘             │
│                                                      │
│  Thread 2: Inference                                 │
│  ┌─────────────────────────────────────┐             │
│  │ 1. 收到 observation                  │             │
│  │ 2. 构建 soft mask W                  │             │
│  │ 3. 5步 denoising + ΠGDM guidance    │             │
│  │ 4. 新 chunk 就绪 → 替换旧 chunk      │             │
│  └─────────────────────────────────────┘             │
│                                                      │
│  Soft Mask:                                          │
│  |████|▓▓▓▓▓▓▓|░░░░░|                               │
│  frozen  decay   free                                │
│  W=1    W→0     W=0                                  │
└──────────────────────────────────────────────────────┘
```

---

## 与其他方法的对比

| 方法 | 需要重训? | 计算开销 | 延迟鲁棒性 | 多模态支持 |
|------|----------|---------|-----------|-----------|
| Synchronous | ❌ | 低 | ❌ 线性退化 | ✅ |
| Temporal Ensembling | ❌ | 低 | ❌ 高延迟崩溃 | ❌ 取平均=灾难 |
| BID | 需要 weak policy | 很高 (64×) | ⚠️ 中等 | ✅ |
| Consistency Policy | ✅ 蒸馏 | 低 | N/A (单步) | ✅ |
| **RTC** | **❌** | **中 (+28%)** | **✅ 几乎免疫** | **✅** |
