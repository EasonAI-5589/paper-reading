# 6. Evaluation Results

> 来源: RoboBrain 2.0 Technical Report

---

## 📄 原文

> 💡 **Section 概览**: 12 个 benchmark，分空间和时序两大类。32B 在 6 个上取得 SOTA。

---

### 6.1 Spatial Reasoning (9 benchmarks)

> 💡 **Table 2 + Table 3 批读**:

**SOTA 结果 (32B)**:
| Benchmark | RoboBrain-32B | 第二名 | 提升 |
|-----------|---------------|--------|------|
| **BLINK** (All) | 83.63 | GPT-o4-mini 83.57 | +0.06 |
| **RoboSpatial** | **72.43** | Gemini-2.5-Pro 59.87 | **+12.56** |
| **RefSpatial-Bench** | **54.00** | Gemini-2.5-Pro 38.16 | **+15.84** |
| **SAT** | **86.67** | GPT-o4-mini 82.00 | +4.67 |
| **Where2Place** | **73.59** | Qwen2.5-VL-72B 39.92 | **+33.67** |
| **ShareRobot Afford.** | **35.28** | Qwen2.5-VL-72B 23.80 | +11.48 |
| **ShareRobot Traj.** | **0.2368** (DFD↓) | Qwen2.5-VL-72B 0.5034 | **-53%** |

> 💡 **关键发现**:
> 1. **Where2Place 领先 33.67 分** — 这是最大的 gap，说明 v2 的 spatial referring 数据起了巨大作用
> 2. **RefSpatial-Bench** — 其他模型基本 < 20（GPT-4o 只有 8.78），RoboBrain 达到 54，说明精确空间定位是专项训练的产物
> 3. **RoboSpatial** — 通用模型 <60，RoboBrain-32B 72.43，说明 embodied spatial reasoning 需要专门数据
> 4. **BLINK 上 7B 比 32B 好** — 7B (83.95) > 32B (83.63)，可能是过拟合或数据分布问题
> 5. **ShareRobot Affordance 从 27.1% (v1) 提升到 35.28% (v2)** — 得益于 PACO-LVIS 561K affordance 数据

---

### 6.2 Temporal Reasoning (3 benchmarks)

> 💡 **Table 4 批读**:

| Benchmark | RoboBrain-32B | 第二名 | 提升 |
|-----------|---------------|--------|------|
| **Multi-Robot Plan** | 80.33 | GPT-4o 74.50 | **+5.83** |
| **EgoPlan2** | **57.23** | Qwen2.5-VL-32B 56.25 | +0.98 |
| **RoboBench** | 68.33 | 7B版 72.16 (SOTA) | — |

> 💡 **关键发现**:
> ```
> Multi-Robot Planning (3 场景):
> ├── Supermarket: 84.42 (32B) / 83.92 (7B)
> ├── Restaurant: 72.36 (32B) / 77.39 (7B) ← 7B 更好!
> └── Household: 85.43 (32B) / 84.42 (7B)
> 
> 注意: RoboBrain-7B-1.0 只有 5.50 分 → v2 的 7B 达到 81.50
> 提升了 76 分! 说明 Multi-Robot 数据 (44K) 效果极其显著。
> ```

> 💡 **RoboBench 上 7B > 32B**: 72.16 vs 68.33
> 这很有趣 — 可能是 7B 在 Stage 2 的 robotic planning 数据上拟合更好。

---

## 💡 Section 总结

### vs v1 性能对比
| Task | v1 | v2-7B | v2-32B |
|------|------|-------|--------|
| Affordance AP | 27.1% | 28.05% | **35.28%** |
| Trajectory DFD | 0.109 | 0.5512 | **0.2368** |
| Multi-Robot | ❌ | 81.50 | 80.33 |
| Spatial Referring | ❌ | 32.50 | **54.00** |

> 💡 **注意 Trajectory DFD**: v1 的 0.109 是在 ShareRobot 测试集上用 T-LoRA 的结果，
> v2 的 0.2368/0.5512 是在 ShareRobot-Bench 上统一模型的结果，评测标准可能不同。
> 这说明**统一模型的 trajectory 预测可能不如专门 LoRA 精确**。

### 核心洞察
1. **空间推理大幅提升** — 得益于 2.5M+ 专项空间数据和 pseudo-3D pipeline
2. **时序推理从无到有** — v1 完全没有，v2 直接 SOTA
3. **7B 在部分任务上 > 32B** — BLINK, RoboBench, Restaurant — 说明大模型不总是更好
4. **超越闭源大模型** — GPT-4o, Gemini-2.5-Pro, Claude-Sonnet-4 在 embodied 任务上都不行
