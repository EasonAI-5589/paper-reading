# Real-Time Execution of Action Chunking Flow Policies (RTC)

**作者**: Kevin Black, Manuel Y. Galliker, Sergey Levine
**机构**: Physical Intelligence, UC Berkeley
**会议**: NeurIPS 2025
**链接**: [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) | [项目页面](https://pi.website/research/real_time_chunking) | [代码 (仿真)](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)

---

## 一句话总结

RTC 是一种纯推理时算法，通过将异步 action chunking 建模为 **inpainting 问题**（冻住已执行 action + soft masking guidance 填充剩余部分），让任何 diffusion/flow-based VLA 实现平滑实时执行，**无需重新训练**。

---

## 核心贡献

1. **Inpainting 框架**: 首次将 diffusion/flow inpainting（ΠGDM guidance）应用于实时机器人控制
2. **Soft Masking**: 指数衰减的 guidance 权重，确保跨 chunk 连续性，比 hard masking 显著更好
3. **Guidance Weight Clipping (β)**: 适配少 denoising steps 的控制场景，防止 action chunk 发散
4. **Kinetix Benchmark**: 12 个高动态仿真任务，填补现有 quasi-static benchmark 的空白
5. **大规模真实实验**: 6 个双臂任务 × 480 episodes × 28h 机器人时间，用 π₀.₅ 验证

---

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1（点火柴演示 + 轨迹对比） |
| [01 - Introduction](sections/01-introduction.md) | 动机：物理世界不等你 + action chunking 的两难 |
| [02 - Preliminaries](sections/02-preliminaries.md) | 符号定义 + flow matching 基础 + 延迟数据 + Figure 2/3 |
| [03 - Method](sections/03-method.md) | **核心方法**：ΠGDM inpainting + soft masking + Algorithm 1 |
| [04 - Experiments](sections/04-experiments.md) | 仿真 12 任务 + 真实 6 任务 + Figure 5/6 |
| [05 - Related Work](sections/05-related-work.md) | 定位：vs 加速推理、MPC、BID、System 1/2 |
| [06 - Discussion](sections/06-discussion.md) | 局限性 + 未来方向 |
| [07 - Appendix](sections/07-appendix.md) | β 消融 + 延迟分解 + soft masking 消融 + 超参数 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| RTC 模型延迟 | 97ms (vs 76ms vanilla π₀.₅) |
| 延迟开销 | +28% (来自反向传播) |
| 控制频率 | 50Hz (Δt = 20ms) |
| 支持的最大延迟 | 300ms+ (d ≈ 16, H=50 的 32%) |
| BID 延迟 | 223ms (RTC 的 2.3x) |
| 仿真任务 | 12 (Kinetix, 高动态) |
| 真实任务 | 6 (双臂, 含 2 个移动操作) |
| 真实评估量 | 480 episodes, 28h |
| β (guidance clipping) | 5 |
| Denoising steps | 5 |

---

## 方法概览

```
执行线程 (50Hz)                推理线程 (后台)
─────────────────              ─────────────────
执行 chunk A:                  
  a₀ a₁ a₂ a₃ ...            收到 o，开始生成 chunk B
  ───────────────              ┌─────────────────────┐
  ↑ frozen (d步)               │ ΠGDM Inpainting:    │
  已执行，不可更改              │  frozen: W=1         │
                               │  soft:   W=exp decay │
  ↑ soft guidance              │  free:   W=0         │
  前一个 chunk 有参考值         └──────────┬──────────┘
                                          │
  ↑ free generation                       ↓
  前一个 chunk 没覆盖            新 chunk B 就绪 → 替换
```

---

## 与相关方法对比

| 方法 | 原理 | 需要训练？ | 计算开销 | 延迟鲁棒性 |
|------|------|-----------|---------|-----------|
| Synchronous | 执行完停下等 | 否 | 1x | ❌ 线性下降 |
| Temporal Ensembling | 多 chunk 取平均 | 否 | 1x | ❌❌ 高延迟直接崩溃 |
| BID | 采样 N 个挑最好 | 需要 weak model | ~50x | ⚠️ 中等 |
| **RTC** | **Inpainting + soft mask** | **否** | **~1.3x** | **✅ 完全鲁棒** |

---

## 📊 Citation Landscape

> ⚠️ Semantic Scholar API 当前不可用，Citation Landscape 数据待补充。

**Connected Papers**: [查看](https://www.connectedpapers.com/main/2506.07339)
