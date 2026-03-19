# Real-Time Execution of Action Chunking Flow Policies (RTC)

**作者**: Kevin Black, Manuel Y. Galliker, Sergey Levine  
**机构**: Physical Intelligence, UC Berkeley  
**会议**: NeurIPS 2025  
**链接**: [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) | [Blog](https://pi.website/research/real_time_chunking) | [Code (sim)](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)

---

## 一句话总结

RTC 是一个纯推理时算法，通过把异步 action chunking 建模为 **inpainting 问题**，让 diffusion/flow VLA 在有推理延迟（甚至 300ms+）的情况下仍能**平滑、实时**地控制机器人，无需重训。

## 核心贡献

1. **RTC 算法**：基于 ΠGDM guidance 的 action chunk inpainting，冻结已执行动作、补全剩余动作
2. **Soft masking**：指数衰减权重替代 hard mask，大幅提升跨 chunk 连续性
3. **Guidance weight clipping ($\beta$)**：解决少步去噪时 guidance 发散的问题
4. **Kinetix benchmark**：12 个高动态模拟任务，填补现有 quasi-static benchmark 的空白
5. **真实世界验证**：6 个双臂灵巧操作任务（含划火柴、插网线），480 episodes，28 小时

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1 (划火柴 demo + 速度/平滑度对比) |
| [01 - Introduction](sections/01-introduction.md) | 动机：VLA 延迟问题 + action chunking 的不足 |
| [02 - Preliminaries](sections/02-preliminaries.md) | 核心定义 ($H$, $s$, $d$) + 为什么现有方法不够 |
| [03 - Method](sections/03-method.md) | ⭐ ΠGDM inpainting + soft masking + Algorithm 1 |
| [04 - Experiments](sections/04-experiments.md) | ⭐ Kinetix 12 任务 + π₀.₅ 6 个真实世界任务 |
| [05 - Related Work](sections/05-related-work.md) | 五类相关工作 + RTC 在 landscape 中的位置 |
| [06 - Discussion](sections/06-discussion.md) | 局限性 + Appendix 亮点 (延迟 breakdown, 超参数) |

## 关键数字

| 指标 | 数值 |
|------|------|
| 基础模型 | π₀.₅ (3B 参数) |
| Prediction horizon $H$ | 50 |
| 控制频率 | 50Hz (20ms) |
| RTC 推理延迟 | 97ms (vs vanilla 76ms, +28%) |
| 对 +200ms 注入延迟 | ✅ 无性能下降 |
| TE 在 +100ms 时 | ❌ 保护性停机 |
| 比 BID 快 | 2.3x (97ms vs 223ms) |
| 真实世界速度提升 | ~20% vs synchronous |
| 评估规模 | 480 episodes / 28h |

## 方法核心图

```
旧 chunk 执行中：  [a₀ a₁ a₂ a₃ | a₄ a₅ ... a₁₀ | a₁₁ ... a₁₅]
                     ↑ frozen (W=1)   ↑ soft mask    ↑ fresh (W=0)
                     已执行，必须匹配    指数衰减        自由生成
                     ← d steps →      ← 过渡区 →     ← s steps →
                     
新 chunk = inpaint(noise, 观测, 旧chunk前缀, W)
```

## 对 VLA-RTC 系列的意义

RTC 是这个方向的**奠基工作**：
- Training-time RTC (2512.05964) → 训练时模拟 delay，消除推理时 inpainting 开销
- VLASH (2512.01031) → 异步推理 + future-state prediction，和 RTC 思路互补
- Leave No Observation Behind (2509.23224) → 类似的 real-time correction 思想

---

*批读完成于 2026-03-19 📚*
