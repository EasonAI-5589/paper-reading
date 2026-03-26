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

**TLDR** (Semantic Scholar): *Results demonstrate that RTC is fast, performant, and uniquely robust to inference delay, significantly improving task throughput and enabling high success rates in precise tasks - such as lighting a match - even in the presence of significant latency.*

**引用统计**: 参考文献 70 篇 | 被引 71 次 | Influential Citations: 12

### 参考文献分组 (Top 5 per category, by citations)

#### Action Chunking / Real-Time Inference
| 论文 | 年份 | 引用 |
|------|------|------|
| Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success | 2025 | 350 |
| FAST: Efficient Action Tokenization for Vision-Language-Action Models | 2025 | 343 |
| Real-Time Neural MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms | 2022 | 193 |
| Control delay in Reinforcement Learning for real-time dynamic systems: A memoryless approach | 2010 | 75 |
| Bidirectional Decoding: Improving Action Chunking via Closed-Loop Resampling | 2024 | 25 |

#### VLA / Robot Policy
| 论文 | 年份 | 引用 |
|------|------|------|
| Diffusion policy: Visuomotor policy learning via action diffusion | 2023 | 2,702 |
| RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | 2023 | 2,621 |
| OpenVLA: An Open-Source Vision-Language-Action Model | 2024 | 1,832 |
| Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | 2023 | 1,449 |
| π0: A Vision-Language-Action Flow Model for General Robot Control | 2024 | 1,327 |

#### Diffusion / Flow / Inverse Problems
| 论文 | 年份 | 引用 |
|------|------|------|
| Denoising Diffusion Probabilistic Models | 2020 | 28,506 |
| Flow Matching for Generative Modeling | 2022 | 3,719 |
| Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | 2022 | 2,510 |
| RePaint: Inpainting using Denoising Diffusion Probabilistic Models | 2022 | 1,954 |
| Consistency Models | 2023 | 1,636 |

#### Benchmarks / Datasets / Embodiment
| 论文 | 年份 | 引用 |
|------|------|------|
| BridgeData V2: A Dataset for Robot Learning at Scale | 2023 | 574 |
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | - | 284 |
| ALOHA Unleashed: A Simple Recipe for Robot Dexterity | 2024 | 176 |
| RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot | 2023 | 171 |
| Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks | 2024 | 27 |

#### Control / Planning
| 论文 | 年份 | 引用 |
|------|------|------|
| Temporal Difference Learning for Model Predictive Control | 2022 | 374 |
| QueST: Self-Supervised Skill Abstractions for Learning Continuous Control | 2024 | 47 |
| Thinking While Moving: Deep Reinforcement Learning with Concurrent Control | 2020 | 42 |
| Planning and Learning in Environments with Delayed Feedback | 2007 | 34 |
| Process Systems Analysis and Control , chapter 18 | 2009 | 0 |

### 推荐论文（Semantic Scholar Recommendations）

| 论文 | 年份 | 引用 | arXiv |
|------|------|------|-------|
| Causal World Modeling for Robot Control | 2026 | 10 | [2601.21998](https://arxiv.org/abs/2601.21998) |
| DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation | 2026 | 5 | [2601.22153](https://arxiv.org/abs/2601.22153) |
| RISE: Self-Improving Robot Policy with Compositional World Model | 2026 | 4 | [2602.11075](https://arxiv.org/abs/2602.11075) |
| VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model | 2026 | 4 | [2602.12063](https://arxiv.org/abs/2602.12063) |
| AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge | 2026 | 3 | [2602.13476](https://arxiv.org/abs/2602.13476) |
| WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL | 2026 | 3 | [2602.13977](https://arxiv.org/abs/2602.13977) |
| World-Gymnast: Training Robots with Reinforcement Learning in a World Model | 2026 | 3 | [2602.02454](https://arxiv.org/abs/2602.02454) |
| How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf | 2026 | 2 | [2602.18397](https://arxiv.org/abs/2602.18397) |
| Learning Native Continuation for Action Chunking Flow Policies | 2026 | 2 | [2602.12978](https://arxiv.org/abs/2602.12978) |
| RL-VLA3: Reinforcement Learning VLA Accelerating via Full Asynchronism | 2026 | 2 | [2602.05765](https://arxiv.org/abs/2602.05765) |

### 🔗 相关链接

- [Connected Papers](https://www.connectedpapers.com/main/2506.07339)
- [Semantic Scholar](https://www.semanticscholar.org/paper/d0f525dba7d3425e36316127424e67fe2c2fdb0d)
- [arXiv](https://arxiv.org/abs/2506.07339)
