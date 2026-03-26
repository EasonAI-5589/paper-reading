# Leave No Observation Behind: Real-time Correction for VLA Action Chunks

**作者**: Kohei Sendai, Maxime Alvarez, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa
**机构**: The University of Tokyo
**期刊**: arXiv 2025
**链接**: [arXiv 2509.23224](https://arxiv.org/abs/2509.23224) | [Semantic Scholar](https://www.semanticscholar.org/paper/91e498568adc30045672c57d3991cbd8ae4d859e) | [Kinetix 代码](https://github.com/k1000dai/a2c2-kinetix) | [LIBERO 代码](https://github.com/k1000dai/a2c2-libero)

---

## 一句话总结

A2C2 不去重写整段 action chunk，而是在每个控制步用一个轻量 correction head 对当前要执行的 base action 做 residual 校正，从而把 chunking policy 丢掉的 closed-loop 反应性补回来。

---

## 核心贡献

1. **延迟形式化**: 明确写出了 action chunking VLA 在异步执行里的 `H / e / d` 关系，以及动作最旧会落后 `d + e` 步 observation
2. **逐步校正头**: 通过 `latest observation + base action + chunk 内时间位置 + base policy 特征` 预测 residual action，不需要重训 base policy
3. **与 RTC 正交**: RTC 修的是 chunk 切换连续性，A2C2 修的是 chunk 执行期间的逐步反应性，两者可以叠加
4. **动态任务验证**: 在 Kinetix 和 LIBERO Spatial 上都证明了对 delay 与 long horizon 的鲁棒收益

---

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：A2C2 的定位、输入输出和核心结果 |
| [01 - Introduction](sections/01-introduction.md) | 动机：大 VLA 延迟、action chunking 的 open-loop 问题、与 RTC/层级架构的区别 |
| [02 - Problem Formulation](sections/02-problem-formulation.md) | `H / e / d` 形式化、等待时间条件与最坏 observation stale 程度 |
| [03 - Method](sections/03-method.md) | 核心方法：correction head、时间特征、残差训练目标 |
| [04 - Experimental Setup](sections/04-experimental-setup.md) | Kinetix / LIBERO 数据与两套 correction head 架构 |
| [05 - Results](sections/05-results.md) | Kinetix 和 LIBERO 上的主结果、Figure 5/6 与 Table 1 |
| [06 - Related Work](sections/06-related-work.md) | imitation learning、VLA、异步 chunk 执行与推理加速 |
| [07 - Conclusion](sections/07-conclusion.md) | 结论、适用边界和远程 client-server VLA 部署视角 |
| [08 - Appendix](sections/08-appendix.md) | 环境、训练细节、表格、推理时间与硬件资源 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| Kinetix 任务数 | 12 |
| LIBERO Spatial 任务数 | 10 |
| Kinetix delay 场景对 RTC 的提升 | `+23` 个百分点（摘要） |
| Kinetix 长 horizon 场景对 RTC 的提升 | `+7` 个百分点（摘要） |
| Kinetix correction head 参数量 | `0.31M` |
| LIBERO correction head 参数量 | `32M` |
| LIBERO base policy | `SmolVLA (450M)` |
| 推理时间对比 | `4.7 ms`（correction head） vs `101 ms`（SmolVLA） |

---

## 方法概览

```text
base policy
  -> 生成 action chunk

latest observation + base action + time feature + base features
  -> correction head
  -> residual action

execution action = base action + residual
```

---

## A2C2 与 RTC

如果把 RTC 看成“在 chunk 与 chunk 之间做平滑切换”，那 A2C2 更像“在 chunk 内每一步持续打补丁”。

- **RTC**: 解决异步切换时的 chunk continuity，核心是 inpainting 和 soft masking
- **A2C2**: 解决 chunk 执行期间观察陈旧的问题，核心是 residual correction
- **共同点**: 都不要求重新训练大模型本体
- **差异点**: RTC 主要是 inference-time chunk stitching；A2C2 额外训练一个小头，但每步都能看最新 observation

1. **action chunking 省掉的是推理次数，不是 observation 新鲜度问题**
2. **即使没有显式注入 delay，长 horizon 本身也会把策略推向 open-loop**
3. **补一个足够小、足够快的 residual head，可能比继续堆大模型更直接**

---

## 📊 Citation Landscape

**TLDR** (Semantic Scholar): *Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA's action chunk, indicates that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control.*

**引用统计**: 参考文献 `27` 篇 | 被引 `8` 次 | Influential Citations `2`

### 参考文献分组

#### Action Chunking / Asynchronous Execution
| 论文 | 年份 | 引用 |
|------|------|------|
| Real-Time Execution of Action Chunking Flow Policies | 2025 | 74 |
| Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling | 2024 | 13 |
| Fast Policy Synthesis with Variable Noise Diffusion Models | 2024 | 30 |
| Streaming Flow Policy: Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories | 2025 | 9 |

#### VLA / Robot Policy
| 论文 | 年份 | 引用 |
|------|------|------|
| SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | 2025 | 220 |
| Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models | 2025 | 150 |
| Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications | 2025 | 58 |
| Gemini Robotics: Bringing AI into the Physical World | 2025 | 265 |

#### Imitation / Action Generation
| 论文 | 年份 | 引用 |
|------|------|------|
| Diffusion policy: Visuomotor policy learning via action diffusion | 2023 | 2710 |
| Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | 2023 | 1457 |
| FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation | 2024 | 78 |
| An Algorithmic Perspective on Imitation Learning | 2018 | 970 |

#### Benchmarks / Systems
| 论文 | 年份 | 引用 |
|------|------|------|
| Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks | 2024 | 27 |
| Real-world robot applications of foundation models: a review | 2024 | 104 |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | 2023 | 5074 |

### 推荐论文（Semantic Scholar Recommendations）

| 论文 | 年份 | 引用 | arXiv |
|------|------|------|-------|
| Causal World Modeling for Robot Control | 2026 | 11 | [2601.21998](https://arxiv.org/abs/2601.21998) |
| DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation | 2026 | 5 | [2601.22153](https://arxiv.org/abs/2601.22153) |
| Learning Native Continuation for Action Chunking Flow Policies | 2026 | 3 | [2602.12978](https://arxiv.org/abs/2602.12978) |
| AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge | 2026 | 3 | [2602.13476](https://arxiv.org/abs/2602.13476) |
| Real-Time Robot Execution with Masked Action Chunking | 2026 | 2 | [2601.20130](https://arxiv.org/abs/2601.20130) |
| RL-VLA3: Reinforcement Learning VLA Accelerating via Full Asynchronism | 2026 | 2 | [2602.05765](https://arxiv.org/abs/2602.05765) |
| How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf | 2026 | 2 | [2602.18397](https://arxiv.org/abs/2602.18397) |
| TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments | 2026 | 1 | [2602.02459](https://arxiv.org/abs/2602.02459) |

### 🔗 相关链接

- [Connected Papers](https://www.connectedpapers.com/main/2509.23224)
- [Semantic Scholar](https://www.semanticscholar.org/paper/91e498568adc30045672c57d3991cbd8ae4d859e)
- [arXiv](https://arxiv.org/abs/2509.23224)
