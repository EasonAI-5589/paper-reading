# VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference

**作者**: Jiaming Tang, Yufei Sun, Yilong Zhao, Shang Yang, Yujun Lin, Zhuoyang Zhang, James Hou, Yao Lu, Zhijian Liu, Song Han
**机构**: MIT, NVIDIA, Tsinghua University, UC Berkeley, UCSD, Caltech
**期刊**: arXiv 2025
**链接**: [arXiv 2512.01031](https://arxiv.org/abs/2512.01031) | [项目代码](https://github.com/mit-han-lab/vlash)

---

## 一句话总结

VLASH 通过在推理前先用旧 chunk 将 robot state 前滚到执行时刻，并配合偏移增强微调让模型真正学会利用 future state，从而在几乎不增加运行时开销、不改主干架构的前提下，实现平滑、稳定且快速的异步控制；此外还可结合 action quantization 进一步压缩执行时间。

---

## 核心贡献

1. **Future-state-aware async inference**: 不直接在 stale state 上预测，而是先用旧 chunk 把 robot state roll forward 到执行时刻。
2. **训练时 offset augmentation**: 固定视觉 observation、联动偏移 state 与 target action，逼模型真正学会使用 proprioceptive state。
3. **Shared-observation fine-tuning**: 多个 offset 共享一次 observation 编码，用 block-sparse attention 明显降低 fine-tuning 成本。
4. **Action quantization**: 在异步推理已经隐藏模型延迟之后，进一步通过action合并压缩执行时间（**速度提升主要来源**）

---

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要与全文定位：future-state-aware async inference |
| [01 - Introduction](sections/01-introduction.md) | 异步推理动机、prediction-execution misalignment、和 RTC/A2C2 的位置关系 |
| [02 - Related Work](sections/02-related-work.md) | VLA、异步推理与 concurrent work 的定位 |
| [03 - Background](sections/03-background.md) | `H / K / Δ` 定义与 prediction interval / execution interval 失配 |
| [04 - Method](sections/04-method.md) | future-state awareness、offset augmentation、shared observation、action quantization |
| [05 - Experiments](sections/05-experiments.md) | Kinetix、LIBERO、真实机器人、reaction speed、fine-tuning efficiency |
| [06 - Conclusion](sections/06-conclusion.md) | 方法边界与整体结论 |
| [07 - Appendix](sections/07-appendix.md) | SmolVLA 泛化、超参数、补充视频与架构细节 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| 相对同步推理速度提升 | 最高 **2.03x** |
| reaction latency 降低 | 最高 **17.4x** |
| Kinetix 延迟 4 步成功率 | **81.7%** |
| 相对 Naive Async 的提升 | **+30.5** 个百分点 |
| shared-observation fine-tuning 单步加速 | **3.26x** |
| SmolVLA 在 LIBERO 的附录 speedup | 最高 **1.35x** |

---

## 方法概览

```
执行线程                          推理线程 (后台)
─────────────────                 ─────────────────
执行 chunk A:                  
  a₀ a₁ a₂ a₃ ...               收到 o_t 和前滚状态 s_{t+Δ}，开始推理
  ───────────────                 ┌─────────────────────┐
  ↑ 已执行动作                     │ Future-State-Aware: │
                                  │   输入: o_t         │
  ↑ 推理延迟 Δ                     |   输入: s_{t+Δ}     │
                                  └──────────┬──────────┘
                                             │
                                             ↓
                                新 chunk B 就绪 → 直接对齐未来的执行时刻
```

---

## 与相关方法对比

| 方法 | 核心机制 | 需要训练？ | 额外运行时开销 |
|------|----------|-----------|----------------|
| Synchronous | 执行完停下等 | 否 | 无 |
| Naive Async | 收到新 chunk 立马切换 | 否 | 无 (但会失稳) |
| A2C2 | 增加额外的 residual correction head | 是 (只训小头) | 有 (额外的网络前向) |
| RTC | 运行时 Inpainting + soft mask | 否 | 有 (额外的 diffusion guidance) |
| **VLASH** | **推理前状态前滚 + 训练时 offset 增强** | **是** | **几乎无额外开销** |

---

## 📊 Citation Landscape

**TLDR** (Semantic Scholar): *VLASH is proposed, a general asynchronous inference framework for VLAs that delivers smooth, accurate, and fast reaction control without additional overhead or architectural changes and empowers VLAs to handle fast-reaction, high-precision tasks such as playing ping-pong and playing whack-a-mole, where traditional synchronous inference fails.*

**引用统计**: 参考文献 33 篇 | 被引 15 次 | Influential Citations: 3

### 参考文献分组 (Top 5 per category, by citations)

#### VLA Foundations / Generalist Policies
| 论文 | 年份 | 引用 |
|------|------|------|
| RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | 2023 | 2668 |
| OpenVLA: An Open-Source Vision-Language-Action Model | 2024 | 1869 |
| π0: A Vision-Language-Action Flow Model for General Robot Control | 2024 | 1365 |
| π0.5: a Vision-Language-Action Model with Open-World Generalization | 2025 | 669 |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | 2025 | 595 |

#### Real-Time / Asynchronous Inference
| 论文 | 年份 | 引用 |
|------|------|------|
| SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | 2025 | 220 |
| Real-Time Execution of Action Chunking Flow Policies | 2025 | 77 |
| Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better | 2025 | 68 |
| Running VLAs at Real-time Speed | 2025 | 14 |
| Leave No Observation Behind: Real-time Correction for VLA Action Chunks | 2025 | 7 |

#### Benchmarks / Data / Embodiment
| 论文 | 年份 | 引用 |
|------|------|------|
| LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | 2023 | 685 |
| DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | 2024 | 620 |
| Gemini Robotics: Bringing AI into the Physical World | 2025 | 268 |
| HITTER: A Humanoid Table Tennis Robot via Hierarchical Planning and Learning | 2025 | 31 |
| Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks | 2024 | 27 |

#### Efficiency / Systems / Infrastructure
| 论文 | 年份 | 引用 |
|------|------|------|
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 | 3897 |
| MLP-Mixer: An all-MLP Architecture for Vision | 2021 | 3470 |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | 2424 |
| SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | 2022 | 1403 |
| PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation | 2024 | 1015 |

### 推荐论文（Semantic Scholar Recommendations）

| 论文 | 年份 | 引用 | arXiv |
|------|------|------|-------|
| RL-VLA3: Reinforcement Learning VLA Accelerating via Full Asynchronism | 2026 | 2 | [2602.05765](https://arxiv.org/abs/2602.05765) |
| FASTER: Rethinking Real-Time Flow VLAs | 2026 | 0 | [2603.19199](https://arxiv.org/abs/2603.19199) |
| AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge | 2026 | 3 | [2602.13476](https://arxiv.org/abs/2602.13476) |
| StreamingVLA: Streaming Vision-Language-Action Model with Action Flow Matching and Adaptive Early Observation | 2026 | 0 | [2603.28565](https://arxiv.org/abs/2603.28565) |
| StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating | 2026 | 0 | [2602.01100](https://arxiv.org/abs/2602.01100) |
| ProbeFlow: Training-Free Adaptive Flow Matching for Vision-Language-Action Models | 2026 | 0 | [2603.17850](https://arxiv.org/abs/2603.17850) |
| FUTURE-VLA: Forecasting Unified Trajectories Under Real-time Execution | 2026 | 0 | [2602.15882](https://arxiv.org/abs/2602.15882) |
| Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement | 2026 | 1 | [2602.03983](https://arxiv.org/abs/2602.03983) |

### 🔗 相关链接

- [Connected Papers](https://www.connectedpapers.com/main/2512.01031)
- [Semantic Scholar](https://www.semanticscholar.org/paper/6e3206b3b9302698c4a2df8df26bce6b9710c1bc)
- [arXiv](https://arxiv.org/abs/2512.01031)
