# Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space (DMLR)

**作者**: Chengzhi Liu*, Yuzhe Yang*, Yue Fan, Qingyue Wei, Sheng Liu†, Xin Eric Wang† (UC Santa Barbara / Stanford / UC Santa Cruz)
**会议/期刊**: Arxiv preprint | **年份**: 2025 (v1: 2025-12-14)
**链接**: [arXiv:2512.12623](https://arxiv.org/abs/2512.12623) | [Project Page](https://mllm-dmlr.github.io/) | [PDF](https://arxiv.org/pdf/2512.12623)

---

## 一句话总结

DMLR 是一个 **训练免费 (training-free) 的 test-time 多模态隐空间推理框架**：用「置信度作奖励」做策略梯度优化 latent think tokens，并通过「动态视觉注入」把最相关的图像 patch 实时塞进 latent 推理流，让 MLLM 像人脑一样按需"瞄一眼图"，而不是固定步骤显式 CoT。

---

## 核心贡献

1. **两个关键观察 (Section 3)**
   - **Takeaway 1+2 (视觉依赖)**: 推理过程中视觉信息只在 **少数 token** 上发挥作用，且不同推理链对视觉的依赖差异巨大——视觉依赖强的链更准。
   - **Observation 1-3 (置信度信号)**: 内部置信度 (token entropy) 同时反映**推理正确性**、**推理质量** (faithful vs spurious) 和**视觉接地性** (hallucination)。
2. **DMLR 框架 (Section 4)**
   - **可优化的 latent think tokens**：注入 L=4 个可学习的 latent embedding 作为 "mental draft"。
   - **置信度引导的策略梯度** (Eq.8-9, REINFORCE-style)：以 1 − truncated entropy 为 reward，对 latent tokens 做 test-time 梯度上升。
   - **动态视觉注入 (DVI)**：每步根据 latent 的 attention 重采样 m=2 个候选 patch，与历史 best patch 拼接后比较 reward 决定是否替换 best。
3. **两条理论保证 (Section 4.3)**
   - Theorem 4.1: 置信度梯度与质量梯度对齐时，沿置信度上升等价于推理质量提升。
   - Theorem 4.2: 视觉注入提升 latent 与视觉的互信息 → 提升期望置信度。
4. **大规模实证 (Section 5)**: 在 7 个 benchmark × 6 种 backbone (Qwen2.5-VL-3B/7B, Qwen3-VL-4B/8B, R1-OneVision, VLAA-Thinking) 上 95%+ 任务取得最佳；R1-OneVision +4.5% (数学) / +3.45% (视觉)；VLAA-Thinking 平均 +2.43%；推理与感知不再 trade-off。
5. **效率优势**: 完全在 latent 空间迭代，不增加生成长度；DVI 只挑相关 patch，避免 ICoT 那种大量视觉 token 的 decode 开销 (Figure 11)。

---

## 📖 批读导航

| Section | 内容 | 关键 Figure/Table |
|---------|------|-----|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 论文动机一句话 | — |
| [01 - Introduction](sections/01-introduction.md) | 三类多模态推理范式对比 + DMLR 定位 | Figure 1 |
| [02 - Related Work](sections/02-related-work.md) | Explicit reasoning vs Latent reasoning 谱系 | — |
| [03 - Preliminary & Motivation](sections/03-preliminary-motivation.md) | RQ1/RQ2：视觉依赖度 & 置信度的实证分析 | Figure 2, 3, 4 |
| [04 - Methodology](sections/04-methodology.md) | DMLR 主体：latent token + 置信度 reward + DVI + 理论 | Figure 5, Algorithm 1 |
| [05 - Experiments](sections/05-experiments.md) | 主实验 + 消融 + 定性分析 + 效率分析 | Table 1, 2, Figure 6-11 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 | — |
| [07 - Appendix](sections/07-appendix.md) | 数据集详情 / 超参 / Case Study | Figure 12, 13 |

---

## 关键数字速查

| 指标 | 数值 |
|------|------|
| Latent think token 数 L | 4 |
| 候选视觉 patch 数 m | 2 |
| 优化迭代步数 T | 15 |
| 学习率 η | 1e-3 |
| Noise 扰动 σ | 10% (decay 0.95) |
| 评测 backbone 数 | 6 (Qwen2.5-VL-3B/7B, Qwen3-VL-4B/8B, R1-OneVision, VLAA-Thinking) |
| Benchmark 数 | 7 (MathVista/MathVision/MM-Math/HallusionBench/MMVP/MMStar/ScienceQA) |
| 实验硬件 | 4× NVIDIA H100，float32，eager attention |
| R1-OneVision +DMLR 数学平均提升 | +4.5% |
| R1-OneVision +DMLR 视觉平均提升 | +3.45% |
| VLAA-Thinking 所有 benchmark 平均提升 | +2.43% |

---

## 📊 Citation Landscape

> 数据来源：[Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/ArXiv:2512.12623) | 拉取于 2026-05-18

### TLDR (Semantic Scholar 自动摘要)
> *DMLR is proposed, a test-time Dynamic Multimodal Latent Reasoning framework that employs confidence-guided latent policy gradient optimization to refine latent think tokens for in-depth reasoning and significantly improves reasoning and perception performance while maintaining high inference efficiency.*

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 53 |
| 被引次数 | 8 |
| Influential Citation Count | 1 |
| Semantic Scholar | [paperId 2affab1847f59b51b277d194e94ad14c1b0d3933](https://www.semanticscholar.org/paper/2affab1847f59b51b277d194e94ad14c1b0d3933) |
| Connected Papers | [访问图谱](https://www.connectedpapers.com/main/2affab1847f59b51b277d194e94ad14c1b0d3933) |

### 参考文献分组（Top 5 by citation count，from 53 refs）

**🧠 Latent Reasoning（18 篇，DMLR 最直接的家族）**
| Paper | Year | Citations | Link |
|---|---|---|---|
| CoCoNut — Training LLMs to Reason in Continuous Latent Space | 2024 | 501 | [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) |
| ThinkAct — VLA Reasoning via Reinforced Visual Latent Planning | 2025 | 104 | [arXiv:2507.16815](https://arxiv.org/abs/2507.16815) |
| Soft Thinking — Continuous Concept Space Reasoning | 2025 | 71 | [arXiv:2505.15778](https://arxiv.org/abs/2505.15778) |
| Machine Mental Imagery (Mirage) | 2025 | 65 | [arXiv:2506.17218](https://arxiv.org/abs/2506.17218) |
| Reducing Hallucinations via Latent Space Steering | 2025 | 55 | — |

**🖼️ Think-with-Image / Visual Reasoning（7 篇）**
| Paper | Year | Citations | Link |
|---|---|---|---|
| Visual Sketchpad — Sketching as Visual CoT | 2024 | 260 | [arXiv:2406.09403](https://arxiv.org/abs/2406.09403) |
| DeepEyes — Thinking with Images via RL | 2025 | 192 | [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) |
| Pixel Reasoner — Curiosity-Driven RL | 2025 | 184 | [arXiv:2505.15966](https://arxiv.org/abs/2505.15966) |
| MVoT — Multimodal Visualization-of-Thought | 2025 | 171 | [arXiv:2501.07542](https://arxiv.org/abs/2501.07542) |
| ReFocus — Visual Editing as CoT | 2025 | 66 | [arXiv:2501.05452](https://arxiv.org/abs/2501.05452) |

**🤖 MLLM / VLM Backbones（5 篇）**
| Paper | Year | Citations | Link |
|---|---|---|---|
| Qwen2.5-VL Technical Report | 2025 | 4537 | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) |
| InternVL3.5 | 2025 | 768 | [arXiv:2508.18265](https://arxiv.org/abs/2508.18265) |
| R1-OneVision | 2025 | 331 | [arXiv:2503.10615](https://arxiv.org/abs/2503.10615) |
| GLM-4.5V/4.1V-Thinking | 2025 | 205 | [arXiv:2507.01006](https://arxiv.org/abs/2507.01006) |
| Qwen3-VL | 2025 | 0 | [Blog](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef) |

**📊 Benchmarks（2 篇被分组到这里，其余在 Other）**
| Paper | Year | Citations | Link |
|---|---|---|---|
| MMStar — Are We Right Way to Evaluate VLMs? | 2024 | 775 | [arXiv:2403.20330](https://arxiv.org/abs/2403.20330) |
| MMVP — Eyes Wide Shut? | 2024 | 712 | [arXiv:2401.06209](https://arxiv.org/abs/2401.06209) |

**📐 CoT / Reasoning General（9 篇）**
| Paper | Year | Citations | Link |
|---|---|---|---|
| ScienceQA — Multimodal Reasoning via Thought Chains | 2022 | 2259 | [arXiv:2209.09513](https://arxiv.org/abs/2209.09513) |
| Multimodal Chain-of-Thought Reasoning | 2023 | 841 | [arXiv:2302.00923](https://arxiv.org/abs/2302.00923) |
| Vision-R1 | 2025 | 519 | [arXiv:2503.06749](https://arxiv.org/abs/2503.06749) |
| CCoT — Compositional CoT Prompting | 2023 | 207 | [arXiv:2311.17076](https://arxiv.org/abs/2311.17076) |
| SFT or RL? — Early Investigation into R1-like VLMs (VLAA-Thinking) | 2025 | 191 | [arXiv:2504.11468](https://arxiv.org/abs/2504.11468) |

**🚨 Hallucination & Confidence（5 篇）**
| Paper | Year | Citations | Link |
|---|---|---|---|
| HallusionBench | 2023 | 515 | [arXiv:2310.14566](https://arxiv.org/abs/2310.14566) |
| More Thinking, Less Seeing? | 2025 | 64 | [arXiv:2505.21523](https://arxiv.org/abs/2505.21523) |
| Look Twice Before You Answer (Memory-Space Visual Retracing) | 2024 | 60 | [arXiv:2410.03577](https://arxiv.org/abs/2410.03577) |
| Seeing Far and Clearly — Attention Causal Decoding | 2025 | 36 | [arXiv:2505.16652](https://arxiv.org/abs/2505.16652) |
| Seeing and Reasoning with Confidence | 2025 | 11 | [arXiv:2503.08308](https://arxiv.org/abs/2503.08308) |

### 🌟 Semantic Scholar 推荐论文（10 篇最相关）

| Paper | Year | Citations | Link |
|---|---|---|---|
| Visual Latents Know More Than They Say — Unsilencing Latent Reasoning in MLLMs | 2026 | 0 | [arXiv:2605.02735](https://arxiv.org/abs/2605.02735) |
| Visual Enhanced Depth Scaling for Multimodal Latent Reasoning | 2026 | 1 | [arXiv:2604.10500](https://arxiv.org/abs/2604.10500) |
| CoLVR — Contrastive Exploratory Latent Visual Reasoning | 2026 | 0 | [arXiv:2605.08802](https://arxiv.org/abs/2605.08802) |
| Decompose, Look, and Reason — RL Latent Reasoning for VLMs | 2026 | 0 | [arXiv:2604.07518](https://arxiv.org/abs/2604.07518) |
| Thinking Diffusion — Visual-Grounded Reasoning in Diffusion MLMs | 2026 | 0 | [arXiv:2604.05497](https://arxiv.org/abs/2604.05497) |
| Rethinking Token-Level Policy Optimization for Multimodal CoT | 2026 | 1 | [arXiv:2603.22847](https://arxiv.org/abs/2603.22847) |
| MedLVR — Latent Visual Reasoning for Medical VQA | 2026 | 0 | [arXiv:2604.09757](https://arxiv.org/abs/2604.09757) |
| LanteRn — Latent Visual Structured Reasoning | 2026 | 0 | [arXiv:2603.25629](https://arxiv.org/abs/2603.25629) |
| Q-Tacit — IQA via Latent Visual Reasoning | 2026 | 0 | [arXiv:2603.22641](https://arxiv.org/abs/2603.22641) |
| Hybrid Latent Reasoning with Decoupled Policy Optimization | 2026 | 0 | [arXiv:2604.20328](https://arxiv.org/abs/2604.20328) |

> 💡 **观察**: 推荐列表里全是 2026 年发表的 **latent visual reasoning** 方向论文，说明 DMLR 处于一个**正在快速涌现**的新方向中心位置。3-5 月这一拨论文几乎都在做"latent + 视觉 + RL/PO"的组合，是这个 niche 最热的窗口期。

---

*Generated via MinerU + 批读 SKILL（参见 `/mnt/eason/paper-reading/PAPER-READER-SKILL.md`）*
