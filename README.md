![Paper Reading Banner](./banner.png)

# Paper Reading 📚

Eason 的文献阅读仓库，按课题组织。每篇论文都有「批读格式」阅读笔记：原文完整保留 + 内嵌批注。

---

## 课题列表

### 📊 MLLM Token Compression
多模态大模型视觉 Token 压缩方法

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| ⭐ [Survey](./MLLM-Token-Compression/%5BArxiv%202507.20198%5D%20Survey-Token-Compression/) | arXiv 2507.20198 | **综述** - 50+ 方法全景，4 位置 × 5 决策维度分类 |
| [Survey (TechRxiv)](./MLLM-Token-Compression/%5BTechRxiv%202025%5D%20Survey-Token-Compression/) | TechRxiv 2025 | **综述** - Token 压缩全景综述 |
| [FastV](./MLLM-Token-Compression/%5BECCV%202024%5D%20FastV/) | ECCV 2024 | 第 2 层后固定剪枝，简单高效 |
| [SparseVLM](./MLLM-Token-Compression/%5BICML%202025%5D%20SparseVLM/) | ICML 2025 | 文本引导 + Token 回收 |
| [PyramidDrop](./MLLM-Token-Compression/%5BCVPR%202025%5D%20PyramidDrop/) | CVPR 2025 | 金字塔式渐进剪枝 |
| [SwiftVLM](./MLLM-Token-Compression/%5BArxiv%202403.12178%5D%20SwiftVLM/) | arXiv 2403.12178 | Bypass 范式 + DP 选层 |
| [DivPrune](./MLLM-Token-Compression/%5BCVPR%202025%5D%20DivPrune/) | CVPR 2025 | MMDP 最大化 token 多样性 |
| [VisionZip](./MLLM-Token-Compression/%5BCVPR%202025%5D%20VisionZip/) | CVPR 2025 | Text-agnostic, [CLS] attention + similarity merging |
| [CDPruner](./MLLM-Token-Compression/%5BNeurIPS%202025%5D%20CDPruner/) | NeurIPS 2025 | DPP 条件多样性剪枝 |
| [SCOPE](./MLLM-Token-Compression/%5BNeurIPS%202025%5D%20SCOPE/) | NeurIPS 2025 | Saliency + Coverage 联合优化 |
| [VScan](./MLLM-Token-Compression/%5BArxiv%202505.22654%5D%20VScan/) | TMLR 2026 | 两阶段 Global+Local Scan + Middle Layer Pruning |
| [VisionTrim](./MLLM-Token-Compression/%5BICLR%202026%5D%20VisionTrim/) | ICLR 2026 | Training-free, DVTS (global-local 选) + TGVC (text-guided 补), 两阶段统一压缩 |
| [HoloV](./MLLM-Token-Compression/%5BNeurIPS%202025%5D%20HoloV/) | NeurIPS 2025 | Crop-wise 自适应分配 + Diversity Variance, 88.9% 剪枝保留 95.8% 性能 |

📖 [方法对比总结](./MLLM-Token-Compression/methods-list.md)

---

### 🎬 Video Chaptering
视频章节生成 - 自动将长视频分割成语义连贯的章节

| 论文 | 会议 | 方法特点 | 性能 (F1) |
|------|------|----------|-----------|
| [SODA](./Video-Chaptering/%5BECCV%202020%5D%20SODA/) | ECCV 2020 | **评估指标** - 考虑故事性的评估框架 | - |
| [VidChapters-7M](./Video-Chaptering/%5BNeurIPS%202023%5D%20VidChapters-7M/) | NeurIPS 2023 | **THE Benchmark** - 817K 视频, 7M 章节 | 25.0 |
| [Chapter-Llama](./Video-Chaptering/%5BCVPR%202025%5D%20Chapter-Llama/) | CVPR 2025 | LLM 文本域方法, Speech-guided 采样 | 45.3 |
| [ARC-Chapter](./Video-Chaptering/%5BarXiv%202025%5D%20ARC-Chapter/) | arXiv 2025 | **SOTA** - Qwen2.5-VL + GRPO, GRACE 指标 | **59.3** |

📖 [Video Chaptering 详细总结](./Video-Chaptering/README.md)

---

### 🤖 VLA (Vision-Language-Action)
视觉-语言-动作模型，机器人操作

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [RoboBrain](./VLA-WM/%5BCVPR%202025%5D%20RoboBrain/) | CVPR 2025 | LLaVA + A-LoRA/T-LoRA, ShareRobot 数据集 |
| [RoboBrain 2.0](./VLA-WM/%5BArxiv%202507.02029%5D%20RoboBrain-2.0/) | arXiv 2507.02029 | Qwen2.5-VL base, 统一空间+时间推理 |
| [RoboBrain 2.5](./VLA-WM/%5BArxiv%202601.14352%5D%20RoboBrain-2.5/) | arXiv 2601.14352 | 精确 3D 空间推理 + 密集时间价值估计 |
| [π0](./VLA-WM/%5BCoRL%202025%5D%20pi0/) | CoRL 2025 | Flow matching policy |
| [π0.5](./VLA-WM/%5BCoRL%202025%5D%20pi0.5/) | CoRL 2025 Oral | VLM-based robot policy |
| [π0.6-RECAP](./VLA-WM/%5BICLR%202026%5D%20Pi0.6-RECAP/) | ICLR 2026 | CoT reasoning robot policy |
| ⭐ [π0.6-MEM](./VLA-WM/%5BPI%202026%5D%20MEM/) | PI 2026-03 | **多尺度具身记忆** - 短时视频记忆(ViT空时注意力)+长时语言记忆，支持15分钟长任务，in-context 自适应 |
| [Cosmos Policy](./VLA-WM/%5BArxiv%202601.16163%5D%20Cosmos-Policy/) | arXiv 2601.16163 | **Video Model → Policy**, Latent Frame Injection, 零架构修改, LIBERO 98.5% / RoboCasa 67.1% / ALOHA 93.6% SOTA |
| [VLAW](./VLA-WM/%5BarXiv%202602.12063%5D%20Iterative%20Co-Improvement%20of%20VLA%20Policy%20and%20World%20Model/) | arXiv 2602.12063 | 迭代修正世界模型过度乐观偏差 + 合成数据提升 VLA，DROID 5任务 +39.2% |
| [WoVR](./VLA-WM/%5BarXiv%202602.13977%5D%20WoVR%20World%20Models%20as%20Reliable%20Simulators%20for%20VLA%20RL/) | arXiv 2602.13977 | Hallucination-aware RL框架：KIR + PACE + Wan 5B World Model，LIBERO +29.3pp，真实机器人 +30pp |
| ⭐ [Robo-Dopamine](./VLA-WM/%5BCVPR%202026%5D%20Robo-Dopamine/) | CVPR 2026 | **通用过程奖励模型 GRM** - hop-based 相对进度 + 多视角融合 + PBRS policy-invariant shaping, 150 rollouts → 95% SR |
| [RLinf](./VLA-WM/%5BarXiv%202509.15965%5D%20RLinf/) | arXiv 2509.15965 | **RL 训练系统** - M2Flow 宏微流变换, 弹性流水线+上下文切换+自动调度, Embodied RL 2.43x加速, LIBERO 97.83% SR |
| ⭐ [Motus](./VLA-WM/%5BArxiv%202512.13030%5D%20Motus/) | arXiv 2512.13030 | **统一 Latent Action World Model** - MoT(Wan5B+Qwen3-VL-2B+Action)三专家共享注意力, 光流latent action跨embodiment, +45% over π0.5 |

---

### 🧠 Agent Memory
Agent 记忆机制

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Survey: Memory in the Age of AI Agents](./Agent-Memory/%5BArxiv%202512.13564%5D%20Memory%20in%20the%20Age%20of%20AI%20Agents/) | arXiv 2512.13564 | **综述** - Forms×Functions×Dynamics 三维分类框架, 400+文献 |
| [Coconut](./Agent-Memory/%5BICLR%202025%5D%20Coconut/) | ICLR 2025 | Chain of Continuous Thought，hidden state 直接反馈做 latent reasoning，涌现 BFS，MemGen 理论基础 |
| [MA-LMM](./Agent-Memory/%5BCVPR%202024%5D%20MA-LMM/) | CVPR 2024 | Dual memory bank (visual+query) + MBC 压缩，在线逐帧处理，plug-and-play，LVU/Breakfast/COIN SOTA |
| [G-Memory](./Agent-Memory/%5BNeurIPS%202025%5D%20G-Memory/) | NeurIPS 2025 Spotlight | 三层图记忆 (Query→Insight→Interaction)，双向遍历+agentic update，LatentMem 的前作 |
| [VisMem](./Agent-Memory/%5BArxiv%202511.11007%5D%20VisMem/) | arXiv 2511.11007 | 短期(视觉)+长期(语义) latent vision memory，特殊 token 按需调用，两阶段 GRPO，+11% |
| [Mirage](./Agent-Memory/%5BArxiv%202506.17218%5D%20Mirage/) | arXiv 2506.17218 | Machine Mental Imagery，hidden state 重铸为 latent visual token，两阶段 SFT+RL，VisMem 的 latent baseline |
| [MemEvolve](./Agent-Memory/%5BArxiv%202512.18746%5D%20MemEvolve/) | arXiv 2512.18746 | Meta-evolution of agent memory systems，自动演化记忆架构 |
| [AgeMem](./Agent-Memory/%5BArxiv%202601.01885%5D%20AgeMem/) | arXiv 2601.01885 | 统一 LTM+STM 为 tool action，三阶段渐进 RL + step-wise GRPO，5 benchmark SOTA |
| [Mem-T](./Agent-Memory/%5BArxiv%202601.23014%5D%20Mem-T/) | arXiv 2601.23014 | 层次化记忆数据库 + 密集化奖励训练 Memory Agent |
| [MemSkill](./Agent-Memory/%5BArxiv%202602.02474%5D%20MemSkill/) | arXiv 2602.02474 | Memory 操作→可学习可进化 skill，Controller(RL)+Executor+Designer 闭环，4 benchmark SOTA |
| [LatentMem](./Agent-Memory/%5BArxiv%202602.03036%5D%20LatentMem/) | arXiv 2602.03036 | 可学习多智能体 latent memory，Memory Composer + LMPO，角色感知+token高效，6 benchmark × 4 MAS SOTA |
| [MemGen](./Agent-Memory/%5BICLR%202026%5D%20MemGen/) | ICLR 2026 | 生成式隐式记忆，推理-记忆交织，超 ExpeL/AWM 38% |

---

### 🔗 LLM Web3 Agent
LLM 驱动的 Web3 智能代理

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Web3Agent](./LLM-Web3-Agent/%5BACM%20TWEB%202026%5D%20Web3Agent/) | ACM TWEB 2026 | LLM Agent 端到端链上操作，六模块流水线 + 领域专用 RAG (Operation/API/Error chunks)，94.1% Query SR / 80.3% Op SR |

---

### 🏥 Medical VQA Benchmark
医学视觉问答基准

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| *待添加* | | |

---

### 🏥 Medical AI (未分组)

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [MedFrameQA](./%5BICLR%202026%20Rejected%5D%20MedFrameQA/) | ICLR 2026 Rejected | **Benchmark** - 多帧医学 VQA, 2851 QA, MLLM 跨图推理 < 55% |
| [Context Clues](./%5BICLR%202026%5D%20Context-Clues/) | ICLR 2026 | 长上下文 EHR FM, Mamba-16k EHRSHOT 9/14 SOTA, EHR 三属性分析 |
| [EHRSHOT](./%5BNeurIPS%202023%5D%20EHRSHOT/) | NeurIPS 2023 | EHR benchmark + CLMBR-T-base FM, 6739 patients, 15 few-shot tasks |

---

### 🧠 Parametric Memory
参数化记忆机制 - 将知识/经验编码到模型参数中

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [MemoryLLM](./Parametric%20Memory/%5BICML%202024%5D%20MemoryLLM/) | ICML 2024 | Transformer 隐空间 memory pool (1B params), self-update 无需反向传播, 百万次更新无退化 |
| [M+](./Parametric%20Memory/%5BICML%202025%5D%20M-Plus/) | ICML 2025 | MemoryLLM + CPU 端 Long-Term Memory + co-trained retriever, 知识保留 20k→160k+ |
| [Titans](./Parametric%20Memory/%5BArxiv%202501.00663%5D%20Titans/) | arXiv 2501.00663 | Neural long-term memory (surprise metric + momentum + weight decay), MAC/MAG/MAL 三变体, 2M+ context |
| [Nested Learning](./Parametric%20Memory/%5BNeurIPS%202025%5D%20Nested-Learning/) | NeurIPS 2025 | 嵌套优化范式, 优化器=联想记忆, Continuum Memory System + Hope 架构 |
| [ParamMem](./Parametric%20Memory/%5BArxiv%202602.23320%5D%20ParamMem/) | arXiv 2602.23320 | 参数化 reflection 记忆, LoRA 编码跨样本反思模式, ~500 样本即有效 |
| [In-Place TTT](./Parametric%20Memory/%5BICLR%202026%5D%20In-Place%20TTT/) | ICLR 2026 | MLP $W_{down}$ 复用为 fast weights, NTP 对齐目标 + chunk-wise 更新, drop-in 增强 LLM 长上下文能力 |
| ⭐ [LaCT (TTT Done Right)](./Parametric%20Memory/%5BArxiv%202505.23884%5D%20TTT-Done-Right/) | arXiv 2505.23884 | **Large Chunk TTT** - chunk size 2K~1M, GPU 利用率 <5%→70%, fast weight 占模型 40%, SwiGLU+Muon, 三模态 (NVS/LM/Video) |
| [TTT-E2E](./Parametric%20Memory/%5BArxiv%202512.23675%5D%20E2E-TTT-LongContext/) | arXiv 2512.23675 | 标准 SWA Transformer + inner-loop NTP TTT + outer-loop meta-learning，**双 E2E**，3B/164B scaling 比肩 full attention，128K prefill 2.7× 加速 |

---

### ⚡ VLA Real-Time Chunking (VLA-RTC)
VLA 实时推理加速 - Action Chunking 的异步执行与校正

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [RTC](./VLA-RTC/%5BNeurIPS%202025%5D%20RTC/) | NeurIPS 2025 | **Real-Time Chunking** - 推理时 inpainting 异步执行，无需重训练，高动态任务 |
| [Training-time RTC](./VLA-RTC/%5BArxiv%202512.05964%5D%20Training-time%20RTC/) | arXiv 2512.05964 | 训练时模拟延迟 + action prefix conditioning，RTC 的 drop-in 替代，更低开销 |
| [VLASH](./VLA-RTC/%5BArxiv%202512.01031%5D%20VLASH/) | arXiv 2512.01031 | MIT Han Lab，Future-State-Aware 异步推理，2.03x 加速 / 17.4x 反应延迟降低 |
| [A2C2](./VLA-RTC/%5BArxiv%202509.23224%5D%20A2C2%20Leave-No-Observation-Behind/) | arXiv 2509.23224 | 轻量校正头，per-step correction，+23% over RTC on Kinetix，plug-in |

---

### 🎯 RLHF
强化学习人类反馈

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [MM-RLHF](./RLHF/%5BICML%202025%5D%20MM-RLHF/) | ICML 2025 | 多模态 RLHF |

---

## 目录结构

```
paper-reading/
├── MLLM-Token-Compression/          # MLLM Token 压缩 (12 篇)
│   ├── [Arxiv 2507.20198] Survey-Token-Compression/
│   ├── [TechRxiv 2025] Survey-Token-Compression/
│   ├── [ECCV 2024] FastV/
│   ├── [ICML 2025] SparseVLM/
│   ├── [CVPR 2025] PyramidDrop/
│   ├── [CVPR 2025] DivPrune/
│   ├── [CVPR 2025] VisionZip/
│   ├── [Arxiv 2403.12178] SwiftVLM/
│   ├── [NeurIPS 2025] CDPruner/
│   ├── [NeurIPS 2025] SCOPE/
│   ├── [Arxiv 2505.22654] VScan/
│   ├── [ICLR 2026] VisionTrim/
│   └── methods-list.md
│
├── Video-Chaptering/                # 视频章节生成 (4 篇)
│   ├── [ECCV 2020] SODA/
│   ├── [NeurIPS 2023] VidChapters-7M/
│   ├── [CVPR 2025] Chapter-Llama/
│   ├── [arXiv 2025] ARC-Chapter/
│   └── README.md
│
├── VLA-WM/                          # VLA + 世界模型 (10 篇)
│   ├── [CVPR 2025] RoboBrain/
│   ├── [Arxiv 2507.02029] RoboBrain-2.0/
│   ├── [Arxiv 2601.14352] RoboBrain-2.5/
│   ├── [CoRL 2025] pi0/
│   ├── [CoRL 2025] pi0.5/
│   ├── [ICLR 2026] Pi0.6-RECAP/
│   ├── [PI 2026] MEM/               ← ⭐ 新加
│   ├── [arXiv 2602.12063] Iterative Co-Improvement of VLA Policy and World Model/
│   ├── [arXiv 2602.13977] WoVR World Models as Reliable Simulators for VLA RL/
│   └── [Arxiv 2512.13030] Motus/     ← ⭐ 新加
│
├── Agent-Memory/                    # Agent 记忆机制 (12 篇)
│   ├── [Arxiv 2512.13564] Memory in the Age of AI Agents/
│   ├── [ICLR 2025] Coconut/
│   ├── [CVPR 2024] MA-LMM/
│   ├── [NeurIPS 2025] G-Memory/
│   ├── [Arxiv 2511.11007] VisMem/
│   ├── [Arxiv 2506.17218] Mirage/
│   ├── [Arxiv 2512.18746] MemEvolve/
│   ├── [Arxiv 2601.01885] AgeMem/
│   ├── [Arxiv 2601.23014] Mem-T/
│   ├── [Arxiv 2602.02474] MemSkill/
│   ├── [Arxiv 2602.03036] LatentMem/
│   └── [ICLR 2026] MemGen/
│
│
├── LLM-Web3-Agent/                  # LLM Web3 Agent (1 篇)
│   └── [ACM TWEB 2026] Web3Agent/
├── Medical VQA Benchmark/           # 医学视觉问答基准 (0 篇)
│
├── [ICLR 2026 Rejected] MedFrameQA/ # Medical AI (未分组, 3 篇)
├── [ICLR 2026] Context-Clues/
├── [NeurIPS 2023] EHRSHOT/
│
├── Parametric Memory/               # 参数化记忆机制 (8 篇)
│   ├── [ICML 2024] MemoryLLM/
│   ├── [ICML 2025] M-Plus/
│   ├── [Arxiv 2501.00663] Titans/
│   ├── [NeurIPS 2025] Nested-Learning/
│   ├── [Arxiv 2602.23320] ParamMem/
│   ├── [ICLR 2026] In-Place TTT/
│   ├── [Arxiv 2505.23884] TTT-Done-Right/
│   └── [Arxiv 2512.23675] E2E-TTT-LongContext/
│
├── VLA-RTC/                         # VLA 实时推理加速 (4 篇)
│   ├── [NeurIPS 2025] RTC/
│   ├── [Arxiv 2512.05964] Training-time RTC/
│   ├── [Arxiv 2512.01031] VLASH/
│   └── [Arxiv 2509.23224] A2C2 Leave-No-Observation-Behind/
│
├── RLHF/                           # 强化学习人类反馈 (1 篇)
│   └── [ICML 2025] MM-RLHF/
│
└── README.md                        # 本文件
```

---

## 论文文件夹结构

每篇论文包含：
```
[会议 年份] 论文名/
├── README.md           # 论文概览 + Section 导航
├── sections/           # 批读笔记（原文 + 内嵌批注）
│   ├── 00-abstract.md
│   ├── 01-introduction.md
│   └── ...
├── full.md             # MinerU 解析的完整内容
├── images/             # 论文图片（MinerU 提取）
├── content_list.json   # 结构化内容
├── layout.json         # 版面分析
└── paper.pdf           # 原始 PDF
```

---

## 命名规范

文件夹命名: `[会议 年份] 论文名`
- 例: `[CVPR 2025] Chapter-Llama`
- 例: `[NeurIPS 2025] CDPruner`
- 例: `[Arxiv 2507.20198] Survey-Token-Compression`

---

## 相关资源

- 📖 [葵花宝典](https://github.com/EasonAI-5589/openclaw-baodian) - OpenClaw 配置文档

---

---

## 📋 待读清单

### Video Generation / World Model 基座

| 论文 | arXiv | 日期 | 关键词 | 课题 |
|------|-------|------|--------|------|
| **Wan** | [2503.20314](https://arxiv.org/abs/2503.20314) | 2025-03 | 阿里通义，1.3B/14B DiT 视频基座，开源 SOTA | VLA-WM |
| **VACE** | [2503.07598](https://arxiv.org/abs/2503.07598) | 2025-03 | All-in-One 视频创作与编辑，Video Condition Unit | VLA-WM |
| **Wan-S2V** | [2508.18621](https://arxiv.org/abs/2508.18621) | 2025-08 | 音频驱动电影级视频生成 (Wan2.2) | VLA-WM |
| **Wan-Move** | [2512.08765](https://arxiv.org/abs/2512.08765) | 2025-12 | 运动可控，Latent Trajectory Guidance | VLA-WM |
| **Open-Sora** | [2412.20404](https://arxiv.org/abs/2412.20404) | 2024-12 | HPC-AI Tech，STDiT 架构，15s/720p 开源 | VLA-WM |
| **Open-Sora 2.0** | [2503.09642](https://arxiv.org/abs/2503.09642) | 2025-03 | $200k 训练成本，对标 HunyuanVideo / Runway Gen-3 | VLA-WM |
| **Open-Sora Plan** | [2412.00131](https://arxiv.org/abs/2412.00131) | 2024-11 | 北大，Wavelet-Flow VAE + Skiparse Denoiser | VLA-WM |
| **LDA-1B** | [2602.12215](https://arxiv.org/abs/2602.12215) | 2026-02 | 北大 EPIC，DINO latent space 建模 dynamics+action，3万小时异构数据，跨本体 scale up | VLA-WM |
| **Interactive World Sim** | [2603.08546](https://arxiv.org/abs/2603.08546) | 2026-03 | Columbia+TRI，action-conditioned video WM，10min@15FPS 单卡4090，policy训练+评估 | VLA-WM |
| **ConRFT** | [2502.05450](https://arxiv.org/abs/2502.05450) | 2025-02 | VLA 一致性策略强化微调，8个真实任务 96.3% SR，比 SFT +144% | VLA-WM |
| **OpenVLA-OFT** | [2502.19645](https://arxiv.org/abs/2502.19645) | 2025-02 | VLA 微调最佳实践（并行解码+action chunking+L1），LIBERO 76.5%→97.1%，26x 加速 | VLA-WM |

### VLA 新方向

| 论文 | arXiv | 日期 | 关键词 | 课题 |
|------|-------|------|--------|------|
| **BagelVLA** | [2602.09849](https://arxiv.org/abs/2602.09849) | 2026-02 | Interleaved VL-Action Generation，Residual Flow Guidance，长程操作 | VLA-WM |
| **DreamZero** | [2602.15922](https://arxiv.org/abs/2602.15922) | 2026-02 | NVIDIA，14B World Action Model，Video Diffusion → Policy，零样本泛化 39.5%，7Hz 实时 | VLA-WM |
| **GigaBrain-0.5M\*** | 待确认 | 2026-02 | RAMP（World Model-conditioned RL），Laundry/Box/Espresso +30%，自进化闭环 | VLA-WM |

---

*由 3号机 协助整理 📚 | 更新: 2026-03-20 | 共 47 篇论文 + 14 篇待读*
