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
| [FastV](./MLLM-Token-Compression/%5BECCV%202024%5D%20FastV/) | ECCV 2024 | 第 2 层后固定剪枝，简单高效 |
| [SparseVLM](./MLLM-Token-Compression/%5BICML%202025%5D%20SparseVLM/) | ICML 2025 | 文本引导 + Token 回收 |
| [PyramidDrop](./MLLM-Token-Compression/%5BCVPR%202025%5D%20PyramidDrop/) | CVPR 2025 | 金字塔式渐进剪枝 |
| [SwiftVLM](./MLLM-Token-Compression/%5BArxiv%202403.12178%5D%20SwiftVLM/) | arXiv 2403.12178 | Bypass 范式 + DP 选层 |
| [DivPrune](./MLLM-Token-Compression/%5BCVPR%202025%5D%20DivPrune/) | CVPR 2025 | MMDP 最大化 token 多样性 |
| [VisionZip](./MLLM-Token-Compression/%5BCVPR%202025%5D%20VisionZip/) | CVPR 2025 | Text-agnostic, [CLS] attention + similarity merging |
| [CDPruner](./MLLM-Token-Compression/%5BNeurIPS%202025%5D%20CDPruner/) | NeurIPS 2025 | DPP 条件多样性剪枝 |
| [SCOPE](./MLLM-Token-Compression/%5BNeurIPS%202025%5D%20SCOPE/) | NeurIPS 2025 | Saliency + Coverage 联合优化 |
| [VScan](./MLLM-Token-Compression/%5BArxiv%202505.22654%5D%20VScan/) | TMLR 2026 | 两阶段 Global+Local Scan + Middle Layer Pruning |
| [VisionTrim](./MLLM-Token-Compression/%5BArxiv%202601.22674%5D%20VisionTrim/) | arXiv 2601.22674 | Training-free, DVTS (global-local 选) + TGVC (text-guided 补), 两阶段统一压缩 |

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
| [RoboBrain](./VLA/%5BCVPR%202025%5D%20RoboBrain/) | CVPR 2025 | LLaVA + A-LoRA/T-LoRA, ShareRobot 数据集 |
| [RoboBrain 2.0](./VLA/%5BArxiv%202507.02029%5D%20RoboBrain-2.0/) | arXiv 2507.02029 | Qwen2.5-VL base, 统一空间+时间推理 |
| [RoboBrain 2.5](./VLA/%5BArxiv%202601.14352%5D%20RoboBrain-2.5/) | arXiv 2601.14352 | 精确 3D 空间推理 + 密集时间价值估计 |
| [π0](./VLA/%5BCoRL%202025%5D%20pi0/) | CoRL 2025 | Flow matching policy |
| [π0.5](./VLA/%5BCoRL%202025%5D%20pi0.5/) | CoRL 2025 Oral | VLM-based robot policy |
| [π0.6-RECAP](./VLA/%5BICLR%202026%5D%20Pi0.6-RECAP/) | ICLR 2026 | CoT reasoning robot policy |

---

### 🧠 Agent Memory
Agent 记忆机制

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [Survey: Memory in the Age of AI Agents](./Agent-Memory/%5BArxiv%202512.13564%5D%20Memory%20in%20the%20Age%20of%20AI%20Agents/) | arXiv 2512.13564 | **综述** - Forms×Functions×Dynamics 三维分类框架, 400+文献 |
| [MemGen](./Agent-Memory/%5BICLR%202026%5D%20MemGen/) | ICLR 2026 | 生成式隐式记忆，推理-记忆交织，超 ExpeL/AWM 38% |
| [Mem-T](./Agent-Memory/%5BArxiv%202601.23014%5D%20Mem-T/) | arXiv 2601.23014 | 层次化记忆数据库 + 密集化奖励训练 Memory Agent |
| [VisMem](./Agent-Memory/%5BArxiv%202511.11007%5D%20VisMem/) | arXiv 2511.11007 | 短期(视觉)+长期(语义) latent vision memory，特殊 token 按需调用，两阶段 GRPO，+11% |
| [Coconut](./Agent-Memory/%5BICLR%202025%5D%20Coconut/) | ICLR 2025 | Chain of Continuous Thought，hidden state 直接反馈做 latent reasoning，涌现 BFS，MemGen 理论基础 |
| [MA-LMM](./Agent-Memory/%5BCVPR%202024%5D%20MA-LMM/) | CVPR 2024 | Dual memory bank (visual+query) + MBC 压缩，在线逐帧处理，plug-and-play，LVU/Breakfast/COIN SOTA |
| [Mirage](./Agent-Memory/%5BArxiv%202506.17218%5D%20Mirage/) | arXiv 2506.17218 | Machine Mental Imagery，hidden state 重铸为 latent visual token，两阶段 SFT+RL，VisMem 的 latent baseline |
| [MemSkill](./Agent-Memory/%5BArxiv%202602.02474%5D%20MemSkill/) | arXiv 2602.02474 | Memory 操作→可学习可进化 skill，Controller(RL)+Executor+Designer 闭环，4 benchmark SOTA |
| [AgeMem](./Agent-Memory/%5BArxiv%202601.01885%5D%20AgeMem/) | arXiv 2601.01885 | 统一 LTM+STM 为 tool action，三阶段渐进 RL + step-wise GRPO，5 benchmark SOTA |
| [LatentMem](./Agent-Memory/%5BArxiv%202602.03036%5D%20LatentMem/) | arXiv 2602.03036 | 可学习多智能体 latent memory，Memory Composer + LMPO，角色感知+token高效，6 benchmark × 4 MAS SOTA |

---

### 🏥 Medical AI (未分组)

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [MedFrameQA](./%5BICLR%202026%20Rejected%5D%20MedFrameQA/) | ICLR 2026 Rejected | **Benchmark** - 多帧医学 VQA, 2851 QA, MLLM 跨图推理 < 55% |
| [Context Clues](./%5BICLR%202026%5D%20Context-Clues/) | ICLR 2026 | 长上下文 EHR FM, Mamba-16k EHRSHOT 9/14 SOTA, EHR 三属性分析 |
| [EHRSHOT](./%5BNeurIPS%202023%5D%20EHRSHOT/) | NeurIPS 2023 | EHR benchmark + CLMBR-T-base FM, 6739 patients, 15 few-shot tasks |

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
├── MLLM-Token-Compression/          # MLLM Token 压缩 (9 篇)
│   ├── [Arxiv 2507.20198] Survey-Token-Compression/
│   ├── [ECCV 2024] FastV/
│   ├── [ICML 2025] SparseVLM/
│   ├── [CVPR 2025] PyramidDrop/
│   ├── [CVPR 2025] DivPrune/
│   ├── [CVPR 2025] VisionZip/
│   ├── [Arxiv 2403.12178] SwiftVLM/
│   ├── [NeurIPS 2025] CDPruner/
│   ├── [NeurIPS 2025] SCOPE/
│   ├── [Arxiv 2601.22674] VisionTrim/
│   └── methods-list.md
│
├── Video-Chaptering/                # 视频章节生成 (4 篇)
│   ├── [ECCV 2020] SODA/
│   ├── [NeurIPS 2023] VidChapters-7M/
│   ├── [CVPR 2025] Chapter-Llama/
│   ├── [arXiv 2025] ARC-Chapter/
│   └── README.md
│
├── VLA/                             # Vision-Language-Action (6 篇)
│   ├── [CVPR 2025] RoboBrain/
│   ├── [Arxiv 2507.02029] RoboBrain-2.0/
│   ├── [Arxiv 2601.14352] RoboBrain-2.5/
│   ├── [CoRL 2025] pi0/
│   ├── [CoRL 2025] pi0.5/
│   └── [ICLR 2026] Pi0.6-RECAP/
│
├── Agent-Memory/                    # Agent 记忆机制 (3 篇)
│   ├── [Arxiv 2512.13564] Memory in the Age of AI Agents/
│   ├── [ICLR 2026] MemGen/
│   ├── [Arxiv 2601.23014] Mem-T/
│   ├── [Arxiv 2511.11007] VisMem/
│   ├── [CVPR 2024] MA-LMM/
│   ├── [Arxiv 2506.17218] Mirage/
│   ├── [Arxiv 2602.02474] MemSkill/
│   └── [Arxiv 2601.01885] AgeMem/
│
├── [ICLR 2026 Rejected] MedFrameQA/ # Medical AI (未分组, 3 篇)
├── [ICLR 2026] Context-Clues/
├── [NeurIPS 2023] EHRSHOT/
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

*由 3号机 协助整理 📚 | 更新: 2026-02-14 | 共 29 篇论文*
