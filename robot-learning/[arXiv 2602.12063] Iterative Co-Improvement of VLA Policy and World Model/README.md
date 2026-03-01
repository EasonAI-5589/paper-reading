# VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model

> **arXiv**: [2602.12063](https://arxiv.org/abs/2602.12063)  
> **PDF**: [paper.pdf](paper.pdf)  
> **Project Page**: https://sites.google.com/view/vlaw-arxiv

---

## 元信息

| 字段 | 内容 |
|------|------|
| **标题** | VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model |
| **作者** | Yanjiang Guo, Tony Lee, Lucy Xiaoyang Shi, Jianyu Chen, Percy Liang, Chelsea Finn |
| **机构** | Stanford University, Tsinghua University |
| **发布时间** | 2026-02（arXiv 2602.12063） |
| **关键词** | VLA, World Model, Robot Manipulation, Iterative Improvement, Flow Matching |

---

## 一句话总结

用少量真实机器人 rollout（含 failure）fine-tune world model，再让 world model 生成大量合成轨迹来改进 VLA policy，迭代两轮后在 5 类 contact-rich 任务上平均成功率从 **46% → 87%**（+39.2 pp），其中合成数据贡献 **+11.6 pp**。

---

## 核心贡献

1. **发现并解决 world model 的 over-optimism 问题**：用包含 failure case 的 online rollout 数据 fine-tune pretrained world model（Ctrl-World），显著提升物理保真度（FVD 225→64，FP 11→1）
2. **VLAW 迭代协同优化 pipeline**：World Model ↔ VLA Policy 相互增强的正反馈循环，每轮用 50 条 real rollout 撬动 500 条合成数据
3. **flow-matching VLA 的 RL 理论框架**：将 binary-filtered BC 与 AWR / regularized RL 联系起来，证明方法有理论依据

---

## 方法概述

```
┌─────────────────────────────────────────────────────┐
│                    VLAW Pipeline                     │
│                                                      │
│  Real World Rollout (K=50/task)                     │
│          │                                           │
│          ▼                                           │
│  Fine-tune World Model (Ctrl-World)                  │
│  + Co-train with DROID dataset                       │
│  Fine-tune Reward Model (Qwen3-VL-4B)               │
│          │                                           │
│          ▼                                           │
│  Generate Synthetic Trajectories (N=500/task)        │
│  → Filter with Reward Model (threshold=0.8)          │
│          │                                           │
│          ▼                                           │
│  Fine-tune VLA Policy (π₀.₅)                        │
│  on D_real+ ∪ D_syn+ (flow-matching loss)            │
│          │                                           │
│          └──── Repeat for K_iter=2 iterations ──────┘
```

---

## 关键结果

### 成功率（50 次评估 / task）

| Method | Stacking | Wiping | Open Book | Scooping | Drawing | **Mean** |
|--------|----------|--------|-----------|----------|---------|---------|
| Base model | 0.62 | 0.46 | 0.56 | 0.44 | 0.22 | 0.460 |
| DSRL | 0.70 | 0.40 | 0.50 | 0.60 | 0.30 | 0.500 |
| Filtered BC-1 | 0.80 | 0.62 | 0.72 | 0.64 | 0.46 | 0.648 |
| Filtered BC-2 | 0.88 | 0.76 | 0.82 | 0.74 | 0.56 | 0.752 |
| **Ours-1** | 0.80 | 0.72 | 0.80 | 0.72 | 0.68 | 0.744 |
| **Ours-2** | **0.92** | **0.86** | **0.86** | **0.92** | **0.78** | **0.868** |

### World Model 质量（action replay，wrist camera）

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ | FP↓ |
|--------|-------|-------|--------|------|------|-----|
| Pretrained Ctrl-World | 16.32 | 0.634 | 0.347 | 41.03 | 225.13 | - |
| + Expert Rollout | 19.87 | 0.748 | 0.189 | 12.76 | 99.98 | 11 |
| + Expert + **Online Rollout** | **21.77** | **0.784** | **0.136** | **9.58** | **64.12** | **1** |

---

## 批读笔记导航

| 文件 | 内容 |
|------|------|
| [00-abstract-intro.md](notes/00-abstract-intro.md) | Abstract + Introduction |
| [01-related-work-preliminaries.md](notes/01-related-work-preliminaries.md) | Related Work + Preliminaries |
| [02-method.md](notes/02-method.md) | Method（4.1~4.3 + Algorithm） |
| [03-experiments.md](notes/03-experiments.md) | Experiments（5.1~5.3 + Ablation） |
| [04-conclusions-appendix.md](notes/04-conclusions-appendix.md) | Conclusions + Appendix A/B/C + 总体评价 |

---

## Citation Landscape

### 本文引用的关键工作

| 论文 | 角色 |
|------|------|
| π₀.₅ (Intelligence et al., 2025b) | Base VLA model |
| Ctrl-World (Guo et al., 2025a) | Base world model |
| π₀.₆* (Intelligence et al., 2025a) | 同类在线 RL for VLA |
| DayDreamer (Wu et al., 2023) | Real-world MBRL 先驱 |
| DROID (Khazatsky et al., 2024) | 实验平台 + 预训练数据集 |
| Qwen3-VL (Team, 2025a) | Reward model 基座 |
| DSRL (Wagenmaker et al., 2025) | Baseline 方法 |
| AWR (Peng et al., 2019) | 理论基础 |

### 同期相关工作（未直接比较）

| 论文 | 简述 |
|------|------|
| WMPO (Zhu et al., 2025) | World model + VLA policy optimization |
| World-Gymnast (Sharma et al., 2026) | 在 world model 里做 RL 训练 |
| VLA-RFT (Li et al., 2025a) | World simulator 里做 RL fine-tuning |
| World4rl (Jiang et al., 2025) | Diffusion world model for RL |

---

## 快速批评

**值得学习**：pipeline 设计简洁、任务选择有挑战性、实验在真实机器人上做、理论联系清晰

**值得质疑**：DSRL baseline 只评 10 次；缺少"不 fine-tune world model 直接用 pretrained Ctrl-World 生成数据"的 ablation；reward model 召回率仅约 45%（阈值=0.8）；计算开销未报告；只迭代 2 次，未见收敛曲线
