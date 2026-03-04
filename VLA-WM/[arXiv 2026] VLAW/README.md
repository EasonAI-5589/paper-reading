# VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model

**作者**: Yanjiang Guo\*, Tony Lee\*, Lucy Xiaoyang Shi\*, Jianyu Chen, Percy Liang, Chelsea Finn  
**单位**: Stanford University, Tsinghua University  
**来源**: arXiv 2602.12063 (Feb 2026)  
**链接**: [arXiv](https://arxiv.org/abs/2602.12063) | [Project Page](https://sites.google.com/view/vlaw-arxiv) | [GitHub](https://github.com/Robert-gyj/Ctrl-World)

## 一句话总结

用少量真实机器人 rollout 数据（含失败案例）修正世界模型的过度乐观偏差，再用修正后的世界模型大量生成合成轨迹，迭代提升 VLA 策略性能（+39.2% 成功率）。

## 核心贡献

1. **问题诊断**: 现有世界模型（如 Ctrl-World）因训练数据以演示为主，缺乏失败案例，存在「过度乐观偏差」
2. **VLAW 迭代框架**: 真实 rollout（含失败）微调世界模型 → 生成合成轨迹（10x 扩增）→ 过滤成功轨迹 → SFT 训练 VLA
3. **奖励模型**: 微调 Qwen3-VL-4B-Instruct，概率阈值 P(yes)>0.8 过滤策略，大幅降低假阳性
4. **AWR for Flow-Matching**: 理论证明「只在成功轨迹 SFT」等价于正则化 RL 框架下的策略优化
5. **真实机器人实验**: DROID 平台 5 类接触丰富任务，+39.2% 绝对成功率，合成数据贡献 +11.6%

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：核心问题、方法、关键结果（+39.2%/+11.6%） |
| [01 - Introduction](sections/01-introduction.md) | 动机、过度乐观偏差诊断、VLAW 方案、贡献总结 + Figure 1 |
| [02 - Related Work](sections/02-related-work.md) | VLA Post-training 路线 + 世界模型用于决策的演进 |
| [03 - Preliminaries](sections/03-preliminaries.md) | 多任务 MDP 定义、符号表、World Model 生成轨迹的形式化 |
| [04 - Method](sections/04-method.md) | 核心方法：世界模型修正 + 迭代 VLA 提升 + AWR 理论分析 + Figures 2-3 + Eqs 1-8 + Algorithm 1 |
| [05 - Experiments](sections/05-experiments.md) | 5 类任务设定 + Table 1 + Figures 4-9 + 消融实验 |
| [06 - Conclusion](sections/06-conclusions.md) | 总结 + 局限性分析（作者说的 + 批注补充的）+ 总体评价 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 总成功率提升（vs base policy） | +39.2% |
| 合成数据额外贡献（vs Filtered BC） | +11.6% |
| 真实 rollout 数 | 50 条/任务/迭代 |
| 合成轨迹数 | 500 条/任务（10x 扩增） |
| 迭代次数 | 2 次 |
| 世界模型微调 | 50K steps |
| 策略微调 | 2K steps, batch_size=256 |
| 奖励阈值 α | 0.8 |
| FP 修正（世界模型混淆矩阵） | 11 → 1 |
| 基础 VLA | π₀.₅ (Physical Intelligence) |
| 基础世界模型 | Ctrl-World (Guo et al., 2025a) |
| 奖励模型 | Qwen3-VL-4B-Instruct (微调) |
| 实验平台 | DROID (Franka Panda) |

---

## 📊 Citation Landscape

> ⚠️ Semantic Scholar API 暂不可用（网络问题），待补充。

**Connected Papers**: [查看](https://www.connectedpapers.com/main/2602.12063)

### 参考文献分组（手动整理）

**VLA Post-training**
| 论文 | 年份 | 要点 |
|------|------|------|
| π₀.₆* (Intelligence et al., 2025a) | 2025 | Advantage-conditioned SL，offline RL for VLA |
| π₀.₅ (Intelligence et al., 2025b) | 2025 | 本文的基础 VLA 策略 |
| π₀ (Black et al., 2024) | 2024 | Vision-language-action flow model |
| VLA-RL (Lu et al., 2025) | 2025 | 可扩展 RL for VLA |
| DSRL (Wagenmaker et al., 2025) | 2025 | 噪声空间 RL（本文 baseline） |

**World Models**
| 论文 | 年份 | 要点 |
|------|------|------|
| Ctrl-World (Guo et al., 2025a) | 2025 | 本文的基础世界模型（DROID 上训练） |
| DayDreamer (Wu et al., 2023) | 2023 | 经典真实机器人 world model RL |
| World4RL (Jiang et al., 2025) | 2025 | Diffusion world model + RL for manipulation |
| WMPO (Zhu et al., 2025) | 2025 | World model policy optimization for VLA |
| World-Gymnast (Sharma et al., 2026) | 2026 | 同期 WM+RL，同方向竞品 |
| Genie 3 (Ball et al., 2025) | 2025 | Google DeepMind 大规模世界模型 |

**Reward & Evaluation**
| 论文 | 年份 | 要点 |
|------|------|------|
| RoboReward (Lee et al., 2026) | 2026 | 通用 VLM 奖励模型（本文奖励模型基础） |
| Qwen3-VL (Team, 2025a) | 2025 | 本文奖励模型底座 |
| DROID (Khazatsky et al., 2024) | 2024 | 大规模真实机器人数据集+平台 |

---

## BibTeX

```bibtex
@article{guo2026vlaw,
  title={VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model},
  author={Guo, Yanjiang and Lee, Tony and Shi, Lucy Xiaoyang and Chen, Jianyu and Liang, Percy and Finn, Chelsea},
  journal={arXiv preprint arXiv:2602.12063},
  year={2026}
}
```
