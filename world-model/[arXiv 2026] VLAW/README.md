# VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model

## 元信息

| 字段 | 内容 |
|------|------|
| **标题** | VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model |
| **作者** | Yanjiang Guo\*, Tony Lee\*, Lucy Xiaoyang Shi\*, Jianyu Chen, Percy Liang, Chelsea Finn |
| **机构** | Stanford University, Tsinghua University |
| **arXiv** | [2602.12063](https://arxiv.org/abs/2602.12063) |
| **投稿日期** | 2026-02-12 |
| **分类** | world-model / VLA post-training |
| **项目主页** | https://sites.google.com/view/vlaw-arxiv |

## 一句话总结

用少量真实机器人 rollout 数据来「修正」世界模型的物理保真度，再用修正后的世界模型大量生成合成轨迹，迭代提升 VLA 策略性能。

## 核心贡献

1. **问题诊断**：现有世界模型（如 Ctrl-World）存在「过度乐观」偏差——因为训练数据以演示为主，缺乏失败案例，无法准确建模接触丰富任务的物理动态。
2. **解决方案 VLAW**：迭代框架——先用真实 rollout（含成功/失败）微调世界模型，再用修正后的世界模型生成大量合成轨迹，过滤成功轨迹后训练 VLA 策略。
3. **奖励模型**：微调 Qwen3-VL-4B-Instruct 为机器人任务奖励模型，用概率阈值过滤（P(yes) > 0.8）显著降低假阳性。
4. **兼容 flow-matching 策略**：把加权 flow-matching 损失解释为正则化 RL 框架（AWR for flow matching），理论上打通了 RL 与 SFT 的连接。
5. **真实机器人实验**：在 DROID 平台 5 类接触丰富任务上，相比 base policy 提升 **+39.2%** 绝对成功率，合成数据贡献 **+11.6%**。

## 方法流程

```
真实 rollout (K=50/任务) 
    ↓
微调世界模型 Ctrl-World（含失败案例）
    ↓
Policy-in-the-loop 生成合成轨迹 (N=500/任务)
    ↓
奖励模型过滤成功轨迹
    ↓
Flow-matching 微调 π₀.₅ 策略
    ↓ (迭代 2 次)
```

## Section 导航

| Section | 文件 |
|---------|------|
| Abstract + Introduction | [notes/01-intro.md](notes/01-intro.md) |
| Related Works | [notes/02-related-works.md](notes/02-related-works.md) |
| Method (核心) | [notes/03-method.md](notes/03-method.md) |
| Experiments | [notes/04-experiments.md](notes/04-experiments.md) |
| Conclusion + Appendix | [notes/05-conclusion-appendix.md](notes/05-conclusion-appendix.md) |

## 关键数据

| 方法 | Stacking | Wiping | Open Book | Scooping | Drawing | Mean |
|------|----------|--------|-----------|----------|---------|------|
| Base model | 0.62 | 0.46 | 0.56 | 0.44 | 0.22 | 0.460 |
| DSRL | 0.70 | 0.40 | 0.50 | 0.60 | 0.30 | 0.500 |
| Filtered BC-2 | 0.88 | 0.76 | 0.82 | 0.74 | 0.56 | 0.752 |
| **Ours-2** | **0.92** | **0.86** | **0.86** | **0.92** | **0.78** | **0.868** |

## 与相关工作的关系

- **Ctrl-World** (Guo et al., 2025a)：本文的基础世界模型，VLAW 在其上微调
- **π₀.₅** (Physical Intelligence, 2025b)：本文的基础 VLA 策略
- **π₀.₆*** (Physical Intelligence, 2025a)：用 advantage-conditioned SL 做 offline RL，本文类似但引入世界模型
- **World4RL / WMPO / World-Gymnast**：同期 world model + RL for VLA 方向
- **DayDreamer**：经典真实机器人 world model RL，但模型能力有限，任务简单

## Citation

```bibtex
@article{guo2026vlaw,
  title={VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model},
  author={Guo, Yanjiang and Lee, Tony and Shi, Lucy Xiaoyang and Chen, Jianyu and Liang, Percy and Finn, Chelsea},
  journal={arXiv preprint arXiv:2602.12063},
  year={2026}
}
```
