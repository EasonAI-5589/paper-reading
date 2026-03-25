# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

**作者**: Lucas Maes*¹, Quentin Le Lidec*², Damien Scieur¹³, **Yann LeCun**², Randall Balestriero⁴
**来源**: arXiv 2603.19312 (Mar 2026) | **机构**: ¹Mila & Université de Montréal, ²New York University, ³Samsung SAIL, ⁴Brown University
**链接**: [arXiv](https://arxiv.org/abs/2603.19312) | [Website](https://yuantianyuan01.github.io/FastWAM/) | [Code](https://github.com/facebookresearch/lewm)

## 一句话总结

LeWM 是首个从原始像素端到端稳定训练的 JEPA 世界模型，仅用两项损失（next-embedding 预测 + SIGReg 正则化），1500万参数单 GPU 可训，规划速度比基于 foundation model 的世界模型快 48 倍，同时在多种 2D/3D 控制任务上保持竞争力。

## 核心贡献

1. **首个稳定端到端 JEPA**: 不依赖 stop-gradient、EMA、预训练编码器或辅助监督，仅用两项损失就能避免表征坍缩
2. **SIGReg 防坍缩**: 通过 Sketched-Isotropic-Gaussian Regularizer 强制 latent 分布匹配各向同性高斯，有可证明的防坍缩保证
3. **极致轻量化**: 15M 参数（ViT-Tiny encoder ~5M + ViT-S predictor ~10M），单 GPU 几小时可训
4. **高效规划**: 规划速度比 DINO-WM 快 **48×**，全部规划在 1 秒内完成
5. **物理理解涌现**: latent space 编码有意义的物理结构（位置、角度），能检测物理违规事件

## 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：首个稳定端到端 JEPA + 两项损失 + 15M 参数 |
| [01 - Introduction](sections/01-introduction.md) | JEPA 坍缩问题 + 现有方案的局限 + LeWM 的简洁方案 |
| [02 - Related Work](sections/02-related-work.md) | 生成式 WM vs JEPA WM vs Planning with Latent Dynamics |
| [03 - Method](sections/03-method.md) | ⭐ 编码器/预测器架构 + 训练目标（MSE + SIGReg）+ Latent Planning (MPC + CEM) |
| [04 - Experiments](sections/04-experiments.md) | ⭐ 四环境评估 + 消融实验 + 物理理解探测 + VoE |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性 + 未来方向 + 总体评价 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 总参数量 | **~15M**（ViT-Tiny encoder ~5M + ViT-S predictor ~10M） |
| Encoder | ViT-Tiny: patch=14, 12 layers, 3 heads, hidden=192 |
| Predictor | ViT-S: 6 layers, 16 heads, 10% dropout |
| Embedding 维度 | 192（[CLS] token + 1-layer MLP projector） |
| 训练超参数 | 仅 1 个有效超参（$\lambda$），默认 $\lambda=0.1$, $M=1024$ projections |
| 训练设备 | **单 GPU**（NVIDIA L40S） |
| 训练 epochs | 10 epochs（所有环境） |
| Frame skip | 5（每 5 步一帧） |
| 规划方法 | CEM (Cross-Entropy Method) + MPC |
| 规划速度 | **0.98s** vs DINO-WM 47s（**48× 加速**） |
| PushT 成功率 | **96%**（vs PLDM 78%, DINO-WM 92%） |
| OGBench-Cube 成功率 | 74%（vs DINO-WM 86%） |

## ⭐ 与其他 JEPA 方法对比

| 特性 | LeWM | PLDM | DINO-WM |
|------|:---:|:---:|:---:|
| 端到端训练 | ✅ | ✅ | ❌（冻结 DINOv2） |
| 从像素学习 | ✅ | ✅ | ✅（但编码器不更新） |
| 需要预训练编码器 | ❌ | ❌ | ✅ (DINOv2) |
| 训练稳定性 | ✅（2 项损失） | ❌（7 项损失，不稳定） | ✅（冻结编码器避免坍缩） |
| 可调超参数数量 | **1** ($\lambda$) | **6** ($\alpha,\beta,\gamma,\zeta,\nu,\mu$) | 0（无需防坍缩） |
| 超参搜索复杂度 | O(log n) | O($n^6$) | — |
| 防坍缩保证 | ✅ 可证明 | ❌ 启发式 | ✅（冻结） |
| 规划速度 | **0.98s** | 快（类似 LeWM） | **47s** |
| PushT 成功率 | **96%** | 78% | 92% |

---

## BibTeX

```bibtex
@article{maes2026leworldmodel,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}
```

## 相关论文

| 论文 | 年份 | 关系 |
|------|------|------|
| JEPA (LeCun) | 2022 | 定义了 JEPA 框架的路线图论文 |
| PLDM (Sobal et al.) | 2022/2025 | 此前唯一的端到端 JEPA，但用 VICReg 7项损失，不稳定 |
| DINO-WM (Zhou et al.) | 2025 | 冻结 DINOv2 做编码器避免坍缩，但不是端到端 |
| V-JEPA 2 (Assran et al.) | 2025 | 视频自监督 JEPA，用 EMA + SG |
| SIGReg / LeJEPA (Balestriero & LeCun) | 2025 | LeWM 的防坍缩正则化来源 |
| Fast-WAM (Yuan et al.) | 2026 | 谢赛宁建议一起看，用视频训练做世界表征但推理时不生成未来 |
| TD-MPC2 (Hansen et al.) | 2024 | 任务特定 WM，需要奖励信号 |
| Dreamer (Hafner et al.) | 2020-2025 | 生成式 WM + RL，需要奖励和重建 |
