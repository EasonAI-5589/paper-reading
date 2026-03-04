# RoboBrain 2.5: Depth in Sight, Time in Mind

**作者**: BAAI RoboBrain Team（核心：Huajie Tan, Enshen Zhou, Zhiyu Li, Yijie Xu, Yuheng Ji, Xiansheng Chen, Cheng Chi, ...）  
**通讯作者**: Shanghang Zhang（仉尚航，北京大学计算机系教授）  
**机构**: BAAI (北京智源人工智能研究院) + 北京大学  
**年份**: 2026 | **Arxiv**: [2601.14352](https://arxiv.org/abs/2601.14352)  
**项目主页**: https://superrobobrain.github.io

## 一句话总结
RoboBrain 2.5 在 RoboBrain 2.0 基础上新增两大核心能力——**精确 3D 空间推理**（从 2D 像素到度量级 3D 关键点轨迹生成）和**密集时间价值估计**（逐步感知的进度/回退反馈，可作为 RL reward），实现了从"语义推理器"到"物理接地智能体"的范式转变。

## 核心贡献
1. **Precise 3D Spatial Reasoning** ("Depth in Sight"): 解耦 $(u,v,d)$ 表示，三个递进技能（Referring→Measuring→Tracing），从单目 RGB 生成无碰撞 3D 操作轨迹
2. **Dense Temporal Value Estimation** ("Time in Mind"): Hop-based 标注策略 + 多视角融合 + 双向一致性检查，提供密集的通用 reward signal
3. **大规模数据工程**: 12.4M 高质量样本，涵盖通用/空间/时间三大域
4. **跨加速器训练**: NVIDIA 和摩尔线程 GPU 双平台训练，性能一致（收敛差距 0.62%）

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：两大新能力概述 |
| [01 - Introduction](sections/01-introduction.md) | 动机（度量盲 + 开环预测）+ 三大贡献 + Figure 1 |
| [02 - New Feature](sections/02-new-feature.md) | **核心方法**：3D 空间推理的 $(u,v,d)$ 表示 + 时间估计的 hop formulation |
| [03 - Training Data](sections/03-training-data.md) | 12.4M 数据构建：General/Spatial/Temporal 三域详解 |
| [04 - Training Strategy](sections/04-training-strategy.md) | 两阶段训练：定性→定量，15% 数据重放抗遗忘 |
| [05 - Infrastructure](sections/05-infrastructure.md) | 混合并行 + 动态内存 + 跨加速器训练 |
| [06 - Evaluation](sections/06-evaluation.md) | 2D/3D 空间 + 时间估计全面评估，TraceSpatial 44 vs baseline 7 |
| [07 - Conclusion](sections/07-conclusion.md) | 总结 + 四个未来方向（世界模型、人形部署等） |
| [08 - Appendix](sections/08-appendix.md) | 定性示例（3D trace + 时间估计可视化）+ Bounded Progress 证明 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 模型规模 | 8B (基于 Qwen3-VL) |
| 总训练数据 | 12.4M 样本 |
| Stage 1 数据 | 8.3M |
| Stage 2 数据 | 4.1M |
| 3D Spatial 数据 | 1.74M (8.08M QA) |
| Dense Value 数据 | 3.5M (from 35M) |
| CrossPoint 提升 | 28.40 → 76.30 (2.7x) |
| TraceSpatial Success | 44 vs baseline 7 (6x) |
| LIBERO VOC+/VOC- | 98.97/98.94 |
| 跨平台收敛差距 | 0.62% |
| GPU (NVIDIA) | 64×8 A800 |
| GPU (MTT) | 128×8 |

## RoboBrain 系列迭代

| 版本 | 会议 | 核心能力 |
|------|------|----------|
| RoboBrain | CVPR 2025 | 通用具身感知 + 推理 |
| RoboBrain 2.0 | Arxiv 2507.02029 | 2D 空间 + 规划 + 闭环 |
| **RoboBrain 2.5** | **Arxiv 2601.14352** | **+ 3D 空间推理 + 密集时间估计** |

---

## BibTeX

```bibtex
@article{tan2026robobrain25,
  title={RoboBrain 2.5: Depth in Sight, Time in Mind},
  author={Huajie Tan and Enshen Zhou and Zhiyu Li and Yijie Xu and Yuheng Ji and Xiansheng Chen and Cheng Chi and Peng Wang and Hao Jia and Yu Ao and others},
  journal={arXiv preprint arXiv:2601.14352},
  year={2026}
}
```
