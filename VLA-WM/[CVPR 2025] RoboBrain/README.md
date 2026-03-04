# RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete

**作者**: Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, ... **Shanghang Zhang** (仉尚航, 通讯作者)  
**会议**: CVPR 2025 | **机构**: 北京大学 + BAAI + 中科院  
**链接**: [arXiv 2502.21257](https://arxiv.org/abs/2502.21257) | [Project](https://robobrain.github.io/)

## 一句话总结

基于 LLaVA 架构的统一机器人操作模型，通过 ShareRobot 数据集和多阶段训练，实现从抽象指令到具体动作的三层能力：**任务规划 → 可供性感知 → 轨迹预测**。

## 核心贡献

1. **RoboBrain 模型**：基于 LLaVA + Qwen2.5-7B，用 A-LoRA/T-LoRA 分别处理 affordance 和 trajectory
2. **ShareRobot 数据集**：从 OXE 筛选 51K 实例，生成 1M QA pairs + 6.5K affordance + 6.9K trajectory 标注
3. **多阶段训练策略**：Phase 1 通用 OV 训练（4 stages）→ Phase 2 机器人专项训练
4. **SOTA 结果**：在 RoboVQA、OpenEQA、ShareRobot 等 benchmark 上超越 GPT-4V 和 RoboMamba

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1 总览图 |
| [01 - Introduction](sections/01-introduction.md) | 动机：MLLM 缺乏三个机器人核心能力 + 四点贡献 |
| [02 - Related Work](sections/02-related-work.md) | MLLM for Planning + 操作数据集发展脉络 + Figure 2 |
| [03 - Dataset](sections/03-dataset.md) | ShareRobot：筛选准则 + 三维度标注 + 数据统计 |
| [04 - Method](sections/04-method.md) | 模型架构（LLaVA + A-LoRA + T-LoRA）+ 多阶段训练 |
| [05 - Experiments](sections/05-experiments.md) | Planning/Affordance/Trajectory 三个任务评测 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + Appendix 概要（消融实验、可视化） |

## 关键数字

| 指标 | 数值 |
|------|------|
| 模型参数 | 8B（Qwen2.5-7B + SigLIP） |
| ShareRobot QA pairs | 1,027,990 |
| RoboVQA BLEU-4 | 55.05（超第二名 18.75） |
| Affordance AP | 27.1%（超 Qwen2-VL 14.6） |
| Trajectory HD 降幅 | 94.2% |

---

## BibTeX

```bibtex
@inproceedings{DBLP:conf/cvpr/JiTSHZZWZ0AXSL025,
  author       = {Yuheng Ji and
                  Huajie Tan and
                  Jiayu Shi and
                  Xiaoshuai Hao and
                  Yuan Zhang and
                  Hengyuan Zhang and
                  Pengwei Wang and
                  Mengdi Zhao and
                  Yao Mu and
                  Pengju An and
                  Xinda Xue and
                  Qinghang Su and
                  Huaihai Lyu and
                  Xiaolong Zheng and
                  Jiaming Liu and
                  Zhongyuan Wang and
                  Shanghang Zhang},
  title        = {RoboBrain: {A} Unified Brain Model for Robotic Manipulation from Abstract
                  to Concrete},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2025, Nashville, TN, USA, June 11-15, 2025},
  pages        = {1724--1734},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2025},
  url          = {https://openaccess.thecvf.com/content/CVPR2025/html/Ji\_RoboBrain\_A\_Unified\_Brain\_Model\_for\_Robotic\_Manipulation\_from\_Abstract\_CVPR\_2025\_paper.html},
  doi          = {10.1109/CVPR52734.2025.00168}
}
```
