# π0.5: a Vision-Language-Action Model with Open-World Generalization

**作者**: Physical Intelligence (Kevin Black, Noah Brown, Danny Driess, Chelsea Finn, Sergey Levine, Karl Pertsch, et al.)
**会议**: CoRL 2025 (Oral) | **年份**: 2025
**链接**: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | [项目主页](https://pi.website/blog/pi05)

## 一句话总结
π0.5 通过在异构数据源（多机器人、语义预测、网络数据、语言指令）上协同训练 VLA，首次实现端到端机器人系统在全新家庭中完成 10-15 分钟的长时域灵巧操作任务。

## 核心贡献
1. **异构数据 Co-training 框架** — 整合 6 类数据源（MM/ME/CE/HL/WD/VI），97.6% 数据不来自目标域
2. **两阶段训练策略** — Pre-training 用离散 FAST token (高效) → Post-training 加 flow matching action expert (精细)
3. **统一层级架构** — 同一模型同时做高层子任务预测和低层动作生成
4. **开放世界泛化验证** — 3 个全新真实家庭中的定量评估，性能接近在测试环境上训练的模型
5. **详尽的 Ablation** — 证明跨机器人迁移、网络数据、语言指令各自的贡献

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：异构 co-training → 开放世界泛化 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 + Figure 1-2 |
| [02 - Related Work](sections/02-related-work.md) | 四方向定位：通用策略/co-training/推理规划/开放泛化 |
| [03 - Preliminaries](sections/03-preliminaries.md) | VLA 基础：模仿学习 + token化 + flow matching |
| [04 - Model & Training](sections/04-model-and-training.md) | 核心方法：架构 + 离散/连续动作 + 5+1 类数据源 + Figure 3-5 |
| [05 - Experiments](sections/05-experiments.md) | 5 个核心问题的实验验证 + Figure 6-13 |
| [06 - Discussion](sections/06-discussion.md) | 局限性 + 未来方向 |
| [07 - Appendix](sections/07-appendix.md) | 评估细节 + 逐任务分析 + 模型规格 + Figure 14-18 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 移动操作数据 | ~400 小时, ~100 个家庭 |
| 非目标域数据占比 | 97.6% (pre-training) |
| Pre-training steps | 280k |
| Post-training steps | 80k |
| VLM backbone | PaliGemma 2B |
| Action Expert | 300M |
| 控制频率 | 50 Hz |
| 动作维度 | 18-19 DoF |
| 推理去噪步数 | 10 |
| 任务持续时间 | 2-15 分钟 |

## 数据源速查

| 简称 | 全称 | Pre-train | Post-train | 核心作用 |
|------|------|-----------|------------|---------|
| MM | Mobile Manipulator | ✅ | ✅ | 目标域直接经验 |
| ME | Multi-Environment | ✅ | ✅ | 多样家庭的跨构型迁移 |
| CE | Cross-Embodiment | ✅ | ❌ | 多任务跨机器人迁移 |
| HL | High-Level | ✅ | ✅ | 子任务推理能力 |
| WD | Web Data | ✅ | ✅ | 语义理解 + OOD 物体泛化 |
| VI | Verbal Instruction | ❌ | ✅ | 高层策略演示 (仅 11% 但影响巨大) |

---

## BibTeX

```bibtex
@inproceedings{black2025pi05,
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and others},
  title        = {pi-0.5: a Vision-Language-Action Model with Open-World Generalization},
  booktitle    = {Conference on Robot Learning ({CoRL})},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2504.16054},
  eprinttype   = {arXiv},
  eprint       = {2504.16054}
}
```
