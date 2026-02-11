# π₀: A Vision-Language-Action Flow Model for General Robot Control

**作者**: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky  
**机构**: Physical Intelligence  
**会议**: CoRL 2025  
**链接**: [Blog](https://physicalintelligence.company/blog/pi0)

## 一句话总结

基于预训练 VLM (PaliGemma) + flow matching 的通用机器人策略，在 10,000+ 小时跨 embodiment 数据上预训练，通过 pre-training/post-training 两阶段训练配方，在叠衣服、清桌子、组装箱子等复杂灵巧任务上达到 SOTA。

## 核心贡献

1. **架构**: VLM backbone + Action Expert (MoE 风格) + flow matching → 首个支持高频 (50Hz) action chunk 的 VLA
2. **训练配方**: 借鉴 LLM 的 pre-training / post-training 范式 → 预训练提供广度和恢复能力，post-training 提供精通
3. **规模**: 10,000 小时机器人数据 + OXE，7 种机器人构型，68 个任务 → 迄今最大的机器人学习实验
4. **评估**: 全面超越 OpenVLA、Octo、ACT、Diffusion Policy，展示数十分钟长的复杂多阶段灵巧任务

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、方法、评估维度 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 三大贡献 + Figure 1 & 2 |
| [02 - Related Work](sections/02-related-work.md) | VLA、diffusion/flow matching、大规模机器人学习三条线 |
| [03 - Overview](sections/03-overview.md) | 框架全局图 (Figure 3)：数据→模型→多 embodiment 输出 |
| [04 - Model](sections/04-model.md) | π₀ 架构详解：VLM + Action Expert + flow matching 公式 |
| [05 - Data & Training](sections/05-data-and-training.md) | 数据集、训练配方、语言策略、7 种机器人平台 |
| [06 - Experiments](sections/06-experiments.md) | 四组实验：out-of-box / 语言跟随 / fine-tuning / 复杂任务 |
| [07 - Discussion](sections/07-discussion.md) | 总结、局限性、未来方向 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 模型参数 | 3.3B (3B VLM + 300M action expert) |
| 预训练数据 | ~10,000 小时 |
| 机器人构型 | 7 种 (单臂/双臂/移动) |
| 任务数 | 68 个 |
| Action chunk 长度 | H=50 (1秒 @50Hz) |
| Flow matching 步数 | 10 步 (δ=0.1) |
| 推理时间 | 73ms on-board (RTX 4090) |
| 预训练步数 | 700k |
| VLM backbone | PaliGemma 3B |

---

## BibTeX

```bibtex
@article{black2024pi0,
  author       = {Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and others},
  title        = {pi-0: A Vision-Language-Action Flow Model for General Robot Control},
  journal      = {CoRR},
  volume       = {abs/2410.24164},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2410.24164},
  eprinttype   = {arXiv},
  eprint       = {2410.24164}
}
```
