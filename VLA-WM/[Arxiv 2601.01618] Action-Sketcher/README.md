# Action-Sketcher: From Reasoning to Action via Visual Sketches for Long-Horizon Robotic Manipulation

**作者**: Huajie Tan*, Peterson Co*, Yijie Xu*, Shanyu Rong, Yuheng Ji, Cheng Chi, Xiansheng Chen, Qiongyu Zhang, Zhongxia Zhao, Pengwei Wang†, Zhongyuan Wang, Shanghang Zhang🖂  
**机构**: Peking University / Beijing Academy of Artificial Intelligence (BAAI) / University of Sydney / Chinese Academy of Sciences  
**会议**: arXiv 2601.01618  
**链接**: [arXiv](https://arxiv.org/abs/2601.01618) | [Project Page](https://action-sketcher.github.io)

## 一句话总结

提出 **Visual Sketch**（点 + 框 + 箭头的显式空间意图中间表达）+ **See-Think-Sketch-Act** 循环框架，解决长时域操作中的空间模糊与时序脆弱问题，比 π₀.5 等 SOTA 方法在长时域复杂任务上有显著提升，且支持人工实时修改 Sketch 进行闭环纠错。

## 核心贡献

1. **Visual Sketch 形式化**: 将空间意图显式表示为图像平面上的几何原语（Boxes + Points + Arrows），作为高层推理与底层控制之间的可验证契约
2. **Action-Sketcher 框架**: See→Think→Sketch→Act 循环，通过 token-gate 机制（`<BOR>`/`<BOA>`）自适应切换 Reasoning Mode 和 Action Mode，支持实时人工干预
3. **多阶段课程训练**: Stage 1（时空预训练）→ Stage 2（reasoning-to-sketch 精调）→ Stage 3（sketch-to-action + mode 自适应）
4. **Human-in-Loop**: 因 Sketch 是显式可编辑的，人工在执行前修改 Sketch 可将成功率推向近乎完美

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 核心问题、方案、结果 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 空间/时序两大瓶颈 + 贡献 + Figure 1 & 2 |
| [02 - Related Work](sections/02-related-work.md) | VLA 分类、Visual Prompting 相关工作 |
| [03 - Method](sections/03-method.md) | Visual Sketch 定义 + Action-Sketcher 框架 + 训练策略 + Figure 2~5 |
| [04 - Experiments](sections/04-experiments.md) | LIBERO / RoboTwin 2.0 / 真机 + Human-in-Loop + 消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 + 局限性 |

## 模型规格

| 项目 | 内容 |
|------|------|
| Backbone | π₀（PaliGemma + Flow Matching Action Expert）|
| Visual Sketch 原语 | Boxes + Points + Arrows（Translation + Rotation）|
| 推理模式切换 | `<BOR>` 触发推理，`<BOA>` 触发动作 |
| 训练阶段 | 3 阶段课程：时空→推理-to-Sketch→Sketch-to-Action |
| Stage 1 数据 | 3.4M 空间理解 + 870k 时序序列 |
| Stage 2 数据 | 21k reasoning-to-sketch 样本（含真机 2.6k episodes）|
| 评估 Benchmark | LIBERO / RoboTwin 2.0 / 真机（Agilex + Galaxea 双臂）|

---

## BibTeX

```bibtex
@article{tan2026actionsketcher,
  author    = {Huajie Tan and Peterson Co and Yijie Xu and Shanyu Rong and Yuheng Ji and Cheng Chi and Xiansheng Chen and Qiongyu Zhang and Zhongxia Zhao and Pengwei Wang and Zhongyuan Wang and Shanghang Zhang},
  title     = {Action-Sketcher: From Reasoning to Action via Visual Sketches for Long-Horizon Robotic Manipulation},
  journal   = {arXiv preprint arXiv:2601.01618},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.01618}
}
```
