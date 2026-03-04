# RoboBrain 2.0: Unified Embodied Spatial Reasoning and Planning

**作者**: Huajie Tan, Yuheng Ji, Jiayu Shi, Xiaoshuai Hao, ... **Shanghang Zhang** (仉尚航, 通讯作者)  
**来源**: Arxiv 2507.02029 | **机构**: 北京大学 + BAAI + 中科院  
**链接**: [arXiv](https://arxiv.org/abs/2507.02029)

## 一句话总结

RoboBrain 系列的第二代，从 LLaVA 换到 **Qwen2.5-VL** 底座，统一空间理解（Spatial）和时序规划（Temporal）两大能力，提供 **7B 和 32B** 两个规模，在 12+ benchmark 上达到 SOTA。

## 核心贡献

1. **统一架构**：基于 Qwen2.5-VL，统一处理 spatial reasoning（pointing/grounding/affordance）和 temporal planning
2. **大规模训练数据**：3 大类数据 — 通用多模态 + 空间理解 + 时序规划，精心设计数据配比
3. **三阶段训练策略**：预训练 → 多任务微调 → GRPO 强化学习
4. **工程优化**：FlagScale 分布式框架 + 5 项训练优化 + VeRL 强化学习基础设施
5. **双规模发布**：7B 和 32B，32B 在多个 benchmark 上超越 GPT-4o

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + Figure 1 雷达图 |
| [01 - Introduction](sections/01-introduction.md) | 三个瓶颈 + 解决方案 + Figure 2 能力全景 |
| [02 - Architecture](sections/02-architecture.md) | Qwen2.5-VL 架构 + 四大组件 + Figure 3 |
| [03 - Training Data](sections/03-training-data.md) | 三大类数据 + 10 个子数据集详解 |
| [04 - Training Strategy](sections/04-training-strategy.md) | 三阶段训练 + Table 1 配置 |
| [05 - Infrastructures](sections/05-infrastructures.md) | FlagScale + 5 项优化 + VeRL + 量化推理 |
| [06 - Evaluation](sections/06-evaluation.md) | 12 个 benchmark + Table 2/3/4 |
| [07 - Conclusion](sections/07-conclusion.md) | 总结 + 未来方向（VLA、系统集成） |
| [08 - Appendix](sections/08-appendix.md) | 概要：定性示例 + prompt 模板 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 模型规模 | 7B / 32B |
| 底座模型 | Qwen2.5-VL |
| 训练数据 | 通用 + Spatial + Temporal |
| 超越 GPT-4o 的 benchmark 数 | 多个（32B） |

---

## BibTeX

```bibtex
@article{DBLP:journals/corr/abs-2507-02029,
  author       = {Mingyu Cao and
                  Huajie Tan and
                  Yuheng Ji and
                  Minglan Lin and
                  Zhiyu Li and
                  Zhou Cao and
                  Pengwei Wang and
                  Enshen Zhou and
                  Yi Han and
                  Yingbo Tang and
                  Xiangqi Xu and
                  Wei Guo and
                  Yaoxu Lyu and
                  Yijie Xu and
                  Jiayu Shi and
                  Mengfei Du and
                  Cheng Chi and
                  Mengdi Zhao and
                  Xiaoshuai Hao and
                  Junkai Zhao and
                  Xiaojie Zhang and
                  Shanyu Rong and
                  Huaihai Lyu and
                  Zhengliang Cai and
                  Yankai Fu and
                  Ning Chen and
                  Bolun Zhang and
                  Lingfeng Zhang and
                  Shuyi Zhang and
                  Dong Liu and
                  Xi Feng and
                  Songjing Wang and
                  Xiaodan Liu and
                  Yance Jiao and
                  Mengsi Lyu and
                  Zhuo Chen and
                  Chenrui He and
                  Yulong Ao and
                  Xue Sun and
                  Zheqi He and
                  Jingshu Zheng and
                  Xi Yang and
                  Donghai Shi and
                  Kunchang Xie and
                  Bochao Zhang and
                  Shaokai Nie and
                  Chunlei Men and
                  Yonghua Lin and
                  Zhongyuan Wang and
                  Tiejun Huang and
                  Shanghang Zhang},
  title        = {RoboBrain 2.0 Technical Report},
  journal      = {CoRR},
  volume       = {abs/2507.02029},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2507.02029},
  doi          = {10.48550/ARXIV.2507.02029},
  eprinttype    = {arXiv},
  eprint       = {2507.02029}
}
```
