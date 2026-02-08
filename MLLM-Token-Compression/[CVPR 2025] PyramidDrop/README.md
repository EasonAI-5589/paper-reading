# PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction

**作者**: Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, Dahua Lin  
**机构**: USTC, Shanghai AI Laboratory, CUHK  
**会议**: CVPR 2025 | **年份**: 2025  
**链接**: [arXiv:2410.17247](https://arxiv.org/abs/2410.17247) | [GitHub](https://github.com/Cooperx521/PyramidDrop)

## 一句话总结
基于 "浅层需要所有视觉 token，深层冗余递增" 的观察，提出 PyramidDrop：将 LVLM 分成多个 stage，每个 stage 末尾按比例丢弃低重要性图像 token，实现训练 40%+ 和推理 55%+ 的加速，性能几乎无损。

## 核心贡献
1. **关键发现**：LVLM 对视觉 token 的理解是逐层递进的 — 浅层全局理解（需要所有 token），深层聚焦关键区域（大部分 token 冗余）
2. **PyramidDrop 方法**：利用 last instruction token 与 image token 的 attention score 排名重要性，渐进式丢弃，token 数量呈金字塔式指数递减
3. **训练 + 推理双加速**：同时适用于训练和推理，也可作为 plug-and-play 推理加速策略
4. **更高分辨率更低成本**：双倍分辨率 + PyramidDrop 的训练成本甚至低于原始分辨率

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、方法、关键数字 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 核心观察 + 贡献 + Figure 1（冗余度层级分析） |
| [02 - Related Work](sections/02-related-work.md) | Token Reduction + LVLM 发展 |
| [03 - Method](sections/03-method.md) | 视觉 token 冗余研究 + PyramidDrop 设计 + 效率分析 + Figure 2 |
| [04 - Experiments](sections/04-experiments.md) | 推理/训练加速实验 + 16 benchmarks + 消融 + Tables 1-8 + Figure 3-4 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结 |
| [06 - Appendix](sections/06-appendix.md) | Stage 数 S 消融实验 + Table 9 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 训练加速（LLaVA-NeXT-7B） | 40.4%（366→218 GPU hours） |
| 训练加速（LLaVA-NeXT-p9） | 44.3%（483→269 GPU hours） |
| 推理 FLOPs 减少（LLaVA-NeXT） | 54%（20.8→9.46T） |
| 推理 FLOPs 减少（LLaVA-1.5） | 53%（3.82→1.78T） |
| 默认配置 | λ=0.5, S=4 |
| 理论计算节省 | 53.2% |
| 训练加速倍数 | 1.82× |
| 推理加速倍数 | 2.22× |
