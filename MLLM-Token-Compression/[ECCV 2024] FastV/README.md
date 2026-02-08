# An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference (FastV)

**作者**: Liang Chen, Haozhe Zhao, Tianyu Liu, Shuang Bai, Junyang Lin, Chang Zhou, Baobao Chang  
**单位**: Peking University, Alibaba Group  
**会议**: ECCV 2024  
**链接**: [arXiv](https://arxiv.org/abs/2403.06764) | [GitHub](https://github.com/pkunlpicler/FastV)

## 一句话总结

发现 LVLM 深层对 visual token 的注意力极其低效（仅为 system prompt 的 0.21%），提出 FastV 在第 2 层后剪枝 50% 的 image token，实现 45% FLOPs 减少且性能几乎不变。

## 核心贡献

1. **发现问题**: 揭示 LVLM 深层视觉注意力低效现象（anchor token 机制）
2. **提出 FastV**: Plug-and-play 的 visual token 剪枝方法，无需重训练
3. **全面验证**: 在图像/视频理解等多任务、多模型上验证有效性，视频场景甚至提升性能

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题、方法、效果概述 |
| [01 - Introduction](sections/01-introduction.md) | 动机、核心发现、三大贡献 + Figure 1 |
| [02 - Related Work](sections/02-related-work.md) | LVLM 架构、LLM 推理优化、VLM token 压缩 |
| [03 - Inefficient Attention](sections/03-inefficient-attention.md) | 注意力分析实验 + anchor token 发现 + Figures 2-4 |
| [04 - FastV Method](sections/04-method.md) | 方法详解：动态剪枝 + FLOPs 估算 + Figures 5-6 |
| [05 - Experiments](sections/05-experiments.md) | 全面实验：Tables 1-7 + Figure 7 + 消融实验 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结 + 局限性分析 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 推荐配置 | K=2, R=50% |
| FLOPs 减少 | ~45% |
| 深层 img attention vs sys prompt | 0.21% (472x 差距) |
| Image token 占输入比例 | ~64% |
| 13B+FastV 延迟 vs 7B | 0.341s ≈ 0.344s |
| 视频 token 数 (Video-LLaVA) | 2048 |
