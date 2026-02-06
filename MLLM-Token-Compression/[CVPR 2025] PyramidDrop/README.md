# PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction

**arXiv**: [2410.17247](https://arxiv.org/abs/2410.17247)  
**会议**: CVPR 2025  
**分类**: Token Dropping / Compression  
**解析工具**: MinerU API (vlm model)  
**解析时间**: 2026-02-06

## 核心贡献

1. **发现**: 浅层需要所有视觉 token，深层冗余逐渐增加
2. **PyramidDrop**: 分阶段逐步 drop 视觉 token
3. **训练+推理加速**: 40% 训练时间减少，55% 推理 FLOPs 加速
4. **Plug-and-play**: 可直接应用于现有模型

## 方法

- 将 LVLM 分成多个 stage
- 每个 stage 结束时按预定比例 drop 部分 image token
- 基于轻量级相似度计算选择要 drop 的 token
- 渐进式减少，形成金字塔结构

## 实验结果

- LLaVA-NeXT 上验证
- 40% 训练时间减少
- 55% 推理 FLOPs 加速
- 性能几乎无损

## 文件说明

| 文件 | 说明 |
|------|------|
| `full.md` | 完整论文 Markdown |
| `*_origin.pdf` | 原始 PDF |
| `content_list_v2.json` | 结构化内容 |
| `images/` | 提取的图片 |

## 代码

GitHub: https://github.com/Cooperx521/PyramidDrop
