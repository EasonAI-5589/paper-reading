# FastV: An Image is Worth 1/2 Tokens After Layer 2

**arXiv**: [2403.06764](https://arxiv.org/abs/2403.06764)  
**会议**: ECCV 2024 Oral  
**分类**: Token Pruning / Compression  
**解析工具**: MinerU API (vlm model)  
**解析时间**: 2026-02-06

## 核心贡献

1. **发现**: 在 LVLM 深层，视觉 token 的注意力计算极其低效
2. **FastV 方法**: 在早期层学习自适应注意力模式，在后续层 pruning 视觉 token
3. **Plug-and-play**: 无需训练，直接应用
4. **高效**: LLaVA-1.5-13B 减少 45% FLOPs，性能几乎无损

## 方法

- 第 2 层后，大部分视觉 token 对输出影响很小
- 根据注意力分数 pruning 不重要的视觉 token
- 可调参数控制 efficiency/performance trade-off

## 实验结果

- 多个 benchmark 验证（NoCaps, Flickr30k, A-OKVQA, MMMU）
- 45% FLOPs 减少，性能几乎不变
- 13B 模型可压缩到比 7B 更低成本但性能更好

## 文件说明

| 文件 | 说明 |
|------|------|
| `full.md` | 完整论文 Markdown |
| `*_origin.pdf` | 原始 PDF |
| `content_list_v2.json` | 结构化内容 |
| `images/` | 提取的图片 |
| `sections/` | 分章节内容 (旧) |
| `figures/` | 论文图片 (旧) |

## 代码

GitHub: https://github.com/pkunlpicler/FastV
