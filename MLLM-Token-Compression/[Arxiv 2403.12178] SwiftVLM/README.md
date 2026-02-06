# SwiftVLM: Efficient Vision-Language Model Inference via Cross-Layer Token Bypass

**arXiv**: [2602.03134](https://arxiv.org/abs/2602.03134)  
**分类**: compress (Token Compression)  
**解析工具**: MinerU API (vlm model)  
**解析时间**: 2026-02-06

## 核心贡献

1. **Bypass 策略**: 不直接丢弃低重要性 token，而是保留并转发到后续层重新评估
2. **Layer-wise 分析**: 揭示不同层对 visual token 重要性判断存在显著差异
3. **动态规划选层**: 选择具有高判别能力的层进行 pruning
4. **Training-free**: 无需额外训练

## 方法

- 在浅层 pruning 后，未选中的 visual token 被保留并 bypass 到深层
- 深层重新评估这些 token 的重要性
- 避免早期 pruning 导致的不可逆信息丢失

## 实验结果

- 在多个 VLM 和 benchmark 上优于现有 pruning 策略
- 更好的 accuracy-efficiency trade-off
- 更准确的 visual token 选择

## 文件说明

| 文件 | 说明 |
|------|------|
| `full.md` | 完整论文 Markdown |
| `*_origin.pdf` | 原始 PDF |
| `content_list_v2.json` | 结构化内容 |
| `layout.json` | 布局信息 |
| `images/` | 提取的图片 |

## 相关工作

- FastV
- PDrop
- SparseVLM
- FEATHER
- VisionZip
