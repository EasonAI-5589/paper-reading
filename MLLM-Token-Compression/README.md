# MLLM Token Compression 📦

多模态大语言模型 (MLLM) 的视觉 Token 压缩技术研究。

## 📋 课题简介

高分辨率图像和长视频会产生大量 visual tokens，导致计算复杂度爆炸、GPU 内存消耗巨大。Token Compression 通过减少 token 数量来提升效率，同时保持关键语义信息。

**核心问题**：
- Where to compress? (Vision Encoder / Projector / LLM)
- How to compress? (Pruning / Merging / Query-based)

## 📂 目录结构

```
MLLM-Token-Compression/
├── repo/               # [submodule] 原始 GitHub 仓库
├── notes.md            # 📝 阅读笔记
└── README.md           # 本文件
```

## 🔗 资源链接

| 资源 | 链接 |
|------|------|
| 📚 原始仓库 | **[yaolinli/MLLM-Token-Compression](https://github.com/yaolinli/MLLM-Token-Compression)** |
| 📄 Survey PDF | [直接下载](https://github.com/yaolinli/MLLM-Token-Compression/releases/download/v1.0_2512/Towards.Efficient.Multimodal.Large.Language.Models.-.A.Survey.on.Token.Compression.-.v2512.pdf) |
| 📖 论文列表 | [100+ Papers](https://github.com/yaolinli/MLLM-Token-Compression#-paper-table) |
| 🔬 TechRxiv | [DOI](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176823010.07236701/v1) |

## 📖 阅读列表

| # | 论文 | 类型 | 状态 | 笔记 |
|---|------|------|------|------|
| 001 | Towards Efficient MLLMs: A Survey on Token Compression | Survey | ✅ 已读 | [notes.md](./notes.md) |
| 002 | FastV | Method | 📋 待读 | - |
| 003 | VisionZip | Method | 📋 待读 | - |
| 004 | PyramidDrop | Method | 📋 待读 | - |

## 📈 阅读进度

- [x] Survey 主体内容
- [x] 分类框架 (Taxonomy)
- [x] 方法选择指南
- [ ] 重点论文深读 (FastV, VisionZip, PyramidDrop...)
- [ ] 与自己研究的结合点

## 💡 Key Takeaways

1. **Vision tokens 冗余极高** — 自然场景只需 ~9 tokens/image
2. **Attention-based 剪枝有位置偏差** — 用 similarity 替代更稳定
3. **Merging vs Dropping 互补** — 混合策略效果最佳
4. **Text-guided vs Purely-visual** — 根据场景选择

## 📝 详细笔记

→ [notes.md](./notes.md)

---

*课题由 3号机 协助整理 📚*
