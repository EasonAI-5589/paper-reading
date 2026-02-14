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
├── repo/                           # [submodule] 原始 GitHub 仓库
├── survey-token-compression/       # 📖 Survey 论文笔记
├── fastv/                          # 📖 FastV 论文笔记
├── pyramiddrop/                    # 📖 PyramidDrop (待读)
└── README.md                       # 本文件
```

## 🔗 资源链接

| 资源 | 链接 |
|------|------|
| 📄 Survey 论文 | [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176823010.07236701/v1) |
| 📚 GitHub | [yaolinli/MLLM-Token-Compression](https://github.com/yaolinli/MLLM-Token-Compression) |
| 📖 论文列表 | [100+ Papers](https://github.com/yaolinli/MLLM-Token-Compression#-paper-table) |

## 📖 阅读列表

| # | 论文 | 类型 | 状态 | 笔记 |
|---|------|------|------|------|
| 001 | Towards Efficient MLLMs: A Survey on Token Compression | Survey | ✅ 已读 | [📝](./survey-token-compression/) |
| 002 | FastV: An Image is Worth 1/2 Tokens After Layer 2 | Method | ✅ 已读 | [📝](./fastv/) |
| 003 | PyramidDrop | Method | 📋 待读 | - |
| 004 | VisionZip | Method | 📋 待读 | - |
| 005 | DART | Method | 📋 待读 | - |
| 006 | VScan: Rethinking Visual Token Reduction | Method | ✅ 已读 | [📝](./%5BArxiv%202505.22654%5D%20VScan/) |

## 📈 阅读进度

- [x] Survey 全文阅读和笔记整理
- [x] 分类框架 (Taxonomy)
- [x] 方法选择指南
- [ ] 重点论文深读 (FastV, VisionZip, PyramidDrop, DART...)
- [ ] 与自己研究的结合点

## 💡 Key Takeaways

1. **Vision tokens 冗余极高** — 自然场景只需 ~9 tokens/image，OCR 需要 144-576
2. **Attention-based 剪枝有位置偏差** — 用 similarity 替代更稳定 (RoPE 的长期衰减特性)
3. **Merging vs Dropping 互补** — 混合策略效果最佳（先 merge 低层，再 drop 高层）
4. **Text-guided 适合单轮 QA，Purely-visual 适合多轮对话**
5. **训练时在前端压缩 (VE/Projector)，推理时在后端压缩 (LLM/KV Cache)**

## 🗂️ 核心分类框架

```
Where to Compress?
├── Vision Encoder (§3.1)
│   ├── Inside-Encoder (Token Dropping/Merging)
│   └── Outside-Encoder (Purely-Vision/Text-guided)
├── Projector (§3.2)
│   ├── Transformation-based (Pooling/PixelShuffle/Conv)
│   ├── Query-based (Q-Former)
│   └── Importance-driven
├── LLM (§3.3)
│   ├── Prefilling Stage (Importance/Learnable/Merging/Fusion)
│   └── Decoding Stage (KV Cache Compression)
└── Hybrid (§3.4)
    ├── Collaborative Compression
    └── Progressive Compression
```

---

*课题由 3号机 协助整理 📚*
*更新日期：2026-02-06*
