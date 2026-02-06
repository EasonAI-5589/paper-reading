![Paper Reading Banner](./banner.png)

# Paper Reading 📚

Eason 的文献阅读仓库，按课题组织。

## 课题列表

### 📊 MLLM Token Compression
多模态大模型视觉 Token 压缩方法

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [FastV](./MLLM-Token-Compression/[ECCV%202024]%20FastV/) | ECCV 2024 | 第2层后固定剪枝，简单高效 |
| [PyramidDrop](./MLLM-Token-Compression/[CVPR%202025]%20PyramidDrop/) | CVPR 2025 | 金字塔式渐进剪枝 |
| [SparseVLM](./MLLM-Token-Compression/[ICML%202025]%20SparseVLM/) | ICML 2025 | 文本引导 + Token回收 |
| [SwiftVLM](./MLLM-Token-Compression/[Arxiv%202403.12178]%20SwiftVLM/) | arXiv | Swift token 压缩 |

📖 [方法对比总结](./MLLM-Token-Compression/methods-list.md)

### 🎬 Video Chaptering
视频章节分割与理解

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| [SODA](./Video-Chaptering/[ECCV%202020]%20SODA/) | ECCV 2020 | 早期视频分段方法 |
| [VidChapters-7M](./Video-Chaptering/[NeurIPS%202023]%20VidChapters-7M/) | NeurIPS 2023 | 大规模视频章节数据集 |
| [Chapter-Llama](./Video-Chaptering/[CVPR%202025]%20Chapter-Llama/) | CVPR 2025 | LLM-based 章节生成 |

---

## 目录结构

```
paper-reading/
├── MLLM-Token-Compression/     # MLLM Token 压缩
│   ├── [ECCV 2024] FastV/
│   ├── [CVPR 2025] PyramidDrop/
│   ├── [ICML 2025] SparseVLM/
│   └── ...
├── Video-Chaptering/           # 视频章节分割
│   ├── [ECCV 2020] SODA/
│   └── ...
└── README.md
```

每篇论文包含：
- `README.md` - 阅读笔记与总结
- `full.md` - MinerU 解析的完整内容
- `images/` - 论文图片
- `*_origin.pdf` - 原始 PDF

---

## 使用说明

### Clone 仓库
```bash
git clone --recursive https://github.com/EasonAI-5589/paper-reading.git
```

### 更新 submodules
```bash
git submodule update --remote
```

---

*由 3号机 协助整理 📚*
