![Paper Reading Banner](./banner.png)

# Paper Reading 📚

Eason 的文献阅读仓库，按课题组织。

---

## 课题列表

### 📊 MLLM Token Compression
多模态大模型视觉 Token 压缩方法

| 论文 | 会议 | 方法特点 |
|------|------|----------|
| ⭐ [Survey](./MLLM-Token-Compression/[TechRxiv%202025]%20Survey-Token-Compression/) | TechRxiv 2025 | **综述** - Token 压缩方法全景 |
| [FastV](./MLLM-Token-Compression/[ECCV%202024]%20FastV/) | ECCV 2024 | 第2层后固定剪枝，简单高效 |
| [PyramidDrop](./MLLM-Token-Compression/[CVPR%202025]%20PyramidDrop/) | CVPR 2025 | 金字塔式渐进剪枝 |
| [SparseVLM](./MLLM-Token-Compression/[ICML%202025]%20SparseVLM/) | ICML 2025 | 文本引导 + Token回收 |
| [SwiftVLM](./MLLM-Token-Compression/[Arxiv%202403.12178]%20SwiftVLM/) | arXiv | Swift token 压缩 |

📖 [方法对比总结](./MLLM-Token-Compression/methods-list.md)

---

### 🎬 Video Chaptering
视频章节生成 - 自动将长视频分割成语义连贯的章节，并生成章节标题

| 论文 | 会议 | 方法特点 | 性能 (F1) |
|------|------|----------|-----------|
| [SODA](./Video-Chaptering/[ECCV%202020]%20SODA/) | ECCV 2020 | **评估指标** - 考虑故事性的评估框架 | - |
| [VidChapters-7M](./Video-Chaptering/[NeurIPS%202023]%20VidChapters-7M/) | NeurIPS 2023 | **THE Benchmark** - 817K视频, 7M章节 + Baseline | 25.0 |
| [Chapter-Llama](./Video-Chaptering/[CVPR%202025]%20Chapter-Llama/) | CVPR 2025 | LLM 文本域方法, Speech-guided 采样 | 45.3 |
| [ARC-Chapter](./Video-Chaptering/[arXiv%202025]%20ARC-Chapter/) | arXiv 2025 | **SOTA** - Qwen2.5-VL + GRPO, GRACE 指标 | **59.3** |

📖 [Video Chaptering 详细总结](./Video-Chaptering/README.md)

---

## 目录结构

```
paper-reading/
├── MLLM-Token-Compression/          # MLLM Token 压缩
│   ├── [TechRxiv 2025] Survey/
│   ├── [ECCV 2024] FastV/
│   ├── [CVPR 2025] PyramidDrop/
│   ├── [ICML 2025] SparseVLM/
│   └── methods-list.md
│
├── Video-Chaptering/                # 视频章节生成
│   ├── [ECCV 2020] SODA/
│   ├── [NeurIPS 2023] VidChapters-7M/
│   ├── [CVPR 2025] Chapter-Llama/
│   ├── [arXiv 2025] ARC-Chapter/
│   └── README.md                    # 领域详细总结
│
└── README.md                        # 本文件
```

---

## 论文文件夹结构

每篇论文包含：
```
[会议 年份] 论文名/
├── README.md           # 阅读笔记与总结
├── full.md             # MinerU 解析的完整内容
├── paper.pdf           # 原始 PDF
├── content_list.json   # 结构化内容
├── layout.json         # 版面分析
└── images/             # 论文图片
```

---

## 命名规范

文件夹命名: `[会议 年份] 论文名`
- 例: `[CVPR 2025] Chapter-Llama`
- 例: `[arXiv 2025] ARC-Chapter`
- 例: `[NeurIPS 2023] VidChapters-7M`

---

## 使用说明

### Clone 仓库
```bash
git clone --recursive https://github.com/EasonAI-5589/paper-reading.git
```

### 更新
```bash
cd paper-reading
git pull
```

---

## 相关资源

- 🍎 [Apple Interview 准备](https://github.com/EasonAI-5589/apple-interview) - 基于 Video Chaptering 的面试
- 📖 [葵花宝典](https://github.com/EasonAI-5589/openclaw-baodian) - OpenClaw 配置文档

---

*由 1号机 协助整理 📚 | 更新: 2026-02-07*
