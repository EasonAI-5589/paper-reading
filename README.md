# Paper Reading Notes

个人论文阅读笔记仓库，按研究方向分类整理。

## 目录结构

```
paper-reading/
├── VLA/                      # Vision-Language-Action 机器人
│   ├── README.md             # 论文列表与元数据
│   ├── pdfs/                 # PDF 文件
│   └── *.md                  # 阅读笔记
├── Diffusion-Efficiency/     # Diffusion 模型效率优化
│   ├── README.md
│   └── pdfs/
├── Video-Generation/         # 视频生成模型
│   ├── README.md
│   └── pdfs/
└── README.md                 # 本文件
```

## 分类概览

| 分类 | 论文数 | 描述 |
|------|--------|------|
| [VLA](VLA/) | 4 | Vision-Language-Action 机器人策略模型 |
| [Diffusion-Efficiency](Diffusion-Efficiency/) | - | Diffusion 加速、蒸馏、量化 |
| [Video-Generation](Video-Generation/) | - | 视频生成模型（Wan、Sora 等） |

## 笔记模板

使用 [paper-reading skill](https://github.com/EasonAI-5589/my-claude-config/tree/main/skills/paper-reading) 生成结构化笔记：

- **Section 1**: Motivation & Problem Definition
- **Section 2**: Related Work
- **Section 3**: Method（含公式解析）
- **Section 4**: Experiments（含消融实验）

## 工具链

- **Zotero**: 文献管理、PDF 存储
- **Claude Code**: 论文阅读、笔记生成
- **MinerU**: PDF 图表提取
- **飞书**: 笔记导出与分享

---

*Last updated: 2025-12-26*
