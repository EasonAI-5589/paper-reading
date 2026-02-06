# 📄 ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries |
| **作者** | Junfu Pu*, Teng Wang*, Yixiao Ge†, Yuying Ge, Chen Li, Ying Shan |
| **机构** | ARC Lab, Tencent PCG |
| **发布** | November 18, 2025 (arXiv) |
| **类型** | Technical Report |
| **arXiv** | https://arxiv.org/abs/2511.14349 |
| **Project** | https://arcchapter.github.io/ |
| **GitHub** | https://github.com/TencentARC/ARC-Chapter |

## 一句话总结

**首个百万级视频章节模型**，提出 GRACE 多对一匹配指标，在 VidChapters-7M 上 F1 提升 14%（45.3→59.3），首次证明 Video Chaptering 的 Scaling Law。

---

## 论文结构

| # | 章节 | 笔记 | 状态 | 重点 |
|---|------|------|------|------|
| 0 | Abstract | [📝](./sections/00-abstract.md) | ✅ | 核心贡献概览 |
| 1 | Introduction | [📝](./sections/01-introduction.md) | ✅ | 三大挑战 + 三大贡献 |
| 2 | Related Works | [📝](./sections/02-related-works.md) | ✅ | 任务演进脉络 |
| 3 | Data Collection | [📝](./sections/03-data-collection.md) | ✅ ⭐ | VidAtlas 数据集 |
| 4 | Method | [📝](./sections/04-method.md) | ✅ ⭐⭐ | GRACE + GRPO |
| 5 | Experiments | [📝](./sections/05-experiments.md) | ✅ ⭐⭐ | Scaling Law |
| 6 | Conclusion | [📝](./sections/06-conclusion.md) | ✅ | 总结 + 未来方向 |

---

## 核心贡献框架

```
ARC-Chapter
├── 📦 VidAtlas 数据集 (§3)
│   ├── 410k+ 视频，115k 小时
│   ├── 中英双语
│   └── 层级标注（短标题→结构化章节→时间戳描述）
│
├── 🏗️ 方法设计 (§4)
│   ├── Base: Qwen2.5-VL-7B
│   ├── 输入: Video (768帧) + ASR (文本)
│   ├── 输出: 18种Prompt模板
│   ├── 训练: SFT + Adaptive Modality Dropping
│   └── 优化: GRPO 强化学习
│
├── 📏 GRACE 评估指标 (§4.3)
│   ├── 多对一匹配（解决粒度歧义）
│   ├── DTW 动态规划找最优
│   └── BERTScore 语义评估
│
└── 🔬 关键发现 (§5)
    ├── Scaling Law: 数据量↑ → 性能持续提升
    ├── 多模态互补: ASR+Video 最佳
    └── GRPO 有效: 时间↑，语义不降反升
```

---

## Key Takeaways

### 1. 数据层面
- **VidAtlas**: 410k+ 视频，115k 小时，是之前研究的 **50 倍**
- **标注流程**: Whisper-v3 (ASR) + Qwen2.5-VL (视觉) → LLM 推理 → 验证

### 2. 方法层面
- **GRACE 指标**: 多对一匹配，解决"粗粒度 vs 细粒度"的标注歧义
- **GRPO**: 用 RL 直接优化时间准确性，KL=0.01 防止语言退化

### 3. 实验层面
- **Scaling Law**: 首次证明 Video Chaptering 性能随数据量持续提升（推翻之前 ~20k 饱和的观察）
- **多模态重要**: Video+ASR 比单模态高 5-8 SODA
- **迁移性强**: YouCook2, ActivityNet Captions 都达到 SOTA

---

## 性能对比

| Benchmark | 指标 | Chapter-Llama | ARC-Chapter | 提升 |
|-----------|------|---------------|-------------|------|
| VidChapters-7M | F1 | 45.3 | **59.3** | +31% |
| VidChapters-7M | SODA | 19.3 | **30.6** | +58% |
| VidChapters-7M | CIDEr | 100.9 | **186.6** | +85% |
| VidAtlas (中文) | F1 | - | **66.2** | - |
| YouCook2 | SODA | 7.2 | **12.5** | +74% |

---

## 对 Apple Assignment 的价值

| 问题 | 参考价值 |
|------|----------|
| **Q1 评估维度** | 时间准确性 + 语义质量 + 层级结构 |
| **Q2 评估指标** | GRACE (多对一匹配)，F1/tIoU/SODA/CIDEr |
| **Q3 人工审核** | 半自动标注流程 + 验证步骤 |
| **Q4 评分标准** | 层级输出提供多粒度评估维度 |
| **Q5 LLM 错误** | 时间戳偏移、粒度不匹配、模态依赖 |

---

## 附件

### 📄 论文 PDF
- **[ARC-Chapter.pdf](./ARC-Chapter.pdf)** ← 直接打开对照阅读

### 📁 完整解析
```
./2511.14349-ec960086-4710-41e0-98a4-fdb14f73ae01/
├── full.md          # 完整 Markdown (85KB)
├── images/          # 34 张论文图片
├── *.pdf            # 原始 PDF (8.6MB)
└── *.json           # 解析元数据
```

### 🔧 VSCode PDF 标注工具

在 VSCode 扩展商店安装:
1. **vscode-pdf** (`tomoki1207.pdf`) - 基础 PDF 预览
2. **pdf-annotate** - PDF 标注 + Markdown 笔记导出

安装后可直接在 VSCode 中打开 PDF 对照阅读和标注。

---

*笔记由 1号机 🤖 整理*  
*首次阅读：2026-02-06*
