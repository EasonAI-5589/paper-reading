# Video Chaptering 视频章节生成

> 📁 研究方向: 自动将长视频分割成语义连贯的章节，并生成章节标题

---

## 🎯 任务定义

**Video Chaptering** = 时间分割 + 标题生成

给定一个长视频，目标是:
1. **Temporal Segmentation**: 找到章节边界时间戳
2. **Title Generation**: 为每个章节生成描述性标题

---

## 📚 核心论文

| 论文 | 会议 | 年份 | 贡献 | 链接 |
|------|------|------|------|------|
| [ARC-Chapter](./ARC-Chapter/) | arXiv | 2025.11 | **最新 SOTA**, GRACE 指标, VidAtlas 数据集 | [arxiv](https://arxiv.org/abs/2511.14349) · [项目](https://arcchapter.github.io/) |
| Chapter-Llama | CVPR | 2025.03 | 小时级视频章节, LLM 文本域方法 | [arxiv](https://arxiv.org/abs/2504.00072) · [项目](https://imagine.enpc.fr/~lucas.ventura/chapter-llama/) |
| YTSEG | EACL | 2024 | YouTube 章节分割, 多粒度 benchmark | [arxiv](https://arxiv.org/abs/2402.17279) |
| [VidChapters-7M](./VidChapters-7M/) | NeurIPS | 2023 | **THE benchmark** (817K 视频, 7M 章节) + Baseline | [arxiv](https://arxiv.org/abs/2309.13952) · [项目](https://antoyang.github.io/vidchapters.html) |

---

## 📊 主要 Benchmarks

| Dataset | 规模 | 平均时长 | 语言 | 链接 |
|---------|------|----------|------|------|
| VidChapters-7M | 817K 视频, 7M 章节 | ~12 min | EN | [arxiv](https://arxiv.org/abs/2309.13952) · [数据](https://antoyang.github.io/vidchapters.html) |
| VidAtlas | 410K 视频 | 16.8 min | EN + ZH | [arxiv](https://arxiv.org/abs/2511.14349) · [项目](https://arcchapter.github.io/) |
| YTSEG | 19K 视频 | - | EN | [arxiv](https://arxiv.org/abs/2402.17279) |

---

## 📈 评估指标

| 指标 | 评估内容 | 备注 | 出处 |
|------|----------|------|------|
| **F1** | 分割准确度 | 主要指标 | 通用 |
| **SODA** | 整体质量 (分割+标题) | 一对一匹配 | [arxiv](https://arxiv.org/abs/2005.03954) |
| **GRACE** | 整体质量 | 多对一匹配 (更灵活) | [ARC-Chapter](https://arxiv.org/abs/2511.14349) |
| CIDEr | 标题质量 | 文本相似度 | [arxiv](https://arxiv.org/abs/1411.5726) |
| mIoU | 时间重叠 | 辅助指标 | 通用 |

---

## 🏆 性能对比 (VidChapters-7M)

| 模型 | 架构 | F1 ↑ | SODA ↑ | 论文 |
|------|------|------|--------|------|
| PDVC | Proposal-based | 18.3 | 8.5 | [2023](https://arxiv.org/abs/2309.13952) |
| Vid2Seq | 端到端 Seq2Seq | 23.1 | 10.8 | [2023](https://arxiv.org/abs/2309.13952) |
| **VidChapters Baseline** | CLIP + GPT-2 | 25.0 | 12.1 | [2023](https://arxiv.org/abs/2309.13952) |
| YTSEG | 多粒度分割 | ~30 | - | [2024](https://arxiv.org/abs/2402.17279) |
| Chapter-Llama | LLaMA + Speech-guided | 45.3 | - | [CVPR 2025](https://arxiv.org/abs/2504.00072) |
| **ARC-Chapter** | Qwen2.5-VL + GRPO | **59.3** | **30.6** | [2025](https://arxiv.org/abs/2511.14349) |

> 💡 **Baseline 说明**: VidChapters-7M 论文 (2023) 自己提出的基准模型，使用 CLIP 编码视频帧 + GPT-2 生成章节边界和标题

---

## 🔗 相关任务

| 任务 | 说明 | 代表 Benchmark |
|------|------|----------------|
| Dense Video Captioning | 密集视频描述 | [ActivityNet Captions](https://arxiv.org/abs/1705.00754), [YouCook2](https://arxiv.org/abs/1703.09788) |
| Temporal Grounding | 给定文本找时间段 | [Charades-STA](https://arxiv.org/abs/1705.02101) |
| Video Summarization | 视频摘要 | TVSum, SumMe |
| Action Segmentation | 动作分割 | Breakfast, 50Salads |

---

## 🍎 Apple Assignment 相关

这个方向直接对应 Apple Data Scientist Assignment Part B:
> LLM 自动生成视频章节 (标题 + 时间戳) 的质量评估框架

答题文档: [`/mnt/eason/apple-interview/Part_B/`](file:///mnt/eason/apple-interview/Part_B/)

---

## 📖 扩展阅读

- [Awesome Long Video Understanding](https://github.com/ttengwang/Awesome_Long_Form_Video_Understanding)
- [Dense Video Captioning Survey](https://arxiv.org/abs/2311.02538)

---

*Created: 2026-02-06 | Updated: 2026-02-07*
