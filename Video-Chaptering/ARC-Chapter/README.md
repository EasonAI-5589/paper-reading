# ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries

> 📅 Paper Date: November 18, 2025  
> 🏢 Affiliation: ARC Lab, Tencent PCG  
> 📎 arXiv: https://arxiv.org/abs/2511.14349  
> 🌐 Project: https://arcchapter.github.io/  
> 💻 Code: https://github.com/TencentARC/ARC-Chapter

---

## 📋 TL;DR

ARC-Chapter 是目前 **最新 SOTA** 的视频章节生成模型，在 VidChapters-7M 上比 Chapter-Llama 高 **14% F1**。

**核心贡献**:
1. 百万级视频章节训练数据 (VidAtlas)
2. 层级式标注: 短标题 → 结构化章节 → 时间戳描述
3. 新评估指标 GRACE (解决标注粒度问题)
4. 用 GRPO 强化学习优化时间戳准确性

---

## 🎯 研究动机

### 现有问题
1. **数据规模小**: 之前最大的 VidChapters-7M 只用了约 2万 样本训练
2. **标注太粗**: 只有简短标题，没有详细描述
3. **评估指标不合理**: SODA 的一对一匹配不适合章节任务

### 解决方案
- 构建 **VidAtlas**: 41万+ 视频，115k 小时内容
- **层级标注**: 短标题 + 摘要 + 详细描述
- 提出 **GRACE**: 多对一匹配，更灵活

---

## 📊 主要结果

### VidChapters-7M Test Set

| Method | F1 | SODA | CIDEr |
|--------|-----|------|-------|
| GPT-4o | 37.6 | 8.1 | 51.0 |
| Gemini-1.5-Pro | 42.2 | 11.4 | 63.2 |
| Chapter-Llama | 45.3 | 19.3 | 100.9 |
| **ARC-Chapter** | **59.3** | **30.6** | **186.6** |

**提升**: +14.0% F1, +11.3% SODA, +85% CIDEr

---

## 🏗️ 方法

### 1. 模型架构

```
输入:
├── Prompt (任务指令)
├── Video Frames (最多 768 帧 @ 1fps)
└── ASR Transcript (带时间戳)

模型: Qwen2.5-VL-7B (冻结 vision encoder, 微调 LLM)

输出 (三种格式):
├── Short Title: 简短章节标题
├── Structural Chapter: 标题 + 摘要 + 介绍
└── Video Description: 带时间戳的完整描述
```

### 2. 数据集 VidAtlas

| 统计 | 数值 |
|------|------|
| 视频数量 | 410k+ |
| 总时长 | 115k 小时 |
| 平均视频长度 | 16.8 分钟 |
| 平均章节数 | 5.5 个/视频 |
| 平均章节时长 | 182 秒 |
| 语言 | 中文 + 英文 |

**数据来源**: 带有用户标注章节的视频平台内容

**标注流程**:
```
1. Whisper-v3 提取 ASR
2. Qwen2.5-VL-7B 提取视觉描述 + OCR
3. LLM 生成层级标注
4. 验证时间戳一致性
```

### 3. GRACE 评估指标

**问题**: SODA 用一对一匹配，但章节标注有粒度差异
- 有人标粗粒度 (按天)
- 有人标细粒度 (按景点)

**解决**: GRACE 用多对一匹配

```
GRACE = Σ φ(Pi, Gi) · BERTScore(Pi, Gi)

其中:
- φ = 时间重叠 IoU
- Pi, Gi = 匹配的章节组
- 用 DTW 动态规划找最优匹配
```

### 4. GRPO 强化学习

- 目的: 直接优化时间戳准确性
- 奖励函数: 只用 GRACE 的时间部分 (去掉 BERTScore)
- 训练数据: 9万视频子集
- KL 系数: 0.01

---

## 🔬 消融实验

### Scaling Law

| 训练数据量 | F1 | SODA |
|-----------|-----|------|
| 20k | 45.3 | 19.3 |
| 100k | 52.1 | 25.4 |
| 400k | **59.3** | **30.6** |

**发现**: 性能随数据量持续提升，没有饱和！

### 模态消融

| 输入 | F1 | SODA |
|------|-----|------|
| ASR only | 54.0 | 25.3 |
| Video only | 51.6 | 22.9 |
| **ASR + Video** | **59.3** | **30.6** |

---

## 💡 对 Apple Assignment 的启示

### Q1: 评估维度
论文提出的 GRACE 考虑:
- **时间准确性** (IoU)
- **语义相似度** (BERTScore)
- **粒度灵活性** (多对一匹配)

### Q2: 评估指标
推荐组合:
- F1 (分割准确度)
- SODA (整体质量)
- **GRACE** (更合理的匹配)
- CIDEr (标题质量)

### Q3: 人工审核
论文的标注流程可参考:
1. 用户原始标注作为 ground truth
2. LLM 生成详细标注
3. 验证时间戳一致性

### Q5: LLM 错误
论文隐含的错误类型:
- 时间戳偏移
- 粒度不匹配
- 多模态信息利用不足

---

## 📚 相关论文

| 论文 | 关系 |
|------|------|
| VidChapters-7M (NeurIPS 2023) | Benchmark |
| Chapter-Llama (CVPR 2025) | 之前 SOTA |
| Vid2Seq | Dense captioning baseline |
| Qwen2.5-VL | Base model |

---

## 🔗 资源

- Paper: https://arxiv.org/abs/2511.14349
- Project: https://arcchapter.github.io/
- Code: https://github.com/TencentARC/ARC-Chapter

---

*Reading Note by 1号机 | 2026-02-06*
