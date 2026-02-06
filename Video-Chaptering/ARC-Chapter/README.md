# ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries

> 📅 **Date**: November 18, 2025  
> 🏢 **Affiliation**: ARC Lab, Tencent PCG  
> 👨‍💻 **Authors**: Junfu Pu*, Teng Wang*, Yixiao Ge†, Yuying Ge, Chen Li, Ying Shan  
> 📎 **arXiv**: https://arxiv.org/abs/2511.14349  
> 🌐 **Project**: https://arcchapter.github.io/  
> 💻 **Code**: https://github.com/TencentARC/ARC-Chapter

---

## 📋 TL;DR

ARC-Chapter 是目前 **最新 SOTA** 的视频章节生成模型，核心贡献：

1. **百万级训练数据** VidAtlas (410k+ 视频, 115k 小时)
2. **层级式标注**: 短标题 → 结构化章节 → 时间戳描述
3. **新评估指标 GRACE**: 多对一匹配，解决标注粒度不一致问题
4. **GRPO 强化学习**: 直接优化时间戳准确性

**性能提升**: VidChapters-7M 上 F1 45.3 → **59.3** (+14%), SODA 19.3 → **30.6** (+58%)

---

## 🎯 研究动机

### 现有问题

1. **数据规模小**: 之前最大的 VidChapters-7M 只用了约 2万样本训练
2. **标注太粗**: 只有简短标题，没有详细描述
3. **评估指标不合理**: SODA 一对一匹配不适合章节任务（标注粒度存在歧义）

### 解决方案

- 构建 **VidAtlas**: 41万+ 视频，115k 小时内容
- **层级标注**: 短标题 + 摘要 + 详细描述
- 提出 **GRACE**: 多对一匹配，更灵活

---

## 📊 主要结果

### VidChapters-7M Test Set (Table 1)

| Method | Finetune | F1 | tIoU | SODA | CIDEr |
|--------|----------|-----|------|------|-------|
| GPT-4o | ✗ | 37.6 | 68.0 | 8.1 | 51.0 |
| Gemini-2.0-Flash | ✗ | 40.2 | 69.3 | 11.4 | 69.7 |
| Gemini-1.5-Pro | ✗ | 42.2 | 70.9 | 11.4 | 63.2 |
| Chapter-Llama | ✓ | 45.3 | 71.8 | 19.3 | 100.9 |
| **ARC-Chapter-asr** | ✓ | 54.5 | 76.7 | 25.3 | 144.0 |
| **ARC-Chapter-vid** | ✓ | 50.2 | 74.3 | 22.9 | 138.3 |
| **ARC-Chapter-vidasr** | ✓ | **59.3** | **79.6** | **30.6** | **186.6** |

**提升**: +14.0% F1, +11.3% SODA, +85% CIDEr

---

## 🏗️ 方法详解

### 1. 整体框架 (Figure 4)

```
输入:
├── Task Prompt (任务指令, 18种模板)
├── Video Frames (最多 768 帧 @ ≤1fps)
└── ASR Transcript (带时间戳, Whisper-v3)

Base Model: Qwen2.5-VL-7B
├── Vision Encoder: 冻结
└── LLM: 微调

输出 (三种格式):
├── Short Title: 简短章节标题
├── Structural Chapter: 标题 + 摘要 + 介绍
└── Video Description: 带时间戳的完整描述
```

### 2. Prompt 设计

18 种模板，基于三个维度：
- **语言**: 英文 / 中文
- **输入模态**: ASR-only / Video-only / ASR+Video
- **输出格式**: Short Title / Structural Chapter / Video Description

### 3. 视频输入处理

- 最多 768 帧 (12.8分钟以下 @1fps，更长视频降采样)
- 动态调整每帧 token 数 (Video-only 用高分辨率，ASR+Video 用低分辨率)
- 随机在帧上叠加时间戳，增强时间感知

### 4. ASR 输入处理

为什么用文本而非音频特征？
- Whisper 产生 50 tokens/秒 → 60分钟 = 180k tokens，超出 context 限制
- ASR 文本信息密度更高，tokens 更少

格式: `start time (hh:mm:ss): <ASR text>`

---

## 📦 数据集: VidAtlas

### 统计信息 (Figure 3)

| 统计项 | 数值 |
|--------|------|
| 视频数量 | 410k+ |
| 总时长 | 115k 小时 |
| 平均视频长度 | 16.8 分钟 |
| 平均章节数 | 5.5 个/视频 |
| 平均章节时长 | 182 秒 (~3分钟) |
| 语言 | 中文 + 英文 |
| 类别 | 16 大类, 100+ 子类 |

### 数据标注流程 (Figure 2)

```
1. 数据来源: 带用户标注章节的视频平台内容
   (用户提供: 时间戳 + 简短标题)

2. 多模态信息提取:
   ├── Whisper-v3: ASR 转录 (带时间戳)
   └── Qwen2.5-VL-7B: 视觉描述 + OCR

3. 时间对齐: 将 ASR 和视觉描述按时间交错

4. LLM 推理: 生成层级标注
   ├── Comprehensive Title
   ├── Abstract
   ├── Introduction
   └── Temporal Boundaries

5. 验证: 确保生成的边界与原始时间戳一致
```

---

## 📏 评估指标: GRACE (Figure 5)

### 为什么需要新指标？

**SODA 的问题**: 一对一匹配
- 不同标注者可能用不同粒度标注同一视频
  - 粗粒度: "第一天" 
  - 细粒度: "参观景点A", "参观景点B"
- 一对一匹配会漏掉有效的章节

### GRACE 设计: 多对一匹配

```
GRACE = Σ φ(Pi, Gi) · BERTScore(Pi, Gi)

其中:
- φ(Pi, Gi) = 平均 IoU (时间重叠)
- Pi, Gi = 匹配的章节组 (可以是多个章节)
- 用 DTW (动态时间规整) 找最优匹配
```

### GRACE 优势

1. **粒度鲁棒**: 不同标注风格都能公平评估
2. **语义保真**: 奖励捕获完整内容的模型
3. **人类对齐**: 更符合人类对章节边界的判断

---

## 🔧 训练策略

### 1. 监督微调 (SFT)

- 训练目标: 标准自回归 next-token prediction
- 数据: VidAtlas + VidChapter-7M
- Vision encoder 冻结, LLM 全参数微调

### 2. Adaptive Modality Dropping

训练时随机选择输入模态:
- Video + ASR
- Video-only
- ASR-only

**好处**: 单一模型可处理各种推理场景

### 3. GRPO 强化学习

**目的**: 直接优化时间戳准确性 (SFT 的交叉熵 loss 无法直接优化)

**奖励函数**: GRACE 的时间部分 (去掉 BERTScore)

```
R = Σ φ(Pi, Gi)
```

**设置**:
- 训练数据: 9万视频子集 (中英双语)
- 仅用 Video 模态
- KL 系数: 0.01 (防止偏离 SFT 能力太远)

---

## 🔬 消融实验

### Scaling Law (Figure 6)

| 训练数据比例 | F1 (VidChapters) | SODA |
|-------------|------------------|------|
| 20% | ~48 | ~20 |
| 40% | ~52 | ~23 |
| 60% | ~55 | ~26 |
| 80% | ~57 | ~28 |
| 100% | **59.3** | **30.6** |

**关键发现**: 性能随数据量持续提升，**没有饱和**！

这推翻了 Chapter-Llama 的观察 (在 ~20k 样本就饱和)

### 模态消融 (Table 2)

| 输入 | F1 | SODA |
|------|-----|------|
| ASR only | 56.5 | 25.9 |
| Video only | 50.0 | 21.6 |
| **ASR + Video** | **62.4** | **30.1** |

**结论**: 多模态互补，ASR + Video 最佳

### GRPO 效果 (Table 6)

| 模型 | F1 | tIoU | CIDEr |
|------|-----|------|-------|
| Base-vidasr (SFT) | 59.3 | 79.6 | 186.6 |
| **GRPO-vidasr (+RL)** | **60.8** (+1.5) | **80.7** (+1.1) | **190.7** (+4.1) |

**关键发现**:
1. GRPO 提升时间准确性 (F1, tIoU)
2. 语义质量 (CIDEr) 不降反升
3. 跨模态迁移：仅用 Video 训练 RL，但 ASR 和 ASR+Video 也提升

---

## 🔄 迁移性 (Table 4)

### Dense Video Captioning

| Method | YouCook2 F1 | YouCook2 SODA | ActivityNet F1 |
|--------|-------------|---------------|----------------|
| Vid2Seq | 27.3 | 7.9 | 52.4 |
| TimeExpert | 33.5 | 7.2 | 40.5 |
| **ARC-Chapter** | **37.9** | **12.5** | **55.9** |

VidAtlas 预训练显著提升下游任务性能

---

## 💡 对 Apple Assignment 的启示

### Q1: 评估维度

论文的三层输出结构对应不同评估维度:
- **Short Title**: 简洁性、信息量
- **Structural Chapter**: 完整性、层次结构
- **Temporal Boundary**: 时间准确性

### Q2: 评估指标

推荐组合 (基于论文):
- **F1**: 分割准确度
- **tIoU**: 时间重叠
- **SODA**: 传统章节+标题联合评估
- **GRACE**: 多对一匹配 (更合理)
- **CIDEr**: 标题语义质量

### Q3: 人工审核

论文的标注流程可参考:
1. 用户原始标注作为 ground truth
2. LLM 生成详细标注
3. **验证步骤**: 确保生成边界与原始时间戳一致

### Q4: 评分标准

层级标注提供多粒度评分:
- Level 1: 只评 Short Title
- Level 2: 评 Title + Abstract
- Level 3: 评完整 Structural Chapter

### Q5: LLM 错误类型

论文隐含的错误:
- **时间戳偏移**: F1/tIoU 下降
- **粒度不匹配**: SODA vs GRACE 差异
- **模态依赖**: Video-only 性能明显低于 ASR+Video
- **幻觉**: Structural Chapter 可能生成不存在的内容

---

## 📚 相关论文

| 论文 | 关系 |
|------|------|
| VidChapters-7M (NeurIPS 2023) | Benchmark |
| Chapter-Llama (CVPR 2025) | 之前 SOTA |
| Vid2Seq (CVPR 2023) | Dense captioning baseline |
| Qwen2.5-VL | Base model |
| GRPO (DeepSeek-R1) | RL 算法 |

---

## 📁 附件

完整论文解析 (含图片) 位于:
```
./2511.14349-ec960086-4710-41e0-98a4-fdb14f73ae01/
├── full.md          # 完整 Markdown
├── images/          # 论文图片
└── *.json           # 解析元数据
```

---

*Reading Note by 1号机 | 2026-02-06*
