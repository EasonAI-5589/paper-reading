# Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs

> **CVPR 2025**

📄 **Paper**: [arxiv:2504.00072](https://arxiv.org/abs/2504.00072)  
🌐 **Project**: https://imagine.enpc.fr/~lucas.ventura/chapter-llama/  
💻 **Code**: https://github.com/antoyang/chapter-llama

---

## 📌 核心贡献

1. **纯文本域方法**: 把视频转换成文本 (ASR + Frame Captions)，利用 LLM 长上下文能力
2. **Speech-guided Frame Selection**: 用 ASR 内容指导关键帧采样，避免穷举所有帧
3. **单次处理 1 小时视频**: 高效扩展到小时级长视频
4. **大幅超越 SOTA**: F1 45.3 vs 26.7 (VidChapters-7M)

---

## 🏗️ 方法架构

```
Video + Audio
    ↓
┌───────────────────────────────────┐
│  1. Speech-based Frame Selection  │
│     - ASR 提取语音转录             │
│     - 训练 speech-only LLM        │
│     - 在预测边界处采样关键帧        │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  2. Frame Captioning              │
│     - MiniCPM-V 生成图像描述       │
│     - 仅对选中的关键帧处理          │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  3. LLM Chaptering               │
│     - Llama-3.1-8B + LoRA        │
│     - 输入: ASR + Captions + 时间戳 │
│     - 输出: 章节边界 + 标题        │
└───────────────────────────────────┘
```

---

## 📊 性能对比 (VidChapters-7M)

| 方法 | F1 ↑ | tIoU ↑ | SODA ↑ | CIDEr ↑ |
|------|------|--------|--------|---------|
| Vid2Seq (baseline) | 26.7 | 58.6 | 11.6 | 55.8 |
| **Chapter-Llama** | **45.3** | **71.8** | **19.3** | **100.9** |
| 提升 | +70% | +23% | +66% | +81% |

### 按视频时长

| 时长 | Vid2Seq F1 | Chapter-Llama F1 |
|------|------------|------------------|
| Short (0-15min) | 33.4 | 45.5 |
| Medium (15-30min) | 19.0 | 46.7 |
| Long (30-60min) | 16.7 | 41.3 |

> 💡 **关键发现**: 长视频提升更大！

---

## 🔑 关键技术

### 1. Speech-based Frame Selection
- 问题：1 FPS 采样 → 3,960 tokens/min (太长)
- 方案：先用 speech-only LLM 预测边界，只在边界处采样帧
- 效果：大幅减少需要 caption 的帧数

### 2. 多模态融合
```
ASR:     "[00:05:23] Today we'll learn about machine learning..."
Caption: "[00:05:23] A person standing in front of a whiteboard..."
```
- 时间戳格式: `HH:MM:SS`
- 按时间交织排列 ASR 和 Caption

### 3. 迭代预测 (长视频)
- 训练时: 限制 15k tokens (~50 min)
- 推理时: 分块处理，合并预测
- 每块 ~25k tokens (~80 min)

---

## 🧪 消融实验

### 模态贡献

| ASR | Captions | F1 |
|-----|----------|-----|
| ✗ | ✓ | 39.1 |
| ✓ | ✗ | 38.5 |
| ✓ | ✓ | **42.6** |

> 两种模态互补，缺一不可

### 帧选择策略

| 策略 | F1 |
|------|-----|
| 等距采样 | 38.7 |
| Random | 38.8 |
| Speech-guided | **42.6** |

---

## 📝 与其他方法对比

| 特点 | Vid2Seq | Chapter-Llama |
|------|---------|---------------|
| 输入 | 100 帧 (固定) | Speech-guided 采样 |
| 模型 | T5 + CLIP | Llama-3.1-8B |
| 视频长度 | 受限 | 小时级 |
| 训练数据 | HowTo100M + VidChapters | 20k videos (2.5%) |

---

## 📂 文件结构

```
[CVPR 2025] Chapter-Llama/
├── README.md           # 本文件
├── full.md             # MinerU 解析的完整论文
├── paper.pdf           # 原始 PDF
├── content_list.json   # 结构化内容
├── layout.json         # 版面分析
└── images/             # 论文图片
```

---

## 📝 引用

```bibtex
@inproceedings{ventura2025chapterllama,
  title={Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs},
  author={Ventura, Lucas and Yang, Antoine and Schmid, Cordelia and Varol, G{\"u}l},
  booktitle={CVPR},
  year={2025}
}
```

---

*解析时间: 2026-02-07*
