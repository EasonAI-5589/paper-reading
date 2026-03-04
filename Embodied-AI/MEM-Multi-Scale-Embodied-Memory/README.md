# MEM: Multi-Scale Embodied Memory for Vision Language Action Models

📅 **Read Date**: 2026-03-04
📂 **Category**: Embodied AI / VLA / Robot Memory
🏢 **Institution**: Physical Intelligence, Stanford, UC Berkeley, MIT
🔗 **PDF**: https://www.pi.website/download/Mem.pdf
🌐 **Project Page**: https://www.pi.website/research/memory
📰 **Status**: 非 arXiv，官网直发（2026.03.03）

---

## 一句话总结

> 给 VLA 装上**多尺度记忆**（短时视频 + 长时语言），让机器人能完成长达 15 分钟的复杂任务（清理厨房、做饭等），同时不牺牲推理速度。

---

## 核心贡献

1. **Multi-Scale Memory 架构**：首次提出将 VLA 的记忆分为短时视频记忆（秒级，处理遮挡/动态）和长时语言记忆（分钟级，追踪任务进度），两者互补
2. **零额外参数的 Video Encoder**：通过修改 ViT 的注意力模式（space-time separable attention）实现视频编码，不引入新参数，可直接从预训练 VLM 初始化
3. **自更新的 Language Memory**：模型自主决定"记什么、忘什么"，通过 LLM 自动生成训练标注，压缩记忆减少 distribution shift
4. **In-context Adaptation**：记忆带来了"失败后切换策略"的 emergent 能力——这在之前的 VLA 中没有观察到
5. **加记忆不掉性能**：MEM 在不需要记忆的精细操作任务上也能 match 无记忆的 SOTA VLA（避免了 causal confusion）

---

## 方法概述

```
┌─────────────────────────────────────────────────┐
│                    MEM System                     │
│                                                   │
│  ┌──────────────┐        ┌──────────────────┐    │
│  │  High-Level   │        │   Low-Level       │    │
│  │  Policy π_HL  │──l_t──▶│   Policy π_LL     │    │
│  │               │        │                   │    │
│  │ Input:        │        │ Input:            │    │
│  │ - o_t (当前)  │        │ - o_{t-K:t} (短时)│    │
│  │ - m_t (记忆)  │        │ - l_t (子任务)    │    │
│  │ - g (目标)    │        │ - g (目标)        │    │
│  │               │        │                   │    │
│  │ Output:       │        │ Output:           │    │
│  │ - l_{t+1}     │        │ - a_{t:t+H}      │    │
│  │ - m_{t+1}     │        │   (action chunk)  │    │
│  └──────────────┘        └──────────────────┘    │
│                                                   │
│  Language Memory (长时)    Video Encoder (短时)    │
│  - 自然语言总结             - Space-time attention │
│  - LLM 自动标注             - 零额外参数          │
│  - 智能压缩                 - 从 VLM ViT 初始化   │
│  - ≤15 min                  - ≤54 sec             │
└─────────────────────────────────────────────────┘
```

---

## 关键实验结果

| 能力 | 无记忆 π₀.₆ | Pool Memory | Proprio Memory | MEM |
|------|-------------|-------------|----------------|-----|
| 长 horizon 任务 | ❌ 很差 | — | — | ✅ 显著提升 |
| Partial observability | ~25% (猜) | 部分有效 | ❌ | ✅ |
| Counting | ~50% (猜) | ✅ 简单任务 | ❌ | ✅ |
| Spatial memory | ❌ | ❌ 长时差 | ❌ | ✅ |
| In-context adaptation | ❌ 反复犯错 | — | — | ✅ 失败后换策略 |
| 精细操作（无需记忆） | ✅ SOTA | — | — | ✅ 持平 |

---

## 技术细节

- **Base Model**: Gemma 3-4B (VLM) + 860M Flow-matching action expert
- **Video Encoder**: SigLIP ViT + space-time separable attention (每 4 层加 temporal attention)
- **Pre-training**: 6 帧 (5s)，混合 robot demo + policy rollout + human correction + video captioning
- **Post-training**: 扩展到 18 帧 (54s)
- **Language Memory**: LLM 自动标注 + 压缩
- **推理**: RTC (Real-Time Chunking)，H100 GPU
- **Action**: FAST token + flow-matching

---

## Section 导航

| 文件 | 内容 |
|------|------|
| [00-Abstract.md](notes/00-Abstract.md) | 摘要 + 作者信息 |
| [01-Introduction.md](notes/01-Introduction.md) | 问题定义与动机 |
| [02-Related-Work.md](notes/02-Related-Work.md) | 相关工作分析 |
| [03-Method.md](notes/03-Method.md) | MEM 架构详解 (核心) |
| [04-Experiments.md](notes/04-Experiments.md) | 实验结果与分析 |
| [05-Conclusion.md](notes/05-Conclusion.md) | 总结与未来方向 |

---

## Citation Landscape

| 论文 | 关系 |
|------|------|
| π₀ (Black et al., 2024) | MEM 的 base VLA |
| π₀.₅ (Physical Intelligence, 2025) | π₀.₆ 的前身 |
| Gemma 3 (Kamath et al., 2025) | VLM backbone |
| TimeSformer (Bertasius et al., 2021) | Space-time separable attention 的灵感来源 |
| ContextVLA (Jang et al., 2025) | Pool Memory baseline |
| TracVLA (Zheng et al., 2024) | 2D point track + causal confusion |
| MemoryVLA (Shi et al., 2025) | 同期工作，latent memory architecture |
| OneTwoVLA (Lin et al., 2025) | 语言记忆的先驱 |
| FAST (Pertsch et al., 2025) | Action tokenization |
| ViViT (Arnab et al., 2021) | Video ViT 先驱 |

---

## 批判性评价

**优点**：
- 多模态记忆设计非常合理，抓住了短/长时记忆的本质区别
- 工程实现优雅：video encoder 零额外参数 + 完美的 backward compatibility
- 实验全面：ablation 充分，每个组件的贡献都有清晰验证
- 加记忆不掉性能，解决了 causal confusion 这个老大难问题

**不足/存疑**：
- Language memory 的信息瓶颈：文本无法描述精确空间信息
- 评估场景偏厨房，泛化性待验证
- 计算门槛高（H100 + π₀.₆），普通实验室难以复现
- 没有与 MemoryVLA 等同期工作的直接对比
- LLM 标注的记忆质量上限问题

---

*批读 by openclaw-read 📚 | 2026-03-04*
