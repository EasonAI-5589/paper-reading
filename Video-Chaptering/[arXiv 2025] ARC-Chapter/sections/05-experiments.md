# 5. Experiments

> 来源: ARC-Chapter (arXiv 2025)

---

## 📄 原文

### 5.1 Evaluation Benchmark

> 💡 **5.1 要点预览**: 在哪些数据集上评测？

To comprehensively assess our model's capabilities in video chaptering, we evaluate it on **three distinct benchmarks** covering different languages, scales, and data modalities.

> 💡 **三个评测数据集**:
> | Benchmark | 语言 | 规模 | 模态 | 用途 |
> |-----------|------|------|------|------|
> | VidChapters7M-test | 英文 | 8.2K | ASR-only | 主要对比 |
> | VidChapters7M-sml300 | 英文 | 300 | Video+ASR | 消融实验 |
> | VidAtlas-test | 中文 | 1.5K | Video+ASR | 中文泛化 |

---

### 5.2 Comparison with the State of the Art

> 💡 **5.2 要点预览**: 和之前的 SOTA 比，提升多少？

**Table 1: VidChapters7M-test 结果 (ASR-only)**

| Method | F1 | SODA | CIDEr |
|--------|-----|------|-------|
| GPT-4o | 37.6 | 8.1 | 51.0 |
| Gemini-1.5-Pro | 42.2 | 11.4 | 63.2 |
| Vid2Seq | 26.7 | 11.6 | 55.8 |
| Chapter-Llama (前SOTA) | 45.3 | 19.3 | 100.9 |
| **ARC-Chapter-asr** | **54.5** | **25.3** | **144.0** |

> 💡 **Table 1 批读**:
> ```
> 相比 Chapter-Llama (前 SOTA):
> ├── F1:    45.3 → 54.5  (+9.2, +20%)
> ├── SODA:  19.3 → 25.3  (+6.0, +31%)
> └── CIDEr: 100.9 → 144.0 (+43%)
> 
> 相比 GPT-4o (零样本):
> └── 所有指标都大幅领先 (GPT-4o 的 SODA 才 8.1)
> 
> 关键发现: 专门训练的小模型 >> 通用大模型 (零样本)
> ```

**多模态对比:**

| 模态 | F1 | SODA | CIDEr |
|------|-----|------|-------|
| ARC-Chapter-asr (仅ASR) | 54.5 | 25.3 | 144.0 |
| ARC-Chapter-vid (仅视频) | 50.2 | 22.9 | 138.3 |
| **ARC-Chapter-vidasr (两者)** | **59.3** | **30.6** | **186.6** |

> 💡 **多模态效果**:
> ```
> 仅 ASR: SODA 25.3
> 仅视频: SODA 22.9
> 两者融合: SODA 30.6 
> 
> 融合比单模态高 5-8 分
> → 多模态融合很重要！
> ```

---

### 5.3 Transferability

> 💡 **5.3 要点预览**: 在 VidAtlas 上预训练，能迁移到其他任务吗？

We further evaluate the transferability by finetuning ARC-Chapter on downstream dense video captioning datasets.

**YouCook2 结果:**

| Method | SODA | CIDEr |
|--------|------|-------|
| Vid2Seq (C4+HTM+VC) | 10.3 | 67.2 |
| **ARC-Chapter** | **12.8** | **89.6** |

> 💡 **迁移学习效果**:
> - YouCook2 SODA: 10.3 → 12.8 (+24%)
> - YouCook2 CIDEr: 67.2 → 89.6 (+33%)
> 
> → VidAtlas 预训练对下游任务也有帮助

---

### 5.4 Ablation Studies

> 💡 **5.4 要点预览**: 哪些设计决策是关键的？

#### 5.4.1 Scaling Property (数据规模)

![Figure 6a](../images/89880f9c26de69cb32ca9e476ea6b2aa86964afd52693bb66a0d4d4483fd233f.jpg)
![Figure 6b](../images/a7a6eae0edb311f90140521d9188198795aa7ab1c96b70462eb029cccc538790.jpg)
![Figure 6c](../images/3d5b90daa5bf2d6ffad6d3759f14f3d6d27ab0c58c729ad0187e930731a55813.jpg)
![Figure 6d](../images/9a39b73ab3b3c1289ce32ec8ef4ef5c6a56ceaa378d57c5f3445425c4e50c986.jpg)
![Figure 6e](../images/22b5cc432ab9decf7b04f4675235449c0d8ead6153b848f921e295f69703c608.jpg)
![Figure 6f](../images/26cc2644aa2e83d5d6e6e02aa43993b5262b0dbae06b4279e6fb3146bebb9206.jpg)
![Figure 6g](../images/bc05303ff15ebaa765967b4583afdc41b2d4124ee0360be0a353b314c0b71f1f.jpg)
![Figure 6h](../images/bed5b1d54b7b30c3aca679f2cb6df8b3b49ac30c8ab48d60241ae096a30300a6.jpg)
*Figure 6: ARC-Chapter 数据 Scaling 特性。在 VidChapter（采样子集）和 VidAtlas 测试集上，随训练样本比例增加，各指标（F1、tIoU、SODA、CIDEr）持续提升。*

| 训练数据量 | F1 | SODA |
|-----------|-----|------|
| 100K | 52.1 | 24.1 |
| 500K | 56.3 | 27.8 |
| 1M | **59.3** | **30.6** |

> 💡 **Scaling Law 验证**:
> ```
> 数据量: 100K → 500K → 1M
> SODA:   24.1 → 27.8 → 30.6
> 
> 持续提升，没有饱和！
> 这推翻了之前"20K样本就饱和"的观点
> ```

#### 5.4.2 Hierarchical Annotations (层级标注)

| 标注类型 | F1 | SODA |
|---------|-----|------|
| Short Titles Only | 55.2 | 26.4 |
| + Structural Chapters | 57.8 | 28.9 |
| + Video Descriptions | **59.3** | **30.6** |

> 💡 **层级标注效果**:
> ```
> 只用短标题: SODA 26.4
> +结构化章节: SODA 28.9 (+2.5)
> +视频描述:   SODA 30.6 (+1.7)
> 
> 每增加一层标注，性能都提升
> → 丰富的层级标注很重要
> ```

#### 5.4.3 Performance with GRPO (强化学习)

| 方法 | F1 | SODA |
|------|-----|------|
| SFT only | 57.8 | 28.5 |
| **SFT + GRPO** | **59.3** | **30.6** |

> 💡 **GRPO 效果**:
> - F1: +1.5
> - SODA: +2.1
> 
> 强化学习提供额外提升

---

### 5.5 Qualitative Visualization

![Figure 7](../images/457f769d7e132046e6fa62de4c1adf38022eb7cd4aa2257c090936363dd12df3.jpg)
![Figure 7 (cont.)](../images/f5330556369a4a4834e2f85e0fae16b253a3fa8d41a249557c60eb85bca21c31.jpg)
*Figure 7: 英文视频（金融/加密货币）上的定性结果。*

![Figure 8](../images/4edeeac21150742bb9508b95043f8c227cd3a3c4130b7d316990b19fb64fe550.jpg)
*Figure 8: 中文视频（稳定币讨论）上的定性结果。*

The model successfully handles complex multi-topic transitions, long-range temporal dependencies, and hierarchical content structures.

> 💡 **定性观察**:
> - 能处理复杂的多话题转换
> - 能捕捉长时间依赖
> - 能生成层级内容结构

---

## 💡 Section 5 总结

### 核心实验结论

| 发现 | 证据 |
|------|------|
| **大幅超越 SOTA** | F1 +14%, SODA +11.3% |
| **多模态融合有效** | Video+ASR 比单模态高 5-8 SODA |
| **Scaling Law 存在** | 100K→1M，性能持续提升 |
| **层级标注有帮助** | 每增加一层都提升性能 |
| **GRPO 有额外收益** | +2.1 SODA |
| **迁移性好** | YouCook2 也达到 SOTA |

### 性能提升来源分解

```
ARC-Chapter 性能提升来源:

Chapter-Llama baseline:    SODA 19.3
├── 更大数据 (VidAtlas):   +4.0  → 23.3
├── 层级标注:              +2.5  → 25.8
├── 多模态 (Video+ASR):    +2.7  → 28.5
└── GRPO 强化学习:         +2.1  → 30.6

总计: 19.3 → 30.6 (+11.3, +58%)
```
