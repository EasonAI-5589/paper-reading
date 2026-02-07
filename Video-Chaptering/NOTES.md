# Video Chaptering 论文笔记

## 论文列表

| 论文 | 会议 | 贡献 |
|------|------|------|
| SODA | ECCV 2020 | **评测框架** - 提出考虑故事顺序的评测指标 |
| VidChapters-7M | NeurIPS 2023 | **大规模数据集** - 817K视频/7M章节 |
| Chapter-Llama | CVPR 2025 | 待阅读 |

---

## SODA: Story Oriented Dense Video Captioning Evaluation Framework

**ECCV 2020 | Tokyo Tech + NTT**

### 核心问题

现有 Dense Video Captioning 评测框架（ActivityNet Challenge 官方评测）存在两个问题：

1. **Loose Matching**: 一个生成 caption 可以匹配多个 reference，忽略了故事顺序
2. **Averaging METEOR**: 只按匹配对数量平均，不惩罚冗余 caption（生成几百个 caption 反而得高分）

### SODA 解决方案

#### 1. 最优匹配 (Dynamic Programming)

用 DP 找到生成 caption 和 reference caption 之间的**时序最优一对一匹配**：

```
S[i][j] = max{
    S[i-1][j],           // 跳过第i个生成caption
    S[i-1][j-1] + C_i,j, // 匹配 (p_i, g_j)
    S[i][j-1]            // 跳过第j个reference
}
```

这样就能找到**保持时序顺序**的最佳匹配，避免乱序。

#### 2. F-measure 惩罚冗余

用 Precision/Recall/F1 代替简单平均：

- **Precision** = Σ METEOR / |P| → 惩罚过多 caption
- **Recall** = Σ METEOR / |G| → 惩罚过少 caption
- **F-measure** = 2 × P × R / (P + R)

#### 3. IoU 加权 (SODA_c)

改进 cost 函数：`C_i,j = IoU(g_i, p_j) × METEOR(g_i, p_j)`

IoU 低的匹配即使 METEOR 高也会被降权。

### 实验结论

| 指标 | m=0.1 | m=1.0 | m=10 | All |
|------|-------|-------|------|-----|
| **Current** | 3.78 | 4.10 | 4.18 | 4.19 |
| **SODA (c)** | 1.47 | **4.02** | 1.70 | 0.63 |

- 旧评测：生成多少 caption 分数都差不多
- SODA：只有 m=1.0（数量匹配）时分数最高

---

## VidChapters-7M: Video Chapters at Scale

**NeurIPS 2023 | Inria + Oxford + CTU Prague**

### 数据集

- **817K 视频** + **7M 章节**
- 平均视频 23 分钟，8.3 个章节
- 从 YouTube 用户标注自动爬取（无人工标注成本）
- 97.3% 视频有 ASR

### 三个任务

| 任务 | 输入 | 输出 |
|------|------|------|
| **Video Chapter Generation** | 完整视频 | 边界 + 标题 |
| **Generation (GT boundaries)** | 视频片段 | 标题 |
| **Video Chapter Grounding** | 视频 + 标题 | 边界 |

### 关键发现

1. **多模态重要**: Speech + Visual 效果最好（ASR 单独比 Visual 单独好）
2. **预训练有效**: VidChapters-7M 预训练后在 YouCook2/ViTT 上达到 SOTA
3. **规模 matters**: 数据集越大，下游性能越好
4. **Chapter ≠ ASR**: 章节标题（5.4词）比 ASR（11.5词）更简洁高层

### 评测

使用 **SODA_c** 作为主要评测指标（验证了 SODA 论文的贡献）

Vid2Seq (Speech+Visual, C4+HowTo100M 预训练) 达到最佳效果。

---

## 两篇论文的关系

```
SODA (ECCV 2020)
    ↓ 提出评测框架
VidChapters-7M (NeurIPS 2023)
    ↓ 采用 SODA_c 作为评测指标
    ↓ 提供大规模数据集
Chapter-Llama (CVPR 2025)
    ↓ 待阅读
```

---

*Updated: 2026-02-07*
