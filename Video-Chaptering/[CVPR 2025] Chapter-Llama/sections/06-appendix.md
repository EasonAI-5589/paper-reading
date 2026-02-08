[← 返回 README](../README.md)

# Appendix 概要

## 📌 预览
Appendix 包含大量实现细节和补充实验。这里只提取关键信息，详见 full.md 原文。

---

## A. Implementation Details

> 💡 **关键超参数**:
> - LoRA: rank=8, α=32, dropout=0.04, 目标模块 Q/V projections
> - Batch size=1, lr=1e-4, 1 epoch, AdamW
> - 训练: 40min on 4×H100; 推理: 100 短视频 30min on 同硬件

**Prompt 模板**:
```
Given the complete transcript of a video of duration {duration},
{task}. Identify the approximate start time of each chapter in the
format 'hh:mm:ss - Title'.
Ensure each chapter entry is on a new line.
Focus on significant topic changes that would merit a new chapter
in a video, but do not provide summaries of the chapters.
{transcript}
```

**输出格式**:
```
00:00:00 - We're at Buckhorn Wash, Utah
00:00:51 - Morrison Knudson (MK) Tunnels
00:01:25 - In Buckhorn Wash, Like a Little Zion
...
```

**Iterative prediction**: 滑动窗口（如 20k tokens），顺序处理，合并预测。

---

## B. Data Analysis and Statistics

> 💡 **数据分析要点**:
> - 58.4% 短视频 (<15min), 21.9% 中等 (15-30min), 11.4% 长 (30-60min), 8.3% 超长 (>1hr)
> - 平均章节数随时长增长，但在 ~60min 处**饱和于 ~13 章节**（标注者心理上限）
> - 训练子集的 category 分布与全量数据接近（均匀采样）
> - 所有 short/medium 视频 <15k tokens; 79% long 视频也满足

---

## C. Additional Quantitative Results（精选）

### C.1 预测时间戳同时输出标题有助于分段质量
- 只预测时间戳: 42.0 F1; 同时预测标题: 42.6 F1 (+0.6)

### C.2 ASR 时间戳表示
- Speech-only: 加 end timestamp 有帮助 (+2.9 F1)
- Speech+Caption: 只用 start timestamp 更好（避免模态间不一致）

### C.3 模态前缀
- 加 "ASR:"/"Caption:" 前缀: 42.6 vs 41.9 F1

### C.7 Llama 变体
| 模型 | F1 (Speech) | F1 (Speech+Caption) |
|------|:-----------:|:-------------------:|
| Llama-3.2-1B | 23.5 | 24.6 |
| Llama-3.2-3B | 35.2 | 34.7 |
| Llama-3.1-8B | 38.5 | **42.6** |
| Llama-3.2-11B | 39.8 | n/a |

### C.8 LoRA rank
- 1k 视频: rank=8 > rank=16 (42.6 vs 39.9)
- 10k 视频: 两者接近 (46.7 vs 46.6)

### C.13 重复标题分析
- GT: 99.6% unique; Chapter-Llama: 96.3%; Vid2Seq: **63.5%**

---

## D. Additional Qualitative Analyses

![Figure A.4](../images/591c8164e682ed79fae9a373b13d94f897c85f7ae1523d6cf8c61e343560cd35.jpg)
*Figure A.4. Segmentation metrics visualization: tIoU and F1 scores calculation examples.*

> 💡 **Figure A.4 批读**: 直观展示了 tIoU 和 F1 的计算过程——先贪心匹配，再计算 IoU 均值/多阈值 F1。

![Figure A.5](../images/6a0b9bf2c8b4df479777c70ad368939888ac24fb64d7f9afa8ba131f741e3ead.jpg)
*Figure A.5. Visualizing captions: Caption 采样和 Chapter-Llama 预测的完整流程可视化。*

> 💡 **Figure A.5 批读**: 展示了 frame selector 预测的边界、采样的 caption、以及最终 Chapter-Llama 的优化预测。可以看到最初 02:00 的冗余章节被最终模型抑制了。

---

## 🔖 Appendix 总结

### 关键 takeaway
1. **Prompt 设计简洁**: ~90 tokens 的固定指令，不依赖复杂 prompt engineering
2. **数据饱和**: 章节数在 ~60min 处饱和于 ~13，反映了人工标注的实际限制
3. **模型规模效应**: 1B→8B 有显著提升，8B→11B 收益递减
4. **Vid2Seq 重复标题严重**: 63.5% unique vs Chapter-Llama 的 96.3%
