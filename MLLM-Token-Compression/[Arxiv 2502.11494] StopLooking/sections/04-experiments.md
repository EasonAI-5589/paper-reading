# 4. Experiments

## 实验设置
- 4 个 MLLM: LLaVA-1.5-7B, LLaVA-Next-7B, Qwen2-VL-7B, MiniCPM-V2.6
- 10 个 image benchmarks: GQA, MMB, MMB-CN, MME, POPE, SQA, VQA-V2, VQA-Text, VizWiz, OCRBench
- 4 个 video benchmarks: TGIF, MSVD, MSRVT
- Default: pruning after layer 2, 8 pivot tokens, K-norm selection

## 4.1 Main Results

### Image Understanding (Table 1 — LLaVA-1.5-7B)

**66.7% reduction (retain 192/576)**:
- DART: 98.8% avg → beat MustDrop (97.2%) by **1.6%**
- FastV only 91.2%, ToMe 88.5%

**77.8% reduction (retain 128/576)**:
- DART: 98.0% → beat MustDrop (95.6%) by **2.4%**

**88.9% reduction (retain 64/576)**:
- DART: **93.7%** → beat FiCoCo-V (91.5%) by **2.2%**
- FastV collapsed to 77.3%, PDrop to 78.1%
- 特别注意 MME: DART 1765 vs FastV 1256，差距巨大

### LLaVA-Next-7B (88.9% reduction)
- DART: **93.9%** → beat all methods by 3.5%+ margin
- 在 VQA-V2 上 DART 79.1 vs next best HiRED 75.7

### Qwen2-VL-7B 和 MiniCPM-V2.6
- DART consistently outperforms FastV by 3-8% across all compression ratios
- MiniCPM-V2.6 在极端压缩下 (88.9%) 退化更严重：DART 76.1% vs FastV 68.4%

### Video Understanding (Table 5)
- 50% token retention on Video-LLaVA
- DART 58.0% avg accuracy vs FastV 57.1%
- 提升相对 modest，video 场景下 temporal 信息可能更关键

> 💡 **最震撼的结果是 88.9% compression 下的对比**。PDrop 从 96.7% (66.7% compression) 暴跌到 78.1% (88.9%)，说明 importance-based progressive pruning 在极端压缩下彻底崩溃。DART 只从 98.8% 降到 93.7%，graceful degradation 非常好。

> 💡 Qwen2-VL 和 MiniCPM 的结果说明 DART 的优势在不同架构上都成立，但绝对性能差异与模型本身的 visual token 冗余度有关。MiniCPM 可能本身对 visual tokens 利用更充分，所以裁剪伤害更大。

## Efficiency Analysis (Table 2 — LLaVA-Next-7B)

| Method | Tokens | Total Time | Prefill Time | Speedup (Total/Prefill) | POPE |
|---|---|---|---|---|---|
| Vanilla | 2880 | 36:16 | 22:51 | 1.00× / 1.00× | 86.5 |
| FastV | 320 | 18:17 | 7:41 | 1.98× / 2.97× | 78.3 |
| SparseVLM | 320 | 23:11 | - | 1.56× / - | 82.3 |
| **DART** | 320 | **18:13** | **7:38** | **1.99× / 2.99×** | **84.1** |

> 💡 DART 和 FastV 的 speedup 几乎相同（因为最终 token 数一样），但 DART 在 POPE 上高出 5.8 分。SparseVLM 虽然 FLOPs 只多 2.8%，但实际 speedup 低 21.6%——这就是不兼容 FlashAttention 的代价。

> 💡 DART 的 token reduction overhead < 0.08s，这是因为只需计算 8 个 pivot tokens 与所有 visual tokens 的 cosine similarity，计算量极小。
