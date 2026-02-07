# 4. Experiments

> 来源: SparseVLM (ICML 2025)

---

## 📄 原文

> 💡 **Section 概览**: 图像理解 + 视频理解两大任务，在 LLaVA / MGM / Qwen2-VL / VideoLLaVA 上验证

---

### 4.1 Image Understanding Tasks

**Benchmarks**: GQA, MMBench, MME, POPE, SQA, SEED, TextVQA, MMVet

**VLMs**: LLaVA-1.5 (7B), Mini-Gemini, Qwen2-VL

#### Table 1: SparseLLaVA 主结果

> 💡 **Table 1 批读**:
> ```
> 576 → 192 tokens (66.7% 压缩):
> ├── SparseVLM: 99.1% 准确率 ⭐ (只掉 0.9%)
> ├── PDrop:     95.9%
> ├── ToMe:      88.9%
> └── FastV:     87.9%
>
> 576 → 128 tokens (77.8% 压缩):
> ├── SparseVLM: 96.7% ⭐
> ├── PDrop:     94.3%
> ├── FastV:     82.4%
> └── ToMe:      81.9%
>
> 576 → 64 tokens (88.9% 压缩):
> ├── SparseVLM: 89.3% ⭐ (+17.3% vs FastV!)
> ├── PDrop:     73.4%
> ├── FastV:     72.0%
> └── ToMe:      71.1%
> ```
>
> **关键发现**:
> 1. 压缩越狠，SparseVLM 优势越明显（从 +11.2% 到 +17.3%）
> 2. PDrop 虽然 FLOPs 低，但 latency 反而更高（43.41ms vs 29.89ms）
> 3. SparseVLM 在 192 tokens 时几乎无损（99.1%）

#### Table 2: Qwen2-VL 结果

| Tokens | MMB | POPE | TextVQA | Avg. |
|--------|-----|------|---------|------|
| Dynamic (~1320) | 80.5 | 86.4 | 84.3 | 83.7 |
| 600 | 79.6 | 86.5 | 80.3 | 82.1 |
| 500 | 78.8 | 86.3 | 79.0 | 81.4 |
| 400 | 79.0 | 85.8 | 77.1 | 80.7 |

> 💡 **批注**: 去掉 54.5% tokens 后准确率保持 98.0%。说明方法对动态分辨率模型也有效。

---

### 4.2 Video Understanding Tasks

**Benchmarks**: TGIF-QA, MSVD-QA, MSRVTT-QA, ActivityNet-QA

**VLM**: VideoLLaVA (2048 video tokens → 194 tokens, 90.5% 剪枝)

#### Table 3: VideoLLaVA 结果

> 💡 **Table 3 批读**:
> ```
> 2048 → 194 tokens (90.5% 压缩):
>
> SparseVLM:  95.0% 平均准确率 ⭐
> FastV:      80.3%
> 差距:       +14.7%!
>
> GPT 评分对比:
> SparseVLM:  -0.04 分
> FastV:      -0.17 分
>
> 各 benchmark:
>   TGIF:        78.8% vs 54.0% (FastV)
>   MSVD:        99.6% vs 81.0%
>   MSRVTT:      98.3% vs 91.6%
>   ActivityNet: 103.4% vs 94.7% (甚至超过了原始模型!)
> ```
>
> **核心发现**: 视频任务差距更大！因为视频帧间冗余更高，text-guided 方法能更精准地保留有用帧的关键 token。

---

## 💡 Section 总结

### 关键数字速查
| 场景 | 压缩率 | SparseVLM | FastV | 差距 |
|------|--------|-----------|-------|------|
| LLaVA 192 tokens | 66.7% | 99.1% | 87.9% | +11.2% |
| LLaVA 128 tokens | 77.8% | 96.7% | 82.4% | +14.3% |
| LLaVA 64 tokens | 88.9% | 89.3% | 72.0% | +17.3% |
| VideoLLaVA 194 tokens | 90.5% | 95.0% | 80.3% | +14.7% |

### 核心洞察
1. **压缩越狠，优势越大** — text guidance 在极端压缩下尤为关键
2. **视频场景更出色** — temporal 冗余被有效处理
3. **泛化性好** — 在 LLaVA, MGM, Qwen2-VL, VideoLLaVA 上都有效
