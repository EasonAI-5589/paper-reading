# 4. Experiments

> 来源: DivPrune (Arxiv 2503.02175)

---

## 📄 原文

> 💡 **Section 概览**: 实验部分非常扎实，涵盖了 insight 可视化、图像理解、视频理解、效率分析和消融实验。重点关注 DivPrune 在极端压缩比下的优势。

---

### 4.1 Experimental Settings

> 💡 **4.1 要点预览**: 实验设置——用了哪些模型、baseline、数据集。

**Baselines and Models:**
- **Plug-and-play（主要竞争对手）**: FastV, VTW, PruMerge
- **需要校准**: FitPrune
- **需要微调**: M³

测试的 LMMs:
- LLaVA 1.5-7B (576 visual tokens)
- LLaVA 1.5-13B (576 visual tokens)
- LLaVA 1.6-7B / LLaVA-NeXT (3-5× more tokens)
- LLaVA-NeXT-Video-7B (144 tokens × 8 frames = 1152)

**Datasets:** 11 image-language + 5 video-language = 16 datasets total.

> 💡 **批注**: 实验设计很全面。特别好的是把 baseline 分成三类（plug-and-play / calibration / fine-tuning），然后主要和同类方法比，但也报告了跨类比较。

**TFLOP ratio 计算:**

$$\frac{K \times (4\mu d^2 - 2\mu^2 d + 2\mu dm) + (T-K) \times (4\tilde{\mu} d^2 - 2\tilde{\mu}^2 d + 2\tilde{\mu} dm)}{T \times (4\mu d^2 - 2\mu^2 d + 2\mu dm)}$$

> 💡 **批注**: 这个公式计算的是剪枝后 vs 剪枝前的 TFLOP 比值。$K$ 是剪枝前正常处理的层数，$T-K$ 是剪枝后处理的层数。TFLOP ratio 越低说明压缩越狠。

> 💡 **4.1 小结**: 硬件 8×V100 32GB，用 lmms-evals 跑 benchmark，batch size=1。

---

### 4.2 Insights

> 💡 **4.2 要点预览**: 通过 t-SNE 可视化直观展示 DivPrune vs FastV 的区别。

The visual tokens in LLaVa 1.5 model are 4096-dimensional vectors. The t-SNE method is utilized to project the visual tokens from a high dimensional to a 2D space.

![Figure 3](../images/4afc43ceb1e35eb69f2a97bedd3231e0fe4ba5b91ba871f31507c3c10f8e9bd8.jpg)
*Figure 3: (a) t-SNE 可视化：原始 token（浅紫色）、DivPrune 选出的 token（红色）、FastV 选出的 token（蓝色）。(b) 1000 个 SeedBench 样本上的 Max-Min 距离直方图。*

> 💡 **Figure 3 批读**:
> ```
> (a) t-SNE 可视化:
> ┌─────────────────────────┐
> │  ○ ○ ○ ○   ← 上方簇    │ ← FastV 完全没选这里的 token！
> │  ○ ○ ○                  │
> │                          │
> │  ● ● ●    ← 主要簇      │ ← FastV 的 token 挤在一起
> │  ★ ★ ★ ★               │ ← DivPrune 均匀分布
> └─────────────────────────┘
> 
> 关键发现:
> - DivPrune 从所有簇中选取 token → 覆盖全面
> - FastV 遗漏了上方整个簇 → 信息丢失
> - FastV 选的 token 互相靠近 → 冗余高
>
> (b) Max-Min 距离直方图:
> - DivPrune 的 max-min 距离分布明显右移
> - 说明 DivPrune 选出的 token 更分散、更多样
> ```

> 💡 **4.2 小结**: 可视化直观验证了 DivPrune 的核心假设——多样性选择 > 重要性选择。

---

### 4.3 Image-Language Understanding

> 💡 **4.3 要点预览**: 在 11 个图像数据集上的主实验。极端压缩（~15% TFLOP）下 DivPrune 碾压 baseline。

**Table 1: 主实验结果（LLaVA 1.5-7B, 13B, 1.6-7B）**

> 💡 **Table 1 批读（LLaVA 1.5-7B, ~15% TFLOP）**:
> ```
> Plug-and-play 方法对比:
> ┌──────────┬────────┬────────┬────────┬────────┬────────┐
> │ 数据集    │ Original│ FastV  │ VTW    │ DivPrune│ 差距   │
> ├──────────┼────────┼────────┼────────┼────────┼────────┤
> │ COCO     │  1.10  │  0.06  │  0.05  │  0.96  │ 16× ⭐ │
> │ GQA      │ 61.96  │ 38.73  │ 38.94  │ 56.85  │ +18    │
> │ MMB      │ 64.09  │ 20.62  │ 21.31  │ 59.19  │ +38 ⭐ │
> │ POPE     │ 85.84  │ 32.84  │ 25.35  │ 86.02  │ +53 ⭐ │
> │ OKVQA    │ 53.39  │ 18.32  │ 18.64  │ 46.98  │ +28    │
> └──────────┴────────┴────────┴────────┴────────┴────────┘
> 
> 关键发现:
> 1. POPE 上 DivPrune 甚至超过原始模型！(86.02 vs 85.84)
> 2. FastV/VTW 在 captioning 任务上几乎归零（CIDEr 0.05-0.06）
> 3. DivPrune 在 COCO 上只掉 12.7%，而 baseline 掉 95%
> 4. MMMU 和 SQA 掉不到 2%
> ```

> 💡 **Table 1 批读（vs 需要校准/微调的方法）**:
> ```
> DivPrune vs FitPrune (需要校准):
> - DivPrune 在几乎所有数据集上都赢
> - POPE: 86.02 vs 60.89 (+25.1%) ⭐
> - 关键: DivPrune 不需要校准数据就能赢
>
> DivPrune vs M³ (需要微调):
> - M³ 整体更好（毕竟微调过）
> - 但 DivPrune 在 MMMU 和 SQA 上更好
> - 考虑到 DivPrune 零训练成本，性价比极高
> ```

> 💡 **Table 1 批读（LLaVA 1.6-7B, ~11% TFLOP）**:
> ```
> 更极端的压缩比（89% TFLOP reduction）:
> - FastV/VTW: POPE F1 暴跌到 7-8%（几乎不能用）
> - DivPrune: POPE F1 = 82.97%（仅掉 3.4%）
> - MMMU 上 DivPrune 甚至超过原始模型！(37.11 vs 36.44)
>
> 结论: token 数越多的模型，DivPrune 优势越大
> ```

> 💡 **4.3 小结**: DivPrune 在 plug-and-play 赛道上全面碾压，甚至在很多场景下超过需要校准的 FitPrune。

---

### 4.4 Video-Language Understanding

> 💡 **4.4 要点预览**: 在 5 个视频数据集上验证 DivPrune 的泛化能力。

**Table 2: LLaVA-NeXT-Video-7B 实验结果**

| | TFLOP ratio | ActivityNet | SeedBench | VChatGPT | NextQA | EgoSchema |
|---|---|---|---|---|---|---|
| Original | 100% | 48.10 | 38.7 | 2.16 | 26.05 | 41.8 |
| VTW | 17.0% | 26.84 | 29.39 | 1.19 | 18.66 | 25.42 |
| FastV | 14.2% | 33.91 | 32.98 | 1.44 | 22.51 | 29.14 |
| **Ours** | **14.1%** | **45.90** | **37.00** | **1.92** | **24.48** | **39.76** |

> 💡 **Table 2 批读**:
> ```
> 性能保留率（vs 原始模型）:
> ├── DivPrune: ~95% (ActivityNet), ~96% (SeedBench), ~95% (EgoSchema)
> ├── FastV:    ~71%, ~85%, ~70%
> └── VTW:      ~56%, ~76%, ~61%
>
> 关键发现:
> - 视频模型的 token 更多，DivPrune 优势更明显
> - "token 数越多，冗余越多，多样性选择越有效"
> ```

> 💡 **4.4 小结**: DivPrune 完美泛化到视频 LMM，token 数越多优势越大。

---

### 4.5 Efficiency Analysis

> 💡 **4.5 要点预览**: 实际的延迟和显存改善。

| | Max GPU mem | Prefill Time | E2E Latency |
|---|---|---|---|
| Original | 14.06 GB | 0.330s | 4.37s |
| FastV | 13.57 GB | 0.150s | 3.63s |
| VTW | 13.63 GB | 0.150s | 3.43s |
| **Ours** | **13.51 GB** | 0.161s | **3.39s** |

> 💡 **批读**:
> ```
> 效率对比:
> - 显存: 省 ~0.5GB（和 baseline 差不多）
> - Prefill: 比 baseline 慢 6-7%（因为要算距离矩阵）
> - E2E: 反而最快！（因为 baseline 每步都要做 token 选择）
>
> 关键: DivPrune 的距离计算只做一次（prefill），
>       而 FastV 每个 decoding step 都要看 attention → E2E 更慢
> ```

---

### 4.6 Ablation Study

> 💡 **4.6 要点预览**: 两个消融：(1) 在哪一层剪枝 (2) 用什么距离度量。

**Table 3: 不同层剪枝效果**

| 层 | MMB | MMMU | POPE | SQA | Avg |
|---|---|---|---|---|---|
| Layer 0 (默认) | 59.19 | 35.89 | 86.02 | 68.27 | **62.34** |
| Layer 1 | 59.02 | 34.89 | 80.67 | 67.18 | 60.44 |
| Layer 2 | 54.90 | 34.22 | 69.27 | 69.56 | 56.99 |
| Layer 3 | 23.97 | 32.67 | 31.82 | 65.94 | 38.60 |

> 💡 **Table 3 批读**:
> ```
> 结论: 越早剪越好！
> Layer 0 >> Layer 1 > Layer 2 >> Layer 3
>
> 原因推测:
> - 越往后层，token 的表示被 LLM 变换过，失去了原始视觉语义
> - 在后面层做 diversity-based 选择可能不再有效
> - 而且越早剪，后续层的计算量节省越多
> ```

**Table 4: 不同距离度量和选择策略**

| 策略 | MMB | POPE | Avg |
|---|---|---|---|
| Cosine (默认) | 59.19 | 86.02 | **62.34** |
| ℓ₁ | 59.71 | 85.40 | 61.94 |
| ℓ₂ | 59.97 | 85.64 | 62.22 |
| Random | 52.66 | 72.78 | 56.66 |
| Min-Max (反向) | 38.57 | 49.26 | 46.53 |

> 💡 **Table 4 批读**:
> ```
> 距离度量: 三种都差不多，不敏感 ✓
> 
> 选择策略排行:
> ├── Max-Min (DivPrune) ⭐ 62.34  ← 最大化多样性
> ├── Random              56.66  ← 随机也有一定多样性
> └── Min-Max              46.53  ← 最小化多样性 = 最大化冗余 → 最差！
>
> 关键发现:
> - Min-Max 比 DivPrune 差 15.8% → 证明冗余有害
> - Random 比 DivPrune 差 5.6% → 证明最大化多样性是有必要的
> - 三重验证: 多样性↑ → 性能↑
> ```

---

## 💡 Section 总结

### 关键实验结论
1. **极端压缩下 DivPrune 碾压**: ~15% TFLOP 下，其他 plug-and-play 方法几乎不能用，DivPrune 保持 85-95% 性能
2. **泛化性好**: 跨模型（7B/13B）、跨架构（LLaVA 1.5/1.6/NeXT-Video）都有效
3. **越多 token 越有效**: LLaVA 1.6 和 Video 模型上优势更大
4. **实际效率提升**: 省显存、降延迟，E2E 最快
5. **越早剪越好**: Layer 0 最优
6. **多样性 > 随机 > 冗余**: 消融实验完美验证了核心假设
