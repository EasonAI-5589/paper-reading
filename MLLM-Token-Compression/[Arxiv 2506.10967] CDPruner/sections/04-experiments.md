# 4 Experiments

> 来源: Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs

---

> 💡 **Section 概览**: 在 LLaVA-1.5、LLaVA-NeXT（高分辨率）、LLaVA-Video（视频）、Qwen2.5-VL（高级架构）上验证 CDPruner，外加效率分析和消融实验。

---

## 4.1 Experimental Setup

> 💡 **4.1 要点预览**: 实验涵盖的模型、基准和对比方法。

**模型架构**:
- LLaVA-1.5（图像理解）、LLaVA-NeXT（高分辨率）、LLaVA-Video（视频）
- Qwen2.5-VL（当前最强开源 MLLM）

**评测基准**: 14 个图像基准 + 4 个视频基准

**对比方法**:
- Attention-based: FastV, PyramidDrop, SparseVLM
- Attention+Similarity: LLaVA-Prumerge, TRIM, VisionZip
- Similarity-based: DART, DivPrune

> 💡 **4.1 小结**: 实验覆盖全面——4种模型架构、18个基准、8种对比方法、3种剪枝比例。

---

## 4.2 Main Results (LLaVA-1.5-7B)

> 💡 **4.2 要点预览**: 在最常用的 LLaVA-1.5 上，CDPruner 在所有剪枝比例下全面领先。

**Table 1: LLaVA-1.5-7B 不同剪枝比例下的性能对比**

| 方法 | 保留 tokens | Rel. (保持%) | 亮点 |
|------|-----------|------------|------|
| 原始模型 | 576 (100%) | 100.0% | 基线 |
| **128 tokens (↓77.8%)** | | | |
| VisionZip | 128 | 97.6% | 第二名 |
| DivPrune | 128 | 97.5% | |
| **CDPruner** | **128** | **99.0%** | ⭐ **几乎无损** |
| **64 tokens (↓88.9%)** | | | |
| FastV | 64 | 74.9% | 崩了 |
| VisionZip | 64 | 94.4% | |
| DivPrune | 64 | 94.7% | |
| **CDPruner** | **64** | **97.0%** | ⭐ **领先 2.3%** |
| **32 tokens (↓94.4%)** | | | |
| DivPrune | 32 | 91.3% | 第二名 |
| **CDPruner** | **32** | **94.3%** | ⭐ **领先 3.0%** |

> 💡 **Table 1 批读**:
> ```
> 剪枝比例越高，CDPruner 的优势越大：
>
> ↓77.8%: CDPruner 99.0% vs VisionZip 97.6%  (差距 1.4%)
> ↓88.9%: CDPruner 97.0% vs DivPrune 94.7%   (差距 2.3%)
> ↓94.4%: CDPruner 94.3% vs DivPrune 91.3%   (差距 3.0%)
>
> 关键发现：
> 1. Attention-based（FastV）在高剪枝比下崩溃（64 tokens 只剩 74.9%）
> 2. CDPruner 在 POPE 上甚至超过未剪枝模型！说明剪枝可能减少幻觉
> 3. VizWiz 上优势不大，因为其问题缺乏信息量（如"What is this?"）
> ```

---

## 4.3 CDPruner for High Resolution (LLaVA-NeXT-7B)

> 💡 **4.3 要点预览**: 高分辨率场景冗余更多，CDPruner 优势更明显。

**Table 2: LLaVA-NeXT-7B (2880 tokens)**

| 剪枝比例 | CDPruner Rel. | 第二名 | 差距 |
|---------|-------------|--------|------|
| ↓77.8% (640) | **100.1%** | VisionZip 99.5% | +0.6% |
| ↓88.9% (320) | **98.0%** | DivPrune 96.0% | +2.0% |
| ↓94.4% (160) | **96.0%** | DivPrune 92.9% | +3.1% |

> 💡 **批读**:
> ```
> 惊人发现：保留 640/2880 tokens 时，CDPruner 性能甚至略超原始模型！
> 说明高分辨率图片确实有大量冗余 token，适当剪枝反而有益。
>
> 剪枝比例越高，CDPruner 和 DivPrune 的差距越大 (0.6% → 2.0% → 3.1%)
> → 指令条件在极端剪枝下尤其重要
> ```

---

## 4.4 CDPruner for Video Understanding (LLaVA-Video-7B)

> 💡 **4.4 要点预览**: 视频场景帧间冗余极高，CDPruner 同样表现最佳。

**Table 3: LLaVA-Video-7B (64帧 × 169 tokens)**

| 剪枝比例 | CDPruner Rel. | 第二名 | 差距 |
|---------|-------------|--------|------|
| ↓62.1% (64×64) | **98.6%** | PDrop 97.1% | +1.5% |
| ↓81.1% (64×32) | **95.0%** | DivPrune 93.0% | +2.0% |
| ↓90.5% (64×16) | **89.7%** | DivPrune 88.3% | +1.4% |

> 💡 **批读**: 每帧只保留 16 个 token 时（极端剪枝），attention-based 方法崩溃到 ~77%，而 CDPruner 还有 89.7%，比 SparseVLM 高 10 个百分点。

---

## 4.5 CDPruner for Advanced Architectures (Qwen2.5-VL-7B)

> 💡 **4.5 要点预览**: 在已经做过内部压缩的高级模型上，CDPruner 仍然最强。

**Table 4: Qwen2.5-VL-7B (1296 tokens)**

| 剪枝比例 | CDPruner Rel. | FastV | DivPrune |
|---------|-------------|-------|----------|
| ↓60.5% (512) | **97.5%** | 97.0% | 96.0% |
| ↓80.2% (256) | **92.8%** | 90.8% | 88.2% |
| ↓90.1% (128) | **85.2%** | 79.0% | 79.9% |

> 💡 **批读**:
> ```
> Qwen2.5-VL 的特殊性：
> - 其 projector 已经内置了视觉 token 压缩（pixel unshuffle）
> - 所以再剪枝时性能下降更明显
> - 但 CDPruner 在 128 tokens 时仍保持 85.2%，远超 FastV (79%) 和 DivPrune (80%)
>
> DivPrune 在 Qwen2.5-VL 上表现不佳：因为不考虑指令，
> 在已经压缩过的 token 上纯靠多样性效果有限
> ```

---

## 4.6 Efficiency Analysis

> 💡 **4.6 要点预览**: CDPruner 在效率和性能上双赢。

**Table 5: LLaVA-NeXT-7B 效率对比 (2880→320 tokens)**

| 指标 | 原始 | CDPruner | 加速比 |
|------|------|---------|--------|
| FLOPs | 39.2T | 4.0T | **×9.8** |
| Prefill 时间 | 250ms | 38ms | **×6.6** |
| Decode 时间 | 21ms | 16ms | **×1.3** |
| KV Cache | 2250MB | 250MB | ×9 |
| GPU 内存 | 18.2GB | 15.1GB | 节省 17% |

> 💡 **批读**: CDPruner 的额外开销 < 10ms（DPP 推断），相对于 prefill 节省的 212ms 完全可以忽略。而且 CDPruner 与 DivPrune 效率相同（都是 pre-LLM pruning），但性能更好。

---

## 4.7 Ablation Study

> 💡 **4.7 要点预览**: 验证 DPP 和 instruction condition 各自的贡献。

![Figure 4](../images/c55e7fe825bbae9bed50a56f05bf4ead332d91389931fd9336b12a311fca5df5.jpg)
*Figure 4: 消融实验。DPPruner = 无条件 DPP（不含指令相关性），CDPruner = 条件 DPP。*

> 💡 **Figure 4 批读**:
> ```
> 性能排名（一致趋势）:
> CDPruner > DPPruner > DivPrune
>
> 两个关键结论：
> 1. DPPruner > DivPrune → DPP 比 MMDP 更好（全局 vs 局部多样性）
> 2. CDPruner > DPPruner → 加入指令条件有额外增益
>
> 两个组件都有贡献，缺一不可。
> ```

---

## 💡 Section 总结

### 关键数字速查
| 模型 | 剪枝比例 | CDPruner 保持率 | vs 第二名 |
|------|---------|---------------|----------|
| LLaVA-1.5-7B | 94.4% | 94.3% | +3.0% |
| LLaVA-NeXT-7B | 94.4% | 96.0% | +3.1% |
| LLaVA-Video-7B | 90.5% | 89.7% | +1.4% |
| Qwen2.5-VL-7B | 90.1% | 85.2% | +5.3% |

### 核心发现
1. **剪枝比例越高，CDPruner 优势越大**——条件多样性在极端情况下尤其重要
2. **POPE 上超过原始模型**——合理剪枝可能减少视觉幻觉
3. **高分辨率场景最适合剪枝**——2880→640 tokens 几乎无损
4. **DPP > MMDP**，**条件 > 无条件**——消融实验清晰验证
