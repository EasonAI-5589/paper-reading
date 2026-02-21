[← 返回 README](../README.md)

# Appendix 精选

## 📌 预览
附录包含 pivot 选择实验细节、大模型验证（72B）、音频模态扩展、VLA 机器人任务扩展。精选最有价值的部分。

---

## A.1 Pivot Token Selection (Table 8)

Table 8 details performance metrics across multiple benchmarks with all experiments retaining 128 vision tokens. Results validate the robustness of DART under various pivot token selection criteria. Key numbers:

| 策略 | GQA | MMB | MME | POPE | VQAV2 | VQAText | Avg. |
|------|-----|-----|-----|------|-------|---------|------|
| Random | 59.0±0.3 | 63.2±0.7 | 1772±17.9 | 80.6±0.49 | 75.2±0.2 | 56.0±0.3 | 96.0% |
| A-Score♠ | 59.2 | 63.1 | 1826 | 81.1 | 75.9 | 55.7 | 96.9% |
| K-norm♠ | 58.7 | 63.2 | 1840 | 80.1 | 75.9 | 56.4 | 96.8% |
| V-norm♠ | 57.3 | 62.5 | 1760 | 76.8 | 75.4 | 55.5 | 94.9% |
| V-norm♡ | 59.4 | 64.3 | 1825 | 81.6 | 76.1 | 56.0 | 97.2% |
| SparseVLM | 56.0 | 60.0 | 1745 | 80.5 | 73.8 | 54.9 | 93.9% |
| FastV | 49.6 | 56.1 | 1490 | 59.6 | 61.8 | 50.6 | 81.5% |

> 💡 Random pivot 的标准差很小（MME ±17.9, 其他 ±0.3~0.7），说明 DART 对 pivot 选择的 variance 很低。V-norm♡（选最小 V-norm 做 pivot）反而最好（97.2%），再次说明 "unimportant" pivot 也完全 work。

---

## B.1 DART in Audio Modalities (Table 10)

DART 在 Phi-4-Multimodal-Instruct 的 ASR 任务上也有效：

| 压缩比 | Random WER | FastV WER | DART WER |
|--------|-----------|-----------|----------|
| 20% | 16.69 | 23.86 | **6.00** |
| 30% | 26.30 | 42.85 | **8.74** |
| 50% | 57.21 | 134.19 | **34.03** |

> 💡 **音频扩展**: DART 在 audio token 上的优势更加明显。FastV 在 50% 压缩下 WER 达到 134.19（完全不可用），而 DART 保持 34.03。这说明 duplication 现象不限于 vision——audio tokens 同样有大量冗余，而 attention-based importance 在 audio 上更加不可靠。

---

## B.2 DART for VLA (Table 11)

在 CogACT 机器人操作任务（SIMPLER 环境）上：

| Method | Retained | Avg. (Visual Match) | Avg. (Variant Agg.) | Speedup |
|--------|----------|--------------------|--------------------|---------|
| CogACT (vanilla) | 256 | 74.8% | 61.3% | 1.00× |
| FastV | 56 | 74.1% | 62.1% | 1.21× |
| VLA-Cache | - | 74.4% | 62.3% | 1.38× |
| **DART** | **56** | **75.2%** | **64.4%** | 1.25× |

> 💡 **VLA 扩展**: DART 在机器人任务上甚至超过 vanilla CogACT（75.2% vs 74.8%），再次验证了 "去除冗余 token 可能减少 noise、提升性能" 的 hypothesis。1.25× speedup 对于实时机器人控制有实际意义。

---

## D. Computational Complexity

Total FLOPs = T × (4nd² + 2n²d + 2ndm) (11)

Post-Pruning FLOPs = L × (4nd² + 2n²d + 2ndm) + (T−L) × (4n̂d² + 2n̂²d + 2n̂dm) (12)

Reduction ratio = 1 − Post-Pruning FLOPs / Total FLOPs (13)

> 💡 **FLOPs 公式**: 关键变量是 n（原始序列长度）和 n̂（pruning 后序列长度）。由于 attention 的 O(n²) 项，n 减半可带来约 75% 的 attention FLOPs reduction。但实际加速还受 FFN（O(n)）和 memory bandwidth 影响。

---

## F. Sparsification Visualization

Figure 9 shows that different pivot selection strategies retain spatially scattered tokens without obvious position bias (spatial uniformity). Different strategies retain significantly different token sets but achieve comparable performance—corroborating the finding that multiple distinct covering sets exist.

> 💡 **可视化总结**: 所有 DART 策略保留的 token 空间分布均匀（无 position bias），而不同策略的 token 集差异很大。这与 §5.2 的定量分析（<50% overlap）完美对应。

---
