# In-Place Test-Time Training

**作者**: Guhao Feng*, Shengjie Luo*, Kai Hua, Ge Zhang, Wenhao Huang, Di He, Tianle Cai
**机构**: ByteDance Seed, Peking University (State Key Laboratory of General AI)
**会议**: ICLR 2026 | **年份**: 2026
**链接**: [OpenReview](https://openreview.net/forum?id=dTWfCLSoyl)

## 一句话总结

将 Transformer MLP 块的 $W_{down}$ 矩阵复用为 TTT 的 fast weights，在推理时原地更新实现动态适应，配合 NTP 对齐的学习目标和 chunk-wise 更新，无需改动架构即可为预训练 LLM 赋予 test-time learning 能力。

## 核心贡献

1. **In-Place 设计**: 复用现有 MLP $W_{down}$ 作为 fast weights，无需引入新层，可直接 "drop-in" 增强预训练 LLM
2. **LM-Aligned 目标**: 用 Conv1D + $W_{target}$ 构造包含未来 token 信息的学习目标，替代传统 reconstruction 目标，与 NTP 任务对齐
3. **理论保证**: Theorem 1 证明 LM-Aligned 目标能增加正确 next token 的 logit，而 reconstruction 目标对此无帮助
4. **Chunk-wise + Context Parallelism**: 更新规则具有结合律，支持 prefix sum 并行化，chunk size 512-1024 最优
5. **广泛验证**: Drop-in 增强 Qwen3-4B 在 128k 上下文达 77.0% RULER；从头预训练击败 GLA/DeltaNet/LaCT

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：问题-方案-结果概览 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + TTT 三大障碍 + In-Place TTT 方案 |
| [02 - Preliminary](sections/02-preliminary.md) | TTT 机制详解 + LLM 生态三大需求 |
| [03 - Method](sections/03-method.md) | 核心方法：MLP 复用 + LM-Aligned 目标 + 理论分析 + 实现细节 |
| [04 - Experiments & Conclusion](sections/04-experiments.md) | RULER/Perplexity/消融实验 + 结论 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Drop-in 基座 | Qwen3-4B-Base (32k → 128k) |
| RULER@128k | 77.0% (vs baseline 74.8%) |
| RULER@64k | 78.7% (vs baseline 74.3%) |
| 外推到 256k | 43.9% (vs baseline 41.7%) |
| LLaMA-3.1-8B @64k 提升 | +2.1% |
| Qwen3-14B @64k 提升 | +2.7% |
| From-scratch 最优 chunk | C = 512 或 1024 |
| 训练数据量 | Stage1: ~20B tokens@32k + Stage2: ~15B tokens@128k |
| 额外开销 | 吞吐量和显存几乎无额外开销 |

## 📊 Citation Landscape

> *注：本文发表于 ICLR 2026，未在 arXiv 上发布，Semantic Scholar 暂无收录。以下信息基于论文参考文献手动整理。*

### 参考文献分组

**Test-Time Training (TTT) 核心**
| 论文 | 年份 | 要点 |
|------|------|------|
| Sun et al. (TTT) | 2020 | TTT 开创性工作，self-supervised objectives at test time |
| Sun et al. (TTT-Linear) | 2024 | Learning to (learn at test time): RNNs with expressive hidden states |
| Zhang et al. (TTT Done Right) | 2025 | Test-time training done right |
| Behrouz et al. (Titans) | 2024 | Learning to memorize at test time，Neural Memory |
| Li et al. (TNT) | 2025 | Improving chunkwise training for test-time memorization |
| Yau et al. | 2025 | Sequential-parallel duality in prefix scannable models |

**Linear Attention / SSM**
| 论文 | 年份 | 要点 |
|------|------|------|
| Yang et al. (GLA) | 2024b | Gated Linear Attention |
| Yang et al. (DeltaNet) | 2024a/d | Delta rule for linear transformers |
| Schlag et al. | 2021 | Linear transformers are secretly fast weight programmers |
| Dao et al. (Mamba) | 2023 | State Space Models |

**MLP as Memory**
| 论文 | 年份 | 要点 |
|------|------|------|
| Geva et al. | 2020 | Transformer feed-forward layers are key-value memories |
| Ba et al. | 2016 | Using fast weights to attend to the recent past |

**Long-Context Methods**
| 论文 | 年份 | 要点 |
|------|------|------|
| Beltagy et al. (Longformer) | 2020 | Sparse attention for long documents |
| Peng et al. (YaRN) | 2023 | Efficient context window extension |
| Child et al. (Sparse Transformer) | 2019 | Generating long sequences with sparse transformers |

### 推荐阅读

| 论文 | 理由 |
|------|------|
| Titans (Behrouz et al., 2024) | 同为 parametric memory 方向，但用独立 Neural Memory 模块 |
| TTT-Linear (Sun et al., 2024) | In-Place TTT 的直接前身，用线性层做 TTT hidden state |
| MemoryLLM (Wang et al., 2024) | 另一种在 Transformer 内嵌入可更新参数记忆的方案 |
| GLA (Yang et al., 2024b) | 对比方法：Gated Linear Attention 作为高效注意力替代 |
| Geva et al. (2020) | 理论基础：MLP 层 = key-value memory |

---

*Connected Papers: 本文暂未在 arXiv 发布，Connected Papers 链接待补充*
