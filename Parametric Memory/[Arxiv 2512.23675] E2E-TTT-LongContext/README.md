# End-to-End Test-Time Training for Long Context

**作者**: Arnuv Tandon*, Karan Dalal*, Xinhao Li*, Daniel Koceja*, Marcel Rød*, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun*
**机构**: Astera Institute · NVIDIA · Stanford · UC Berkeley · UC San Diego
**发布**: arXiv 2512.23675 (2025-12-29) | **年份**: 2025
**链接**: [arXiv](https://arxiv.org/abs/2512.23675) · [代码](https://github.com/test-time-training/e2e)

## 一句话总结

把长上下文语言建模重新框定为**持续学习 (continual learning)** 问题：用一个标准 Transformer + 滑动窗口注意力 (SWA) 作为骨架，在**测试时**继续用 next-token prediction 的梯度下降把上下文"压"进 MLP 的权重里；同时在**训练时**用 meta-learning（gradients of gradients）学习一个"更擅长被 TTT"的初始化 —— 这就是 **TTT-E2E**：**测试时 E2E**（直接用最终 NTP loss）、**训练时 E2E**（outer loop 直接优化 TTT 之后的 loss）。3B / 164B tokens 下，TTT-E2E 在 context 长度上能像 full attention 那样 scale，同时保持 RNN 般的**常数推理延迟**（128K 上 2.7× 快于 full attention）。

## 核心贡献

1. **Continual learning 视角**：不再把长上下文当架构问题（改 attention 复杂度），而是当 continual learning 问题——模型在测试时继续学当前这条序列，把"读过的内容"压缩到权重里。
2. **E2E at Test Time**：inner loop 直接优化最终 next-token prediction loss，而不是像 TTT-KVB / Titans / MesaNet / Nested Learning 那样用 layer-wise 的 reconstruction / KV binding loss。
3. **E2E at Training Time**：outer loop 不再像 dynamic evaluation 那样 mimic static loss，而是用 meta-learning 通过 gradients-of-gradients 直接优化"经过 TTT 后的 loss"。
4. **架构几乎不改**：只把 full attention 换成 SWA (窗口 k=8K)，只 TTT 最后 1/4 的 block 中的 MLP，每个 TTT block 加一个 static 第二 MLP 作为 pre-train 知识的"safe storage"。
5. **三组 scaling 实验**：与 Mamba 2、Gated DeltaNet、TTT-KVB、SWA、full attention、Hybrid SWA+Full 全面对比，TTT-E2E 是**唯一**在长 context 上保持对 full attention 优势的方法。
6. **对偶推导**：从 TTT-KVB（E2E at train time）出发，把 KVB loss 换成 NTP loss，就能推出 TTT-E2E（E2E at test time）—— 两个方向殊途同归。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：continual learning 框架 + 核心结果一页看懂 |
| [01 - Introduction](sections/01-introduction.md) | 动机：Transformer vs RNN 的困境 + 为什么是"压缩" + Figure 1 (key results) |
| [02 - Method](sections/02-method.md) | 2.1 NTP 形式的 TTT / 2.2 Learning to Learn / 2.3 Mini-batch + SWA + 实现细节 / 2.4 从 KVB 出发的对偶推导 |
| [03 - Main Results](sections/03-main-results.md) | 3.1 Setup / 3.2 超参消融 / 3.3 训练 compute scaling / 3.4 context length scaling / 3.5 NIAH / 3.6 长序列 decode / 3.7 效率 |
| [04 - Related Work](sections/04-related-work.md) | Continual Learning / TTT 三种形态 / Fast Weight Programmers / Meta-Learning |
| [05 - Conclusion](sections/05-conclusion.md) | 结论 + 作者贡献说明（带项目时间线） |

## 关键数字

| 指标 | 数值 |
|------|------|
| 主模型规模 | 3B 参数 |
| 预训练 tokens | 164B (DCLM-Baseline) |
| 预训练 context | 8K |
| 扩展 context | 最长 128K (Books 数据集 fine-tune) |
| 滑动窗口 k | 8K |
| TTT mini-batch b | 1K |
| TTT 更新比例 | 最后 **1/4** 的 block 的 MLP |
| 每个 TTT block | 2 个 MLP (一个更新, 一个 static "safe storage") |
| 128K prefill 加速 | **2.7×** 快于 full attention |
| Hidden state 大小 (760M) | 88M (vs TTT-E2E-all-layers-MH 的 18M, 5× 更大) |
| Prefill latency (760M) | 0.0086 sec/1K token (vs TTT-E2E-all-layers-MH 的 0.017) |
| 训练延迟 (128K) | 仍比 full attention 快 1.2× |
| 训练延迟 (8K) | 比 full attention 慢 **3.4×** ⚠️ 当前实现瓶颈 |
| scaling 转折点 (model size) | 760M（之后趋势与 full attention 一致） |
| scaling 转折点 (tokens) | 48B |
| NIAH (recall 任务) | ⚠️ 明显劣于 full attention — 压缩的代价 |

## 📊 Citation Landscape

> Semantic Scholar paperId: `c081ead0ce870dae8237cd4d8fa031a2b195acf2` · [Connected Papers](https://www.connectedpapers.com/main/c081ead0ce870dae8237cd4d8fa031a2b195acf2) · [Semantic Scholar](https://www.semanticscholar.org/paper/c081ead0ce870dae8237cd4d8fa031a2b195acf2)

### TLDR (Semantic Scholar auto)

> This paper forms a form of Test-Time Training (TTT), is End-to-End (E2E) both at test time (via next-token prediction) and training time (via meta-learning), in contrast to previous forms.

### 引用统计

| 指标 | 数值 |
|------|------|
| Reference count | 104 |
| Citation count | 13 |
| Influential citations | 2 |
| Venue | arXiv.org (2025) |

### 参考文献分组（Top 5 / 组，按 citationCount 排序）

**🧠 TTT / Meta-Learning / Fast Weights** (26 refs)

| 论文 | 年份 | Cites | 要点 |
|------|------|-------|------|
| MAML (Finn et al.) | 2017 | 14.1K | Model-Agnostic Meta-Learning，outer loop 学 inner loop 初始化 — TTT-E2E 的直接思想来源 |
| Locally Weighted Regression (Cleveland) | 1979 | 11.4K | TTT 最古老的形式之一，给定 test 点在邻居上重训 |
| Meta-Learning with Memory-Augmented NN (Santoro et al.) | 2016 | 2.1K | 外存 + 内存 meta-learning 基座 |
| Sun et al. (TTT with Self-Supervision) | 2019 | 1.2K | Yu Sun 本人的 TTT 开山作，CV 领域 |
| Bengio et al. (Synaptic learning rule) | 1990 | 646 | Learning to learn 早期尝试 |
| Sun et al. (Learning to learn at test time) | 2023 | — | arXiv:2310.13807，本文前身 |

**🔄 RNN / Linear Attention / SSM** (16 refs)

| 论文 | 年份 | Cites | 要点 |
|------|------|-------|------|
| LSTM (Hochreiter & Schmidhuber) | 1997 | 103K | RNN 的基础，gating 灵感源头 |
| Empirical Eval of GRU (Chung et al.) | 2014 | 14.4K | 系统研究 gated RNN |
| Mamba (Gu & Dao) | 2023 | 6.4K | 选择性 SSM，本文主要 RNN baseline 之一 |
| Transformers are RNNs (Katharopoulos et al.) | 2020 | 2.6K | Linear attention，和 FWP 同构 |
| Mamba 2 / SSM-Transformer duality (Dao & Gu) | 2024 | 1.3K | Mamba 2 论文，baseline |
| Schmidhuber (Fast-weight memories) | 1992 | 549 | FWP 开山，所有 linear attention 的祖师爷 |

**💾 Parametric Memory / Memory Models** (2 refs)

| 论文 | 年份 | Cites | 要点 |
|------|------|-------|------|
| Titans (Behrouz et al.) | 2024 | 198 | Learning to memorize at test time，TTT-KVB 家族代表 |
| Nested Learning (Behrouz et al.) | 2025 | 31 | 把深度网络看成嵌套的 associative memory |
| MesaNet (von Oswald et al.) | 2025 | — | 每 token locally optimal TTT step |
| Zhang et al. (TTT Done Right, TTT-KVB) | 2025 | — | arXiv:2505.23884，TTT-KVB 的最新最强版，本文第 6 号 baseline + 2.4 节推导起点 |

**📏 Continual Learning** (10 refs)

| 论文 | 年份 | Cites | 要点 |
|------|------|-------|------|
| EWC (Kirkpatrick et al.) | 2016 | 9.5K | 解决 catastrophic forgetting 的经典正则化 |
| LwF (Li & Hoiem) | 2016 | 5.4K | Learning without Forgetting |
| GEM (Lopez-Paz & Ranzato) | 2017 | 3.3K | Gradient Episodic Memory |
| CL Survey (De Lange et al.) | 2019 | 2.3K | 定义了 CL 的经典设定 |
| Wang et al. (CL Survey) | 2023 | 1.3K | 更现代的综述 |

**🪟 Long Context / Sparse / SWA** (6 refs)

| 论文 | 年份 | Cites | 要点 |
|------|------|-------|------|
| Longformer (Beltagy et al.) | 2020 | 5.2K | SWA 的开山，baseline #2 的架构 |
| RULER (Hsieh et al.) | 2024 | 767 | S-NIAH 评测来源（Section 3.5） |
| Native Sparse Attention (Yuan et al.) | 2025 | 305 | 训练时稀疏注意力 |
| RingAttention (Liu et al.) | 2024 | 163 | 百万长度 context 训练基础设施 |
| Gemma 2 (Riviere et al.) | 2024 | 1.8K | Hybrid SWA:Full = 5:1 pattern，baseline #3 来源 |

**⚙️ LLM Pre-training / Scaling Recipes** (6 refs)

| 论文 | 年份 | Cites | 要点 |
|------|------|-------|------|
| Llama 3 (Dubey et al.) | 2024 | 14K | Tokenizer 来源（paper 实测影响 +0.01 loss advantage） |
| Scaling Laws (Kaplan et al.) | 2020 | 7.5K | 为什么 small budget 下 Transformer 弱 |
| GPT-3 (Brown et al.) | 2020 | 56K | Basic recipe |
| DeepSeek-V3 | 2024 | 2.9K | 多阶段训练流程 |
| Chinchilla (Hoffmann et al.) | 2022 | 7.5K | 预训练 token 数选择依据 |

### 🌱 Semantic Scholar 推荐阅读 (10 篇最相关)

| 论文 | ArXiv | 年份 | Cites |
|------|-------|------|-------|
| **In-Place Test-Time Training** | [2604.06169](https://arxiv.org/abs/2604.06169) | 2026 | 0 |
| **GradMem**: Learning to Write Context into Memory with Test-Time Gradient Descent | [2603.13875](https://arxiv.org/abs/2603.13875) | 2026 | 0 |
| Learning When to Attend: Conditional Memory Access for Long-Context LLMs | [2603.17484](https://arxiv.org/abs/2603.17484) | 2026 | 1 |
| SR-TTT: Surprisal-Aware Residual Test-Time Training | [2603.06642](https://arxiv.org/abs/2603.06642) | 2026 | 0 |
| AllMem: A Memory-centric Recipe for Efficient Long-context Modeling | [2602.13680](https://arxiv.org/abs/2602.13680) | 2026 | 0 |
| Reinforced Fast Weights with Next-Sequence Prediction | [2602.16704](https://arxiv.org/abs/2602.16704) | 2026 | 0 |
| Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion | [2604.05688](https://arxiv.org/abs/2604.05688) | 2026 | 0 |
| Effective Distillation to Hybrid xLSTM Architectures | [2603.15590](https://arxiv.org/abs/2603.15590) | 2026 | 0 |
| **Doc-to-LoRA**: Learning to Instantly Internalize Contexts | [2602.15902](https://arxiv.org/abs/2602.15902) | 2026 | 1 |
| HiCI: Hierarchical Construction-Integration for Long-Context Attention | [2603.20843](https://arxiv.org/abs/2603.20843) | 2026 | 0 |

### 本地相关论文（同一 Parametric Memory 目录）

| 论文 | 关系 |
|------|------|
| [ICLR 2026] In-Place TTT | **最直接兄弟论文**：同样用 MLP 当 fast weights，但保持 architecture in-place, 无 meta-learning outer loop |
| [Arxiv 2501.00663] Titans | **主要对比对手**：TTT-KVB 家族代表，本文 Section 2.4 证明 TTT-E2E ≥ TTT-KVB |
| [NeurIPS 2025] Nested-Learning | 同样把 inner/outer loop 显式化的 memory 视角 |
| [ICML 2024] MemoryLLM / [ICML 2025] M+ | Parametric memory pool 路线 (vs TTT-E2E 的权重更新路线) |
| [Arxiv 2602.23320] ParamMem | 用 LoRA 编码 reflection 模式（跨样本），与 TTT-E2E 的 in-context 更新是两条正交路线 |
