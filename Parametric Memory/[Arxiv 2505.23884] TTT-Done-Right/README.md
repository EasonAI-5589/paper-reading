# Test-Time Training Done Right

**作者**: Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, Hao Tan  
**机构**: Massachusetts Institute of Technology & Adobe Research  
**arXiv**: [2505.23884](https://arxiv.org/abs/2505.23884) | **年份**: 2025  
**项目主页**: https://tianyuanzhang.com/projects/ttt-done-right/

---

## 一句话总结

LaCT (Large Chunk Test-Time Training) 将 TTT 的 mini-batch 从 16~64 tokens 扩大到 2K~1M tokens，GPU 利用率提升数个数量级，同时支持高达模型参数量 40% 的非线性 fast weight，在 Novel View Synthesis、语言模型、自回归视频生成三个模态上均达到 SOTA。

---

## 核心贡献

1. **大块更新范式 (LaCT)**：将 TTT 的 chunk size 从 16-64 扩大到 2K-1M，GPU FLOP 利用率从 <5% 提升至 70%，只需几十行 PyTorch 代码，无需定制 CUDA kernel
2. **大状态非线性 Fast Weight**：支持 SwiGLU-MLP 作为 fast weight，状态大小可达模型参数量的 40%（比现有方法大 10-40×），显著提升记忆容量
3. **Muon 优化器集成**：将 Muon（Newton-Schulz 迭代正交化梯度）用于 online memory update，一致优于 GD 和 Momentum
4. **跨模态统一框架**：通过不同的 Update/Apply 顺序实现不同注意力掩码，统一支持图像集（无序 set）、文本（1D 序列）、视频（图像序列）三种数据结构
5. **极长上下文扩展**：NVS 任务处理 128 张 960×536 图片（共 1M tokens），14B 参数视频扩散模型处理 56K visual tokens

---

## 📖 批读导航

| Section | 文件 | 内容简介 |
|---------|------|---------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 | TTT 问题动机 + LaCT 核心思想 |
| [01 - Introduction](sections/01-introduction.md) | 引言 | 长上下文需求 + Figure 1 (GPU 利用率 vs 性能) |
| [02 - Preliminary](sections/02-preliminary.md) | 预备知识 | TTT 数学定义 + 计算效率瓶颈分析 + Figure 2 (LaCT Block) |
| [03 - Method: LaCT Architecture](sections/03-method.md) | 方法主体 | Large-Chunk TTT Layer + SwiGLU fast weight + Muon + Window Attention + Context Parallelism |
| [04 - Applications (N-D Data)](sections/04-applications.md) | N维数据应用 | NVS (图像集) / 语言模型 (文本) / 自回归视频扩散 的 LaCT 设计 |
| [05 - Experiments](sections/05-experiments.md) | 实验结果 | 三任务实验 + 消融分析 (状态大小、优化器、线性/非线性) |
| [06 - Related Work & Conclusion](sections/06-related-conclusion.md) | 相关工作与结论 | TTT 系列 / Chunk+Recurrence 方法 / 局限性分析 |

---

## 关键数字速查

| 指标 | 数值 |
|------|------|
| Chunk Size 范围 | 2K ~ 1M tokens |
| GPU FLOP 利用率 (A100) | 提升至 70%（原始 TTT <5%） |
| Fast weight 占模型参数比 | **40%**（现有方法 0.1%~5%） |
| NVS 最大上下文 | 1M tokens (128 张 960×536 图片) |
| 视频模型参数量 | 14B |
| 视频最大序列长度 | 56,160 visual tokens |
| 代码实现 | 几十行纯 PyTorch，无需 custom kernel |
| arXiv 引用量 | 73 次（截至 2026-04） |

---

## 📊 Citation Landscape

### TLDR (Semantic Scholar Auto-Summary)
> An extremely large chunk update is used, ranging from 2K to 1M tokens across tasks of varying modalities, which is referred to as Large Chunk Test-Time Training (LaCT), which improves hardware utilization by orders of magnitude, and more importantly, facilitates scaling of nonlinear state size (up to 40% of model parameters), hence substantially improving state capacity.

### 引用统计

| 项目 | 数值 |
|------|------|
| 参考文献数 | 68 |
| 被引次数 | 73 |
| Influential Citations | 5 |
| Connected Papers | [查看图谱](https://www.connectedpapers.com/main/2505.23884) |
| Semantic Scholar | [论文页面](https://www.semanticscholar.org/paper/c039d5c73107dcc6fd61bbec2a1363a8cb8634af) |

### 核心参考文献分组

#### 🧠 TTT & Sequence Modeling（按引用量排序 Top 5）

| 论文 | 年份 | Citations | 要点 |
|------|------|-----------|------|
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | 6,411 | 线性时间序列建模，本文对比基线 |
| [RetNet](https://arxiv.org/abs/2307.08621) | 2023 | 608 | Retention 机制，TTT 相关 |
| [GLA (Gated Linear Attention)](https://arxiv.org/abs/2312.06635) | 2023 | 370 | 硬件高效线性注意力，语言模型对比基线 |
| [DeltaNet](https://arxiv.org/abs/2406.06484) | 2024 | 232 | Delta rule 线性 Transformer，语言模型对比基线 |
| [Titans](https://arxiv.org/abs/2501.00663) | 2024 | 198 | 测试时记忆，本文引用的 Momentum 实现 |

#### 🎯 Novel View Synthesis

| 论文 | 年份 | Citations | 要点 |
|------|------|-----------|------|
| NeRF | - | 7,992 | 基础方法 |
| [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | 2023 | 7,793 | 本文超越的优化方法基线 |
| [GS-LRM](https://arxiv.org/abs/2404.19702) | 2024 | 289 | 数据驱动 NVS，本文方法基础 |
| [LVSM](https://arxiv.org/abs/2410.17242) | 2024 | 129 | 大视图合成模型，本文 Tokenization 设计参考 |
| [LongLRM](https://arxiv.org/abs/2410.12781) | 2024 | 75 | 长序列 NVS 对比基线 |

#### ⚙️ 优化器 & 训练技术

| 论文 | 年份 | Citations | 要点 |
|------|------|-----------|------|
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 2023 | 2,469 | 高效注意力实现，启发了大块并行设计 |
| [Weight Normalization](https://arxiv.org/abs/1602.07868) | 2016 | 2,080 | L2 fast weight normalization 来源 |
| [SwiGLU](https://arxiv.org/abs/2002.05202) | 2020 | 1,709 | fast weight 网络结构选择 |
| [Mamba2/SSM](https://arxiv.org/abs/2405.21060) | 2024 | 1,333 | Transformers are SSMs，视频对比基线 |
| [InfiniAttention](https://arxiv.org/abs/2404.07143) | 2024 | 188 | 最接近 LaCT 的先前工作 |

#### 🎬 视频生成

| 论文 | 年份 | Citations | 要点 |
|------|------|-----------|------|
| [CogVideoX](https://arxiv.org/abs/2408.06072) | 2024 | 1,670 | 视频扩散 Transformer |
| [Wan 2.1](https://arxiv.org/abs/2503.20314) | 2025 | 1,356 | 本文 fine-tune 的预训练视频模型 |

### 推荐相关论文（Semantic Scholar Recommendations）

| 论文 | 年份 | 要点 |
|------|------|------|
| [Attention Residuals](https://arxiv.org/abs/2603.15031) | 2026 | 注意力残差机制 |
| [LoGeR: Long-Context 3D Reconstruction](https://arxiv.org/abs/2603.03269) | 2026 | LaCT 思路用于几何重建 |
| [tttLRM: TTT for 3D Reconstruction](https://arxiv.org/abs/2602.20160) | 2026 | TTT 用于长上下文自回归 3D 重建 |
| [TTT with KV Binding = Linear Attention](https://arxiv.org/abs/2602.21204) | 2026 | 揭示 TTT 与线性注意力的等价关系 |

---

## BibTeX

```bibtex
@article{zhang2025testtime,
  title={Test-Time Training Done Right},
  author={Zhang, Tianyuan and Bi, Sai and Hong, Yicong and Zhang, Kai and Luan, Fujun and Yang, Songlin and Sunkavalli, Kalyan and Freeman, William T. and Tan, Hao},
  journal={arXiv preprint arXiv:2505.23884},
  year={2025}
}
```
