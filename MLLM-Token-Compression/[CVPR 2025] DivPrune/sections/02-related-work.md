# 2. Related Works

> 来源: DivPrune (Arxiv 2503.02175)

---

## 📄 原文

> 💡 **Section 概览**: Related Work 分三条线梳理：LMM 基础、高效 LMM 技术、视觉 token 剪枝方法。重点在第三部分，将现有 token pruning 方法分成三类来分析不足。

---

### 2.1 Large Multimodal Models (LMMs)

> 💡 **2.1 要点预览**: LMM 的基本分类——image-based vs video-based。

LMMs handle diverse data types, including text, audio, image, and, video [5, 21, 24, 25, 32, 42, 55]. This work focuses on open-source LMMs that support language and visual inputs. These LMMs can be categorized into two types: image-based and video-based LMMs. The image-based LMMs [24, 25] address image-language understanding tasks, like image captioning, visual question answering, and image reasoning. On the other hand, video-based LMMs are geared towards video understanding [21, 55] tasks, like video captioning, video summarization, and video question answering.

> 💡 **2.1 小结**: 标准分类，DivPrune 两种都测了（LLaVA 1.5/1.6 + LLaVA-NeXT-Video）。

---

### 2.2 Efficient LMMs

> 💡 **2.2 要点预览**: 除了 token pruning，还有哪些让 LMM 变快的方法。

Several techniques are proposed to improve inference efficiency specifically for LMMs. The first technique is to change the model architecture in LMMs. For example, [35] proposed to replace transformer-based LLMs with Mamba model [13]. [52, 56] retrained LMMs with small scale LLMs to improve their efficiency. [48] used knowledge distillation to train a small LMM. In addition to changing the architecture, it is shown in [39] that skipping some blocks or layers within LMMs can improve the inference speed without damaging the model's performance. Furthermore, efficient decoding techniques such as speculative decoding are proposed to make LMM inference more efficient [11].

> 💡 **2.2 小结**: 高效 LMM 的技术谱系：
> ```
> 高效 LMM 技术
> ├── 架构替换: Mamba 替代 Transformer
> ├── 小模型: 用小 LLM backbone / 知识蒸馏
> ├── 层跳过: 跳过冗余 decoder 层
> ├── 高效解码: 投机解码 (speculative decoding)
> └── Token 剪枝: ← DivPrune 属于这一类
> ```

---

### 2.3 Visual Token Pruning

> 💡 **2.3 要点预览**: 现有 token pruning 方法分三大类，各有什么问题。

**第一类：基于注意力的方法**

Visual token pruning methods are proposed to reduce the inference complexity for LMMs. The first group of methods uses attention scores to prune tokens [4, 38]. PruMerge [38] introduces a token pruning method for the vision encoder where the visual tokens are clustered and merged based on their attention sparsity. In addition, FastV [4] prunes tokens within a specific layer of the LLM based on the magnitude of attention scores in an earlier layer. It is shown that pruning tokens based on attention scores are not optimal [14, 23], especially at higher pruning ratios.

> 💡 **批注**: FastV 和 PruMerge 是最常见的 baseline：
> | 方法 | 在哪剪 | 怎么选 | 问题 |
> |------|--------|--------|------|
> | FastV | LLM 内某层 | 前面层的 attention score | 高压缩比下暴跌 |
> | PruMerge | Vision encoder 后 | 注意力稀疏度+聚类合并 | 压缩比不可控（variable） |

**第二类：基于校准的方法**

Calibration-based methods offer another line of work, where pruning layers and/or ratios are determined by analyzing the LLM outputs for a calibration dataset [23, 50]. For example, FitPrune [50] calculates a pruning recipe based on the observed attention divergence before and after pruning. VTW [23] argues that visual tokens can be entirely removed after a certain layer within LLM. The layer to remove the visual tokens is chosen using a calibration dataset.

> 💡 **批注**: 
> - **FitPrune**: 用校准数据找最优剪枝策略，效果好但要额外跑一遍校准
> - **VTW**: 发现视觉 token 过了某些层后就不需要了，直接整层删除。思路有趣但粗暴

**第三类：需要微调的方法**

Some previous works proposed token pruning with the need for fine-tuning. M³ [3] applies model fine-tuning to produce nested visual token representations at multiple granularities, allowing users to select token lengths dynamically during inference. In [19], a projector layer trained using a large-scale dataset is proposed that packs finer detailed information into compact token representations.

> 💡 **批注**: M³ (Matryoshka) 的思路很有意思——训练出"套娃"式表示，推理时自由选长度。但代价是需要微调整个模型。

---


> 💡 **Figure 2 批读**:
> ```
> LMM 架构流程:
> Image → Vision Encoder → Projector → Visual Tokens (E_v)
>                                                    ↓
>                                          DivPrune 在这里剪枝
>                                                    ↓
> Text → Text Encoder → Text Tokens (E_t) → 合并 → LLM → 输出
>
> DivPrune 步骤（右侧）:
> 1. 计算所有 token 间的余弦距离
> 2. 用 MMDP 算法选出最多样的子集
> 3. 丢弃剩余 token
> ```

---

## 💡 Section 总结

### 方法对比谱系
```
Visual Token Pruning 方法
├── 基于注意力 (Attention-based)
│   ├── FastV: LLM 层内剪，按 attention score
│   └── PruMerge: Vision encoder 后剪，聚类+合并
├── 基于校准 (Calibration-based)
│   ├── FitPrune: 校准数据优化剪枝策略
│   └── VTW: 校准找最佳删除层
├── 基于微调 (Fine-tuning-based)
│   ├── M³: 套娃式多粒度表示
│   └── TokenPacker: 训练紧凑 projector
└── 基于多样性 (Diversity-based) ← DivPrune
    └── MMDP 最大化 token 多样性，无需训练
```

### 核心洞察
DivPrune 的定位很清晰：在 **plug-and-play**（无需训练/校准）这个赛道上，用数学上更优的选择策略（多样性 vs 重要性）来超越 FastV 和 VTW。
