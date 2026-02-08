[← 返回 README](../README.md)

# 2. Related Works

## 📌 预览
Related Work 分三部分：LMM 概述、高效 LMM 方法、视觉 token 剪枝方法。重点是 token pruning 方法的分类对比。

---

## 2.1. Large Multimodal Models (LMMs)

LMMs handle diverse data types, including text, audio, image, and, video [5, 21, 24, 25, 32, 42, 55]. This work focuses on open-source LMMs that support language and visual inputs. These LMMs can be categorized into two types: image-based and video-based LMMs. The image-based LMMs [24, 25] address image-language understanding tasks, like image captioning, visual question answering, and image reasoning. On the other hand, video-based LMMs are geared towards video understanding [21, 55] tasks, like video captioning, video summarization, and video question answering.

> 💡 **批注**: LMM 分为 image-based（如 LLaVA）和 video-based（如 LLaVA-NeXT-Video）。本文的方法对两类都适用。

---

## 2.2. Efficient LMMs

Several techniques are proposed to improve inference efficiency specifically for LMMs. The first technique is to change the model architecture in LMMs. For example, [35] proposed to replace transformer-based LLMs with Mamba model [13]. [52, 56] retrained LMMs with small scale LLMs to improve their efficiency. [48] used knowledge distillation to train a small LMM. In addition to changing the architecture, it is shown in [39] that skipping some blocks or layers within LMMs can improve the inference speed without damaging the model's performance. Furthermore, efficient decoding techniques such as speculative decoding are proposed to make LMM inference more efficient [11].

> 💡 **批注**: 高效 LMM 的路线图：
> - **换架构**: Mamba 替代 Transformer
> - **小模型**: 小 LLM + 知识蒸馏
> - **跳层**: 跳过部分 LLM 层
> - **加速解码**: speculative decoding
>
> DivPrune 属于第五类——**token pruning**，与上述方法正交，可以组合使用。

---

## 2.3. Visual Token Pruning

Visual token pruning methods are proposed to reduce the inference complexity for LMMs. The first group of methods uses attention scores to prune tokens [4, 38]. PruMerge [38] introduces a token pruning method for the vision encoder where the visual tokens are clustered and merged based on their attention sparsity. In addition, FastV [4] prunes tokens within a specific layer of the LLM based on the magnitude of attention scores in an earlier layer. It is shown that pruning tokens based on attention scores are not optimal [14, 23], especially at higher pruning ratios.

> 💡 **批注**: **Attention-based 方法**（FastV、PruMerge）是最直觉的做法——attention 分数低的 token「不重要」所以删掉。但问题是 attention score 不等于真正的信息量，而且高 attention 的 token 往往彼此相似。

---

Calibration-based methods offer another line of work, where pruning layers and/or ratios are determined by analyzing the LLM outputs for a calibration dataset [23, 50]. For example, FitPrune [50] calculates a pruning recipe based on the observed attention divergence before and after pruning. VTW [23] argues that visual tokens can be entirely removed after a certain layer within LLM. The layer to remove the visual tokens is chosen using a calibration dataset. These methods rely on calibration datasets and require custom calibration for each LMM, which can be costly and cumbersome for new models.

> 💡 **批注**: **Calibration-based 方法**（FitPrune、VTW）需要跑一遍校准数据来决定怎么剪。问题：换模型就要重新校准，成本高。

---

Some previous works proposed token pruning with the need for fine-tuning. M³ [3] applies model fine-tuning to produce nested visual token representations at multiple granularities, allowing users to select token lengths dynamically during inference. In [19], a projector layer trained using a large-scale dataset is proposed that packs finer detailed information into compact token representations. These methods need significant computational resources for training, limiting their use across various scenarios.

> 💡 **批注**: **Fine-tuning-based 方法**（M³、TokenPacker）效果好但代价大——需要重新训练模型。

> 💡 **2.3 小结 — Token Pruning 方法分类**:
>
> | 类别 | 代表方法 | 优点 | 缺点 |
> |------|----------|------|------|
> | Attention-based | FastV, PruMerge | Plug-and-play | 高压缩比下冗余严重 |
> | Calibration-based | FitPrune, VTW | 较优的剪枝策略 | 每个模型需要单独校准 |
> | Fine-tuning-based | M³, TokenPacker | 性能最好 | 训练成本高 |
> | **Diversity-based** | **DivPrune** | **Plug-and-play + 高压缩比下优异** | *（本文方法）* |

---

## 🔖 Section 总结

### 核心洞察
1. Token pruning 方法可分为 attention-based、calibration-based、fine-tuning-based 三类
2. DivPrune 开创了第四类——diversity-based，兼具 plug-and-play 的便利性和高压缩比下的鲁棒性
3. 现有 plug-and-play 方法（FastV、PruMerge）在极端压缩（≥80%）下性能急剧下降
