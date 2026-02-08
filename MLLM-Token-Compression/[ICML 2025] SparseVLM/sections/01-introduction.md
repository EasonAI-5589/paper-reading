[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
Introduction 阐述了 VLM 中视觉 token 冗余的问题背景、现有方法的局限（缺乏文本引导、需要训练），并引出 SparseVLM 的核心设计和主要贡献。

---

Benefiting from advancements in large language models (LLMs) (Radford et al., 2019; Brown et al., 2020; Touvron et al., 2023; Peng et al., 2023; Zhang et al., 2024a), the realm of vision-language models (VLMs) has undergone significant progress. To combine visual signals with textual semantics, the mainstream practice in VLMs (Team et al., 2023; Bai et al., 2023; Chen et al., 2024b; Li et al., 2024c; 2023a) employs sequential visual representation, where images are extracted into visual tokens and sent into an LLM decoder. With modal alignment and instruction fine-tuning (Du et al., 2022; Liu et al., 2024a; Zhu et al., 2024b), recent VLMs successfully adapt LLMs to the vision domain and inherit their perception and reasoning abilities.

> 💡 **背景**: VLM 的主流范式是将图像编码为视觉 token 序列，送入 LLM decoder 进行多模态推理。

---

Despite the promising performance, further incorporation of visual tokens inevitably introduces a huge memory and computational overhead when compared to LLMs, particularly for high-resolution images (Li et al., 2024c) and long videos (Lin et al., 2024). For instance, a 672×672 image in LLaVA (Liu et al., 2024b) yields 2304 visual tokens that span over half of the context length. However, the information in images is typically more sparse than in natural languages (Marr, 2010), resulting in inefficiency when naïvely processing both modalities. To address this, existing methods extract more compact image representations by modifying the image encoder or projector (Alayrac et al., 2022; Li et al., 2024b; Dai et al., 2023; Cha et al., 2024). While some recent works further sparsify visual tokens during the decoding (Ye et al., 2025; Chen et al., 2024a; Shang et al., 2024), they still ignore the guidance from the language tokens, which contradicts the multimodality paradigm. We argue that visual tokens should be sparsified adaptively based on the question prompt, as the model might focus on different parts (e.g., foreground or background) when dealing with various questions as shown in Figure 1. Furthermore, current approaches generally train a network to prune redundant visual tokens and require additional training data (Li et al., 2024b; Ye et al., 2025; Cai et al., 2025).

> 💡 **问题与动机**:
> - 672×672 图像 → 2304 视觉 token，占据超过一半的上下文长度
> - 图像信息比文本更稀疏（Marr, 2010），直接处理两种模态效率低
> - **现有方法两大问题**：
>   1. 缺乏文本引导 — 不管问什么问题，都用同样的方式裁剪视觉 token
>   2. 需要额外训练 — 学一个网络来裁剪，增加训练数据和参数
> - **核心论点**: 视觉 token 应该根据问题自适应裁剪

---

![Figure 1](../images/20340f3afcca8ba9339a442121159f10c42a1c2ca730a7e4389f478f5b2ec8d6.jpg)
*Figure 1. Comparison of visual token sparsification methods. Unlike previous methods with text-agnostic visual sparsification (c) e.g., VocoLLaMA (Ye et al., 2025), our SparseVLM (b) is guided by question prompts to select relevant visual patches from the image (a).*

> 💡 **Figure 1 批读**:
> - (a) 原始图像
> - (b) SparseVLM: 根据不同问题保留不同的视觉区域（text-aware）
> - (c) VocoLLaMA 等: 不管问什么都保留相同的 token（text-agnostic）
> - **关键对比**: 同一张图，问不同问题时，关注的区域应该不同

---

In this paper, we introduce a text-guided training-free framework dubbed SparseVLM for efficient vision language model inference. We reuse the self-attention matrix of visual-text tokens directly from the decoder layers without extra training parameters for sparsification. We ascertain that not all prompt tokens should be considered as some could be less relevant, which leads to inaccurate correlation results and downgrades the performance of sparse inference. Specifically, our SparseVLM first identifies text tokens strongly correlated with visual signals via cross-attention. Then, we measure the contribution of visual tokens to the selected visual-relevant text tokens (i.e., "raters") and adaptively prune the insignificant visual tokens. Instead of directly discarding the pruned tokens, we further recycle and cluster them to reconstruct more compact tokens to minimize the loss of information. Due to the information density varying for different image inputs, we employ the rank of the attention matrix to indicate the redundancy level and set an adaptive sparsification ratio accordingly.

> 💡 **SparseVLM 方法概览**:
> 1. **复用自注意力矩阵** — 无需额外参数，直接用 decoder 层已有的注意力
> 2. **筛选 text raters** — 并非所有文本 token 都适合做评判者，先找视觉相关的
> 3. **评估视觉 token 重要性** — 用选出的 raters 对视觉 token 打分
> 4. **自适应裁剪** — 用注意力矩阵的 rank 决定每层裁剪比例
> 5. **Token recycling** — 被裁剪的 token 聚类压缩，减少信息损失

---

The proposed method is simple yet practical. It can act as a plug-and-play module to improve the efficiency of VLMs without additional fine-tuning. Extensive experiments demonstrate that our SparseVLM effectively reduces computational overhead of various VLMs without sacrificing their performance in a wide range of image and video understanding tasks. For instance, LLaVA (Liu et al., 2024b) when armed with SparseVLM achieves a 4.5× compression rate while maintaining 97% of its original performance. Alternatively, the CUDA latency can decrease by 37% with only a 0.9% drop in accuracy. To investigate the effectiveness of our method in video tasks, we further apply SparseVLM to VideoLLaVA (Lin et al., 2024) to compress frames with temporal dimension. Without complex design changes, SparseVLM can sparsify video frames into an adaptive number of visual tokens and outperform existing methods in video question-answering benchmarks. Our approach consistently outperforms prior state-of-the-art FastV method (Chen et al., 2024a) by 11.2-17.3% on LLaVA, 9.2-20.4% on MiniGemini, and 14.7% on VideoLLaVA when both have similar latencies.

> 💡 **关键数字**:
> - LLaVA + SparseVLM: **4.5× 压缩率**，保持 **97%** 原始性能
> - CUDA latency 减少 **37%**，精度仅降 **0.9%**
> - 超过 FastV: LLaVA 上 **11.2-17.3%**，MiniGemini 上 **9.2-20.4%**，VideoLLaVA 上 **14.7%**

---

Our main contributions are summarized as follows:

• We introduce a novel sparsification framework dubbed SparseVLM. To the best of our knowledge, it is the first training-free approach that explores text-aware guidance for efficient VLM inference.

• Particularly, we propose a strategy to select relevant text tokens as raters of visual tokens, a method to assess the significance of visual tokens followed by pruning of redundant visual tokens with a recycling mechanism to minimize the loss of information.

• When applied to a number of VLMs, SparseVLM consistently outperforms prior state-of-the-art methods in various image and video understanding benchmarks.

> 💡 **三大贡献**:
> 1. **首个 training-free + text-aware** 的 VLM 推理加速框架
> 2. **Text rater 选择 + 视觉 token 评估 + Token recycling** 的完整流程
> 3. 在多个 VLM（LLaVA, MGM, Qwen2-VL, VideoLLaVA）上一致超越 SOTA

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| LLaVA 672×672 视觉 token 数 | 2304 |
| 压缩率 | 4.5× |
| 性能保持 | 97% |
| Latency 减少 | 37% |
| 精度下降 | 0.9% |
| vs FastV (LLaVA) | +11.2-17.3% |
| vs FastV (VideoLLaVA) | +14.7% |

### 核心洞察
1. 视觉 token 信息稀疏，但现有裁剪方法要么忽视文本引导，要么需要额外训练
2. SparseVLM 是首个结合 text-aware 和 training-free 的方案
3. 即插即用，适用于图像和视频理解任务
