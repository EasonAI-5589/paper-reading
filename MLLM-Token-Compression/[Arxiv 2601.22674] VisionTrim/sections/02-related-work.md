[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
两个小节：(1) MLLM 发展背景及 visual token 瓶颈；(2) 视觉 token 压缩方法综述，从 efficient projector 到 training-free pruning/merging。

---

## 2.1 Multimodal Large Language Models

Large Language Models (LLMs) (Vicuna, 2023; Touvron et al., 2023; Bai et al., 2023a; Team, 2023; Achiam et al., 2023) have garnered significant attention due to their powerful capabilities in natural language processing tasks such as text understanding, generation, and question answering. Nonetheless, the reliance on purely textual data limits their applicability, as human perception is inherently multimodal. This has spurred the development of Multimodal LLMs (MLLMs) (Liu et al., 2023; Bai et al., 2023b; Chen et al., 2024c; Reid et al., 2024; Yu et al., 2025; Lin et al., 2024; Li et al., 2024a; Liu et al., 2024c; Wang et al., 2025b), which integrate LLMs with visual encoders to augment performance in multimodal tasks. The typical image- and video-based MLLMs (Liu et al., 2024a; Cheng et al., 2024; Lin et al., 2023) utilize an MLP to project visual information encoded by a Vision Transformer (ViT) (Dosovitskiy, 2020) into a space interpretable by LLMs, improving performance on visual-language tasks through visual instruction tuning. However, this paradigm requires a large number of visual tokens to represent visual information, particularly with high-resolution images and long-context video inputs, which further exacerbates the issue. The resulting increase in computational demands and inference times poses significant challenges, hindering the practical deployment of MLLMs in real-world applications.

> 💡 **2.1 批注**: 标准 MLLM pipeline: ViT → MLP projector → LLM。核心瓶颈在于 visual token 数量：
> - 标准分辨率: 576 tokens（LLaVA-1.5）
> - 高分辨率: 2880 tokens（LLaVA-NeXT）
> - 视频: 2048+ tokens（Video-LLaVA，8帧×256）

---

## 2.2 Vision Token Compression for MLLMs

The quadratic complexity inherent in Transformer networks (Vaswani et al., 2017), which scales with the sequence length of input tokens in MLLMs, remains a widely acknowledged challenge. To address this issue, several methods (Li et al., 2023a; Bai et al., 2023b; Cha et al., 2024; Li et al., 2024b; Yao et al., 2024; Hu et al., 2024; Chu et al., 2023; 2024) explore efficient visual projectors that enable compact visual representations using fewer visual tokens before feeding them into the LLM. While these approaches have demonstrated promising performance, they often necessitate architectural modifications and extensive training. Alternatively, recent works (Shang et al., 2024; Chen et al., 2024a; Zhang et al., 2024a; 2025b; Yang et al., 2025) aim to reduce visual tokens in a training-free manner and mainly focus on either the vision encoding or LLM decoding stages. LLaVA-PruMerge (Shang et al., 2024) utilizes class-spatial similarity for pruning, and FasterVLM (Zhang et al., 2024a) evaluates token importance via attention scores between the [CLS] token and image tokens, both operating before sending the vision tokens to the LLM. FastV (Chen et al., 2024a) and SparseVLM (Zhang et al., 2025b) prune redundant tokens at a specific layer of LLM based on attention scores solely during the LLM decoding stage. Additionally, most existing approaches overlook the alignment between visual token selection and textual information. While CrossGET (Shi et al., 2024) and Turbo (Ju et al., 2024) directly leverage text-visual attention to aid token selection, they place excessive emphasis on text tokens, which can lead to hallucinations and disrupt multi-round interactions. In contrast, our approach considers the entire MLLM pipeline and simultaneously integrates both global semantic significance and local spatial continuity to preserve visual integrity. Furthermore, we introduce a text-guided visual complement mechanism to ensure alignment with textual instructions, offering a more comprehensive and effective solution to the challenge of vision token compression.

> 💡 **2.2 方法分类**:
>
> | 类别 | 代表方法 | 特点 | 局限 |
> |------|---------|------|------|
> | **Efficient Projector** | BLIP-2, MobileVLM, TokenPacker | 架构修改 + 训练 | 需要重新训练 |
> | **Vision Encoding 端** | LLaVA-PruMerge, FasterVLM, VisionZip | [CLS] attention 选 token | 只在 encoder 端 |
> | **LLM Decoding 端** | FastV, SparseVLM | LLM attention 剪枝 | 只在 decoder 端 |
> | **Text-guided** | CrossGET, Turbo | 直接用 text-visual attention | 过度依赖文本，幻觉 |
> | **VisionTrim** | 本文 | 两阶段 + text-guided complement | — |
>
> **关键对比**: CrossGET/Turbo **直接**用文本引导选 token → 过度强调文本 → 幻觉；VisionTrim 先视觉选（DVTS），再文本补（TGVC），更稳健。

---

## 🔖 Section 总结

### 核心洞察
1. Training-free 方法的主流思路是 attention-based pruning，但分为 encoder 端和 decoder 端两个流派
2. Text-guided 是个双刃剑：直接用会过度依赖文本导致幻觉，VisionTrim 的做法是"先选后补"更稳健
3. VisionTrim 在 Related Work 的定位：既不是纯 vision-based 也不是纯 text-guided，而是 two-stage + complementary
