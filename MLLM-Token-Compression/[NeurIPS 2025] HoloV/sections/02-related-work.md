[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览

Related Work 从三个角度综述：MLLM 及其挑战、视觉冗余识别、视觉 token 压缩与剪枝。核心对比在于 HoloV 与现有方法的差异——保持 Flash-Attention 兼容性，且在激进剪枝下保留全局视觉上下文。

---

## 2.1 MLLMs and Their Challenges

The recent remarkable success of Large Language Models (LLMs) [60, 93, 70, 18, 54] has spurred the trend of applying their strong capabilities to multimodal comprehension tasks, fostering the development of MLLMs [1, 67]. Leveraging open-source LLMs such as LLaMA families [70, 71, 18], MLLMs [6, 46, 47] have demonstrated enhanced adaptability across a range of visual understanding tasks, leading to a more profound ability to interpret the world. While this empowers LLMs with the capability of visual perception, the incorporation of lengthy visual tokens significantly escalates the computational burdens. Moreover, studies have shown that existing MLLMs still suffer from certain visual deficiencies [69, 32] and some hallucinations [29, 28]. Some work mitigates these issues by increasing the resolution of input images or videos [53, 84], but this further exacerbates the computational overhead. For example, LLaVA-1.5 [48] encodes a 336-resolution image into 576 visual tokens, while LLaVA-NeXT [47] doubles the resolution and generates 2,880 tokens. LLaVA-OneVision [37] represents an image using 7,290 visual tokens, and Video-LLaVA [44] faces even higher costs, as it must process numerous visual tokens from multiple frames during inference. These visual tokens occupy a large portion of the context window of their LLMs. In this work, we conducted experiments and analysis on these representative models to verify HoloV's applicability.

> 💡 **2.1 要点**: 视觉 token 数量的膨胀趋势：
> | 模型 | 分辨率 | 视觉 Token 数 |
> |------|--------|--------------|
> | LLaVA-1.5 | 336 | 576 |
> | LLaVA-NeXT | 672 | 2,880 |
> | LLaVA-OneVision | 更高 | 7,290 |
> | Video-LLaVA | 多帧 | 更多 |
>
> 提高分辨率可以改善视觉缺陷和幻觉，但代价是 token 数量暴增。这就是 token 剪枝的必要性来源。

---

## 2.2 Visual Redundancy Identification

In MLLMs, visual redundancy identification facilitates the distillation of visual tokens with high informativeness for faster inference. There are two main research directions: a) Vision-centric strategies analyze the image's structure and feature distribution to discard less relevant visual tokens [13, 75]. Existing approaches include spatial-similarity clustering (e.g., TokenLearner [63]), dynamic pruning based on attention scores [25, 87, 82], and using information bottleneck or entropy metrics during the prefilling stage to estimate background redundancy. b) Instruction-centric strategies typically use cross-modal attention analysis or gradient accumulation to identify redundant tokens [49, 99, 66]. Tokens with low attention or negligible gradient impact are deemed redundant [26]. Building on this, some studies explore learned importance scoring, training a lightweight end-to-end model to predict each patch's "instruction relevance," enabling even finer-grained pruning [31, 73, 89]. As the existence of language bias in LLM may cause hallucinations, we use a vision-centric scheme.

> 💡 **2.2 要点 — 两大流派对比**:
> | 流派 | 信号来源 | 代表方法 | 优势 | 劣势 |
> |------|---------|---------|------|------|
> | Vision-centric | 图像结构/[CLS] attention | TokenLearner, FasterVLM | 无语言偏差 | 不考虑指令相关性 |
> | Instruction-centric | Cross-modal attention/梯度 | FastV, FocusLLaVA | 指令感知 | 语言偏差导致幻觉 |
>
> **HoloV 选择 vision-centric 路线**，原因明确：LLM 的语言偏差会引入幻觉。

---

## 2.3 Visual Token Compression and Pruning

The inclusion of visual information in MLLMs introduces long token sequences, leading to high computation and memory costs. For example, mini-Gemini-HD [41] generates 2880 tokens from high-definition images, creating inference bottlenecks. To address this, research has focused on token compression and pruning techniques in Vision Transformers [10] and MLLMs [27]. Methods like LLaMA-VID [40] and DeCo [88] address this by modifying models and adding training, which increases computational costs. ToMe [11] reduces tokens without training but disrupts early cross-modal interactions [81]. LLaVA-PruMerge [64] selectively retains key tokens while merging less critical ones based on key similarity. FasterVLM [91] utilizes [CLS] attention scores from the visual encoder to re-rank and retain top visual tokens. FastV [13] and SparseVLM [96] focus on token selection using attention scores or cross-modal guidance, but overlook the role of token duplication and lack Flash-Attention [16, 15]. Our proposed HoloV maintains hard acceleration compatibility (e.g., Flash-Attention), and effectively retains visual holistic context during aggressive pruning.

> 💡 **2.3 要点 — 方法对比矩阵**:
> | 方法 | 需要训练? | Flash-Attention 兼容? | 策略 |
> |------|----------|---------------------|------|
> | LLaMA-VID, DeCo | ✅ 是 | — | 修改模型架构 |
> | ToMe | ❌ 否 | — | Token 合并（但破坏早期交互） |
> | LLaVA-PruMerge | ❌ 否 | — | 剪枝+合并（基于 key 相似度） |
> | FasterVLM | ❌ 否 | ✅ 是 | [CLS] attention 排序 |
> | FastV, SparseVLM | ❌ 否 | ❌ 否 | Attention 分数选择 |
> | **HoloV** | ❌ 否 | ✅ 是 | Crop-wise 自适应分配 |
>
> HoloV 的两大差异化优势：(1) Flash-Attention 兼容（因为在 LLM 之前剪枝）; (2) 保留全局上下文而非局部显著性。

---

## 🔖 Section 总结

### 核心洞察
1. 视觉 token 压缩有两条路线：vision-centric（无语言偏差）和 instruction-centric（指令感知但有偏差）
2. 现有方法的共性缺陷：在高剪枝率下忽略全局语义关系
3. HoloV 创新点：在 ViT 编码器之后、LLM 之前进行剪枝，兼容 Flash-Attention，且通过 crop 机制保留空间多样性
