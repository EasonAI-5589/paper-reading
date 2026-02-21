[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览

Related Work 从三个角度梳理：(1) MLLMs 的计算挑战；(2) 视觉冗余识别的两个方向；(3) 具体的 token 压缩和剪枝方法。

---

## 2.1 MLLMs and Their Challenges

The recent remarkable success of Large Language Models (LLMs) [60, 92, 70, 18, 54] has spurred the trend of applying their strong capabilities to multimodal comprehension tasks, fostering the development of MLLMs [1, 67]. Leveraging open-source LLMs such as LLaMA families [70, 71, 18], MLLMs [6, 46, 47] have demonstrated enhanced adaptability across a range of visual understanding tasks, leading to a more profound ability to interpret the world. While this empowers LLMs with the capability of visual perception, the incorporation of lengthy visual tokens significantly escalates the computational burdens. Moreover, studies have shown that existing MLLMs still suffer from certain visual deficiencies [69, 32] and some hallucinations [29, 28]. Some work mitigates these issues by increasing the resolution of input images or videos [53, 84], but this further exacerbates the computational overhead.

> 💡 **MLLMs 的两难**: 提高分辨率能改善视觉理解和减少幻觉，但代价是更多的 visual tokens → 更高的计算开销。这正是 token pruning 的存在意义。

For example, LLaVA-1.5 [48] encodes a 336-resolution image into 576 visual tokens, while LLaVA-NeXT [47] doubles the resolution and generates 2,880 tokens. LLaVA-OneVision [37] represents an image using 7,290 visual tokens, and Video-LLaVA [44] faces even higher costs, as it must process numerous visual tokens from multiple frames during inference. These visual tokens occupy a large portion of the context window of their LLMs. In this work, we conducted experiments and analysis on these representative models to verify HoloV's applicability.

> 💡 **Token 数量对比**:
> | 模型 | Token 数 | 倍数 |
> |------|---------|------|
> | LLaVA-1.5 | 576 | 1× |
> | LLaVA-NeXT | 2,880 | 5× |
> | LLaVA-OneVision | 7,290 | 12.7× |
> | Video-LLaVA | 更多（多帧） | >> |

---

## 2.2 Visual Redundancy Identification

In MLLMs, visual redundancy identification facilitates the distillation of visual tokens with high informativeness for faster inference. There are two main research directions: a) Vision-centric strategies analyze the image's structure and feature distribution to discard less relevant visual tokens [13, 75]. Existing approaches include spatial-similarity clustering (e.g., TokenLearner [63]), dynamic pruning based on attention scores [25, 86, 82], and using information bottleneck or entropy metrics during the prefilling stage to estimate background redundancy. b) Instruction-centric strategies typically use cross-modal attention analysis or gradient accumulation to identify redundant tokens [49, 98, 66]. Tokens with low attention or negligible gradient impact are deemed redundant [26]. Building on this, some studies explore learned importance scoring, training a lightweight end-to-end model to predict each patch's "instruction relevance," enabling even finer-grained pruning [31, 73, 88]. As the existence of language bias in LLM may cause hallucinations, we use a vision-centric scheme.

> 💡 **两大方向对比**:
> | 方向 | 信号来源 | 代表方法 | 优点 | 缺点 |
> |------|---------|---------|------|------|
> | Vision-centric | ViT 内部 attention/特征 | TokenLearner, FasterVLM | 不受文本偏置影响 | 不知道指令关注什么 |
> | Instruction-centric | LLM text-vision attention | FastV, MustDrop | 知道指令关注什么 | 受 LLM 语言偏置影响 |
> 
> HoloV 选择 vision-centric，理由是 LLM 的语言偏置会导致幻觉。这个选择和 CDPruner（instruction-centric）相反，是一个有趣的设计分歧。

---

## 2.3 Visual Token Compression and Pruning

The inclusion of visual information in MLLMs introduces long token sequences, leading to high computation and memory costs. For example, mini-Gemini-HD [41] generates 2880 tokens from high-definition images, creating inference bottlenecks. To address this, research has focused on token compression and pruning techniques in Vision Transformers [10] and MLLMs [27]. Methods like LLaMA-VID [40] and DeCo [87] address this by modifying models and adding training, which increases computational costs. ToMe [11] reduces tokens without training but disrupts early cross-modal interactions [81]. LLaVA-PruMerge [64] selectively retains key tokens while merging less critical ones based on key similarity. FasterVLM [90] utilizes [CLS] attention scores from the visual encoder to re-rank and retain top visual tokens. FastV [13] and SparseVLM [95] focus on token selection using attention scores or cross-modal guidance, but overlook the role of token duplication and lack Flash-Attention [16, 15]. Our proposed HoloV maintains hard acceleration compatibility (e.g., Flash-Attention), and effectively retains visual holistic context during aggressive pruning.

> 💡 **方法谱系梳理**:
> - **需要训练**: LLaMA-VID, DeCo → 改模型结构，计算成本高
> - **不需训练 (Training-free)**:
>   - ToMe: token merging，但破坏早期跨模态交互
>   - LLaVA-PruMerge: 保留关键 token + merge 次要 token
>   - FasterVLM: [CLS] attention 评分
>   - FastV / SparseVLM: attention-based 选择
> - **HoloV 的定位**: Training-free + 兼容 Flash-Attention + 在 LLM 前剪枝（更高效）

---

## 🔖 Section 总结

### 核心洞察
1. Token pruning 是解决 MLLMs 计算瓶颈的关键技术
2. Vision-centric vs Instruction-centric 是两条主线，各有优劣
3. HoloV 选择 vision-centric 路线，避免 LLM 语言偏置
4. 在 LLM 前剪枝比在 LLM 内部剪枝更高效（避免 KV cache 浪费）
