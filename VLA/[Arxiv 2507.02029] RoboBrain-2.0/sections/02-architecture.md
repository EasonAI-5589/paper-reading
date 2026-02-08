# 2 Architecture

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 📄 原文

RoboBrain 2.0 employs a modular encoder-decoder architecture that unifies perception, reasoning, and planning for complex embodied tasks. As shown in Figure 3, it processes multi-view visual observations and natural language instructions through four core components: (1) a tokenizer for textual/structured inputs, (2) a vision encoder, (3) an MLP projector mapping visual features to the language model's token space, and (4) a language model backbone initialized from Qwen2.5-VL [5]. Unlike conventional VLMs [2, 22] focused on general static VQA, RoboBrain 2.0 maintains strong general VQA capabilities while specializing in embodied reasoning tasks like spatial perception, temporal modeling, and long-chain causal reasoning. The architecture encodes high-resolution images, multi-view inputs, video frames, language instructions, and scene graphs into a unified multimodal token sequence for comprehensive processing.

> 💡 **架构概览**:
> ```
> 四大组件:
> ├── (1) Tokenizer: 文本 + 结构化输入 (scene graph)
> ├── (2) Vision Encoder: ~689M params, 处理图像/视频
> ├── (3) MLP Projector: 视觉特征 → LLM token space
> └── (4) LLM Decoder: Qwen2.5-VL 7B/32B
> ```
> **关键**: 基于 Qwen2.5-VL 初始化，不是从头训练。这和 1.0 用 Qwen2.5-7B + LLaVA 架构不同——2.0 直接用了 Qwen2.5-VL 这个已经训好的 VLM。

![](../images/93df68325fc81b8640962a5376b2650b4c748c881b59d7ca7d573614181eda36.jpg)
*Figure 3: The Architecture of RoboBrain 2.0. 支持 multi-image, long video, high-resolution visual inputs + task instructions + structured scene graphs. 所有输入统一为 token sequence 输入 LLM decoder.*

> 💡 **Figure 3 批读**:
> ```
> 输入侧:
> ├── Visual: multi-view images / video frames
> │   → Vision Encoder → MLP Projector → visual tokens
> ├── Text: task instructions
> │   → Tokenizer → text tokens
> └── Structured: scene graphs (JSON)
>     → Tokenizer → structured tokens
>
> 处理:
> └── LLM Decoder (Qwen2.5-VL): unified token sequence → CoT reasoning
>
> 输出:
> ├── Free-form text (plans, dialogue)
> ├── Spatial coordinates (points, bboxes, trajectories)
> └── Reasoning traces (CoT, optional)
> ```

---

### 2.1 Input Modalities and Tokenization

RoboBrain 2.0 supports a diverse set of input modalities tailored for embodied AI tasks:

• Language instructions: Natural language commands describing high-level goals or low-level actions. RoboBrain 2.0 processes natural language commands spanning different abstraction levels: from high-level, spatially grounded instructions (e.g., "Carry the apple to the nearest table, aligned with the leftmost cup") to low-level motor commands (e.g., "Navigate to the nearest table", "Grasp the apple", "Detect position aligned with the leftmost cup", "Place the apple into the box").

> 💡 **指令粒度**: 从 high-level goal ("把苹果搬到桌子上") 到 low-level motor commands ("抓取苹果")，支持多层次。

• Scene graph: A structured JSON representation of the explored environment, containing information about discovered objects, their categories, spatial locations, and embodiment configuration (e.g., name: KitchenTable1, type: table, object: [basket, knife], robot: RealMan-single-arm).

> 💡 **Scene Graph 输入**: 这是 2.0 的一个新特性。1.0 没有 scene graph 输入。JSON 格式包含物体、类别、位置、机器人配置——这对 multi-robot planning 至关重要。

• Multi-view static images: Images captured from multiple viewpoints, such as head-mounted cameras, wrist-mounted cameras, or multi-view projections from a 3D environment. These are processed independently by the vision encoder and concatenated into a unified token sequence.

• Video frames: Video sequences (e.g., egocentric views from the agent), optionally annotated with timestamp tokens [5] to facilitate temporal grounding and reasoning.

> 💡 **视觉输入多样性**: multi-view + video + 单图，覆盖了 embodied 场景的主要视觉需求。timestamp tokens 来自 Qwen2.5-VL 的设计。

Language instructions and scene graphs are tokenized using the language tokenizer. Visual inputs—including multi-view images and video frames—are processed by the vision encoder into dense visual embeddings, which are then projected into the LLM's token space through an MLP projector, enabling unified multi-modal reasoning within the decoder.

---

### 2.2 Vision Encoder and Projection

RoboBrain 2.0 vision encoder supports dynamic-resolution image and video inputs through adaptive positional encoding and windowed attention mechanisms [5]. This design choice enables efficient processing of high-resolution and multi-view visual observations common in embodied tasks.

> 💡 **直接继承 Qwen2.5-VL 的 vision encoder**: dynamic resolution + windowed attention，不是自研的。

To accommodate the long-horizon and temporally grounded nature of such tasks, we adopt frame-wise visual tokenization with multi-dimensional RoPE [5] for spatiotemporal encoding. Each visual embedding is projected via a lightweight MLP into the token space of the language model. For multi-view scenarios, visual tokens from different camera perspectives are serialized and augmented with view-specific positional identifiers before being fused with other input modalities.

> 💡 **多视角处理**: 每个视角的 visual tokens 加上 view-specific positional identifiers，然后串联。这是一个简单但有效的方案——没有用复杂的 cross-view attention。

---

### 2.3 LLM Decoder and Output Representations

RoboBrain 2.0 employs a decoder-only language model designed to unify high-level reasoning and spatially grounded output generation. Unlike conventional VLMs that primarily return short-form answers to static prompts, RoboBrain 2.0 flexibly supports both concise responses and multi-step chain-of-thought reasoning. This capability enables deeper understanding of complex instructions and physical scenes.

To enable the decoder to handle embodied tasks, the decoder is trained to produce a diverse range of outputs, including semantically grounded expressions (e.g., referring to objects or actions), spatial coordinates (e.g., absolute positions or bounding boxes), and intermediate reasoning traces. Rotary positional encodings and temporally conditioned tokens allow the model to maintain coherence across multi-round perception-action loops, which are essential for long-horizon planning in dynamic environments. Output formats supported by RoboBrain 2.0 include: (1) Free-form text: Used for task decomposition, scene graph updates, agent invocation, and human-agent dialogue. (2) Spatial coordinates: Used to represent point locations, bounding boxes, or trajectories in the image space for downstream controllers. (3) Reasoning traces (Optional): Long-chain-of-thought explanations to support deep problem solving and decision transparency.

> 💡 **三种输出格式**:
> ```
> (1) Free-form text → planning, dialogue, scene graph update
> (2) Spatial coordinates → points, bboxes, trajectories
> (3) Reasoning traces → CoT (可选)
> ```
> **关键**: 坐标输出是 absolute coordinates (绝对像素坐标)，不是归一化坐标。这对 downstream robot control 更友好。

This unified decoding formulation allows RoboBrain 2.0 to effectively handle a wide range of embodied tasks, from spatial grounding and visual understanding to long-horizon multi-agent planning and causal reasoning.

---

## 💡 Section 总结

### 架构特点
| 组件 | 设计 | 来源 |
|------|------|------|
| Vision Encoder | ~689M, dynamic resolution, windowed attention | Qwen2.5-VL |
| Projector | Lightweight MLP | 标准设计 |
| LLM Decoder | 7B / 32B, decoder-only | Qwen2.5-VL |
| Positional Encoding | Multi-dimensional RoPE | Qwen2.5-VL |

### 核心洞察
1. **架构不是创新点**: 基本就是 Qwen2.5-VL + scene graph 输入，没有结构性创新
2. **创新在于输入/输出的扩展**: scene graph 输入、multi-view 处理、坐标输出
3. **与 1.0 的区别**: 1.0 = LLaVA + LoRA；2.0 = Qwen2.5-VL full fine-tune
