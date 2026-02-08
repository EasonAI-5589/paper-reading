[← 返回 README](../README.md)

# 7 Applications

## 📌 预览
Token 压缩在四大应用领域的实际价值：GUI Agent、医疗影像、机器人/自动驾驶、高效推理。

---

The potential of multimodal token compression extends beyond technical enhancements, emerging as a universal efficiency engine for data-intensive AI systems. Multimodal models frequently process extreme-length token sequences exhibiting high task-agnostic redundancy according to empirical analyses. Capitalizing on recent breakthroughs, we delineate four high-impact application domains:

> 💡 Token 压缩不只是学术研究——在实际应用中，它是处理海量多模态数据的必要条件。

---

## 7.1 GUI Agents and Human-Computer Interaction

Graphical user interface (GUI) agents perceive and interact with visual interfaces, interpret natural language instructions, analyze GUI states, and execute corresponding actions. These agents have to parse screen streams in real-time, producing extensive token sequences that often exceed computational limits (Zhang et al., 2024b; Wang et al., 2024a). Multimodal token compression enhances the efficiency of GUI agents. This approach mitigates context overflow in extended operation sequences by dynamically compressing redundant visual elements (e.g., extra white space or simple backgrounds). For some small but important control elements, it should also eliminate other irrelevant visual elements and highlight their importance. For instance, ShowUI (Lin et al., 2025a) is the first model to apply token selection strategy to GUI agents. ShowUI segments GUI screenshots into connected components by clustering pixels with similar RGB values, significantly reducing the total number of discrete elements. During both training and inference phases, the system employs an adaptive token selection strategy that probabilistically prunes redundant tokens within these components, thereby optimizing computational efficiency while preserving functional semantics However, excessive compression risks inducing operational ambiguity, necessitating careful calibration.

> 💡 **GUI Agent 应用**:
> - GUI 截图中大量空白/简单背景是冗余的，但小控件（按钮等）信息密度高
> - **ShowUI**: 首个在 GUI Agent 中用 token 选择策略的模型——RGB 聚类分割连通区域 → 自适应剪枝冗余 token
> - 风险：过度压缩可能导致操作歧义（丢失小但关键的 UI 元素）

---

## 7.2 Healthcare and Medical Imaging

The effective synthesis of multimodal medical data is pivotal to advancing contemporary medical diagnosis and research. MLLMs can integrate radiographic findings, medical histories, and ancillary diagnostic tests to generate differential diagnoses, which clinicians can correlate with patient records and physician notes to enhance diagnostic accuracy (Liang et al., 2024). Furthermore, MLLMs can automatically draft preliminary radiology reports, potentially reducing the workload of radiologists (Beddiar & Oussalah, 2023; Bazi et al., 2023; He et al., 2020). A major challenge for MLLMs in medical imaging is the processing of high-resolution images, such as Whole-slide Images (WSIs) in pathology, which can contain billions of pixels. To overcome this, TCP-LLaVA (Lyu et al., 2025) uses a set of trainable compression tokens to aggregate and condense crucial information from thousands of visual and textual inputs. Instead of feeding every single image patch token into the language model, only these compressed tokens are forwarded for answer generation. Token compression holds vast potential for widespread application in the field of healthcare and medical imaging, enabling the efficient analysis of complex, high-resolution images.

> 💡 **医疗影像应用**:
> - 病理 WSI 可达数十亿像素 → 必须压缩才能用 MLLM 处理
> - **TCP-LLaVA**: trainable compression tokens 聚合数千个视觉+文本 token → 只送压缩 token 给 LLM
> - 医疗场景对 token 压缩的需求是刚性的（不压缩根本无法运行）

---

## 7.3 Robotics and Autonomous Systems

Leveraging the significant capabilities of video LLMs in long-form video comprehension enables their deployment in robotics (Wei et al., 2025) and autonomous driving systems (Ma et al., 2024b; Zhou et al., 2024; Zhu et al., 2025b). However, the inherent computational complexity of long-duration video processing creates fundamental latency-efficiency tradeoffs that challenge real-time implementation. Token compression addresses this by prioritizing salient spatio-temporal dynamics (e.g., agent movements, action trajectories) and fine-grained per-frame details, enabling computationally efficient video understanding for these domains. VTS (Ma et al., 2024b) proposes a token pruning strategy for autonomous driving scenarios. VTS employs a proposal model based on a lightweight convolutional neural network that is able to adaptively identify keyframes and pry less informative tokens (e.g., invariant backgrounds and stationary objects). StreamVLN (Wei et al., 2025) further enhances inference efficiency for real-time navigation by employing a voxel-based spatial pruning strategy at test time to reduce memory tokens. This approach makes real-time navigation feasible.

> 💡 **机器人/自动驾驶应用**:
> - 实时性要求极高 → 延迟-效率权衡是核心矛盾
> - **VTS**: 轻量 CNN 识别关键帧 + 剪枝不变背景/静止物体
> - **StreamVLN**: 体素空间剪枝策略减少 memory token → 实现实时导航

---

## 7.4 Efficient Reasoning

Token compression improves efficiency by removing redundant input tokens. However, in many cases, the main source of computational cost shifts from input to output, most notably in reasoning models (Team et al., 2025; Guo et al., 2025a; Jaech et al., 2024), where lengthy generation chains are common. The "slowthinking" paradigm improves reasoning ability but results in lengthy reasoning chains (Feng et al., 2025a; Sui et al., 2025; Chen et al., 2025; Feng et al., 2025c;b). Some efficient reasoning methods compress these chains using similar techniques (e.g., attention mechanisms, semantic importance) (Ma et al., 2025a; Xia et al., 2025; Liu et al., 2024b; Fang et al., 2025), typically requiring fine-tuning via Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL). Beyond token compression, other approaches improve reasoning efficiency by compressing model (Magister et al., 2022; Li et al., 2023b; Feng et al., 2024; Zhang et al., 2025e) or accelerating decoding (Sun et al., 2024c; Ma et al., 2024a; Luo et al., 2025a; Xu et al., 2025a; Ding et al., 2025).

> 💡 **高效推理应用**:
> - 推理模型（如 DeepSeek-R1, o1）的瓶颈从输入转向输出——推理链很长
> - Token 压缩思路可以迁移到输出侧：压缩 CoT 推理链
> - 方法：attention-based 选择关键推理步骤、语义重要性筛选
> - 需要 SFT/RL 微调（不像输入压缩可以 training-free）

---

## 🔖 Section 总结

### 四大应用领域速查
| 领域 | 核心需求 | 代表方法 |
|------|---------|---------|
| GUI Agent | 实时屏幕流解析 | ShowUI |
| 医疗影像 | 超高分辨率图像处理 | TCP-LLaVA |
| 机器人/自动驾驶 | 低延迟实时视频理解 | VTS, StreamVLN |
| 高效推理 | 压缩输出推理链 | CoT-Valve |
