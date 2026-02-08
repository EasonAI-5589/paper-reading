# 3 Training Data

> 来源: RoboBrain 2.0 Technical Report (Arxiv 2507.02029)

---

## 📄 原文

As shown in Figure 4, RoboBrain 2.0 is trained on a diverse and extensive dataset designed to enhance its capabilities in spatial understanding, temporal modeling and long-chain causal reasoning in embodied settings. The training data encompasses a wide range of modalities, including high-resolution images, multi-view inputs, video sequences, scene graph and natural language instructions. This comprehensive dataset is meticulously categorized into three primary types: general multimodal understanding, spatial perception, and temporal modeling, ensuring the model can effectively perceive, reason, and plan in complex physical environments.

> 💡 **数据三大类**:
> ```
> ├── General MLLM VQA (通用多模态)
> ├── Spatial Data (空间感知)
> └── Temporal Data (时序建模)
> ```

![](../images/b38f5604013974ce27e5ef9ea43fa84c8219f8609d0b9f858c1c4d9ebe7f5eed.jpg)
*Figure 4: Training Data Distribution for RoboBrain 2.0. 三大类数据分布：general multimodal understanding, spatial perception, temporal modeling.*

> 💡 **Figure 4 批读**: 饼图展示了数据分布。Spatial data 占比最大（因为有 pointing/affordance/spatial understanding/spatial referring 多个子任务），Temporal data 次之。

---

### 3.1 General MLLM VQA

> 💡 **3.1 要点预览**: 通用 VQA 数据 873K samples，来自 LLaVA-665K + LRV-400K。不是新数据，是已有数据的清洗和整合。

High Quality Data. The general training dataset for RoboBrain 2.0 includes 873K high-quality samples, primarily derived from LLaVA-665K [33] and LRV-400K [32], spanning standard Visual Question Answering(VQA), region-level queries, OCR-based VQA, and visual dialogues. (1) LLaVA-665K serves as the primary source and contains diverse VQA-style data, including standard VQA datasets, OCR-based questions, region-level queries, visual conversations, and language-only dialogues. To improve training efficiency, multiple question-answer(QA) pairs from the same image merge into single conversations; invalid ShareGPT [10] entries are filtered out, and overly long conversations (> 2048 tokens) are truncated (resulting in 40K valid samples). Specifically, A-OKVQA [54] samples are augmented by duplicating choices to balance multiple-choice formats, OCR-VQA [41] contributes 80K sampled conversations focused on scene text understanding, Visual Genome(VG) [27] provides dense object-level annotations limited to 10 entries per image with additional captions, and RefCOCO [76] dialogues are split into short multi-turn segments (< 10 exchanges). Language-only conversations, which are generally longer than visual ones, are sampled in single-modality batches to improve throughput by 25% without performance degradation. After removing bounding-box-dependent QA pairs, 531K high-quality samples are retained from this source. (2) LRV-400K is synthetically generated using GPT-4 [44] under a few-shot instruction-following setting. It produces 400K image-conditioned instructions across 16 vision-language tasks with textual answers. Unlike prior works that rely on sparse image captions, this dataset leverages the dense annotations in VG (e.g., bounding boxes, dimensions, and ~21 object regions per image). GPT-4 generates both declarative and interrogative prompts for each image, with 10 tasks randomly sampled per instance. After filtering out bounding-box-related QA pairs, 342K samples are selected for training.

> 💡 **3.1 数据构成**:
> ```
> LLaVA-665K → 清洗后 531K:
> ├── A-OKVQA: 选择题格式平衡
> ├── OCR-VQA: 80K 场景文字
> ├── Visual Genome: 每图 ≤10 object annotations
> ├── RefCOCO: 短多轮对话 (<10 exchanges)
> └── ShareGPT: 过滤无效 + 截断 >2048 tokens
>
> LRV-400K → 清洗后 342K:
> └── GPT-4 生成, 16 个 VL 任务, dense VG annotations
>
> 总计: 531K + 342K = 873K
> ```
> **注意**: 去掉了所有 bounding-box-dependent QA pairs。为什么？可能是为了避免和后面 Spatial Data 的 grounding 数据冲突。
>
> **效率优化亮点**: language-only 对话用 single-modality batches，throughput 提升 25%。

---

### 3.2 Spatial Data

> 💡 **3.2 要点预览**: 五个子任务的空间数据，这是论文最重要的数据贡献。

**Visual Grounding.** The visual grounding dataset is constructed to enhance multimodal understanding through precise object-level localization, leveraging the extensive annotations from LVIS [19]. We carefully curate 152K high-resolution images from LVIS, ensuring broad coverage of diverse object categories and complex visual scenes. Each object annotation is converted into standardized bounding box coordinates $(x_1, y_1, x_2, y_2)$ representing the top-left and bottom-right corners, enabling consistent spatial referencing. To facilitate rich visual dialogue, we generated 86K conversational sequences, each containing multiple rounds of QA pairs that progressively explore visual relationships, attribute reasoning, and contextual understanding. The dataset maintains a balanced distribution across object categories while preserving challenging cases of occlusion, viewpoint variation, and rare instances to support robust visual grounding.

> 💡 **Visual Grounding**: 152K images from LVIS → 86K conversations, bbox 格式 $(x_1,y_1,x_2,y_2)$

**Object Pointing.** The object pointing dataset is constructed to enable RoboBrain 2.0 to identify the locations of specified objects through pointing within an image. We leverage the Pixmo-Points [13] dataset, which includes 2.3M point annotations across 223K images as our data source. However, direct utilization of Pixmo-Points data for RoboBrain 2.0 training presents challenges due to densely repeated object instances (e.g., books on a shelf). To address this, we implement a two-step filtering process: (1) we discard annotations with more than ten labeled points to simplify training, and (2) we use GPT-4o [22] as a scene analyzer to select only indoor-relevant objects, such as kitchenware, furniture, and decorations, excluding irrelevant or outdoor scenes. This process yields 190K QA pairs for 64K images with reduced clutter, making the data more suitable for embodied contexts. To construct QA pairs for pointing tasks, we construct 28 human-designed templates, such as "Point out all instances of {label} in the image." or "Help me find {label} in the image by pointing to them." Here, {label} refers to object categories from the annotations. Templates are randomly selected to ensure linguistic diversity and improve the model's generalization ability in referencing tasks. For object reference pointing, we incorporate object reference data sourced from RoboPoint [77], which includes 347K QA annotations across 288K images. To address the potential issue of excessive points hindering training convergence, we randomly sample up to ten points per question. Additionally, the normalized coordinates are converted into absolute values to better support RoboBrain 2.0 training.

> 💡 **Object Pointing**:
> ```
> Pixmo-Points: 2.3M points, 223K images
> ├── 过滤: >10 points 的丢弃 + GPT-4o 过滤非室内场景
> └── 结果: 190K QA, 64K images
>
> RoboPoint: 347K QA, 288K images
> ├── 过滤: ≤10 points per question
> └── 归一化坐标 → 绝对坐标
>
> 28 个 human-designed templates 增加语言多样性
> ```
> **关键设计**: 限制 ≤10 points/question 防止训练不收敛。用 GPT-4o 做场景过滤很聪明——只保留 indoor 相关物体。

**Affordance.** The affordance dataset focuses on understanding object functionality and spatial vacant areas for placement. For object affordance recognition, we utilize part-level annotations from PACO-LVIS [51], covering 75 object categories and 200 part categories across 46K images. Bounding boxes and segmentation masks are extracted for both whole objects and their functional parts. These annotations are transformed into bounding box coordinates $(x_1, y_1, x_2, y_2)$, serving as ground truth labels for affordance prediction tasks. Questions are constructed using GPT-4o [22] to query object functionality and part usage, e.g., "Which part of a handbag can be grasped to carry it?" for the handle of a handbag. For whole-object affordances, questions avoid naming the object directly, such as "What device can be moved to control the cursor on a screen?" for a mouse (computer equipment). This automatic process results in 561K QA pairs. For spatial affordance learning, we include region reference data from RoboPoint [77]. This dataset consists of 270K images with 320K QA pairs and 14 spatial relationship labels. Each annotation is converted into a set of absolute coordinates $[(x_1, y_1), (x_2, y_2), ...]$, and ground truth points are resampled to a maximum of ten points per answer for optimization. This dataset enables RoboBrain 2.0 to reason about spatial affordances for object placement in real-world settings.

> 💡 **Affordance** (两部分):
> ```
> Object Affordance:
> ├── PACO-LVIS: 75 object + 200 part categories, 46K images
> ├── GPT-4o 生成 QA → 561K pairs
> └── 巧妙设计: 提问时不直接说物体名（"什么设备可以控制光标？" 而非 "鼠标的哪个部分..."）
>
> Spatial Affordance:
> ├── RoboPoint: 270K images, 320K QA, 14 spatial relationship labels
> └── ≤10 points/answer
> ```

**Spatial Understanding.** To enhance RoboBrain 2.0's 3D spatial reasoning, we present the Spatial Understanding Dataset, comprising 826K samples. This dataset emphasizes object-centric spatial attributes (e.g., position, orientation) and inter-object relations (e.g., distance, direction), covering both qualitative and quantitative aspects. It covers 31 distinct spatial concepts, substantially surpassing the ~15 typically found in previous datasets. We partially adopt the RefSpatial [81] pipeline to construct 2D web image and 3D video datasets via automated template- and LLM-based generation: (1) 2D web images aim to provide core spatial concepts and depth perception across diverse indoor and outdoor scenes. To bridge scale and category gaps between these domains, we utilize the large-scale OpenImage [28] dataset. Since direct 3D reasoning from 2D images is challenging, we convert them into pseudo-3D scene graphs. Specifically, after filtering 1.7M images to 466K, we first use RAM [79] for object category prediction and GroundingDINO [34] for 2D boxes Detection. Then we enhance using Qwen2.5-VL [50] and a heuristic method to generate hierarchical captions given the 2D bounding box, ranging from coarse (e.g., "cup") to fine-grained (e.g., "the third cup from the left"). This enables unambiguous spatial referring in cluttered environments and captures both coarse and fine-grained spatial references. Next, we use UniDepth V2 [48] and WildeCamera [84] for depth and camera intrinsics to enable 3D point cloud reconstruction. Finally, combining this with object boxes from GroundingDINO [34] and masks from SAM 2.1 [52], each scene graph includes object labels, 2D boxes, instance masks, and object-level point clouds, yielding axis-aligned 3D boxes. Object captions serve as nodes, and spatial relations form the edges. QA pairs are generated via templates and LLMs (e.g., QwQ [66]), including object-location questions derived from the hierarchical captions. (2) 3D scene-based videos integrates multimodal 3D scene understanding data from five original datasets: MMScan [38], 3RScan [69], ScanQA [3], SQA3D [39], and SpaceR [46]. We conduct template-based question filtering through rigorous data processing to ensure task relevance, perform multi-stage quality screening (e.g., consistency checks, outlier removal), and standardize all formats into a unified representation. This curation enables fine-grained environmental perception with enhanced reliability, supporting tasks ranging from object localization to complex spatial reasoning in 3D scenes. (3) 3D embodied videos focus on fine-grained spatial understanding in indoor environments. We leverage the CA-1M [29] dataset, filtering 2M frames to 100K high-quality ones. Compared to 2D, the availability of accurate 3D bounding boxes allows us to construct richer scene graphs with more diverse spatial relations, thereby generating more quantitative QA pairs (e.g., size, distances).

> 💡 **Spatial Understanding (826K samples, 31 concepts)** — 最复杂的数据 pipeline:
> ```
> (1) 2D Web Images (OpenImage):
>     1.7M → 466K images
>     Pipeline: RAM → GroundingDINO → Qwen2.5-VL hierarchical captions
>              → UniDepth V2 + WildeCamera → 3D point cloud
>              → SAM 2.1 masks → pseudo-3D scene graph
>     QA: templates + QwQ (LLM)
>
> (2) 3D Scene Videos:
>     5 datasets: MMScan, 3RScan, ScanQA, SQA3D, SpaceR
>     统一格式 + 多阶段质量筛选
>
> (3) 3D Embodied Videos (CA-1M):
>     2M frames → 100K high-quality
>     真 3D bbox → 更丰富的空间关系
> ```
> **核心创新**: 用 2D 图像重建 pseudo-3D scene graph 的 pipeline 很巧妙——组合了 RAM + GroundingDINO + depth estimation + SAM 2.1，全自动化。
>
> **31 个 spatial concepts vs 之前的 ~15**：翻倍，这是一个重要的数据贡献。

**Spatial Referring.** After enhancing foundational 3D spatial understanding, we extend these capabilities to physical-world interactions by introducing the Spatial Referring Dataset [81], consisting of 802K samples. Unlike prior datasets in visual grounding or object pointing, which often deal with ambiguous or multiple referents, this dataset targets a single unambiguous target, aligning with robotic applications such as precise pick-and-place that demand accurate object identification and localization. Following the RefSpatial [81] construction pipeline, for location data, we sample caption-point pairs from scene graphs built on 2D web images (OpenImage [28]) and 3D embodied videos (CA-1M [29]), using hierarchical captions. For placement data, we leverage fully annotated 3D datasets to generate top-down occupancy maps encoding object positions, orientations, and metric spatial relations (e.g., "10cm right of the chair"), facilitating accurate spatial referring.

> 💡 **Spatial Referring (802K samples)**:
> - 关键区别: **单一无歧义目标** (vs grounding/pointing 可能有多个)
> - 适合 robot pick-and-place: 需要精确定位单个物体
> - 数据来源: 复用 Spatial Understanding 的 scene graphs + RefSpatial pipeline

---

### 3.3 Temporal Data

> 💡 **3.3 要点预览**: 六类时序数据，涵盖 ego-view planning、ShareRobot、AgiBot、multi-robot、close-loop interaction。

**Ego-View Planning.** We construct Ego-View Planning dataset by partially processing the EgoPlan-IT [9] dataset, which contains 50K automatically generated samples. For each selected task instance, we extract multiple frames from prior actions to represent task progress, and one frame to capture the current viewpoint. To enhance linguistic variety, we use multiple prompt templates that describe the task goal, video context, and current observation. Each question includes the correct next action along with up to three distractor actions randomly sampled from negative examples. This setup supports multimodal instruction tuning with diverse visual and textual input, aimed at improving egocentric task planning performance.

> 💡 **Ego-View Planning**: 50K samples from EgoPlan-IT，多选题格式（正确动作 + 3 个干扰项）

**ShareRobot Planning.** The ShareRobot dataset [23] is a large-scale, fine-grained resource for robotic manipulation, offering multi-dimensional annotations tailored for task planning. Its planning component provides detailed low-level instructions aligned with individual video frames, effectively transforming high-level task descriptions into structured and executable sub-tasks. Each data instance includes precise planning annotations to support accurate and consistent task execution. The dataset comprises 1M QA pairs from 51K instances, spanning 102 diverse scenes across 12 robot embodiments and 107 atomic tasks filtered according to the Open-X-Embodiment taxonomy [47]. All planning data were meticulously annotated by human experts following the RoboVQA [55] format, enabling models to learn robust multi-step planning strategies grounded in diverse real-world scenarios. The scale, quality, and diversity of ShareRobot help improve the model's ability to perform fine-grained reasoning and task decomposition in complex embodied environments.

> 💡 **ShareRobot**: 和 RoboBrain 1.0 一样的数据集！1M QA, 51K instances, 12 robots, 107 tasks。这是 1.0 的核心贡献，在 2.0 中作为 temporal data 的一部分继续使用。

**Agitbot Planning.** The AgiBot Planning dataset is a large-scale robotics task planning dataset built upon the AgiBot-World [6] dataset, comprising 9,148 QA pairs across 19 manipulation tasks with 109,378 first-person perspective images. Each sample contains 4-17 consecutive frames documenting task progression with multimodal conversational format. AgiBot-Planning provides step-by-step planning instructions that transform high-level goals into executable sub-tasks. Each data point includes current objectives, historical steps, and required subsequent actions. The dataset covers diverse scenarios from household refrigerator operations to supermarket shopping tasks across different environments. The meticulously crafted annotations use standardized conversational formats, enabling models to learn from varied real-world contexts. Through continuous visual sequences and fine-grained action plans, AgiBot-Planning enhances RoboBrain 2.0's ability to perform long-horizon task planning and spatial reasoning in complex embodied scenarios.

> 💡 **AgiBot Planning**: 9,148 QA, 19 tasks, 109K images。规模比 ShareRobot 小很多，但有连续视频帧 (4-17 frames/sample)，适合 long-horizon planning。

**Multi-Robot Planning.** The Multi-Robot Planning dataset is constructed by simulating collaborative task scenarios across three environments—household, supermarket, and restaurant—based on RoboOS [61]. Each sample is generated using structured templates that specify a detailed scene graph, robot specifications, and associated tool lists. For every scenario, we design high-level, long-horizon collaborative task goals that require coordination among multiple robots present in the scene, and generate corresponding workflow graphs that decompose the tasks into subtasks with detailed reasoning explanations. Based on these decompositions, we further generate agent-specific robotic tool plans that translate high-level task goals into precise low-level Observation-Action pairs for each subtask. Specifically, we define 1,659 types of multi-robot collaboration tasks across the three environments and produce 44,142 samples using DeepSeek-V3 [31].

> 💡 **Multi-Robot Planning**: 
> ```
> 3 environments: household, supermarket, restaurant
> 1,659 task types → 44,142 samples
> 生成: DeepSeek-V3 (不是人工标注)
> 格式: scene graph + robot specs + tool lists → workflow graph → OA pairs
> ```
> **关键**: 这是 RoboOS 体系的数据。用 DeepSeek-V3 生成，不是人工标注——效率高但质量需要验证。

**Close-Loop Interaction.** The Close-Loop Interaction dataset is designed to facilitate advanced embodied reasoning [80], featuring a large-scale collection of synthesized Observation-Thought-Action (OTA) trajectories that combine first-person visual observations with structured thought tokens. It spans 120 diverse indoor environments—including kitchens, bathrooms, bedrooms, and living rooms—containing over 4,000 interactive objects and receptacles. The dataset is constructed within the AI2Thor [25] simulator through a rigorous multi-stage pipeline based on Embodied-Reasoner [78], which includes: (1) crafting task instructions from constrained templates to ensure scene-appropriate validity; (2) deriving key action sequences from an object-affiliation graph encoding functional relationships; and (3) strategically incorporating search actions to emulate realistic exploration. To enrich the depth of reasoning, GPT-4o generates detailed thought processes—covering situational analysis, spatial reasoning, self-reflection, task planning, and verification—which are seamlessly integrated between observations and actions, forming coherent reasoning chains that guide models through complex, long-horizon interactive tasks.

> 💡 **Close-Loop Interaction**:
> ```
> 环境: AI2Thor simulator, 120 indoor scenes, 4000+ objects
> 格式: Observation-Thought-Action (OTA) trajectories
> Pipeline:
> ├── (1) 模板生成 task instructions
> ├── (2) Object-affiliation graph → action sequences
> └── (3) GPT-4o 生成 thought processes (situational analysis + spatial reasoning + self-reflection)
> ```
> **评价**: OTA 格式很好——比单纯的 OA (Observation-Action) 多了 Thought，能让模型学到推理过程。这对 Stage 3 的 CoT 训练很重要。

---

## 💡 Section 总结

### 数据规模汇总
| 数据类别 | 子任务 | 规模 |
|----------|--------|------|
| General VQA | LLaVA + LRV | 873K |
| Spatial - Grounding | LVIS | 86K conv |
| Spatial - Pointing | Pixmo-Points + RoboPoint | 190K + 347K QA |
| Spatial - Affordance | PACO-LVIS + RoboPoint | 561K + 320K QA |
| Spatial - Understanding | OpenImage + 3D scenes + CA-1M | 826K |
| Spatial - Referring | RefSpatial pipeline | 802K |
| Temporal - Ego-View | EgoPlan-IT | 50K |
| Temporal - ShareRobot | ShareRobot | 1M QA |
| Temporal - AgiBot | AgiBot-World | 9.1K QA |
| Temporal - Multi-Robot | RoboOS + DeepSeek-V3 | 44.1K |
| Temporal - Close-Loop | AI2Thor + GPT-4o | OTA trajectories |

### 核心洞察
1. **Spatial 数据是最大亮点**: 826K spatial understanding + 802K spatial referring，31 个 spatial concepts
2. **数据工程是核心贡献**: 大量使用 GPT-4o/DeepSeek-V3/QwQ 自动生成 + pipeline 化
3. **ShareRobot 延续**: 1.0 的核心数据在 2.0 中依然重要
4. **Close-Loop OTA 格式**: Observation-Thought-Action 比传统 OA 更丰富，为 CoT 训练提供基础
