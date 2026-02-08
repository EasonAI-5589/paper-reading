# RoboBrain 2.0 Technical Report

# BAAI RoboBrain Team

Please see Contributions and Author List for more author details.

# Abstract

We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent longhorizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.

![](images/a808da69c76bee61e7c520fa20705382a73db1fa534e82b58996e4ca135aa768.jpg)  
Figure 1 Benchmark comparison across spatial and temporal reasoning. RoboBrain2.0-32B achieves best performance on both spatial and temporal reasoning benchmarks across BLINK-Spatial, RoboSpatial, RefSpatial-Bench, Where2Place, EgoPlan2 and Multi-Robot-Plan, outperforming prior open-source models and proprietary models.

# Contents

# 1 Introduction 3

# 2 Architecture 4

2.1 Input Modalities and Tokenization 5 5   
2.2 Vision Encoder and Projection   
2.3 LLM Decoder and Output Representations 6

# 3 Training Data 6

3.1 General MLLM VQA 6   
3.2 Spatial Data 7   
3.3 Temporal Data 8

# 4 Training Strategy 9

4.1 Stage 1: Foundational Spatiotemporal Learning 9   
4.2 Stage 2: Embodied Spatiotemporal Enhancement 10   
4.3 Stage 3: Chain-of-Thought Reasoning in Embodied Contexts 10

# Infrastructures 11

5.1 Large-Scale Training Infrastructure 11   
5.1.1 Multi-Dimensional Hybrid Parallelism 11   
5.1.2 Pre-Allocate Memory 11   
5.1.3 Data Pre-Processing 11   
5.1.4 Distributed Data Loading 1 2   
5.1.5 Fault Tolerance . 12   
5.2 Reinforcement Fine-Tuning Infrastructure 12   
5.3 Inference Infrastructure 1 2

# Evaluation Results 13

6.1 Spatial Reasoning Capability 13   
6.2 Temporal Reasoning Capability 15

# 7 Conclusion and Future Works 16

# 8 Contributions and Author List 22

# A Qualitative examples 23

A.1 Examples for Pointing 23   
A.2 Examples for Affordance 40   
A.3 Examples for Trajectory 42   
A.4 Examples for EgoPlan2 44   
A.5 Examples for Close-Loop Interaction 47   
A.6 Examples for Multi-Robot Planning 51   
A.7 Examples for Synthetic Benchmarks 52

# B Prompts Details 54

B.1 Spatial Understanding: Coordinates – Pointing 5 4   
B.2 Spatial Understanding: Coordinates – Trajectory 5 4   
B.3 Spatial Understanding: Bounding Box – Affordance 5 4   
B.4 Spatial Understanding: Freeform Q&A – General Spatial Analysis 55   
B.5 Temporal Understanding: Long-horizon Planning 55   
B.6 Temporal Understanding: Closed Loop Conversation 55   
B.7 Temporal Understanding: Multi-Robot Planning 55

# 1 Introduction

In recent years, large language models (LLMs) and vision-language models (VLMs) have emerged as key driving forces in the advancement of general artificial intelligence (AGI). Within digital environments, these models have demonstrated remarkable capabilities in perception [5, 16, 83], understanding [22, 73], and reasoning [2, 17, 18, 45, 65], and have been widely applied in tasks such as multimodal question answering [35, 60], image generation and editing [24, 57], GUI control [37, 71], and video understanding [7, 63, 72]. They have also seen early adoption in practical domains such as education, healthcare, search, and intelligent assistants [11, 21, 82].

However, bridging the gap between “digital intelligence” and “physical intelligence”—enabling models to perceive their surroundings, understand embodied tasks, and interact with the real world—remains a critical challenge on the path toward AGI. Embodied foundation models [4, 64, 74] represent a promising research direction toward physical intelligence. Several recent efforts have extended the capabilities of LLMs and VLMs to embodied scenarios, advancing multimodal fusion, perception, and action execution. While these models have achieved encouraging progress, they still face three fundamental capability bottlenecks when deployed in complex and open-ended real-world environments: (1) Limited spatial understanding: Current models struggle to accurately model relative and absolute spatial relationships and identify affordances in physical environments, which hinders real-world applicability; (2) Weak temporal modeling: The lack of understanding of multi-stage, cross-agent temporal dependencies and feedback mechanisms limits long-horizon planning and closed-loop control; (3) Insufficient reasoning chains: Existing models are often incapable of extracting causal logic from complex human instructions and aligning it with dynamic environmental states, restricting their generalization to open-ended embodied tasks.

To address these challenges, we present RoboBrain 2.0, our latest generation of embodied vision-language foundation models, tailored to bridge perception, reasoning, and planning in physically environments. RoboBrain 2.0 processes visual observations and language instructions in a unified architecture, enabling holistic understanding of the environment, goal-directed reasoning, and long-horizon planning. We release two variants of the model: the lightweight RoboBrain 2.0–7B and the full-scale RoboBrain 2.0–32B, designed to meet different deployment needs under varying resource constraints. On both spatial reasoning and temporal reasoning benchmarks, the 32B variant mostly achieves state-of-the-art performance, outperforming prior open-source and proprietary models, as shown in Figure 1. Model capabilities are summarized in Figure 2.

This report provides a systematic overview of the design principles, core components and key innovations. In particular, we highlight the extensive data contributions that support spatial understanding, temporal reasoning, and causal inference, which form the foundation of RoboBrain 2.0’s capabilities. To address the scarcity of spatial data, we develop a spatial data synthesis pipeline that constructs large-scale, high-quality datasets spanning tasks such as pointing, affordance prediction, and trajectory generation. To improve temporal reasoning and feedback modeling, we design multi-robot coordination templates across common scenarios via RoboOS [61], generate cross-agent long-horizon planning trajectories using external models [31], and simulate randomized failure events to collect closed-loop feedback data that enhances model robustness. To further enrich reasoning data, we extract step-by-step thought traces from powerful reasoning VLMs [22], conditioned on spatiotemporal task contexts. These traces serve as supervision signals for learning causal chains across vision, language, and action.

RoboBrain 2.0 adopts a high-efficiency heterogeneous architecture and a progressive multi-stage training strategy to support spatial understanding, temporal modeling, and long-chain causal reasoning in embodied settings. The model comprises a lightweight vision encoder with approximately 689M parameters and a decoder-only language model with 7B/32B parameters. It is trained using a three-stage curriculum—covering foundational spatiotemporal learning, embodied spatiotemporal enhancement, and chain-of-thought reasoning—on large-scale multimodal and embodied datasets. Training is conducted using our open-source framework FlagScale, which integrates hybrid parallelism, pre-allocated memory optimization, high-throughput I/O pipelines, and robust fault tolerance. These infrastructure innovations significantly reduce training and deployment costs while ensuring scalability for large-scale multimodal models. We evaluate RoboBrain 2.0 on over 12 public benchmarks covering spatial understanding, temporal modeling and multimodal reasoning, achieving state-of-the-art results on 6 of them despite its compact size. We release code, checkpoints, and benchmarks as open-source resources to benefit the research community. These materials facilitate reproducible research, accelerate embodied AI development, and enable practical deployment in robotic systems.

![](images/f466c7e5c8dd007b4314b4c72f1fc1eceb9e3a6e941419b684e552bc3036f09b.jpg)  
Figure 2 The overview of RoboBrain 2.0’s Capabilities. RoboBrain 2.0 supports interactive reasoning with long-horizon planning and closed-loop feedback, spatial perception for precise point and bounding box prediction from complex instructions, temporal perception for future trajectory estimation, and scene reasoning through real-time scene graph construction and updating.

To provide a comprehensive view of RoboBrain 2.0’s architecture, training methodology, and capabilities, this report is organized as follows: Section 2 introduces the overall model design, including the coordination between the vision encoder and language model, as well as image and video input strategies. Section 3 describes the data curation and construction process, covering three major categories: general multimodal understanding, spatial reasoning, and temporal modeling. Section 4 presents our multi-stage training strategies, including foundational spatiotemporal learning, embodied enhancement, and chain-of-thought reasoning. Section 5 outlines the infrastructure stack supporting scalable training and inference, including hybrid parallelization, memory optimization, data loading, and failure recovery. Section 6 reports extensive evaluation results on public benchmarks, highlights RoboBrain 2.0’s capabilities in spatial reasoning, temporal feedback, and embodied planning. Finally, Section 7 discusses current limitations, and outlines future research directions.

# 2 Architecture

RoboBrain 2.0 employs a modular encoder-decoder architecture that unifies perception, reasoning, and planning for complex embodied tasks. As shown in Figure 3, it processes multi-view visual observations and natural language instructions through four core components: (1) a tokenizer for textual/structured inputs, (2) a vision encoder, (3) an MLP projector mapping visual features to the language model’s token space, and (4) a language model backbone initialized from Qwen2.5-VL [5]. Unlike conventional VLMs [2, 22] focused on general static VQA, RoboBrain 2.0 maintains strong general VQA capabilities while specializing in embodied reasoning tasks like spatial perception, temporal modeling, and long-chain causal reasoning. The architecture encodes high-resolution images, multi-view inputs, video frames, language instructions, and scene graphs into a unified multimodal token sequence for comprehensive processing.

![](images/93df68325fc81b8640962a5376b2650b4c748c881b59d7ca7d573614181eda36.jpg)  
Figure 3 The Architecture of RoboBrain 2.0. The model supports multi-image, long video, and high-resolution visual inputs, along with complex task instructions and structured scene graphs on the language side. Visual inputs are processed via a vision encoder and an MLP projector, while textual inputs are tokenized into a unified token stream. All inputs are fed into an LLM decoder that performs long-chain-of-thought reasoning and generates a variety of outputs depending on the task, including structured plans, spatial relations, or relative and absolute coordinates.

# 2.1 Input Modalities and Tokenization

RoboBrain 2.0 supports a diverse set of input modalities tailored for embodied AI tasks:

• Language instructions: Natural language commands describing high-level goals or low-level actions. RoboBrain 2.0 processes natural language commands spanning different abstraction levels: from high-level, spatially grounded instructions (e.g., “Carry the apple to the nearest table, aligned with the leftmost cup”) to low-level motor commands (e.g., “Navigate to the nearest table”, “Grasp the apple”, “Detect position aligned with the leftmost cup”, “Place the apple into the box”).   
• Scene graph: A structured JSON representation of the explored environment, containing information about discovered objects, their categories, spatial locations, and embodiment configuration (e.g., name: KitchenTable1, type: table, object: [basket, knife], robot: RealMan-single-arm).   
• Multi-view static images: Images captured from multiple viewpoints, such as head-mounted cameras, wristmounted cameras, or multi-view projections from a 3D environment. These are processed independently by the vision encoder and concatenated into a unified token sequence.   
• Video frames: Video sequences (e.g., egocentric views from the agent), optionally annotated with timestamp tokens [5] to facilitate temporal grounding and reasoning.

Language instructions and scene graphs are tokenized using the language tokenizer. Visual inputs—including multi-view images and video frames—are processed by the vision encoder into dense visual embeddings, which are then projected into the LLM’s token space through an MLP projector, enabling unified multi-modal reasoning within the decoder.

# 2.2 Vision Encoder and Projection

RoboBrain 2.0 vision encoder supports dynamic-resolution image and video inputs through adaptive positional encoding and windowed attention mechanisms [5]. This design choice enables efficient processing of highresolution and multi-view visual observations common in embodied tasks.

To accommodate the long-horizon and temporally grounded nature of such tasks, we adopt frame-wise visual tokenization with multi-dimensional RoPE [5] for spatiotemporal encoding. Each visual embedding is projected via a lightweight MLP into the token space of the language model. For multi-view scenarios, visual tokens from different camera perspectives are serialized and augmented with view-specific positional identifiers before being fused with other input modalities.

# 2.3 LLM Decoder and Output Representations

RoboBrain 2.0 employs a decoder-only language model designed to unify high-level reasoning and spatially grounded output generation. Unlike conventional VLMs that primarily return short-form answers to static prompts, RoboBrain 2.0 flexibly supports both concise responses and multi-step chain-of-thought reasoning. This capability enables deeper understanding of complex instructions and physical scenes.

To enable the decoder to handle embodied tasks, the decoder is trained to produce a diverse range of outputs, including semantically grounded expressions (e.g., referring to objects or actions), spatial coordinates (e.g., absolute positions or bounding boxes), and intermediate reasoning traces. Rotary positional encodings and temporally conditioned tokens allow the model to maintain coherence across multi-round perception-action loops, which are essential for long-horizon planning in dynamic environments. Output formats supported by RoboBrain 2.0 include: (1) Free-form text: Used for task decomposition, scene graph updates, agent invocation, and human-agent dialogue. (2) Spatial coordinates: Used to represent point locations, bounding boxes, or trajectories in the image space for downstream controllers. (3) Reasoning traces (Optional): Long-chain-of-thought explanations to support deep problem solving and decision transparency.

This unified decoding formulation allows RoboBrain 2.0 to effectively handle a wide range of embodied tasks, from spatial grounding and visual understanding to long-horizon multi-agent planning and causal reasoning.

# 3 Training Data

As shown in Figure 4, RoboBrain 2.0 is trained on a diverse and extensive dataset designed to enhance its capabilities in spatial understanding, temporal modeling and long-chain causal reasoning in embodied settings. The training data encompasses a wide range of modalities, including high-resolution images, multi-view inputs, video sequences, scene graph and natural language instructions. This comprehensive dataset is meticulously categorized into three primary types: general multimodal understanding, spatial perception, and temporal modeling, ensuring the model can effectively perceive, reason, and plan in complex physical environments.

![](images/b38f5604013974ce27e5ef9ea43fa84c8219f8609d0b9f858c1c4d9ebe7f5eed.jpg)  
Figure 4 Training Data Distribution for RoboBrain 2.0. This figure illustrates the distribution of training data supporting RoboBrain 2.0’s capabilities, including interactive reasoning with long-horizon planning and closed-loop feedback, spatial perception for precise point and bounding box prediction from complex instructions, and multi-agent collaboration tasks, which is meticulously categorized into three primary types: general multimodal understanding, spatial perception, and temporal modeling.

# 3.1 General MLLM VQA

High Quality Data. The general training dataset for RoboBrain 2.0 includes 873K high-quality samples, primarily derived from LLaVA-665K [33] and LRV-400K [32], spanning standard Visual Question Answering(VQA), region-level queries, OCR-based VQA, and visual dialogues. (1) LLaVA-665K serves as the primary source and contains diverse VQA-style data, including standard VQA datasets, OCR-based questions, regionlevel queries, visual conversations, and language-only dialogues. To improve training efficiency, multiple question-answer(QA) pairs from the same image merge into single conversations; invalid ShareGPT [10] entries are filtered out, and overly long conversations ( $>$ 2048 tokens) are truncated (resulting in 40K valid samples). Specifically, A-OKVQA [54] samples are augmented by duplicating choices to balance multiple-choice formats, OCR-VQA [41] contributes 80K sampled conversations focused on scene text understanding, Visual Genome(VG) [27] provides dense object-level annotations limited to 10 entries per image with additional captions, and RefCOCO [76] dialogues are split into short multi-turn segments ( $< 1 0$ exchanges). Languageonly conversations, which are generally longer than visual ones, are sampled in single-modality batches to improve throughput by 25% without performance degradation. After removing bounding-box-dependent QA pairs, 531K high-quality samples are retained from this source. (2) LRV-400K is synthetically generated using GPT-4 [44] under a few-shot instruction-following setting. It produces 400K image-conditioned instructions across 16 vision-language tasks with textual answers. Unlike prior works that rely on sparse image captions, this dataset leverages the dense annotations in VG (e.g., bounding boxes, dimensions, and ${ \sim } 2 1$ object regions per image). GPT-4 generates both declarative and interrogative prompts for each image, with 10 tasks randomly sampled per instance. After filtering out bounding-box-related QA pairs, 342K samples are selected for training.

# 3.2 Spatial Data

Visual Grounding. The visual grounding dataset is constructed to enhance multimodal understanding through precise object-level localization, leveraging the extensive annotations from LVIS [19]. We carefully curate 152K high-resolution images from LVIS, ensuring broad coverage of diverse object categories and complex visual scenes. Each object annotation is converted into standardized bounding box coordinates $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ representing the top-left and bottom-right corners, enabling consistent spatial referencing. To facilitate rich visual dialogue, we generated 86K conversational sequences, each containing multiple rounds of QA pairs that progressively explore visual relationships, attribute reasoning, and contextual understanding. The dataset maintains a balanced distribution across object categories while preserving challenging cases of occlusion, viewpoint variation, and rare instances to support robust visual grounding.

Object Pointing. The object pointing dataset is constructed to enable RoboBrain 2.0 to identify the locations of specified objects through pointing within an image. We leverage the Pixmo-Points [13] dataset, which includes 2.3M point annotations across 223K images as our data source. However, direct utilization of Pixmo-Points data for RoboBrain 2.0 training presents challenges due to densely repeated object instances (e.g., books on a shelf). To address this, we implement a two-step filtering process: (1) we discard annotations with more than ten labeled points to simplify training, and (2) we use GPT-4o [22] as a scene analyzer to select only indoor-relevant objects, such as kitchenware, furniture, and decorations, excluding irrelevant or outdoor scenes. This process yields 190K QA pairs for 64K images with reduced clutter, making the data more suitable for embodied contexts. To construct QA pairs for pointing tasks, we construct 28 human-designed templates, such as “Point out all instances of {label} in the image.” or “Help me find {label} in the image by pointing to them.” Here, {label} refers to object categories from the annotations. Templates are randomly selected to ensure linguistic diversity and improve the model’s generalization ability in referencing tasks. For object reference pointing, we incorporate object reference data sourced from RoboPoint [77], which includes 347K QA annotations across 288K images. To address the potential issue of excessive points hindering training convergence, we randomly sample up to ten points per question. Additionally, the normalized coordinates are converted into absolute values to better support RoboBrain 2.0 training.

Affordance. The affordance dataset focuses on understanding object functionality and spatial vacant areas for placement. For object affordance recognition, we utilize part-level annotations from PACO-LVIS [51], covering 75 object categories and 200 part categories across 46K images. Bounding boxes and segmentation masks are extracted for both whole objects and their functional parts. These annotations are transformed into bounding box coordinates $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ , serving as ground truth labels for affordance prediction tasks. Questions are constructed using GPT-4o [22] to query object functionality and part usage, e.g., “Which part of a handbag can be grasped to carry it? ” for the handle of a handbag. For whole-object affordances, questions avoid naming the object directly, such as “What device can be moved to control the cursor on a screen? ” for a mouse (computer equipment). This automatic process results in 561K QA pairs. For spatial affordance learning, we include region reference data from RoboPoint [77]. This dataset consists of 270K images with 320K QA pairs and 14 spatial relationship labels. Each annotation is converted into a set of absolute coordinates $[ ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) , . . . ]$ , and ground truth points are resampled to a maximum of ten points per answer for optimization. This dataset enables RoboBrain 2.0 to reason about spatial affordances for object placement in real-world settings.

Spatial Understanding. To enhance RoboBrain 2.0’s 3D spatial reasoning, we present the Spatial Understanding Dataset, comprising 826K samples. This dataset emphasizes object-centric spatial attributes (e.g., position, orientation) and inter-object relations (e.g., distance, direction), covering both qualitative and quantitative aspects. It covers 31 distinct spatial concepts, substantially surpassing the ${ \sim } 1 5$ typically found in previous datasets. We partially adopt the RefSpatial [81] pipeline to construct 2D web image and 3D video datasets via automated template- and LLM-based generation: (1) 2D web images aim to provide core spatial concepts and depth perception across diverse indoor and outdoor scenes. To bridge scale and category gaps between these domains, we utilize the large-scale OpenImage [28] dataset. Since direct 3D reasoning from 2D images is challenging, we convert them into pseudo-3D scene graphs. Specifically, after filtering 1.7M images to 466K, we first use RAM [79] for object category prediction and GroundingDINO [34] for 2D boxes Detection. Then we enhance using Qwen2.5-VL [50] and a heuristic method to generate hierarchical captions given the 2D bounding box, ranging from coarse (e.g., “cup”) to fine-grained (e.g., “the third cup from the left”). This enables unambiguous spatial referring in cluttered environments and captures both coarse and fine-grained spatial references. Next, we use UniDepth V2 [48] and WildeCamera [84] for depth and camera intrinsics to enable 3D point cloud reconstruction. Finally, combining this with object boxes from GroundingDINO [34] and masks from SAM 2.1 [52], each scene graph includes object labels, 2D boxes, instance masks, and object-level point clouds, yielding axis-aligned 3D boxes. Object captions serve as nodes, and spatial relations form the edges. QA pairs are generated via templates and LLMs (e.g., QwQ [66]), including object-location questions derived from the hierarchical captions. (2) 3D scene-based videos integrates multimodal 3D scene understanding data from five original datasets: MMScan [38], 3RScan [69], ScanQA [3], SQA3D [39], and SpaceR [46]. We conduct template-based question filtering through rigorous data processing to ensure task relevance, perform multi-stage quality screening (e.g., consistency checks, outlier removal), and standardize all formats into a unified representation. This curation enables fine-grained environmental perception with enhanced reliability, supporting tasks ranging from object localization to complex spatial reasoning in 3D scenes. (3) 3D embodied videos focus on fine-grained spatial understanding in indoor environments. We leverage the CA-1M [29] dataset, filtering 2M frames to 100K high-quality ones. Compared to 2D, the availability of accurate 3D bounding boxes allows us to construct richer scene graphs with more diverse spatial relations, thereby generating more quantitative QA pairs (e.g., size, distances).

Spatial Referring. After enhancing foundational 3D spatial understanding, we extend these capabilities to physical-world interactions by introducing the Spatial Referring Dataset [81], consisting of 802K samples. Unlike prior datasets in visual grounding or object pointing, which often deal with ambiguous or multiple referents, this dataset targets a single unambiguous target, aligning with robotic applications such as precise pick-and-place that demand accurate object identification and localization. Following the RefSpatial [81] construction pipeline, for location data, we sample caption-point pairs from scene graphs built on 2D web images (OpenImage [28]) and 3D embodied videos (CA-1M [29]), using hierarchical captions. For placement data, we leverage fully annotated 3D datasets to generate top-down occupancy maps encoding object positions, orientations, and metric spatial relations (e.g., “10cm right of the chair”), facilitating accurate spatial referring.

# 3.3 Temporal Data

Ego-View Planning. We construct Ego-View Planning dataset by partially processing the EgoPlan-IT [9] dataset, which contains 50K automatically generated samples. For each selected task instance, we extract multiple frames from prior actions to represent task progress, and one frame to capture the current viewpoint. To enhance linguistic variety, we use multiple prompt templates that describe the task goal, video context, and current observation. Each question includes the correct next action along with up to three distractor actions randomly sampled from negative examples. This setup supports multimodal instruction tuning with diverse visual and textual input, aimed at improving egocentric task planning performance.

ShareRobot Planning. The ShareRobot dataset [23] is a large-scale, fine-grained resource for robotic manipulation, offering multi-dimensional annotations tailored for task planning. Its planning component provides detailed low-level instructions aligned with individual video frames, effectively transforming high-level task descriptions into structured and executable sub-tasks. Each data instance includes precise planning annotations to support accurate and consistent task execution. The dataset comprises 1M QA pairs from 51K instances, spanning 102 diverse scenes across 12 robot embodiments and 107 atomic tasks filtered according to the Open-X-Embodiment taxonomy [47]. All planning data were meticulously annotated by human experts following the RoboVQA [55] format, enabling models to learn robust multi-step planning strategies grounded in diverse real-world scenarios. The scale, quality, and diversity of ShareRobot help improve the model’s ability to perform fine-grained reasoning and task decomposition in complex embodied environments.

Agitbot Planning. The AgiBot Planning dataset is a large-scale robotics task planning dataset built upon the AgiBot-World [6] dataset, comprising 9,148 QA pairs across 19 manipulation tasks with 109,378 firstperson perspective images. Each sample contains 4-17 consecutive frames documenting task progression with multimodal conversational format. AgiBot-Planning provides step-by-step planning instructions that transform high-level goals into executable sub-tasks. Each data point includes current objectives, historical steps, and required subsequent actions. The dataset covers diverse scenarios from household refrigerator operations to supermarket shopping tasks across different environments. The meticulously crafted annotations use standardized conversational formats, enabling models to learn from varied real-world contexts. Through continuous visual sequences and fine-grained action plans, AgiBot-Planning enhances RoboBrain 2.0’s ability to perform long-horizon task planning and spatial reasoning in complex embodied scenarios.

Multi-Robot Planning. The Multi-Robot Planning dataset is constructed by simulating collaborative task scenarios across three environments—household, supermarket, and restaurant—based on RoboOS [61]. Each sample is generated using structured templates that specify a detailed scene graph, robot specifications, and associated tool lists. For every scenario, we design high-level, long-horizon collaborative task goals that require coordination among multiple robots present in the scene, and generate corresponding workflow graphs that decompose the tasks into subtasks with detailed reasoning explanations. Based on these decompositions, we further generate agent-specific robotic tool plans that translate high-level task goals into precise low-level Observation-Action pairs for each subtask. Specifically, we define 1,659 types of multi-robot collaboration tasks across the three environments and produce 44,142 samples using DeepSeek-V3 [31].

Close-Loop Interaction. The Close-Loop Interaction dataset is designed to facilitate advanced embodied reasoning [80], featuring a large-scale collection of synthesized Observation-Thought-Action (OTA) trajectories that combine first-person visual observations with structured thought tokens. It spans 120 diverse indoor environments—including kitchens, bathrooms, bedrooms, and living rooms—containing over 4,000 interactive objects and receptacles. The dataset is constructed within the AI2Thor [25] simulator through a rigorous multi-stage pipeline based on Embodied-Reasoner [78], which includes: (1) crafting task instructions from constrained templates to ensure scene-appropriate validity; (2) deriving key action sequences from an objectaffiliation graph encoding functional relationships; and (3) strategically incorporating search actions to emulate realistic exploration. To enrich the depth of reasoning, GPT-4o generates detailed thought processes—covering situational analysis, spatial reasoning, self-reflection, task planning, and verification—which are seamlessly integrated between observations and actions, forming coherent reasoning chains that guide models through complex, long-horizon interactive tasks.

# 4 Training Strategy

RoboBrain 2.0 achieves embodied capabilities (spatial understanding, temporal modeling, and chain-of-thought reasoning) through a progressive three-phase training strategy, as shown in Table 1. Starting from a robust vision-language foundation, we introduce escalating complexity in embodied supervision, enabling the model to evolve from static perception to dynamic reasoning and actionable planning in real-world environments.

# 4.1 Stage 1: Foundational Spatiotemporal Learning

The first stage focuses on building general capabilities in spatial perception and temporal understanding. We fine-tune the model on large-scale multimodal datasets covering dense captioning, object localization, interleaved image-text documents, and basic video QA, along with referring expression comprehension. These

Table 1 Detailed configuration for each training stage of the RoboBrain 2.0.   

<table><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>Stage-1</td><td rowspan=1 colspan=1>Stage-2</td><td rowspan=1 colspan=2>Stage-3</td></tr><tr><td rowspan=1 colspan=1>SFT</td><td rowspan=1 colspan=1>SFT</td><td rowspan=1 colspan=1>COT-SFT</td><td rowspan=1 colspan=1>RFT (RLVR)</td></tr><tr><td rowspan=1 colspan=2>    Dataset#Samples</td><td rowspan=1 colspan=1>Foundation4.8M</td><td rowspan=1 colspan=1>Embodied224K</td><td rowspan=1 colspan=1>Embodied (Phase 1)195K</td><td rowspan=1 colspan=1>Embodied (Phase 2)45K</td></tr><tr><td rowspan=1 colspan=2>0    Trainable Part#Tunable Parameters</td><td rowspan=1 colspan=1>Full Model8.29B or 33.45B</td><td rowspan=1 colspan=1>Full Model8.29B or 33.45B</td><td rowspan=1 colspan=1>Full Model8.29B or 33.45B</td><td rowspan=1 colspan=1>Full Model8.29B or 33.45B</td></tr><tr><td rowspan=3 colspan=2>Per-device Batch SizeGradient AccumulationLR: {ψvit, φLLM }</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>1×10−4</td><td rowspan=1 colspan=1>1 ×10−5</td><td rowspan=1 colspan=1>1 ×10−5</td><td rowspan=3 colspan=1>1 ×10−63AdamW</td></tr><tr><td rowspan=1 colspan=2>Epoch</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=2>Optimizer</td><td rowspan=1 colspan=1>AdamW</td><td rowspan=1 colspan=1>AdamW</td><td rowspan=1 colspan=1>AdamW</td></tr><tr><td rowspan=3 colspan=2>Deepspeed    Weight DecayWarmup Ratio</td><td rowspan=1 colspan=1>−</td><td rowspan=1 colspan=1>−</td><td rowspan=1 colspan=1>Zero3</td><td rowspan=1 colspan=1>Zero3</td></tr><tr><td rowspan=1 colspan=1>Decay</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=2>LR Schedule</td><td rowspan=1 colspan=1>Cosine</td><td rowspan=1 colspan=1>Cosine</td><td rowspan=1 colspan=1>Cosine</td><td rowspan=1 colspan=1>Cosine</td></tr><tr><td rowspan=4 colspan=2>Max Seq. LengthMax Compl. LengthNum. of Compl.GPU Nums</td><td rowspan=1 colspan=1>16384</td><td rowspan=1 colspan=1>16384</td><td rowspan=1 colspan=1>32768</td><td rowspan=1 colspan=1>32768</td></tr><tr><td rowspan=2 colspan=1>−−</td><td rowspan=1 colspan=1>−</td><td rowspan=1 colspan=1>−</td><td rowspan=1 colspan=1>1024</td></tr><tr><td rowspan=1 colspan=1>−</td><td rowspan=1 colspan=1>−</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>16/64 × 8</td><td rowspan=1 colspan=1>16/64 × 8</td><td rowspan=1 colspan=1>4 × 8</td><td rowspan=1 colspan=1>4 × 8</td></tr></table>

datasets span common physical scenes and interaction patterns, helping the model develop fundamental grounding for objects, spatial relations, and motion events. This stage lays the groundwork for understanding egocentric video streams and spatially anchored instructions.

# 4.2 Stage 2: Embodied Spatiotemporal Enhancement

To better align the model with embodied tasks, we introduce a carefully curated collection of high-resolution, multi-view, and egocentric video datasets, along with instruction-augmented navigation and interaction data. Tasks include viewpoint-aware referring expressions, 3D affordance estimation, and object-centric scene graph construction. This stage of training emphasizes the modeling of long-horizon temporal dependencies, enabling the model to reason over extended sequences of actions and observations. Additionally, it incorporates multi-agent coordination scenarios, where the model learns to interpret and predict the behaviors of other agents in shared environments. To support these capabilities, we employ extended sequence lengths and multi-camera input encoding, allowing the model to process and fuse visual information from multiple viewpoints simultaneously. Through this training stage, the model can integrate historical visual cues with current instructions, fostering more coherent long-horizon planning, robust scene understanding, and adaptive decision-making in dynamic, interactive settings.

# 4.3 Stage 3: Chain-of-Thought Reasoning in Embodied Contexts

In the third stage, we augment the model’s high-level reasoning capabilities using Chain-of-Thought (CoT) methodology, following the two-phase framework of Reason-RFT [62]: CoT-based Supervised Fine-Tuning (CoT-SFT) and Reinforcement Fine-Tuning (RFT). We leverage multi-turn reasoning examples from both synthetic and real-world embodied scenarios, encompassing long-horizon task planning, manipulation prediction, closed-loop interaction, spatiotemporal understanding, and multi-robot collaboration, sourced from Section 3. Specifically, (1) CoT-SFT Phase: We annotate $1 0 \%$ of the constructed training data with CoT rationales annotated by GPT-4o [22] with custom prompts, then perform supervised fine-tuning for initial model from Stage 2. (2) RFT Phase: An additional 10% of the constructed training data is sampled to collect model’s responses, with incorrect answers curated into a reformatted training set (e.g., multiple-choice questions or LaTeX/numerical answers). Optimization employs Group Relative Policy Optimization (GRPO) [17], guided by a composite reward function evaluating both answer accuracy and format correctness.

# 5 Infrastructures

# 5.1 Large-Scale Training Infrastructure

To improve the efficiency and stability of multimodal model training, we have developed and integrated a series of key optimization techniques, including hybrid parallelism strategies, memory pre-allocation, distributed data loading, kernel fusion, and fine-grained compute-communication overlapping. These optimizations significantly enhance both resource utilization and training throughput. For data preprocessing, we build upon the Megatron–Energon framework [30] and incorporate custom optimization strategies. Our system supports dynamic mixing of multiple datasets containing diverse modalities, including plain text, single image, multiple images, and video, while also allowing for strict sample order preservation within each dataset. A custom WebDataset-based format [1] enables compatibility with various data modalities and greatly reduces preprocessing time while improving flexibility and scalability in data handling.

# 5.1.1 Multi-Dimensional Hybrid Parallelism

Multimodal models differ significantly from conventional LLMs in both architecture and data characteristics [33]. On the architectural side, multimodal models are inherently heterogeneous: the vision module (e.g., ViT with Adaptor) is typically a small-scale encoder-only component, while the language module is a much larger decoder-only transformer. On the data side, training samples include plain text, single images, multi-image sequences, and videos. The number of image tokens, text tokens, and the length of the fused token sequence can vary dramatically between samples.

These heterogeneities pose substantial challenges to distributed training frameworks. To address this, we implemented several targeted strategies in our custom framework, FlagScale [12]:

• Non-uniform Pipeline Parallelism [43]: Since the ViT module appears early in the model and has relatively low computational cost, we reduce the number of LLM layers in the first pipeline stage, thereby improving training throughput without increasing memory overhead.

Separate Recompute Strategy: During the annealing stage, the vision input may contain up to 20,000–30,000 tokens, frequently causing an Out-of-Memory (OOM) error in the ViT module. To mitigate this, we enable recompute [8, 26] only in the ViT module to reduce memory usage of intermediate activations, while disabling recompute in the LLM module to preserve computational efficiency.

# 5.1.2 Pre-Allocate Memory

In the supervised fine-tuning training process of RoboBrain 2.0, input lengths vary significantly across samples. PyTorch’s default caching memory allocator [49] can lead to memory fragmentation under such dynamic input conditions, frequently resulting in OOM errors. A common but inefficient workaround is to call torch.cuda.empty_cache() before every forward pass, which severely degrades performance.Instead, we take a more efficient approach by analyzing PyTorch’s memory allocation mechanism. Fragmentation often results from the lack of a sufficiently large and contiguous cached memory block for new tensors, prompting new allocations and worsening fragmentation. To address this, we introduce a memory pre-allocation strategy: we compute the maximum sequence length across the entire dataset before training, and pad all samples to this maximum length in the first step. This ensures that tensors can reuse pre-allocated memory blocks, reducing fragmentation and maintaining throughput.

# 5.1.3 Data Pre-Processing

We adopt native Megatron-Energon [30] for unified data loading, eliminating the need for external training frameworks. Additionally, we optimized the preprocessing pipeline to reduce time consumption by up to $9 0 \%$ . We evaluated and compared two preprocessing strategies:

• Preprocessing Both JSON and Images. Using the default Megatron-Energon data pipeline, both JSON metadata and images are compressed into binary files for WebDataset. However, this approach suffers from two major issues: (1) Low efficiency: Preprocessing 320,000 samples can take over 2 hours. (2) Inconsistent image readers: Megatron-Energon uses cv2, while models such as RoboBrain 2.0 use PIL, introducing subtle differences that may affect training performance.

• Preprocessing JSON Only (Recommended). In our optimized pipeline, only JSON files are preprocessed, and images are kept in their original form. Image preprocessing is deferred to the TaskEncoder module using the same preprocessor as Qwen2.5-VL. (1) High efficiency: Preprocessing 320,000 samples takes less than 10 minutes. (2) Alignment with model input: Ensures image handling is fully aligned between preprocessing and training, eliminating inconsistency and improving model performance.

# 5.1.4 Distributed Data Loading

To minimize the I/O burden on compute nodes, we reduce redundant data loading in large-scale distributed training. Unlike single-node setups, GPUs in distributed training systems play different roles depending on the chosen parallel strategy. Data loading typically occurs along the data parallel (DP) dimension, where each DP rank handles a unique data shard. However, in multi-dimensional hybrid parallelism (e.g., DP-PP-TP), only a subset of GPU processes actually need to load data: (1) In each Pipeline Parallel (PP) [42] group, only the first and last stages need to perform data loading. (2) Within Tensor Parallel (TP) [58] groups, only one GPU per group is required to load data, with others receiving data via broadcast. This design significantly reduces redundant I/O operations and improves overall data throughput.

# 5.1.5 Fault Tolerance

To handle both hardware and software failures during training, we co-designed fault-tolerant mechanisms between our FlagScale [12] training framework and the system platform. Common errors, such as LostCard, KubeNodeNotReady, are automatically detected and trigger automatic job recovery and restart, ensuring minimal disruption. Furthermore, our custom DataLoader module based on Megatron-Energon supports full data state recovery, allowing seamless resumption from the most recent checkpoint with complete consistency in data loading and sample shuffling states.

# 5.2 Reinforcement Fine-Tuning Infrastructure

We employ Reinforcement Learning with Verifiable Rewards (RLVR) to enhance RoboBrain 2.0 using VeRL [68], an open-source RL framework specifically designed for post-training LLMs and VLMs. Based on the HybridFlow architecture [56], VeRL features a hybrid-controller model that integrates both a global controller for inter-RL-role dataflow coordination and distributed controllers for intra-RL-role parallel processing. This architecture enables efficient execution of complex post-training workflows while ensuring scalability. VeRL’s support for multiple RL algorithms (e.g., GRPO) and seamless LLM integration makes it particularly suitable for RoboBrain 2.0’s reinforcement fine-tuning (RFT) requirements. The framework enables high-performance model tuning with minimal overhead through its optimized dataflow management and parallel processing capabilities. Its efficient handling of large-scale training tasks and rigorous reward verification establishes VeRL as an ideal platform for advancing RoboBrain 2.0’s capabilities via RLVR.

# 5.3 Inference Infrastructure

To improve the efficiency of model inference, we adopt FlagScale [12], also a multi-backend inference framework, which can automatically search for the optimal inference engine and configuration parameters based on the performance characteristics of different models on heterogeneous hardware accelerators, thereby effectively reducing inference latency. Given the high sensitivity of embodied AI models to accuracy, we further introduce a mixed-bit quantization strategy [40, 70]. This strategy enhances inference efficiency and resource utilization while maintaining model performance. Specifically, the vision encoder retains full-precision floating-point computation to ensure the accuracy of key feature extraction. In contrast, during the language module, weights are quantized to 8-bit integers, while activations are preserved in 16-bit floating-point format. This mixed-precision approach significantly reduces computational overhead and memory usage with negligible impact on model accuracy. Moreover, the quantization process is minimally invasive to existing inference pipelines and can be flexibly integrated into current systems. In end-to-end embodied tasks, weight-only quantization alone achieves approximately a $3 0 \%$ reduction in inference latency, demonstrating the effectiveness and practicality of the proposed method in real-world deployment scenarios.

# 6 Evaluation Results

We conducted a comprehensive evaluation of RoboBrain-2.0, focusing on its performance across spatial and temporal reasoning capabilities on embodiment. To ensure consistency and rigor in evaluation, we adopted FlagEvalMM [20], our flexible framework for systematic multimodal model assessment. Evaluations on spatial reasoning benchmarks (e.g., CV-bench [67], Blink [15], Where2Place [77], ShareRobot-Bench [23]), presented in Section 6.1, underscore the model’s strengths in embodied spatial reasoning. An in-depth analysis of multi-robot collaboration [61] and long-horizon planning (e.g., EgoPlan2 [9], RoboBench) capabilities is provided in Section 6.2, highlighting the model’s advancements in temporal reasoning tasks. Qualitative examples and prompt details are provided in Section A and Section B, respectively.

# 6.1 Spatial Reasoning Capability

RoboBrain-32B-2.0 and RoboBrain-7B-2.0 demonstrate exceptional performance across nine spatial reasoning benchmarks: BLINK, CV-Bench, EmbSpatial, RoboSpatial, and RefSpatial-Bench (Table 2), as well as SAT, VSI-Bench, Where2Place, and ShareRobot-Bench (Table 3). Below is a detailed analysis highlighting their state-of-the-art (SOTA) achievements and near-SOTA competitive results.

Table 2 Performance across five spatial reasoning benchmarks. The best results among different models ar highlighted in bold, while the second-best results are underlined.   

<table><tr><td rowspan="2">Models / Metrics</td><td colspan="3">BLINK</td><td>CV-Bench</td><td>EmbSpatial</td><td>RoboSpatial</td><td colspan="3">RefSpatial-Bench</td></tr><tr><td>Dep.</td><td>Spa.</td><td>All↑</td><td>All↑</td><td>All↑</td><td>All ↑</td><td>Loc.</td><td>Pla.</td><td>All↑</td></tr><tr><td colspan="10">General Baselines</td></tr><tr><td>Gemini-2.5-Pro-preview-05-06 [16]</td><td>79.03</td><td>84.62</td><td>81.83</td><td>84.59</td><td>78.74</td><td>59.87</td><td>44.58</td><td>31.73</td><td>38.16</td></tr><tr><td>Gemini-2.5-Flash-preview-04-17 [16]</td><td>77.42</td><td>79.02</td><td>78.22</td><td>84.03</td><td>74.75</td><td>54.10</td><td>37.50</td><td>23.00</td><td>30.25</td></tr><tr><td>GPT-04-mini-2025-05-16 [45]</td><td>79.03</td><td>88.11</td><td>83.57</td><td>85.21</td><td>78.29</td><td>51.25</td><td>15.00</td><td>19.58</td><td>17.29</td></tr><tr><td>GPT-4o-2024-11-20 [22]</td><td>72.58</td><td>83.22</td><td>77.90</td><td>78.63</td><td>71.92</td><td>44.42</td><td>8.00</td><td>9.55</td><td>8.78</td></tr><tr><td>Claude-Sonnet-4-2025-05-14 [2]</td><td>75.81</td><td>80.42</td><td>78.12</td><td>78.43</td><td>64.26</td><td>51.26</td><td>5.00</td><td>10.37</td><td>7.69</td></tr><tr><td>Qwen2.5-VL-32B-Instruct [50]</td><td>77.42</td><td>85.31</td><td>81.37</td><td>81.59</td><td>74.45</td><td>52.16</td><td>16.83</td><td>10.60</td><td>13.72</td></tr><tr><td>Qwen2.5-VL-72B-Instruct [50]</td><td>74.19</td><td>78.32</td><td>76.26</td><td>82.68</td><td>73.30</td><td>48.33</td><td>23.50</td><td>15.83</td><td>19.67</td></tr><tr><td colspan="10">Embodied Baselines</td></tr><tr><td>Cosmos-Reason1-7B [4]</td><td>63.71</td><td>73.43</td><td>68.57</td><td>74.71</td><td>65.22</td><td>38.81</td><td>9.84</td><td>1.04</td><td>5.44</td></tr><tr><td>VeBrain-8B [36]</td><td>78.23</td><td>81.12</td><td>79.68</td><td>78.57</td><td>70.52</td><td>42.48</td><td>0.03</td><td>0.57</td><td>0.30</td></tr><tr><td>Magma-8B [74]</td><td>65.32</td><td>66.43</td><td>65.88</td><td>60.98</td><td>64.59</td><td>33.71</td><td>1.00</td><td>8.00</td><td>4.50</td></tr><tr><td>RoboBrain-7B-1.0 [23]</td><td>75.81</td><td>78.32</td><td>77.07</td><td>76.22</td><td>68.13</td><td>51.53</td><td>14.43</td><td>5.41</td><td>9.92</td></tr><tr><td>RoboBrain-7B-2.0</td><td>84.68</td><td>83.22</td><td>83.95</td><td>85.75</td><td>76.32</td><td>54.23</td><td>36.00</td><td>29.00</td><td>32.50</td></tr><tr><td>RoboBrain-32B-2.0</td><td>79.84</td><td>87.41</td><td>83.63</td><td>83.92</td><td>78.57</td><td>72.43</td><td>54.00</td><td>54.00</td><td>54.00</td></tr></table>

• BLINK. In the BLINK [15] benchmark, models are evaluated on depth perception (Dep.) and spatial relation understanding (Spa.). RoboBrain-7B-2.0 achieves a SOTA average score of 83.95 (Dep.: 84.68, Spa.: 83.22), outperforming all general baselines, including GPT-o4-mini-2025-05-16 (83.57), Gemini2.5-Pro-preview-05-06 (81.83), Qwen2.5-VL-32B-Instruct (81.37), Claude-Sonnet-4-2025-05-14 (78.12), GPT-4o-2024-11-20 (77.90), and Qwen2.5-VL-72B-Instruct (76.26), as well as embodied baselines like VeBrain-8B (79.68) and Cosmos-Reason1-7B (68.57). RoboBrain-32B-2.0 follows closely with an average of 83.63 (Dep.: 79.84, Spa.: 87.41), surpassing all general and embodied baselines except RoboBrain-7B-2.0, demonstrating strong spatial reasoning capabilities.

• CV-Bench. The CV-Bench [67] benchmark assesses a model’s accuracy in 2D/3D spatial understanding and visual processing. RoboBrain-7B-2.0 secures a SOTA accuracy of 85.75, slightly ahead of RoboBrain-32B-2.0 (83.92), both outperforming all general baselines, including GPT-o4-mini-2025-05-16 (85.21), Gemini-2.5- Pro-preview-05-06 (84.59), Qwen2.5-VL-72B-Instruct (82.68), Qwen2.5-VL-32B-Instruct (81.59), GPT-4o2024-11-20 (78.63), and Claude-Sonnet-4-2025-05-14 (78.43), as well as embodied baselines like VeBrain-8B (78.57) and Cosmos-Reason1-7B (74.71).

EmbSpatial. The EmbSpatial [14] benchmark evaluates models on embodied spatial tasks. RoboBrain-32B2.0 achieves a near SOTA accuracy of 78.57, slightly less than Gemini-2.5-Pro-preview-05-06 (78.74) and surpassing all other general baselines, including GPT-o4-mini-2025-05-16 (78.29), Qwen2.5-VL-32B-Instruct (74.45), Qwen2.5-VL-72B-Instruct (73.30), GPT-4o-2024-11-20 (71.92), and Claude-Sonnet-4-2025-05-14 (64.26). RoboBrain-7B-2.0 follows with a competitive score of 76.32, outperforming most general baselines and all embodied baselines, indicating strong embodied spatial reasoning.

Table 3 Performance across four spatial reasoning benchmarks. The best results among different models are highlighted in bold, while the second-best results are underlined.   

<table><tr><td rowspan="2">Models / Metrics</td><td>SAT</td><td>VSI-Bench</td><td colspan="3">Where2Place*</td><td colspan="2">ShareRobot-Bench</td></tr><tr><td>All↑</td><td>All↑</td><td>Seen</td><td>Unseen</td><td>All↑</td><td>Afford. ↑</td><td>Traj.(DFD ↓)</td></tr><tr><td colspan="8">General Baselines</td></tr><tr><td>Gemini-2.5-Pro-preview-05-06 [16]</td><td>79.33</td><td>47.81</td><td>42.92</td><td>41.13</td><td>42.38</td><td>10.26</td><td>0.7666</td></tr><tr><td>Gemini-2.5-Flash-preview-04-17 [16]</td><td>74.00 82.00</td><td>48.83</td><td>31.54</td><td>21.73</td><td>28.60</td><td>2.50</td><td>0.9087</td></tr><tr><td>GPT-04-mini-2025-05-16 [45]</td><td></td><td>41.96</td><td>26.63</td><td>26.49</td><td>26.59</td><td>8.27</td><td>0.5726</td></tr><tr><td>GPT-4o-2024-11-20 [22]</td><td>66.67</td><td>43.60</td><td>20.28</td><td>20.71</td><td>20.41</td><td>6.00</td><td>0.6850</td></tr><tr><td>Claude-Sonnet-4-2025-05-14 [2]</td><td>75.33</td><td>47.02</td><td>21.56</td><td>35.11</td><td>25.63</td><td>8.00</td><td>0.7591</td></tr><tr><td>Qwen2.5-VL-32B-Instruct [50]</td><td>80.00</td><td>36.07</td><td>18.22</td><td>32.55</td><td>22.52</td><td>11.97</td><td>0.9222</td></tr><tr><td>Qwen2.5-VL-72B-Instruct [50]</td><td>58.67</td><td>35.51</td><td>35.74</td><td>49.65</td><td>39.92</td><td>23.80</td><td>0.5034</td></tr><tr><td colspan="8">Embodied Baselines</td></tr><tr><td>Cosmos-Reason1-7B [4]</td><td>60.67</td><td>25.64</td><td>5.07</td><td>6.53</td><td>5.51</td><td>9.98</td><td>0.8524</td></tr><tr><td>VeBrain-8B [36]</td><td>58.00</td><td>26.30</td><td>12.27</td><td>9.17</td><td>11.34</td><td>3.66</td><td>1.1659</td></tr><tr><td>Magma-8B [74]</td><td>71.33</td><td>12.65</td><td>9.93</td><td>13.14</td><td>10.89</td><td>−</td><td>0.7478</td></tr><tr><td>RoboBrain-7B-1.0 [23]</td><td>59.33</td><td>31.12</td><td>54.58</td><td>49.45</td><td>53.04</td><td>10.20</td><td>0.6248</td></tr><tr><td>RoboBrain-7B-2.0</td><td>75.33</td><td>36.10</td><td>64.33</td><td>61.88</td><td>63.59</td><td>28.05</td><td>0.5512</td></tr><tr><td>RoboBrain-32B-2.0</td><td>86.67</td><td>42.69</td><td>73.95</td><td>72.74</td><td>73.59</td><td>35.28</td><td>0.2368</td></tr></table>

• RoboSpatial. The RoboSpatial [59] benchmark measures spatial reasoning in robot environments, such as object localization and manipulation. RoboBrain-32B-2.0 achieves a clear SOTA score of 72.43, substantially ahead of general baselines like Gemini-2.5-Pro-preview-05-06 (59.87), Qwen2.5-VL-72B-Instruct (48.33), GPT-o4-mini-2025-05-16 (51.25), and Claude-Sonnet-4-2025-05-14 (51.26). RoboBrain-7B-2.0 scores 54.23, outperforming all general baselines except RoboBrain-32B-2.0, demonstrating significant improvements in spatial reasoning for robotic tasks.

• RefSpatial-Bench. The RefSpatial-Bench [81] benchmark evaluates models on spatial referring expressions, requiring precise point predictions under spatial constraints, with metrics for Location (Loc.) and Placement (Pla.) accuracy. RoboBrain-32B-2.0 achieves SOTA scores of 54.00 (Loc.) and 54.00 (Pla.), significantly outperforming all general baselines, including Gemini-2.5-Pro-preview-05-06 (44.58, 31.73), Qwen2.5-VL72B-Instruct (23.50, 15.83), Qwen2.5-VL-32B-Instruct (16.83, 10.60), GPT-o4-mini-2025-05-16 (15.00, 19.58), GPT-4o-2024-11-20 (8.00, 9.55), and Claude-Sonnet-4-2025-05-14 (5.00, 10.37). RoboBrain-7B-2.0 scores 36.00 (Loc.) and 29.00 (Pla.), outperforming all general baselines except RoboBrain-32B-2.0, showing competitive precision in complex spatial referring tasks.

• SAT. The SAT [53] benchmark measures general spatial reasoning abilities across various scenes and tasks. RoboBrain-32B-2.0 achieves a clear SOTA score of 86.67, significantly outperforming all general baselines, including GPT-o4-mini-2025-05-16 (82.00), Gemini-2.5-Pro-preview-05-06 (79.33), Qwen2.5- VL-72B-Instruct (58.67), and Claude-Sonnet-4-2025-05-14 (75.33). RoboBrain-7B-2.0 achieves 75.33, surpassing most general and embodied baselines, showcasing its strong spatial reasoning capability.

• VSI-Bench. The VSI-Bench [75] evaluates visual-spatial integration capabilities. Gemini-2.5-Flash-preview04-17 achieves the best performance with 48.83. RoboBrain-32B-2.0 achieves 42.69, outperforming most general and embodied baselines, including GPT-o4-mini-2025-05-16 (41.96) and Qwen2.5-VL-72B-Instruct (35.51). RoboBrain-7B-2.0 reaches 36.10, indicating solid visual-spatial integration skills.

• Where2Place. The Where2Place [77] benchmark measures a model’s ability to predict object placements in both seen and unseen scenarios under spatial constraints. RoboBrain-32B-2.0 achieves a SOTA average of 73.59 (Seen: 73.95, Unseen: 72.74), substantially surpassing all general and embodied baselines, including Qwen2.5-VL-72B-Instruct (39.92), Gemini-2.5-Pro-preview-05-06 (42.38), Claude-Sonnet-4-2025-05-14 (25.63), and VeBrain-8B (11.34). RoboBrain-7B-2.0 also performs strongly with an average of 63.59 (Seen:

64.33, Unseen: 61.88), outperforming all baselines except RoboBrain-32B-2.0.

• ShareRobot-Bench-Affordance. The ShareRobot Affordance task [23] evaluates models on object functionality and interaction understanding. RoboBrain-32B-2.0 secures a SOTA performance with an accuracy of 35.28, ahead of all general baselines, including Qwen2.5-VL-72B-Instruct (23.80), Qwen2.5-VL-32B-Instruct (11.97), GPT-4o-2024-11-20 (6.00), and Claude-Sonnet-4-2025-05-14 (8.00). RoboBrain-7B-2.0 achieves 28.05, outperforming all general and embodied baselines except RoboBrain-32B-2.0.

• ShareRobot-Bench-Trajectory. The ShareRobot Trajectory task [23] assesses navigation and motion prediction, using Dynamic Fréchet Distance (DFD), where lower values denote better performance. RoboBrain-32B-2.0 achieves a SOTA DFD of 0.2368, outperforming all general and embodied baselines, including Qwen2.5-VL-72B-Instruct (0.5034), GPT-o4-mini-2025-05-16 (0.5726), and Gemini-2.5-Propreview-05-06 (0.7666). RoboBrain-7B-2.0 follows with a competitive DFD of 0.5512, demonstrating strong path-planning capabilities.

# 6.2 Temporal Reasoning Capability

RoboBrain-32B-2.0 and RoboBrain-7B-2.0 exhibit outstanding performance across three critical measures of temporal reasoning benchmarks: Multi-Robot Planning, Ego-Plan2, and RoboBench, as shown in Table 4. Below is a detailed analysis highlighting their state-of-the-art (SOTA) achievements and near-SOTA results.

Table 4 Performance across three temporal reasoning benchmarks. The best results among different models ar highlighted in bold, while the second-best results are underlined.   

<table><tr><td rowspan="2">Models / Metrics</td><td colspan="4">Multi-Robot Planning</td><td colspan="6">Ego-Plan2</td><td>RoboBench</td></tr><tr><td>Super.</td><td>Rest.</td><td>House.</td><td>All↑</td><td>Daily.</td><td>Hobbies.</td><td>Rec.</td><td>Work.</td><td>All ↑</td><td>Plan. ↑</td></tr><tr><td colspan="9">General Baselines</td><td></td></tr><tr><td>Gemini-2.5-Pro-preview-05-06 [16]</td><td>63.51</td><td>54.77</td><td>78.39</td><td>65.39</td><td>44.19</td><td>43.05</td><td>46.45</td><td>39.60</td><td>42.85</td><td>63.49</td></tr><tr><td>Gemini-2.5-Flash-preview-04-17 [16]</td><td>59.44</td><td>55.78</td><td>76.88</td><td>63.86</td><td>38.72</td><td>35.59</td><td>43.72</td><td>33.42</td><td>37.09</td><td>69.33</td></tr><tr><td>GPT-04-mini-2025-05-16 [45]</td><td>63.32</td><td>55.28</td><td>78.89</td><td>65.50</td><td>47.61</td><td>35.93</td><td>42.62</td><td>37.13</td><td>41.11</td><td>70.01</td></tr><tr><td>GPT-4o-2024-11-20 [22]</td><td>77.89</td><td>67.34</td><td>79.40</td><td>74.50</td><td>47.38</td><td>40.00</td><td>44.81</td><td>35.64</td><td>41.79</td><td>68.60</td></tr><tr><td>Claude-Sonnet-4-2025-05-14 [2]</td><td>73.08</td><td>61.81</td><td>80.40</td><td>71.30</td><td>43.51</td><td>41.02</td><td>42.62</td><td>38.87</td><td>41.26</td><td>70.21</td></tr><tr><td>Qwen2.5-VL-32B-Instruct [50]</td><td>67.84</td><td>61.81</td><td>75.38</td><td>68.00</td><td>64.46</td><td>51.53</td><td>57.92</td><td>50.00</td><td>56.25</td><td>45.92</td></tr><tr><td>Qwen2.5-VL-72B-Instruct [50]</td><td>77.39</td><td>68.34</td><td>79.40</td><td>74.67</td><td>60.36</td><td>48.14</td><td>63.39</td><td>46.29</td><td>53.75</td><td>66.94</td></tr><tr><td colspan="9">Embodied Baselines</td><td></td></tr><tr><td>Cosmos-Reason1-7B [4]</td><td>35.17</td><td>25.62</td><td>40.70</td><td>33.66</td><td>30.75</td><td>27.12</td><td>31.69</td><td>20.30</td><td>26.87</td><td>53.17</td></tr><tr><td>VeBrain-8B [36]</td><td>41.70</td><td>35.67</td><td>39.69</td><td>38.83</td><td>31.79</td><td>35.31</td><td>31.19</td><td>34.43</td><td>27.30</td><td>46.77</td></tr><tr><td>Magma-8B [74</td><td></td><td></td><td></td><td>—</td><td>4.56</td><td>3.39</td><td>6.56</td><td>2.97</td><td>4.09</td><td>−</td></tr><tr><td>RoboBrain-7B-1.0 [23]</td><td>4.52</td><td>7.04</td><td>5.03</td><td>5.50</td><td>—</td><td></td><td>—</td><td></td><td></td><td>38.93</td></tr><tr><td>RoboBrain-7B-2.0</td><td>83.92</td><td>77.39</td><td>84.42</td><td>81.50</td><td>39.41</td><td>32.20</td><td>33.88</td><td>26.98</td><td>33.23</td><td>72.16</td></tr><tr><td>RoboBrain-32B-2.0</td><td>84.42</td><td>72.36</td><td>85.43</td><td>80.33</td><td>64.01</td><td>53.22</td><td>57.92</td><td>52.48</td><td>57.23</td><td>68.33</td></tr></table>

• Multi-Robot Planning. In the Multi-Robot Planning task [61], models are evaluated on their ability to coordinate multiple robots across different scenarios: Super (Supermarket), Rest (Restaurant), and House (Household). RoboBrain-32B-2.0 achieves a SOTA average score of 80.33 (Super: 84.42, Rest: 72.36, House: 85.43), significantly outperforming all general baselines, including GPT-4o-2024-11-20 (74.50), Qwen2.5-VL-72B-Instruct (74.67), Claude-Sonnet-4-2025-05-14 (71.30), Gemini-2.5-Pro-preview-05-06 (65.39), and Qwen2.5-VL-32B-Instruct (68.00). It also surpasses the embodied baseline RoboBrain-7B-2.0 (81.50). RoboBrain-7B-2.0 follows closely with an average of 81.50 (Super: 83.92, Rest: 77.39, House: 84.42), outperforming all general baselines and matching the performance of RoboBrain-7B-1.5-OS in Rest and House scenarios.

• Ego-Plan2. The Ego-Plan2 [9] benchmark assesses a model’s capability to plan daily activities across four categories: Daily (Daily Routines), Hobbies, Rec (Recreation), and Work. RoboBrain-32B-2.0 secures a SOTA average score of 57.23 (Daily: 64.01, Hobbies: 53.22, Rec: 57.92, Work: 52.48), significantly outperforming all general and embodied baselines, including Qwen2.5-VL-32B-Instruct (56.25), Qwen2.5- VL-72B-Instruct (53.75), Gemini-2.5-Pro-preview-05-06 (42.85), GPT-4o-2024-11-20 (41.79), ClaudeSonnet-4-2025-05-14 (41.26), GPT-o4-mini-2025-05-16 (41.11), VeBrain-8B (27.30), and Cosmos-Reason1- 7B (26.87). In contrast, RoboBrain-7B-2.0 achieves an average of 33.23 (Daily: 39.41, Hobbies: 32.20,

Rec: 33.88, Work: 26.98), which is lower than general baselines like Qwen2.5-VL-32B-Instruct and Qwen2.5-VL-72B-Instruct but surpasses embodied baselines such as VeBrain-8B and Cosmos-Reason1-7B.

• RoboBench. The RoboBench Benchmark (Planning part) evaluates a model’s ability to plan robotic mobile manipulation tasks according to their pre-defined skills across three categories: cross-embodiment, crossobject, and cross-view. On this benchmark, RoboBrain-7B-2.0 achieves a state-of-the-art (SOTA) score of 72.16, surpassing all general and embodied baselines, including Claude-Sonnet-4-2025-05-14 (70.21), GPTo4-mini-2025-05-16 (70.01). The performance of RoboBrain-32B-2.0, with a score of 68.33, outperforming several general baselines like GPT-4o-2024-11-20 (68.60) and Qwen2.5-VL-72B-Instruct (66.94), as well as other embodied baselines such as Cosmos-Reason1-7B (53.17) and VeBrain-8B (46.77).

# 7 Conclusion and Future Works

In this report, we introduced RoboBrain 2.0, our latest generation of embodied vision-language foundation models, developed to support unified perception, reasoning, and planning in complex physical environments. Built on a modular architecture with a dedicated vision encoder and a decoder-only language model, RoboBrain 2.0 enables high-resolution image and video comprehension, as well as spatial and temporal reasoning. Through a progressive three-stage training strategy—encompassing foundational spatiotemporal learning, embodied enhancement, and chain-of-thought reasoning—the model demonstrates strong generalization across a wide variety of challenging embodied tasks. Despite its compact size, RoboBrain 2.0 achieves state-of-the-art results on most of public embodied spatial and temporal reasoning benchmarks, outperforming both open-source and proprietary models in spatial understanding, closed-loop interaction, and long-horizon planning. Its capabilities span a broad spectrum of embodied scenarios, including affordance prediction, spatial referring, trajectory forecasting, multi-agent coordination, and scene graph construction and updating.

We regard RoboBrain 2.0 as a solid foundation toward developing more general embodied AI, emphasizing the importance of tightly integrated perception, reasoning, and planning. Moving forward, we plan to expand RoboBrain 2.0 along two key directions:

• Embodied VLM-powered VLA: We aim to integrate cutting-edge embodied VLMs into the Vision-LanguageAction (VLA) framework. By harnessing the powerful spatiotemporal perception and high-level reasoning capabilities of VLMs, this direction seeks to substantially enhance the generality and robustness of action generation. The resulting system will support more nuanced understanding and precise execution of complex, open-ended instructions in real-world scenarios.   
• System-Level Integration: To improve RoboBrain 2.0’s practical utility, we will pursue tight integration with advanced robotics platforms and operating systems. This will enable serverless deployment, adaptation-free skill registration, and low-latency real-time control. In parallel, we envision building a collaborative embodied AI ecosystem—an “intelligence app store”—that supports plug-and-play components for perception, reasoning, and control in real-world robotic systems.

We release RoboBrain 2.0 at https://superrobobrain.github.io, including model checkpoints, training recipes, and evaluation tools, to support broader research and downstream applications in embodied AI. We hope this work bridges the gap between vision-language intelligence and real-world physical interaction.

# References

[1] Thomas Breuel Alex Aizman, Gavin Maltby. Webdataset: High-performance data loading for deep learning, 2020. URL https://webdataset.github.io/webdataset/.   
[2] Anthropic. Claude sonnet 4. 2025.   
[3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In CVPR, pages 19129–19139, 2022.   
[4] Alisson Azzolini, Hannah Brandon, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, Francesco Ferroni, Rama Govindaraju, et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025.   
[5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
[6] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.   
[7] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472–19495, 2024.   
[8] Tianqi et al. Chen. Gradient checkpointing in pytorch, 2018. URL https://pytorch.org/docs/stable/ checkpoint.html.   
[9] Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu. Egoplan-bench: Benchmarking multimodal large language models for human-level planning, 2024. URL https://arxiv.org/abs/2312.06722.   
[10] Wei-Lin Chiang, Zhuohan Xu, Hao Zhao, Shuyang Zhuang, Zi Lin Li, Yonghao Lin, Isaac Safo, Eric Singh, Rishi Taori, Noah Shinn, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, 2023. URL https://arxiv.org/abs/2306.05685.   
[11] Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S Yu, et al. Llm agents for education: Advances and applications. arXiv preprint arXiv:2503.11733, 2025.   
[12] FlagScale Contributors. Flagscale: A unified meta-framework enabling adaptive heterogeneous computing for the llm ecosystem. https://github.com/FlagOpen/FlagScale, 2024. Accessed: 2025-06-26.   
[13] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.   
[14] Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In ACL, 2024.   
[15] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In ECCV, 2024.   
[16] Google. Gemini 2.5 pro preview: even better coding performance. https://developers.googleblog.com/en/ gemini-2-5-pro-io-improved-coding-performance/, 2025. Accessed: 2025-05-06.   
[17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[18] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.   
[19] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364, 2019.   
[20] Zheqi He, Yesheng Liu, Jing shu Zheng, Xuejing Li, Jin-Ge Yao, Bowen Qin, Richeng Xuan, and Xi Yang. Flagevalmm: A flexible framework for comprehensive multimodal model evaluation. 2025. URL https://arxiv. org/abs/2506.09081.   
[21] Yining Huang, Keke Tang, Meilian Chen, and Boyuan Wang. A comprehensive survey on evaluating large language model applications in the medical industry. arXiv preprint arXiv:2404.15777, 2024.   
[22] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.   
[23] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1724–1734, 2025.   
[24] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6007–6017, 2023.   
[25] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474, 2017.   
[26] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models, 2022. URL https://arxiv. org/abs/2205.05198.   
[27] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.   
[28] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 2020.   
[29] Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, and Afshin Dehghan. Cubify anything: Scaling indoor 3d object detection. arXiv preprint arXiv:2412.04458, 2024.   
[30] Xuechen Li, Yifan Mai, Percy Liang, and Matei Zaharia. Energon: Scaling megatron-lm training with data and expert parallelism, 2023. URL https://github.com/HazyResearch/megatron-energon.   
[31] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
[32] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023.   
[33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.   
[34] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024.   
[35] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.   
[36] Gen Luo, Ganlin Yang, Ziyang Gong, Guanzhou Chen, Haonan Duan, Erfei Cui, Ronglei Tong, Zhi Hou, Tianyi Zhang, Zhe Chen, et al. Visual embodied brain: Let multimodal large language models see, think, and control in spaces. arXiv preprint arXiv:2506.00123, 2025.   
[37] Run Luo, Lu Wang, Wanwei He, and Xiaobo Xia. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.   
[38] Ruiyuan Lyu, Tai Wang, Jingli Lin, Shuai Yang, Xiaohan Mao, Yilun Chen, Runsen Xu, Haifeng Huang, Chenming Zhu, Dahua Lin, and Jiangmiao Pang. Mmscan: A multi-modal 3d scene dataset with hierarchical grounded language annotations. arXiv preprint arXiv:2406.09401, 2024.   
[39] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. In ICLR, 2023. URL https://openreview.net/forum?id=IDJx97BC38.   
[40] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training, 2018. URL https://arxiv.org/abs/1710.03740.   
[41] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE, 2019.   
[42] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the international conference for high performance computing, networking, storage and analysis, pages 1–15, 2021.   
[43] NVIDIA. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2021. URL https://github.com/NVIDIA/Megatron-LM.   
[44] OpenAI. Gpt-4 technical report, 2023. URL https://doi.org/10.48550/arXiv.2303.08774.   
[45] OpenAI. Gpt-4v(ision) system card. https://openai.com/index/introducing-o3-and-o4-mini/, 2025. Accessed: 2025-04-16.   
[46] Kun Ouyang. Spatial-r1: Enhancing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.   
[47] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.   
[48] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv, 2025.   
[49] PyTorch Developers. Cuda memory management, 2023. URL https://pytorch.org/docs/stable/notes/cuda. html#cuda-memory-management.   
[50] Qwen Team. Qwen2.5-vl: Multimodal llms from alibaba, 2025. URL https://github.com/QwenLM/Qwen2.5-VL.   
[51] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts and attributes of common objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7141–7151, 2023.   
[52] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. ICLR, 2025.   
[53] Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. Sat: Spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755, 2024.   
[54] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer, 2022.   
[55] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, et al. Robovqa: Multimodal long-horizon reasoning for robotics. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 645–652. IEEE, 2024.   
[56] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297. ACM, March 2025. doi: 10.1145/3689031.3696075. URL http://dx.doi.org/10.1145/3689031.3696075.   
[57] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.   
[58] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.   
[59] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15768–15780, 2025.   
[60] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question answering over text, tables and images. arXiv preprint arXiv:2104.06039, 2021.   
[61] Huajie Tan, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Yaoxu Lyu, Mingyu Cao, Zhongyuan Wang, and Shanghang Zhang. Roboos: A hierarchical embodied framework for cross-embodiment and multi-agent collaboration. arXiv preprint arXiv:2505.03673, 2025.   
[62] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.   
[63] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025.   
[64] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.   
[65] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.   
[66] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm. github.io/blog/qwq-32b/.   
[67] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. NeurIPS, 2024.   
[68] volcengine. verl: Volcano engine reinforcement learning for llms, 2024. URL https://github.com/volcengine/ verl.   
[69] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. Rio: 3d object instance re-localization in changing indoor environments. In ICCV, pages 7658–7667, 2019.   
[70] Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, and Song Han. Haq: Hardware-aware automated quantization with mixed precision, 2019. URL https://arxiv.org/abs/1811.08886.   
[71] Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, et al. Gui agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890, 2024.   
[72] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024.   
[73] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.   
[74] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14203–14214, 2025.   
[75] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.   
[76] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016.   
[77] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics, 2024. URL https://arxiv.org/abs/2406.10721.   
[78] Wenqi Zhang, Mengna Wang, Gangao Liu, Xu Huixin, Yiwei Jiang, Yongliang Shen, Guiyang Hou, Zhe Zheng, Hang Zhang, Xin Li, et al. Embodied-reasoner: Synergizing visual search, reasoning, and action for embodied interactive tasks. arXiv preprint arXiv:2503.21696, 2025.   
[79] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In CVPR, 2024.   
[80] Enshen Zhou, Qi Su, Cheng Chi, Zhizheng Zhang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, and He Wang. Code-as-monitor: Constraint-aware visual programming for reactive and proactive robotic failure detection. arXiv preprint arXiv:2412.04455, 2024.   
[81] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308, 2025.   
[82] Hao Zhou, Chengming Hu, Ye Yuan, Yufei Cui, Yili Jin, Can Chen, Haolun Wu, Dun Yuan, Li Jiang, Di Wu, et al. Large language model (llm) for telecommunications: A comprehensive survey on principles, key techniques, and opportunities. IEEE Communications Surveys & Tutorials, 2024.   
[83] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.   
[84] Shengjie Zhu, Abhinav Kumar, Masa Hu, and Xiaoming Liu. Tame a wild camera: in-the-wild monocular camera calibration. NIPS, 2023.

# Core Contributors

# Model Training

• Mingyu Cao∗   
• Huajie Tan∗   
• Yuheng Ji∗   
• Xiansheng Chen∗†   
• Minglan Lin∗†   
• Zhiyu Li   
• Zhou Cao   
• Pengwei Wang†

# Data & Evaluation

• Enshen Zhou • Yi Han • Yingbo Tang • Xiangqi Xu • Wei Guo • Yaoxu Lyu • Yijie Xu • Jiayu Shi • Mengfei Du • Cheng Chi† • Mengdi Zhao • Xiaoshuai Hao

# Research Leads

• Yonghua Lin • Zhongyuan WangB • Tiejun Huang • Shanghang ZhangB

# Contributors

# Real-Robot Experiments

• Junkai Zhao • Xiaojie Zhang • Shanyu Rong • Huaihai Lyu • Zhengliang Cai • Yankai Fu • Ning Chen • Bolun Zhang • Lingfeng Zhang • Shuyi Zhang • Dong Liu

# Product & Operations

• Xi Feng • Songjing Wang • Xiaodan Liu • Yance Jiao

# Infrastructure

• Mengsi Lyu • Zhuo Chen • Chenrui He • Yupu Feng • Yulong Ao

# Evaluation

• Xue Sun • Zheqi He • Jingshu Zheng • Xi Yang

# System Management

• Donghai Shi • Kunchang Xie • Bochao Zhang • Shaokai Nie • Chunlei Men

# Appendix

# A Qualitative examples

This section provides a comprehensive set of qualitative examples that illustrate the capabilities of RoboBrain 2.0 in various embodied AI tasks. These examples demonstrate the model’s proficiency in spatial reasoning, temporal planning, and interactive reasoning, showcasing its potential for real-world applications.

# A.1 Examples for Pointing

In the pointing task, RoboBrain 2.0 is required to identify and point to specific objects within an image based on complex spatial instructions. For instance, given the instruction “Please point out the orange box,” the model accurately identifies the orange box in the image. Similarly, for more complex instructions such as “Please point out the brown box on the shelf,” RoboBrain 2.0 demonstrates its ability to understand spatial relationships and accurately points to the correct object. The model’s proficiency in this task is further exemplified by its performance on a variety of pointing examples, as shown in Figure 5-Figure 20. These examples highlight the model’s robust spatial reasoning capabilities, enabling it to handle a wide range of pointing tasks with high precision. Whether the instructions involve simple object identification or more intricate spatial relationships, RoboBrain 2.0 consistently demonstrates its ability to accurately locate and point to the specified objects. This capability is crucial for applications in robotics and automation, where precise object localization is essential for effective interaction with the physical environment.

![](images/5199e0fb9c1ba6a4bd429003f1e686291b66a6a67c9d6010473d6603fab118c7.jpg)  
Figure 5 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

![](images/7c676fbee81cf6b3b551c5244bf6c3055d4176edf198477d948f9a26419d72bf.jpg)  
Reasoning $S t e p = 1$

Please point to the farthest white cabinet.

![](images/20f85a78c4d1dc6a851ce65d3bd544b5679cfa72b3b109131a31889bb4f8c6a5.jpg)  
Reasoning Step = 2

Please point to the top piece of paper on the white table.

![](images/72a221505e4044bce69008d45bfcb8c347196d05107208ee0e81ea7b55648c05.jpg)  
Reasoning Step $= 2$

Please point to the left pillow on the sofa

![](images/0529ec1c72ca54dde490c67a76688dcf48aac24609a28fd5d0f1d15b4957e047.jpg)  
Reasoning Step $= 3$

![](images/34228a7114eced7df07fc7f5316cb5272c8c6ce888e309d89f3209603ca79e6c.jpg)  
Reasoning Step = 2

Please point out the leftmost black object on the same platform as the micro-wave oven.

![](images/a5c3fc5c47a657565c93a50134ad18262db6de90665fbe9a37f6528c084f901d.jpg)  
Reasoning Step = 3

Please point out the orange box on the white table on the left.

Please point out the white cup on the shelf behind the chair.

![](images/660df230436329f985d7d9a0feb5b530770b2f005a3134e762e2b7d0b8f39389.jpg)

# Reasoning Step $= 2$

![](images/99d2b419d00bea52610a211410b05f002d54d46d362061d10769b8a041ee8868.jpg)  
Reasoning Step $= 3$

Please point to the rightmost blue box on the refrigerator.

Please point out the cardboard box under the bed which is closest box to the viewer.

![](images/f4712c9675dd52fdc4ccb9764d1906fca61a3b3b77b1f7125a93e4d472007af8.jpg)  
Reasoning Step = 2   
Figure 6 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the blue object on the table.

![](images/bb960538bea1431e3dffa4ec5f5f01c5dafeb3ca8aae7d06d080dac125b0e9b4.jpg)  
Reasoning $S t e p = 3$

Please point to the white bottle on the table that is closest to the green bottle on the left.

![](images/3f238d8c9f7985b0634c30390b230a5921f9399495c1bdf86fa7c9b9127f5c06.jpg)  
Reasoning $S t e p = 2$

Please point out the second object from the left to the right on the nearest platform.

![](images/da42c6c96ead10ed2388e038a94169f48bed7efecbce5ce66cdff28801e423d6.jpg)  
Reasoning Step = 2

Please point out the object on the right of the shovels.

![](images/7ffc5265b97e7a918dc1c50a457b032424e09799904f091ad938a8002e69f292.jpg)  
Reasoning Step = 2

![](images/a969622356830a650573479e6b0c66b9f491370e1da8a9e59f83e1ad8de392e9.jpg)  
Reasoning Step = 2

Please point to the pillow closest to the right nightstand.

![](images/4f95dcf470e23908d0cf99362f35253518dd12ebce274d1cc340f4dc10e79904.jpg)  
Reasoning Step $= 3$ Please point out the object between the white box and the farthest black pot.

Please point to the pillow closest to the remote controller.

![](images/218b0113a0683ff227bd5fa1ecf4b74f19834921829f012c6d8f7fd306a78f8d.jpg)

# Reasoning Step $= 3$

![](images/869e1b0c5bf0fa89230d98d6f9ef83d18ea466a1a3f957877059a8aa558ce3af.jpg)  
Reasoning Step = 2

Please point out the black object that is on the same platform as the TV.

Please point out the vase closest to the TV.

![](images/e6b8d0678f6ca273da3a82961a3a2f3aadaeca0940d834cceefd3763a87ea024.jpg)  
Reasoning Step = 3   
Figure 7 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point to the rightmost box at the bottom of the shelf.

![](images/346632dafce3b56380528f7496d0c735756a8c2e13cfaed78289fb057065d622.jpg)  
Reasoning $S t e p = 1$

![](images/c87e220ab9f1211445767999f9a17663741323509d01c741ed34ccfbbdd2b6d4.jpg)  
Reasoning Step = 1

Please point out the second silver box from left to right.

![](images/84b7826ca3616d7874da33ecd2a2f1aa82669e423b87fb9d48f4c077c4827300.jpg)  
Reasoning Step = 3

Please point to the wooden plate on the far left.

Please point out the black framed painting on the right of the lamp.

![](images/780e7ff7df7e629e3f6d22177d83275e5a232ace4e20b5d745cb32ce938e6d5b.jpg)  
Reasoning Step = 1

![](images/1dcf7ce854c9049283a7f55079ad0f402c2255faa0a30199b98e7fafcad3674c.jpg)  
Reasoning $S t e p = 2$ Please point out the green towel on the upper right with yellow object on top.

Please point out the chair closest from the viewer.

![](images/d854956af48302da3fde8aa79b14f381d123c7dceaf764c692cd1410db0efe1b.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$

Please point out the brown sofa, which is the closest sofa to the viewer.

![](images/d616ec9e0c0214bbc5fe55ce8cbcd6b78282a0408ca91d51037f3c65a4c34a30.jpg)

# Reasoning Step $= 2$

Please point out the object on the windowsill farthest from the viewer.

![](images/274e1e30339975a21c782463f66b7d974255382a14df6c4243f8e2d95aab9717.jpg)  
Reasoning Step = 1

# Reasoning Step = 2

![](images/04b65904ef358c5b9eba784b100cc7cd4d56d8ad88b0da81cd00941157ad802b.jpg)  
Figure 8 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the black object which is farthest from the shelf.

Please point out the paper tube closest to the viewer.

![](images/d5d73c3707af4f2fa0095a28fcd3f24fcba66a8c068e82ef35ed77fc40bb208a.jpg)

# Reasoning $S t e p = 1$

Please point out the sofa on the right side that is closest to the viewer.

![](images/72d2832c735673f73c58c8165e377a0943e75266fa31be2276e9aa08cad70bb7.jpg)  
Reasoning Step = 1

Please point out the sofa farthest from the viewer.

![](images/26ee5b59217b71b66bfc46088b6570b9140372e7dcae79f9373fb4d2fa3a08ae.jpg)  
Reasoning Step = 2

Plea se point out the painting hanging on the wall.

![](images/fc522065660aa4ea17d76ecff9952df0ac592e43e96f4df02e3b7bbb08831fbb.jpg)  
Reasoning $S t e p = 2$

Please point out the blue toothbrush farthest from the faucet.

![](images/3093855e686184466ada056399eae290a8c7c6e3a8967d5330bbc9cddd6aa22d.jpg)  
Reasoning Step = 2

Please point out the card closest to the wooden door.

![](images/efbc75654e0f2ef2da249c1776c933e014dab4b28cb7d3c282d901b455e50e8d.jpg)  
Reasoning Step ${ \tt = } 2$

Please point to the third card from right to left on the cabinet.

![](images/76bf601fb4b9e4bdf514ca8b8b42006f153543d57420cb9020efc342fa133631.jpg)  
Reasoning Step = 1

Please point out the brown object farthest from the viewer.

![](images/2e7d9432a4a2b671bc9a562ecd59fd79272b981dd9b03cadb220e2b03720e332.jpg)  
Reasoning Step $= 2$

Please point out the white object on the cabinet farthest from the viewer.

![](images/9bb4d42d348f3d11f5f22f39248fed07f766d55808b473ebb4b232d2922ad5bc.jpg)  
Reasoning $S t e p = 3$

Please point out the white object adjacent to the left side of the picture frame on the cabinet

# Reasoning Step = 2

![](images/812d01d5e80fd9e44bdbac2e2cbda5e3a61ecbad4eaabb41a9475508b45f0046.jpg)

Please point out the closest red box to the blue box.

# Reasoning Step = 1

![](images/233eb4cde66bfae36e1eced93a370d83ed01aaa9b580d889f2b167dbaf5ae8c6.jpg)  
Figure 9 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

![](images/0b9e035eb257e663d2ae3cc6b23517e883bfda610ae6ab56cb2d6a18e4be8515.jpg)  
Reasoning Step = 2

Please point out the second closest cup to the viewer.

Please point out the stool which is farthest from the white table.

![](images/5a17671ef46332c2e17d7b11074856b453b097b6c2b0396889517a630ee99962.jpg)

# Reasoning $S t e p = 2$

Please point out the free space on the second shelf of the wooden shelf.

![](images/0b96cef1661433e79db16b4515a212f694cb1c46702935eeaba5c54c1f259964.jpg)  
Reasoning Step $= 3$

Please point out the free space in front of the brown object on the shelf.

![](images/19cb3ea0ef6dbb806bf2b468c46e8ca29bb5fc5d47520d5cee70df841f56bd24.jpg)  
Reasoning Step $= 3$

Please point out the free space between toilet and shelf.

![](images/fa48011ccbe99cc1a421d53cc6957d95e11b179df69bc3e75731b04c5c77ef57.jpg)  
Reasoning Step = 2

![](images/2e59be75deb0c1369c634cf228d8cdc3d201660d8c797de790488bf077191ad3.jpg)  
Reasoning Step = 2

Please point out the free space on the white table at center.

![](images/20baad4d23e0d3b40dda3ee8affe131941964b27efb7320fcca31110264c96cd.jpg)  
Reasoning Step $= 4$

Please point out the empty area to the right of the leftmost stool.

Please point out the free space in front of the blue box which is on the top of the shelf.

![](images/377e8abf3bd5e2f50ce5cec1970a786bb4447b37c73fe944013fd5abdb02e887.jpg)

# Reasoning Step $= 4$

Please point out the free space in front of the white vase which is on the top of the shelf.

![](images/10cf27b0134865d42ee0403d23443dec7c5aebdc79474c8fd7611b805fbec5d0.jpg)  
Reasoning Step = 2

![](images/656deda54d364b80b6fd54f37f9bbbf130a4853eecffdc3f22f295cf352bdfce.jpg)  
Figure 10 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

# Reasoning Step = 3

Please point out the free space between the cat tree and litter box.

Please point out the free space in front of the litter box.

![](images/d9626387faddc54790396e0567e5975e7ad0f40c9e43b987b4c28e3573458f4d.jpg)  
Reasoning $S t e p = 2$

Please point out the free space on the lowest shelf of the shelf.

![](images/10f1b60abc7615e8b0bc642fcb57bbcd132b580b6573c56fdf9b1fd6ab757b46.jpg)  
Reasoning Step = 3 Please point out the free space between the black water bottle and the pot lid.

![](images/d087d40e62bef6581c985a632b9d165032fe5b261a32658817868dcbbbb93fd4.jpg)  
Reasoning Step = 4

Please point out the free space between the black water bottle, the pot lid, and the scissors.

![](images/8a9802b14a244b9cdec7c335258d26097e0335b9cb3d0e524c49941c47063288.jpg)  
Reasoning Step = 2

![](images/5ffe28621fd80529b876525b7b73e0e3772e5bac13baf5eec2fa5707e8beab44.jpg)  
Reasoning Step = 2

Please point out the free space on the right of the farthest pot.

![](images/34b9a672095565478dc50491a766b8eb4b996014e06fe7833244ebb770569756.jpg)  
Reasoning Step $= 4$

Please point out the free space inside the closest pot.

Please point out the free space between the black plate, blue can and closest water glass.

![](images/6e683f1cde0f6228506e4783fd0216ac8906783ad863fe8975c4ef4befbf26bb.jpg)

# Reasoning Step $= 2$

![](images/405041e1618f3109291f9d9dfcaaca528af0ca18fe03021914f788ca0304437e.jpg)  
Reasoning $S t e p = 3$

Please point out the free space in the top corner of the table.

Please point out the free area on the table in facing direction of the second chair from the front on the right side.

![](images/d2a9dd38c78f24aae2f30260ec10d81dceb222af25aaefd28f33d47eb8530697.jpg)  
Reasoning Step $= 3$   
Figure 11 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free area on the table that the second chair from the front on the left side is directly facing.

![](images/c05da10c6165bf5ed748802ffc864a2754fd3c8d90ffd4642e62258f0a287a4b.jpg)  
Reasoning $S t e p = 3$

![](images/1adf84568dc3c72fc7b4ae7c1934260974b61cdfe29ec9c307d5402f2c710f55.jpg)  
Reasoning $S t e p = 4$

Please point out the free space between the scissors and the microwave.

![](images/b28212aee4cc59fb3e04b75535410d76556a474ef7664b680eed1a640f1c28c2.jpg)  
Reasoning Step $= 5$

Please point out the free space between the headphones farthest from the monitor and the keyboard.

Please point out the free space between the black cloth box to the bottom-right of the monitor and the keyboard.

![](images/12528f8ff2f2dd42069a78828a3c17820b356338bc768147b966bff03b379af0.jpg)  
Reasoning $S t e p = 3$

![](images/c6924a90cec151b0efb34c1c823454eaf36379041db608280e70b4a4ba477838.jpg)  
Reasoning Step = 3

Please point out the free area on the table that the first chair from the front on the left side is directly facing.

![](images/54025dc1f2ed63efb116c243434331349b1d870d52138e9b571d25f8c7304384.jpg)  
Reasoning Step $= 2$

Please point out the free area on the table that the first chair from the front on the right side is directly facing.

Please point out the free area in the top-left corner of the table.

![](images/78191decd8e115a80b7b60b47b2e2806f624515a9f99d34df983b12567a3a90b.jpg)

# Reasoning Step $= 3$

Please point out the free space on the stovetop in front of the black pot suitable for placing another pot.

# Reasoning Step $= 3$

![](images/de64e717be5f51888b61b7a5a7c3af05bfd8cbf587613db839ea2d44c1658503.jpg)  
Reasoning Step = 2

![](images/ec58e71bffeb6e27b17ad5e11f9a81830803c935ed2cb2abfef288498c4dfd89.jpg)  
Figure 12 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free area between the black container for spatulas and the black object on its right side.

Please point out the free space in front of the cat tree.

![](images/1b59768bcd84c887ce232cc23b92f987a6a252ebc98ace40379f89f00250f823.jpg)

# Reasoning $S t e p = 4$

Please point out the free space on the table between the keyboard and the viewer.

![](images/9a4965072e397b8873b2011e9aedceb22774a4e37254e42163f92ac579128fd9.jpg)  
Reasoning Step = 4 Please point out the free space on the toilet between the blue bottle and the red can.

![](images/daa317a851acd8507e8ef24db0c6ff304c50f9b1dd767f8ac7f7a46ea4ca5479.jpg)

# Reasoning Step = 3

Please point out the free space between the bathtub and the toilet.

![](images/7bd5708235422a193851d26be2d7c3456eae4c94bd12e9cb35749849b537e910.jpg)  
Reasoning Step = 3

![](images/923ca3d42ebccba6691debf9a371ce500791a98ee4aa4e34269b4f02014ce4a2.jpg)  
Reasoning Step $= 2$

Please point out the free space between the cat tree and the chair.

![](images/936f18cd94fa7fe8e41207eb516e001aed6953ddeac437ab1808363da6dd7ebc.jpg)  
Reasoning Step $= 3$

Please point out the free space on the lowest shelf of the cat tree.

Please point out the free space between the purple vacuum cleaner and the cabinet on the left.

![](images/b497ed114dd5a49e4f118991a34185075227308fa111568ef2a78b1c957dbe3b.jpg)

# Reasoning Step $= 2$

![](images/b9c3632261546fa6b8ae2a2984f69c4c6a947c479fe8d6b4785ab463a81af502.jpg)  
Reasoning Step = 2

Please point out the free space below the table.

Please point out the free space on the sofa cushion.

![](images/98669d61aed0695ef8e32f3e5f656b624a7d2fae5e4c44c2510b4fb1ec89c682.jpg)  
Reasoning Step = 3   
Figure 13 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free space between the table and the sofa.

![](images/bbedbf71e81f9a1f0ce9efc2adbbfdd5e168ad896bcaff3b747869d7aacd2735.jpg)  
Reasoning $S t e p = 3$ Please point out the free space on the cabinet to in front of the brown vase.

![](images/27e17638395190b84b3c2fb26282633952c32f63e2ca55929ea50212b8cdf9a0.jpg)  
Reasoning $S t e p = 4$ Please point out the free space on the cabinet between the brown vase and white bottle.

![](images/25f132244e363ff2ad00a40367925c24668923e14a9db2dcb1854073b5ff6512.jpg)  
Reasoning Step = 3

Please point out the free spot between the blue water kettle and the orange.

![](images/97b32ae92f59629d653e3b6984969eb35b178a21ed3fbde26966850a7e0070da.jpg)  
Reasoning $S t e p = 5$

![](images/6b9adf1ee5d6e43f365dbd9e701c2059049e9a3d45e174bbb91ee50927076397.jpg)  
Reasoning Step = 2

Please point out the free space on the table between the speaker to the right of the monitor and the mouse.

![](images/e4b094a901dd9dca4022ecd8e9f609354313ef0deee1c84408facf1a3be41c7e.jpg)  
Reasoning Step $= 5$

Please point out the free space on the corner of the black table that is closest to the viewer.

Please point out the free space on the right part of the table between the mouse and the picture frame.

![](images/4a197e9c1a2220cb39ef38728d8a72452de6b9d1ea10fa4a52a01eb51f0bda45.jpg)

# Reasoning Step $= 4$

Please point out the free space on the table between the pillow and the brown bowl.

![](images/4695e9e5418be261779016c3005604ba00639c1d54dca8ef3b758c20f2a7b3cb.jpg)  
Reasoning Step = 2

![](images/2ffe0037ce09e33f641551e31ed2210d0745feeac53229afddd31919d000e367.jpg)  
Figure 14 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

# Reasoning Step = 3

Please point out the free area on the stovetop to the left of the pot.

Please point out the free space on the left of the brown shelf.

![](images/cf4607d88761bf82166488040551b7129fd28cc1ce4c07e7d5bf648645a23f0e.jpg)  
Reasoning Step $= 5$

Please point out the free spot on the table to the left of the two pink dumbbells where another dumbbell can be placed at an equal distance.

![](images/712a75d18fa5fa33c4519a5791e078610cc92d127457c83089198a870c4f8274.jpg)

# Reasoning Step = 4

Please point out the free spot, equidistant from both the blue bowl and the red bowl, and between them, where another bowl can be placed.

![](images/b4c594d0968d07bd89ea95e0bd88b8671cc7a35ef845b45a65c9f8476a07fa63.jpg)

# Reasoning Step = 4

Please point out the free spot behind the pink cup, such that distance to the pink cup is equal to distance from the pink cup to the red bowl.

![](images/776034ded50e4fa9054fefb7e23e331d6969a5d01a98b0a80a9b0e8fdc43f432.jpg)  
Reasoning $S t e p = 4$

Please point out the free space midway between the first and second green cups from the left.

![](images/5079420708891471edc619810ad295c09cf9f945e54c6b6927f1f52bba21f252.jpg)  
Reasoning Step $= 2$

![](images/b2d55d15aaf60494e144fb91b21abac941b93e7adcd8229cf446427a1d02c206.jpg)  
Reasoning Step $= 3$

Please point out the free area in the direction of the handle of the rightmost green cup.

Please point out the free space between the mouse and the green cup.

![](images/00571b014fa32d3c7718093f98e7592422303c47de2079c83a6b73c688ce5ddc.jpg)  
Reasoning Step $= 2$

Please point out the free space in the direction of the handle of the second closest cup to the viewer.

![](images/b88a27519b8f5d6d285775d2b21fba1e49a5b60003fc542a6eb9135f1e7b2fae.jpg)

Reasoning Step = 4 Please point out the free spot to the right of the cloth bag, where an object of the same size can be placed at an equal.

![](images/2fb7656a0c55516d749087a7314c09be1936700236388490707e52c1eeacd33a.jpg)  
Reasoning Step = 2

Please point out the free space in the direction of the handle of the transparent glass cup.

![](images/f18af154471e74e5dbd81683674010308c33cbbba2b428ae6ea12ec24eb688ce.jpg)

# Reasoning Step = 2

Please point out the free space in the facing direction of the purple bag.

![](images/b3743968424f1b7f1ead8848c8b4f93eca4766cf167e3ad9432242c487e65298.jpg)  
Reasoning Step = 2

Please point out the free space in the facing direction of the orange box.

![](images/1a1cff70700c17f435f25de7f911722e8a2b1d8fa5296279f1e09460b6d63f72.jpg)  
Reasoning Step $= 3$   
Figure 15 Pointing Examples of RoboBrain 2.0. The blue point represents the model’s spatial referring prediction.

Please point out the free space between the red box on the left and the black box.

![](images/208466a467142ba6b47a616ecfeacb4e2fbe7c7cc03bebada73ac8e00cb09310.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

What part of a mug holds the liquid inside for drinking?

![](images/8474bc73150c49de08542d596819a30b7a0ecd18f8c9cfeafb2e3b43c715f111.jpg)

# Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

Which part of the yellow bottle can be removed to access its content?

![](images/f28e36ad2211b4800ce37aeb8bb001339fe33b8c59766ee27e97187acee61e32.jpg)

# Reasoning Step = 1

What utensil can be used to scoop and transfer food to your mouth?

![](images/89900d5acc9a67199144177873e6dd701f930fd5eb4e418410e4ec9e5b7520e4.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

What part of a mug should be gripped to lift it?

![](images/91571c139fc5e9fa876f637aa38328ca86dbca1d63e870b017d2799c7a03bd91.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

What object can be used to hold and drink beverages?

![](images/99ed4554e299d3dbc75c8ba8452440f8b07d2d08bcf536ca2d5b0081005578f2.jpg)  
Reasoning Step $\sp { \bullet 1 }$

What object can be used to input text and commands into a computer by pressing its keys?

![](images/8db46d8ce64f23b1ad878ebea221d50444f17e70cb4f6088c70dce455cd9af74.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](images/33321b465884e9a5eeeca152b89c410a534513a4eec319da52cbb251b829c2e1.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](images/b3f498ce384605a6d57e23c007b963d90c94a46ccbc975d074677f74fce30db9.jpg)  
Reasoning Step $\sp { \bullet 1 }$

What part of the cabinet should be pulled to open it?

Which part of the bowl can be filled with soup or salad?

What part of a mug should be gripped to lift it?

![](images/4ef5bcf478b1c66c268446bc68388d440597a8c75969e4d8e5d742f73e5e64d2.jpg)

# Reasoning Step = 1

![](images/793b79b51509128360ee4961786fb3c97285b2c9f7f0ff8c43486f43397ace5d.jpg)

# Reasoning Step = 1

# Reasoning Step = 1

![](images/d31d6dce16aafc18ac70d256e4c12702bf7d3b870043bd8819e9c7af955dd8d5.jpg)  
Figure 16 Pointing Examples of RoboBrain 2.0. The objects or their parts are pointed according to their affordances queried in the instruction.

What object can be stacked or used as a building block in a structure?

What part of pan should be gripped to lift it safely?

What part of a faucet should be turned to control the water flow?

![](images/45974ac92d1d2f67592302af9e720aaca7bb5f301aeba1ed246ea522810223c9.jpg)  
Reasoning Step $= 2$

Mark the yellow bottle nearest to the bowl in the image.

![](images/ef89c8415a19caad13a2587286082748dfe3c024cd11c6fdc9090d0c70fea79f.jpg)  
Reasoning Step = 2

Determine several points on plate nearest to the cup in the image.

![](images/4000a4857c8b04d2145201b975246ce2e5839476df9375324473151e8b24920f.jpg)  
Reasoning Step $= 2$

Indicate the plate nearest to the cup in the image.

![](images/738e0edabd72c97e6b7ba8974d6984ed3b027d852c4e605a647167cd01229edf.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](images/765b4319069643ed3bf9b1bbe88d66b99226b275bc844438bca96221579edf37.jpg)  
Reasoning Step = 1

![](images/6ebb79fce885d420d1c54ec6e9eedbc5afdf5d11ec451bd613d2fed47d1ba63c.jpg)  
Reasoning Step = 2

Highlight the middle mug in the image.

Find several points on the front can in the image.

Identify the bowl right to the peach in the image.

![](images/b1517e4e1f7ad92b90da692a226cf8f0e4f1421af6bf898abffb835ced86cdb9.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

Mark the left sink in the image.

![](images/29c0d5d3c90c9297c9442709197b95a455ad765c7820ff7b1d7115a57c8bfd77.jpg)  
Reasoning Step $= 2$

Highlight several points on the notebook near to the plastic container in the image.

![](images/4b746af7c2970b5e03a706eed74a52d49d7d6ecb6a9f3a0208de02733ee9372a.jpg)

# Reasoning Step = 2

Spot several points on the can left next to the apple in the image.

![](images/a6164ffc1535728e1646ebb190fa25ec3578e5b61131776fe47e82c96a057cf5.jpg)

# Reasoning Step = 3

![](images/aa2dae4db9580866b6a6b1339ee00f958fb5f4f7d0adb33f3f18f2e845c50df8.jpg)  
Reasoning Step = 1

# Reasoning Step = 2

Identify the cup right to the bowl below the cabinet in the image.

![](images/2d8d2baa711dd980b0b3ebe742a242102d8bc7346ecd6e81d7be6b1dccce5c87.jpg)  
Figure 17 Pointing Examples of RoboBrain 2.0. The objects referred by spatial relations or object attributes are pointed out.

Highlight the rightmost fruit in the image.

Determine several points on the plate in the right sink in the image.

![](images/6654ef4c14e72db1f45081625cab3ada7202e410deebc370ad6af62d5e13f5f1.jpg)  
Reasoning Step $= 2$

![](images/c6392a9fc2fd66752ed4b333ff73ac23b4cd287aa741d88a3df675b72c44a31b.jpg)  
Reasoning Step = 2

![](images/9d2f6802874b25b9344eeb72758e2d3195146882b197de0c4d57dc1cbf8e9551.jpg)  
Reasoning Step $\sp { \bullet 1 }$

Highlight several points on the bowl right to the orange in the image.

Highlight the object right to can in the image.

Mark the rightmost object in the image.

![](images/c89c1273f16f2af926f63f70c77bf7e0dfef3504da70fec3ef312967139597c7.jpg)  
Reasoning Step = 2

![](images/d97c6260a768e19fc5d30648ac7569dc25a8847d31d09573f8442ec9c9b7262e.jpg)  
Reasoning Step = 2

![](images/91f3f9dba21a4a0a31dcd8a6602a344093ddbc79c01a94cc54fc4280fb3e0b28.jpg)  
Reasoning Step $= 2$

Pinpoint the chair right to the table in the image.

Recognize several points on the box nearest to the pan in the image.

Pinpoint the bottle in front of the can in the image.

![](images/a4520bd41eb446bcaf2fc3407b78c2060e2f21382e2467f0471a55055fa2a767.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \lambda }$

![](images/be0c837bb6142ac97ee31b82c6a87c0a978e07f85c37af217bf4f20535be7adb.jpg)  
Reasoning Step = 2

![](images/a78c52568b6c856fd8658ffd3a0262066f7adac312960b3da796b308bcf1973b.jpg)  
Reasoning Step $= 2$

Identify several points in the front sink in the image.

Pinpoint the box left to the yellow can in the image.

Highlight the bottle left to the lamp in the image.

![](images/e3f59a69b500cd5d9fda4c4d57386a67f011883adc3098faec77a139e60dfc60.jpg)

# Reasoning Step = 1

![](images/d0750768e71d16c463d61997496dde17f9b7b09616e2d3034d3d0d5b687721ea.jpg)  
Reasoning Step = 2

Indicate several points on the top drawer in the image.

Highlight the box behind the pan in the image.

![](images/f383b56ad07f75a3c2407abe7ac50e00bfd48ba65dfa8f5422e8f6d9940f53b6.jpg)  
Reasoning Step $= 2$   
Figure 18 Pointing Examples of RoboBrain 2.0. The objects referred by spatial relations or object attributes are pointed out.

Locate several points on the building block in the bottom-right corner in the image.

![](images/ae4e1297c2d35ba4609728f8922c793644e5601ab0035832c504f32cf49e5701.jpg)  
Reasoning Step $= 3$

Identify several spots within the vacant space that’s between the two mugs.

![](images/c68d64a814c8b611ad32dc8e11c476e2b3ad658c913c1a9dd88b765add517eb5.jpg)  
Reasoning $S t e p = 2$

Locate several points within the vacant space positioned to the left of the yellow mustard bottle.

![](images/b00028429695de20acf045db5aed70c1bdc0e16ef3baea37dc7780aa2a56e7f5.jpg)  
Reasoning Step $= 3$

Locate several points within the vacant area that is situated to the right of the teal plate.

![](images/a89108e0db2f6b3e6cbf97f45b1479de6c93d87704a4867ce339ec37d3d7d619.jpg)  
Reasoning Step $= 3$

Select one or more locations within the vacant area that is in front of the mug in the middle.

![](images/eaaab94989e37cdc2da82afa6203607f2b9f04928aa31ef3e88ec818452bdeab.jpg)  
Reasoning Step $= 3$ Locate a few places in the free space between the orange and the plastic cup.

![](images/361fcdec395c8086a2e42bc0cda22fc6408b1493878d45634146f9049df749ab.jpg)  
Reasoning Step $= 2$

Locate several points within a vacant area on the back side of the stove.

![](images/f106902f300fc2df636e41440cbb7dc6cc9e3a6a6e122b795563c1310b6c0639.jpg)  
Reasoning Step $= 2$

Locate several points within a vacant area on the front portion of the stove.

![](images/9c75e9df1e1613d4874a941e75a1de70465ba403226898096ad94883196f728c.jpg)  
Reasoning Step = 2

Locate a few spots within the unoccupied space behind the mouse.

![](images/5a26a6bf0040741de9e0b0ad162b68445243efdadb214ae5d1aa6c4b64b11779.jpg)  
Reasoning Step $= 2$

Locate a few spots within the unoccupied area inside the cabinet.

![](images/284b2af54d1499fda91d975ff857ea3503116b4e61c7e0f90619354a53e4cdcd.jpg)

# Reasoning Step = 2

Locate several spots within the unoccupied area beneath the apple.

![](images/bb275058c0bb1535fcfa8fee246f4f2a1381b908be3cac631ec79c7005c66067.jpg)  
Reasoning Step $= 3$

Locate several spots within the vacant area that is in front of the bowl on the left.

![](images/6be89d46e68d375b855f6f32ad67f0a105fea897e961f80e49a8d7a8fda81daa.jpg)  
Figure 19 Pointing Examples of RoboBrain 2.0. The free space indicated by spatial relations and the referenced objects are pointed out.

# Reasoning Step $= 2$

Locate several spots within the vacant area that is in front of the teal bowl.

![](images/f934257620fd7f453e1690d1dab7c8c10dd8ddadedbf2d42e3a85b667e865aee.jpg)  
Reasoning Step $= 2$

Locate several points within the vacant area that lies before the plastic container.

![](images/1c87a7a89ffba5c2581218c799ba5d9878dcbb0ed6384a21d0aba0d1ce71f709.jpg)  
Reasoning $S t e p = 2$

Locate several points within the vacant area that is in front of the blue cup.

![](images/4a1a2e4d093a131cd3170cad6891a0542ce84c2f9f54251fcb11ca15f366b514.jpg)  
Reasoning Step $= 3$

Locate several spots within the vacan space situated above the leftmost item.

![](images/e75d8117ad9be2b30691bd13b20f131247280ae96ab00eacceb05c6e22d169b0.jpg)  
Reasoning Step $= 2$

![](images/9adeecfa3d13694079c5ed3d53225262b8f3cc25e4b8b448ce87a11d78eab6b8.jpg)

Pinpoint several spots within the vacant area located to the righthand of the green container.

Reasoning $S t e p = 4$ Identify several points within the vacant area that lies between the blue cup and the teal bowl on the table.

![](images/e69fa32e2a17aee56d476abb13cc283874881e7b8421de0bb3aead34d9da360b.jpg)  
Reasoning Step $= 4$

Locate a few points within the unoccupied space that lies before the leftmost fruit on the table.

![](images/b12a40413993ce71de829347aee23e8418590bc0936e7b718f3e23b6eb844e50.jpg)  
Reasoning Step $= 2$

Locate a few points within the vacant space to the left of the frying pan.

![](images/1d33dac865d2a229ebbb1038c5d60e642b7023cdd9be4c4a7559bd151be223d3.jpg)  
Reasoning Step $= 2$

Locate several spots within the vacant area situated to the left side of the orange.

![](images/be25417eeb223411bacdc2aa03e1d7f41985f84d3210beaec3bb0a70f923777b.jpg)  
Reasoning Step = 3

Locate several spots in a vacant area next to the white mug.

![](images/1cc81ca6de542e1d13136beaec42398fb9179b68449a6b63c068450e4b7c1d1d.jpg)

# Reasoning Step $= 2$

Locate several points within the vacant space situated on the left part of the cabinet shelf.

![](images/0f686052a0aa1b0426207d6736c2de2ff9cc99ddddd7557770bffb91fc496f85.jpg)  
Reasoning Step $= 2$

Locate several points within a vacant area on the front side of the table.

![](images/33d8c7ad387f87e68a8e07ec548e590574fe7d9372dd88e86d4b76ba3a9f4827.jpg)  
Reasoning Step $= 3$ 3   
Figure 20 Pointing Examples of RoboBrain 2.0. The free space indicated by spatial relations and the referenced objects are pointed out.

Find a few points in the free space in front of the window on the left.

![](images/7d015b2968a5e65ddfbf19fd3dadd546b89413ee9636e26f0950515f1b81e1a8.jpg)  
Figure 21 Pointing Examples of RoboBrain 2.0. The free space indicated by spatial relations and the referenced objects are pointed out.

# A.2 Examples for Affordance

The affordance task assesses RoboBrain 2.0’s understanding of object functionalities and interaction possibilities. For example, when asked “What part of a mug holds the liquid for drinking?” the model correctly identifies the interior of the mug as the part that holds the liquid. In another example, the instruction “Which part of a handbag can be grasped to carry it?” is accurately answered by identifying the handle of the handbag. These examples showcase the model’s ability to reason about object affordances, making it capable of understanding how objects can be interacted with in the real world. As shown in Figure 22-Figure 23, the model demonstrates its proficiency in identifying functional parts of objects and their potential uses.

![](images/5b21a6bf6b55f0bc41bef58800e481d789a8e2ff8bf72cfbc825bdbf742ec72b.jpg)  
Figure 22 Affordance Examples of RoboBrain 2.0. The purple bounding boxes denote the actionable affordance areas for specific tasks.

![](images/97d4eaf53f2a5149983bf6487c3c30e33550c4b13c182fcbd62978188c0edb94.jpg)

# Reasoning Step = 2

Please identify the affordance area for pouring the wine glass.

![](images/aed2977d5b0da1552cf63ec2ecc21f534a40194105d1a539066feb85b3a8c775.jpg)

# Reasoning Step = 3

Please identify the affordance area for lying on the bench.

![](images/cb92c079dee680f94a3394a9b0abec3908f33b4573fef94198414d53b16f2437.jpg)

# Reasoning Step = 2

Please identify the affordance area for holding the knife.

![](images/e51ccceb693cd95c99f9a0832b23f798d21d1487080528a72eccbbce3685219e.jpg)

![](images/a6bff0fffa697e707af85745bfe5a184274d61241c6e8af3ffeca2da4c8db38e.jpg)  
Reasoning Step = 2

![](images/6f2487039a92aae3a5c10fd1c51c38f11a4bf9f781c37f6d7bd76ed3bfeb6b0c.jpg)

Please identify the affordance area for opening the bottle.

# Reasoning Step = 2

Please identify the affordance area for the pen to write.

# Reasoning Step = 2

Please identify the affordance area for picking up the suitcase.

![](images/72198db976d5b7f7accc0a3bff276c6f22c9f27f47a2278db47025a1ef207421.jpg)

# Reasoning Step = 2

Please identify the affordance area for sipping the cup.

![](images/32caf0cd855e56136514f9a96f2fecb5f35af8ca440ac3e9e10544842b0630cd.jpg)

# Reasoning Step = 2

Please identify the affordance area for holding the cup.

![](images/1b8f6081cb764bf4d65be0dacd377a997cea71988d402c793cea2ef856f666f1.jpg)

# Reasoning $S t e p = 2$

Please identify the affordance area for opening the refrigerator.

![](images/263c2ddabe052c8ed938e3a9a6dd9457362ac7458b7f5269311dcf5412235ccf.jpg)

# Reasoning Step = 2

![](images/4738db6832f0c2a932c5ca6af3344064dc50fdbe23c82395888673788406d304.jpg)

Please identify the affordance area for holding the cup.

![](images/dc90a282db2224cad89e59a1c66351afa9a7b34b5d031e42140c9c6a6cfc7c4f.jpg)  
Figure 23 Affordance Examples of RoboBrain 2.0. The purple bounding boxes denote the actionable affordance areas for specific tasks.

# Reasoning Step = 2

Please identify the affordance area for sitting on the bicycle.

# A.3 Examples for Trajectory

The trajectory task evaluates the model’s ability to predict and navigate paths based on given instructions. For instance, given the instruction “Please provide the trajectory to move the robot arm to grasp the apple,” RoboBrain 2.0 generates a smooth and efficient path for the robot arm to follow. The model’s trajectory predictions are accurate and take into account the spatial constraints and obstacles in the environment, demonstrating its proficiency in spatial and temporal reasoning for navigation tasks. As shown in Figure 24- Figure 25, the model effectively plans and executes trajectories that are both optimal and collision-free.

![](images/d599d972d714beca9108e9dd24eedfd3373cf3c08f9dd3e7bac9cb3b5d20c55c.jpg)  
Figure 24 Trajectory Examples of RoboBrain 2.0. The blue trajectories, composed of key trajectory points, represent the model-predicted paths for task completion.

![](images/d6af0d9ec885820005f04ce2e69aff671b45ddccdc0236f348d065ae91ea169b.jpg)  
Reasoning Step $= 2$

Please predict the key trajectory points for moving green cube to the top of yellow cube.

![](images/50fdf987306e7eb33f8c023118c2a306e63943e5878b5f8860a1baf15a900c52.jpg)

# Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$

Please predict the key trajectory points for closing middle drawer.

![](images/099d9968a6819418cc5e6813d25bb2d50a355fcab9376374a936968f0556be4f.jpg)

# Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$

Please predict the key trajectory points for closing top drawer.

![](images/dd37b766d0ffc6d6f8a1e92a446f36841d40ef3b36ace7b48cc8799dd753d4fc.jpg)  
Reasoning Step $\mathbf { \Omega } = \pmb { \mathit { 1 } }$ Please predict the key trajectory points for destacking purple yellow cube.

![](images/fb797afaca32c4fb97dec44fc1415745322f392e4b6f499093f8d298e2a7f2ea.jpg)  
Reasoning Step $= 2$ Please predict the key trajectory points for making a cup of coffee with the Keurig machine.

![](images/9bd2760d826b96d633d3fc0ff0cbbfade1d903cf1a825db1ab37c4183952aca4.jpg)  
Reasoning Step $= 2$ Please predict the key trajectory points for moving red circle closer towards blue cube.

![](images/e5bdc5f9c8aca07ae00d3cfad284810efcdb16de88a74f8781247f6d52c4f854.jpg)  
Reasoning $S t e p = 2$ Please predict the key trajectory points for moving red spoon to just below green towel.

![](images/b465194017fdc7a5f761c84b54d2de9676c733f7bb5f507d8c911e7428f21061.jpg)

# Reasoning $S t e p = 2$

![](images/22146398b2af83273eb4eb81cee90ed3a910665047e1c016585f3387f5943c5b.jpg)  
Reasoning Step $= 2$ Please predict the key trajectory points for moving rxbar chocolate near coke can.

Please predict the key trajectory points for moving red spoon to upper right corner of table.

# Reasoning $S t e p = 2$

![](images/3f0a234fd8692acac90ed5c1508c6c9ea190cb4e6dec3e6cbac772375bd046c7.jpg)

Please predict the key trajectory points for moving spatula onto the towel.

![](images/9eeae4fd4d44fd01828b81f20bc491f6ecc4fc5350f9d786d8fc4715b7eff68b.jpg)

# Reasoning $S t e p = 2$

![](images/b493d51a180d40c0b2de6d7ad43ff07fe7987bfa5a5cd659ca7f053ef6193ef2.jpg)  
Figure 25 Trajectory Examples of RoboBrain 2.0. The blue trajectories, composed of key trajectory points, represent the model-predicted paths for task completion.

Please predict the key trajectory points for moving blue triangle below the green star

# Reasoning $S t e p = 2$

Please predict the key trajectory points for picking green can from middle shelf of fridge.

# A.4 Examples for EgoPlan2

The EgoPlan2 task focuses on planning daily activities from an egocentric perspective. For instance, given the instruction “Plan the steps to prepare a cup of coffee,” RoboBrain 2.0 outlines a detailed sequence of actions, including locating the coffee machine, fetching the coffee beans, and following the steps to brew the coffee. The model’s ability to break down complex tasks into actionable steps demonstrates its proficiency in task decomposition and sequential planning. As shown in Figure 26-Figure 28, the model effectively plans and executes multi-step tasks, showcasing its capabilities in long-horizon planning and task execution.

![](images/a4fc9add15e40b08408460b3454e1d6b9a27e4ba1d78f05d472b3c32a2efb61a.jpg)

Question: I am tasked with Move the saucepan and light up the candle. The task's progress is demonstrated in the   
provided video. What should be my next action?   
(A). move saucepan (B). close drawer (C). adjust hob (D). open drawer

Thinking: From the visual input, I observe an open drawer containing various utensils such as spoons and a whisk. My focus remains on completing the task of moving the saucepan and lighting a candle. The current step involves handling objects from this drawer. Based on this, the logical next action is to close the drawer after retrieving the necessary item (likely utensils) for cooking or handling the saucepan, ensuring workspace orderliness and safety.

The available options include: (A) Moving the saucepan, (B) Closing the drawer, (C) Adjusting the hob, and (D) Opening another drawer. Given that the drawer is already open and no further items appear necessary at the moment, closing it aligns with maintaining organization and preventing obstruction during subsequent steps.

Therefore, the most reasonable action is option (B), closing the drawer.

Answer: (B). close drawer

![](images/d678186c89fbb824a180b94cc3042a281e2073c4c5983bd9797c9d2b6d81a941.jpg)

Question: My current task is to Wash and rinse two mugs. The task's progress is demonstrated in the provided video. What should be my next action?   
(A). put mug (B). rinse mug (C). pick up mug (D). close tap

Thinking: From the visual input, I observe that I am positioned near a sink within what appears to be a kitchen environment. My task involves washing and rinsing two mugs. Currently, I am holding one red mug and a sponge, actively cleaning it. Other items like a blue plate are visible on the counter nearby, suggesting this is part of a larger cleaning task.

The sink contains various other dishware, including another mug and additional kitchen utensils, indicating my current focus on washing multiple mugs. Given the visual context, the next logical step after cleaning the mug would be to rinse it thoroughly before placing it aside or moving on to wash the second mug.

Considering the options provided—(A) put mug, (B) rinse mug, (C) pick up mug, (D) close tap—the most immediate and sensible action aligns with rinsing the mug to ensure it is properly cleaned and ready for further steps, such as drying or placing it away.

Therefore, based on my visual analysis of the scene and the task at hand, the appropriate next action is to rinse the mug, which corresponds to option (B).

Answer: (B). rinse mug

Figure 26 EgoPlan Examples of RoboBrain 2.0. The key reasoning process and final answer related to specific tasks are depicted.

![](images/3540ff9fc0fe72fe929909024cda61c617f0899308776e719f297a0afda0eb50.jpg)

Question: I am in charge of Take mozzarella, open the bag, spread it onto the dough and add salt, and the video reveals the task's advancement, What is the most rational next move? (A). spread mozzarella onto dough (B). put down knife (C). take mozzarella (D). open mozzarella bag

Thinking: From the visual input, I recognize that my task involves preparing ingredients for a dish on a wooden countertop. I observe a loaf of mozzarella, which is currently being held by me, and an open packet containing additional mozzarella slices placed near the edge of the countertop. A flat, baked surface with sliced mozzarella arranged neatly appears to be the target area for spreading the cheese.

Given this setup, my next logical step is to apply the cheese directly onto the prepared flat surface with the slices of mozzarella already laid out, ensuring even distribution over the crust. This aligns with the objective of adding cheese as part of the preparation process.

# Comparing my options:

- Option (A) "spread mozzarella onto dough" directly addresses the current task at hand, as it involves positioning the cheese appropriately on the target area.   
- Option (B) "put down knife" doesn\'t fit the immediate need since the task focuses on spreading rather than cutting. - Option (C) "take mozzarella" is redundant, as I already have the mozzarella in hand.   
- Option (D) "open mozzarella bag" is unnecessary since the bag is already open, and the focus is on applying the cheese.

Thus, based on the visual scene analysis and task requirements, the most rational next move is to spread the mozzarella onto the dough.Therefore, the selected action is (A) "spread mozzarella onto dough.

Answer: (A). spread mozzarella onto dough

![](images/9de168dab6d43efd6f8ef18adf6ec2655ac2eaf7f37a0953cfdc42ce4c9223c6.jpg)

Question: My assignment is to Preparing onion and boiling water, with the task's progress evident in the video. What should I do next? Please provide a reasonable and succinct recommendation.   
(A). take onion (B). open container (C). take knife (D). open tap

Thinking: From the visual input, the bowl containing onions is held and is the primary focus of the scene. The onion pieces are visible inside the bowl, indicating that it is already opened and ready for use. The surrounding context includes kitchen items such as a spatula, a knife on the counter, and a grater nearby, but these are not directly related to the task here. My task involves preparing an onion, which requires handling or slicing the onion contents. Given this, the logical next step would be to focus on utilizing the already accessible onion by taking it out of the bowl or using it for cooking.

The options provided include (A) taking the onion itself, (B) opening a container (which has been done), (C) taking a knife (not relevant at this stage since the onion is already handled), and (D) opening the tap (irrelevant as no water preparation is observed). Among these, option (A) aligns most directly with the immediate need to prepare the onion.

Therefore, based on the observed situation, the most reasonable next action is to take the onion from the bowl, ensuring readiness for further steps in the cooking process.

Answer: (A). take onion

Figure 27 EgoPlan Examples of RoboBrain 2.0. The key reasoning process and final answer related to specific tasks are depicted.

![](images/4d620b409cadd9ca5cdcf217125f8e4f577464053d5c0eda270eeb352faa6f3b.jpg)  
Question: As I am in the process of Prepare and slice more onions, with my progress visible in the video. what do you suggest as my next move? (A). take knife (B). turn knife (C). throw onion skin into bin (D). continue cutting onion

Thinking: From the visual input, the person is focused on slicing an onion on a green cutting board. The onion's white layers have been separated, and some skin or layers are visible on the cutting surface. To maintain safety and efficiency, continuing to slice the onion is logical. The person holds the onion steady, and their fingers are positioned to maintain control during further cuts.

Analyzing the current task, the objective remains to chop the onion neatly without causing any unnecessary mess. The presence of a knife suggests readiness for this action. With the onion already sliced partially, the next logical step is to complete the task by continuing to slice.

Considering the options provided:   
(A) indicates taking the knife again, which might be redundant given its current proximity.   
(B) suggests turning the knife, which may not be necessary unless there's an issue with the orientation.   
(C) proposes throwing away the onion skin, which could happen after cutting but not as part of the ongoing slicing process.   
(D) confirms the need to continue cutting the onion, aligning well with the observed task progression.

Based on the visual scene and task context, option (D) to continue cutting the onion is the appropriate choice, as it supports the process of preparing and slicing the onion efficiently.

Answer: (D). continue cutting onion

![](images/17bd2eedaaeccaf681a96e11c8ffee9258cb04e7c2822c8e198fde819257254e.jpg)

Question: Given my responsibility to Clean the sink and counter, the video shows the progress What is the most logical   
next step?   
(A). take spoon (B). rinse spoon (C). take cloth (D). rinse cloth

Thinking: From the visual input, I am observing a kitchen environment where cleaning activities are taking place. The focus is on a sink area that contains various cleaning tools, including utensils and dishes. A spoon is being held over the sink, which is filled with water.The faucet is nearby, suggesting its use for rinsing or further cleaning actions.

Given the current task of cleaning, the spoon appears to be in the process of being washed, as it is already oriented towards the water source. The fact that this spoon is already being handled indicates readiness for further actions, such as rinsing or placing it elsewhere after cleaning. The surrounding environment includes a drying rack holding other utensils, reinforcing the notion of completing cleaning tasks sequentially.

Analyzing the sequence and progression of cleaning actions, the spoon is currently within a position conducive to rinsing under running water. This action is the next logical step before moving on to other items or utensils, to ensure all surfaces are appropriately cleaned for safe handling or placement. The other options—such as taking a cloth or additional utensils—do not align with the immediate visual context, as the spoon is already in an appropriate state for this activity.

Therefore, based on the visual analysis, the most logical next step is to rinse the spoon (Option B). This action ensures proper cleaning and aligns with the observed process within the image.

Answer: (B). rinse spoon

Figure 28 EgoPlan Examples of RoboBrain 2.0. The key reasoning process and final answer related to specific tasks are depicted.

# A.5 Examples for Close-Loop Interaction

Close-loop interaction examples showcase RoboBrain 2.0’s ability to engage in interactive reasoning with feedback. For example, in a scenario where the model is asked to “Find a muff cup and pour coffee into it,” it not only needs to navigate and search for the mug multiple times within the task environment but also must operate the coffee machine based on feedback to complete the pouring process. This iterative process highlights the model’s capability to refine its actions based on real-time feedback, ensuring more accurate and reliable performance in interactive tasks. As shown in Figure 29-Figure 32, the model demonstrates its ability to adapt and improve its responses through iterative feedback loops.

![](images/79d710b126705eb90857298015cd17ba3d6567da7879f140f12951e8111bb9ed.jpg)  
Task: Find a pen and place it to box, and then find a pillow, place it to arm chair.   
Figure 29 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

![](images/5aa21544a6ac70a225234e28029faf546eb1346b5e247b91cb34d91f11b6529f.jpg)  
Figure 30 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

![](images/04b217dc891ec5ee0e828ae5c6b57ce50da45420aab8aa9aa78a16d3e6ff4d72.jpg)  
Task: Find a muff cup and place it to sink, and then find a potato, place the potato into Fridge, and then pick up the egg from fridge and place it to Garbage.   
Figure 31 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

![](images/119c9833b455e51b7a653ea3c2aea5ee27d4595a4ae8be3b236b690d1eff2274.jpg)  
Task: Find an egg and heat it with microwave, and then find a muff cup, pour coffee into it and pick it up.   
Figure 32 Close-loop planning Examples of RoboBrain 2.0. The key planning steps related to specific tasks are depicted.

# A.6 Examples for Multi-Robot Planning

In multi-robot planning scenarios, RoboBrain 2.0 coordinates the actions of multiple robots to achieve a common goal. For example, in a supermarket scenario, the model plans the movements of multiple robots to efficiently restock shelves. The planning involves assigning specific tasks to each robot, coordinating their movements to avoid collisions, and ensuring that the overall goal is achieved in a timely manner. These examples highlight the model’s advanced capabilities in multi-agent coordination and long-horizon planning.

As shown in Figure 33, the model demonstrates its ability to orchestrate complex multi-robot activities with high precision and efficiency. In the restaurant setting (Figure 33(a)), a Unitree G1 humanoid and Agilex dual-arm robot collaborate on burger preparation and delivery for the command “I’m hungry and order a normal burger,” with RoboBrain 2.0 performing scene-aware task decomposition. The household scenario (Figure 33(b)) features a Realman single-arm and Agilex dual-arm robot executing commands like “Give me an orange and a knife.” In the supermarket (Figure 33(c)), RoboBrain 2.0 assists customers with gift selection by analyzing dimensions and bag compatibility, coordinating the Realman robot for gift placement and the Agilex executing VLA-cerebellum skills like “open the gift bag.” Please refer to RoboOS [61] for more details.

![](images/65483ad8bd31e04858177c969a40dc30721d49fa23ff119539f7347774a485af.jpg)  
(c) Global Task: I want to give a small gift to my friend, please help me to choose one. [Supermarket]   
Figure 33 We showcase multi-robot collaboration in three scenarios: (a) Restaurant: Unitree G1 and Agilex robots prepare burgers. (b) Household: Realman and Agilex robots fetch items. (c) Supermarket: Robots coordinate gift selection and packaging.

# A.7 Examples for Synthetic Benchmarks

Synthetic benchmarks are used to evaluate RoboBrain 2.0’s performance on a variety of spatial and temporal reasoning tasks. For instance, in the BLINK benchmark, which assesses depth perception and spatial relation understanding, the model achieves high accuracy in identifying the relative positions and distances of objects. In the CV-Bench benchmark, which evaluates 3D spatial understanding, RoboBrain 2.0 demonstrates its ability to accurately process and reason about 3D scenes. These synthetic benchmarks provide a comprehensive evaluation of the model’s capabilities across different reasoning dimensions. As shown in Figure 34-Figure 35, the model consistently performs well across various synthetic benchmarks, showcasing its robust abilities.

![](images/a0185fb659793abef858b65aea1a37853e132722c355af76d8f2a2bfbdcc0abb.jpg)  
Figure 34 CVbench Benchmark Examples of RoboBrain 2.0. The solid circle in the diagram represents the selected point.

![](images/1ae575a6ffdc31d5976997d5f23f92994603808e220e6e9f8f2329c66ef81439.jpg)  
Figure 35 BLINK Benchmark Examples of RoboBrain 2.0. The solid circle in the diagram represents the selected object.

# B Prompts Details

This section outlines the system prompts for various spatial understanding and planning tasks assigned to a robot with advanced visual and analytical capabilities. Each task requires simulating visual-spatial reasoning, leveraging visual inputs as if directly perceiving the scene, and generating step-by-step reasoning processes within <think></think> tags, with answers in <answer></answer> tags. Reasoning is kept concise (200–500 words) and follows a five-stage process tailored to each task. Instructions emphasize direct visual-spatial language, avoiding abstract references to input data (e.g., “based on the description”) and maintaining the robot’s role.

# B.1 Spatial Understanding: Coordinates -- Pointing

The robot is tasked with identifying specific points within an image based on visual criteria, such as locating points in a vacant area on a delineated plane. The reasoning process includes:

Object Analysis Observe the object’s shape, size, and spatial relationships (e.g., a red rectangular border delineating a plane with vacant patches).

Capability Assessment Relate visual processing capabilities to identifying vacant regions and pinpointing coordinates.

Contextual Relevance Focus on the task requirement to select points within the vacant area.

Verification Ensure selected coordinates lie within boundaries and are distinct.

nt Conclusion Output coordinates as a list of tuples, justified by visual analysi

Example Prompt: Locate points within a vacant area on a plane outlined by a red border. Your answer should be formatted as a list of tuples, i.e. $[ ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) , . . . ]$ , where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should indicate the normalized pixel locations of the points in the image.

Example Output w/ Thinking: <think> . . . </think><answer>[(296, 282), (321, 256), . . . ]</answer>.

# B.2 Spatial Understanding: Coordinates -- Trajectory

The robot predicts a sequence of key trajectory points to achieve a goal, such as reaching an object. The reasoning process includes:

Object Analysis Identify the target object’s properties and spatial relationships (e.g., a banana on a plate with potential obstacles nearby).

Capability Assessment Use joint control to plan smooth end-effector paths, avoiding obstacles.

ontextual Relevance Ensure the trajectory aligns with the goal (e.g., reaching the banana).

Verification Confirm the path avoids obstacles and reaches the target.

Trajectory Conclusion Output trajectory points as $[ [ x _ { 1 } , y _ { 1 } ] , [ x _ { 2 } , y _ { 2 } ] , \dots ]$ , justified by visual and kinodynamic analysis.

Example Prompt: You are a robot using the joint control. The task is “Reach for a banana on a plate”. Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. $[ [ x _ { 1 } , y _ { 1 } ] , [ x _ { 2 } , y _ { 2 } ] , \dots ]$ , where each tuple contains the x and y coordinates of a point."

Example Output w/ Thinking: <think> . . . </think><answer>[[116, 114], [153, 97], . . . ].</answer>.

# B.3 Spatial Understanding: Bounding Box -- Affordance

The robot identifies an affordance area for interaction with an object, such as grasping a handle. The reasoning process includes:

Object Analysis Describe the object’s shape, size, and material properties (e.g., a blue coffee mug with a handle, inferred as ceramic from sheen).

Capability Assessment Assess gripper compatibility with the object’s features (e.g., handle size vs. gripper opening).

Contextual Relevance Align with the task goal (e.g., preparing coffee by grasping the mug).

erification Confirm the affordance area suits the interaction and is within reach.

Affordance Conclusion Output the bounding box as $[ x _ { m i n } , y _ { m i n } , x _ { m a x } , y _ { m a x } ]$ , justified by visual compatibility.

Example Prompt: You are a robot using the joint control. The task is “hold a coffee mug”. Please predict a possible affordance area of the end effector.

Example Output w/ Thinking: <think> . . . </think><answer>[915, 408, 1109, 533].</answer>.

# B.4 Spatial Understanding: Freeform Q&A -- General Spatial Analysis

The robot answers questions about spatial relationships or action outcomes based on one or more images. The reasoning process includes:

Scene Perception Detail prominent features and their spatial arrangement (e.g., a metallic gripper above a green book on a shelf).

Task Objective Interpretation Clarify the question’s focus (e.g., predicting the outcome of a gripper’s trajectory).

Focused Visual Analysis Examine relevant scene elements or project actions (e.g., a yellow trajectory toward a lower shelf).

Relational Reasoning Synthesize observations to form a hypothesis, evaluating provided options.

Conclusion Derivation Output the answer, justified by visual evidence and logical reasoning.

Example Prompt: Predict the outcome of a gripper following a yellow trajectory. Options: (A) place book on lower shelf; (B) place book on upper shelf.

Example Output w/ Thinking: <think> . . . </think><answer>(A)</answer>.

# B.5 Temporal Understanding: Long-horizon Planning

The robot determines the next action in a task (e.g., cooking) based on a sequence of images and the current view. The reasoning process includes:

Task Progress Analysis Interpret completed actions from the sequence (e.g., onions peeled and sliced on a cutting board).

Current Scene Analysis Describe the current view’s objects and state (e.g., frying pan on hob, oil container nearby).

ontextual Relevance Align with the task goal (e.g., cook onions by preparing the pan).

Action Option Evaluation Assess options for suitability (e.g., pour oil vs. peel onion, considering onions are already prepared).

Next Action Conclusion Output the next action, justified by visual evidence and task flow.

Example Prompt: Prepare and cook onions; choose the next action (options: pour oil, turn up hob, etc.).   
Example Output w/ Thinking: <think> . . . </think><answer>Pour oil.</answer>.

# B.6 Temporal Understanding: Closed Loop Conversation

The robot answers a question within a conversation history, leveraging prior visual inputs and responses. The reasoning process includes:

Task Progress Recall Recap previous actions and their outcomes (e.g., opened the fridge to access ingredients).   
Initial Analysis Focus on current visual input relevant to the question (e.g., a coffee machine on the countertop).   
Contextual Relevance Align with the current task goal (e.g., flipping the coffee machine switch).

Action Option Evaluation Assess options for logical progression based on history and current state.

Next Action Conclusion Output the action, justified by visual evidence and conversation context.

Example Prompt: The task is “Flip the coffee machine switch after opening the fridge.” After you have finishe <action $>$ , you can see <image $>$ , and the feedback of final action is xxx. What is your next action?

Example Output w/ Thinking: <think> . . . </think><answer>Toggle on Coffee Machine.</answer>.

# B.7 Temporal Understanding: Multi-Robot Planning

The robot coordinates actions with other robots to achieve a common goal, devided into global task decomposition and agent-based tool-calling.

Example Prompt for Global Task Decomposition: Please Refer to Figure 36.   
Example Prompt for Agent-based Tool-calling: Please Refer to Figure 37.   
Example Output w/ Thinking: <think> . . . </think><answer>Graph of TaskFlow</answer>.

# System Prompt for Global Task Decomposition

# # You are a robotics expert specializing in task decomposition.

Your role is to decompose tasks into subtasks based on the task description and assign them to different robots for execution.

# ## Example 1:

Current Robot: realman_1, singlearm_1, doublearm_1   
Current Task: All the robots go to the table and bring an apple to the fridge respectively.   
Your answer:   
\`\`\`json   
[ {{'robot_id': 'realman_1', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': $" 0 " \}$ , {{'robot_id': 'singlearm_1', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': $" 0 \%$ , {{'robot_id': 'doublearm_1', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': $" 0 " \}$ ,

# ## Example 2:

Current Robot: realman_1, doublearm_1   
Current Task: realman take the basket from table_1 to table_2, then doublearm take the apple into basket in table_2, then realman take the   
basket back to table_1.   
Your answer:   
\`\`\`json {{'robot_id': 'realman_1', 'task': 'bring the basket from table_1 to table_2.', 'task_order': '0'}}, {{'robot_id': 'doublearm_1', 'task': 'pick an apple into the basket.', 'task_order': '1'}}, {{'robot_id': 'realman_1', 'task': 'bring the basket from table_2 to table_1.', 'task_order': '2'}},   
\`\`\`

## Note: 'subtask_order' means the order of the sub-task.

If the tasks are not sequential, please set the same 'task_order' for the same task. For example, if two robots are assigned to the two tasks, both of which are independance, they should share the same 'task_order’. If the tasks are sequential, the 'task_order' should be set in the order of execution. For example, if the task_2 should be started after task_1, they should have different 'task_order'.

# # Now it's your turn !!!

We will provide more scenario information and robot information. Based on the following robot information and scene information, please break down the given task into sub-tasks, each of which cannot be too complex, make sure that a single robot can do it. It can't be too simple either, e.g. it can't be a sub-task that can be done by a single step robot tool. Each sub-task in the output needs a concise name of the sub-task, which includes the robots that need to complete the sub-task. Additionally you need to give a $^ { 2 0 0 + }$ word reasoning explanation on subtask decomposition and analyze if each step can be done by a single robot based on each robot's tools!

# ## The output format is as follows, in the form of a JSON structure:

{ "reasoning_explanation": xxx, "subtask_list": [ {"robot_id": xxx, "subtask": xxx, "subtask_order": xxx}, {"robot_id": xxx, "subtask": xxx, "subtask_order": xxx}, {"robot_id": xxx, "subtask": xxx, "subtask_order": xxx}, ]   
}

# ## Robot Information:

Robot in Scene: {Robot List}. Robot positional states:{Robotic Memory}. Robot available tools:{Robotic Tool Libraries

# ## Scene Information:

# {Scene Graph}

# The task to be completed is:{Global Task}. Your output answer:

Figure 36 Prompt for global task decomposition.

# System Prompt for Agent-based Tool Calling

# # You are an expert assistant who can solve any task using tool calls.

You will be given a task to solve as best you can. To do so, you have been given access to some tools.

The tool call you write is an action: after the tool is executed, you will get the result of the tool call as an "observation".   
This Action/Observation can repeat N times, you should take several steps when needed.

You can use the result of the previous action as input for the next action. The observation will always be a string: it can represent a file, like "image_1.jpg". Then you can use it as input for the next action. You can do it for instance as follows:

Observation: "image_1.jpg"

Action:

{{ "name": "image_transformer", "arguments": {{"image": "image_1.jpg"}}}}

To provide the final answer to the task, use an action blob with "name": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:

Action:

{{"name": "final_answer", "arguments": {{"answer": "insert your final answer here"}}}} "arguments": "image.png"}}

# # Here are a few examples using notional tools:

Task: "What is the result of the following operation: $5 + 3 + 1 2 9 4 . 6 7 8 ?$ “

Action:

$\{ \{$ "name": "python_interpreter", "arguments": {{"code": "5 + 3 + 1294.678"}}}} Observation: 1302.678

Action:

{{ "name": "final_answer", "arguments": "1302.678" }} # Above example were using notional tools that might not exist for you. You only have access to these tools

{%- for tool in tools.values() %}   
- {{ tool.name }}: {{ tool.description }}   
Takes inputs: {{tool.inputs}}   
Returns an output of type: {{tool.output_type}}   
$\{ \% \}$ - endfor %}

# # Here are the rules you should always follow to solve your task:

1. ALWAYS provide a tool call, else you will fail.   
2. Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.   
3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.   
If no tool call is needed, use final_answer tool to return your answer.   
4. Never re-do a tool call that you previously did with the exact same parameters.

# # Now Begin! If you solve the task correctly, you will receive a reward of \$1,000,000.

Task: {Subtask}   
The tool you have used are: {Tool-Calling_History}   
Observation: {Observation}   
Your next action is:

Figure 37 Prompt for agent-based tool calling.