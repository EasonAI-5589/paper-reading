[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
两个方向的相关工作：视觉能力增强（四范式）和记忆增强。定位 VisMem 在两个交叉点上的独特性。

---

## 2.1. Visual Capacities Enhancement

As demonstrated in Fig. 1, existing methods to alleviate "visual processing bottleneck" of VLMs broadly fall into four main categories: (a) direct training paradigm, which directly optimizes model parameters for target visual tasks, as in SFT, Visual-RFT [35], VLM-R1 [44], Vision-R1 [26], and PAPO [66]. Nonetheless, these methods suffer from catastrophic forgetting, specifically manifested as the degradation of general capabilities and overspecialization in specific visual cognition tasks [74, 89]; (b) image-level paradigm, which either leverages bounding boxes to denote visual evidence, represented by methods as Visual CoT [42], DeepEyes [87], SpatialVTS [33], VGR [58], and GRIT [13], or externally generate the iterative visual inputs via predefined tools, as seen in Sketchpad [24], VPRL [69], PyVision [85], OpenAI o3 [40], PixelReasoner [48], MVoT [29], and OpenThinkImg [49]. Nevertheless, modifying visual inputs incurs extremely high computational costs, accompanied by high latency and reliance on external tools and concretized images; (c) token-level paradigm, which select original representations and cannot modify visual evidences, thus restricted by insufficiently refined information and suboptimal selection strategies, as in ICoT [16], MINT-CoT [8], SCAFFOLD [28], LLaVA-AURORA [6], VPT [75], Chameleon [54], (d) latent space paradigm, which employs latent states to optimize autoregressive generation, but its focus remains on pure language models, e.g., Coconut [21], MemGen [81], LatentSeek [30], SoftCoT [68], CODI [47]. Although Mirage [70] attempts to construct a latent vision space, requiring substantial manually labeled images. Our VisMem also belongs to this paradigm, but differs from existing methods by integrating latent vision memory within generation processes, characterized by a short and long memory system.

> 💡 **Latent Space 方法对比**:
> - **Coconut** [21]: 纯语言隐空间 CoT，无视觉
> - **MemGen** [81]: Agent 推理中的生成式 latent memory，纯文本
> - **SoftCoT** [68]: 软化的 CoT token，纯语言
> - **Mirage** [70]: 唯一尝试视觉 latent space 的，但需要大量人工标注图片
> - **VisMem**: 不需要额外标注，双路记忆（视觉+语义），LoRA 挂载不改动核心参数

---

## 2.2. Memory Empowerment

Another mechanism closely tied to our approach involves endowing models with memory functionality. One intuitive strategy entails directly optimize models on prior trajectories, exemplified by [14, 45, 80], or to store them into the external memory repositories [53, 61]. Besides, some models inject persistently stored, retrieval-augmented knowledge from external environments, such as Expel [83] and MemoryBank [88], others, such as SkillWeaver [86] and Alita [41], distill prior knowledge as reusable tools. Currently, latent memory, as an implicit memory representation with better cross-domain generalization, efficiently encodes deep semantic associations, including M+ [65] and MemGen [81]. Nevertheless, these memory paradigms fail to ideally accommodate visual information, which manifests as a continuous, high-dimensional perceptual input. Consequently, the exploration of efficient visual memory mechanisms remains a largely uncharted territory. Thus, we propose a more human-aligned latent vision memory paradigm.

> 💡 **记忆范式光谱**:
> ```
> 显式记忆（外部存储）                     隐式记忆（参数内化）
> ├── 轨迹回放 [14,45,80]                  ├── M+ [65]: 扩展 LLM 记忆
> ├── 外部数据库 [53,61]                    ├── MemGen [81]: 生成式隐记忆
> ├── RAG (Expel, MemoryBank)              └── VisMem: 视觉隐记忆 ← NEW
> └── 工具蒸馏 (SkillWeaver, Alita)
> ```
> 
> VisMem 的独特定位：**视觉维度的隐式记忆**。现有 latent memory 方法都不处理视觉信息，而视觉恰恰是连续、高维的感知输入，更需要隐式编码。

---

## 🔖 Section 总结

### 核心洞察
1. 四范式分类是全面的，VisMem 填补了 latent space paradigm 中视觉维度的空白
2. 记忆增强领域中，MemGen 和 M+ 是最接近的工作，但都不处理视觉
3. Mirage 是唯一的视觉 latent space 前辈，但依赖人工标注——VisMem 通过 RL 自动学习记忆内容
