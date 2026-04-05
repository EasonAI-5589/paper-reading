[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
两个方向的相关工作：视觉能力增强（四范式）和记忆增强。定位 VisMem 在两个交叉点上的独特性。

---

## 2.1. Visual Capacities Enhancement

As demonstrated in Fig. 1, existing methods to alleviate "visual processing bottleneck" of VLMs broadly fall into four main categories: (a) direct training paradigm, which directly optimizes model parameters for target visual tasks, as in SFT, Visual-RFT [35], VLM-R1 [44], Vision-R1 [26], and PAPO [66]. Nonetheless, these methods suffer from catastrophic forgetting, specifically manifested as the degradation of general capabilities and overspecialization in specific visual cognition tasks [74, 89]; (b) image-level paradigm, which either leverages bounding boxes to denote visual evidence, represented by methods as Visual CoT [42], DeepEyes [87], SpatialVTS [33], VGR [58], and GRIT [13], or externally generate the iterative visual inputs via predefined tools, as seen in Sketchpad [24], VPRL [69], PyVision [85], OpenAI o3 [40], PixelReasoner [48], MVoT [29], and OpenThinkImg [49]. Nevertheless, modifying visual inputs incurs extremely high computational costs, accompanied by high latency and reliance on external tools and concretized images; (c) token-level paradigm, which select original representations and cannot modify visual evidences, thus restricted by insufficiently refined information and suboptimal selection strategies, as in ICoT [16], MINT-CoT [8], SCAFFOLD [28], LLaVA-AURORA [6], VPT [75], Chameleon [54], (d) latent space paradigm, which employs latent states to optimize autoregressive generation, but its focus remains on pure language models, e.g., Coconut [21], MemGen [81], LatentSeek [30], SoftCoT [68], CODI [47]. Although Mirage [70] attempts to construct a latent vision space, requiring substantial manually labeled images. Our VisMem also belongs to this paradigm, but differs from existing methods by integrating latent vision memory within generation processes, characterized by a short and long memory system.

> 💡 **四范式代表方法**:
> - **(a) Direct Training**：SFT、Visual-RFT、VLM-R1、Vision-R1、PAPO → 直接优化参数，有灾难性遗忘问题
> - **(b) Image-level**：Visual CoT、DeepEyes、Sketchpad、PixelReasoner、OpenThinkImg、MVoT → 操作像素空间，计算成本极高，依赖外部工具
> - **(c) Token-level**：ICoT、MINT-CoT、SCAFFOLD、LLaVA-AURORA、VPT、Chameleon → 在已编码的视觉 token 上选择，无法修改视觉证据，受限于初始编码质量
> - **(d) Latent Space**：Coconut、MemGen、SoftCoT、CODI、LatentSeek（纯语言隐空间）；Mirage（唯一尝试视觉 latent space，但需大量人工标注）→ VisMem 属于此范式，补上了无需标注的视觉记忆缺口

---

## 2.2. Memory Empowerment

Another mechanism closely tied to our approach involves endowing models with memory functionality. One intuitive strategy entails directly optimize models on prior trajectories, exemplified by [14, 45, 80], or to store them into the external memory repositories [53, 61]. Besides, some models inject persistently stored, retrieval-augmented knowledge from external environments, such as Expel [83] and MemoryBank [88], others, such as SkillWeaver [86] and Alita [41], distill prior knowledge as reusable tools. Currently, latent memory, as an implicit memory representation with better cross-domain generalization, efficiently encodes deep semantic associations, including M+ [65] and MemGen [81]. Nevertheless, these memory paradigms fail to ideally accommodate visual information, which manifests as a continuous, high-dimensional perceptual input. Consequently, the exploration of efficient visual memory mechanisms remains a largely uncharted territory. Thus, we propose a more human-aligned latent vision memory paradigm.

> 💡 **记忆增强范式**:
> - **轨迹回放**：直接在历史轨迹上优化模型参数 [14, 45, 80]
> - **外部存储**：将经验存入外部 memory repository [53, 61]
> - **RAG 式**：Expel、MemoryBank → 从外部环境检索增强知识
> - **工具蒸馏**：SkillWeaver、Alita → 将先验知识提炼为可复用工具
> - **Latent Memory**：M+、MemGen → 隐式编码语义关联，泛化性更好，但均不处理视觉信息
> - **VisMem**：视觉维度的 latent memory，填补视觉这一连续高维感知输入在隐式记忆中的空白

---

## 🔖 Section 总结

### 核心洞察
1. 四范式分类是全面的，VisMem 填补了 latent space paradigm 中视觉维度的空白
2. 记忆增强领域中，MemGen 和 M+ 是最接近的工作，但都不处理视觉
3. Mirage 是唯一的视觉 latent space 前辈，但依赖人工标注——VisMem 通过 RL 自动学习记忆内容
