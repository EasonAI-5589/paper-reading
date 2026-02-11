[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
三部分：Image-language models（BLIP-2/LLaVA/Flamingo）、Video-language models（Video-LLaMA/VideoChatGPT）、Long-term video models（MeMViT/S4/ViS4mer）。MA-LMM 的定位是在 BLIP-2 基础上加 memory bank 实现长视频理解。

---

**Image-language models.** Inspired by the success of powerful language models [1–6], recent image-language models tend to incorporate pre-trained language models with image encoders to support the multimodal reasoning ability [7–10, 22]. Flamingo [22] proposes to connect powerful pre-trained vision-only and language-only models and achieve state-of-the-art performance in few-shot learning tasks. BLIP-2 [7] introduces a lightweight querying transformer to bridge the modality gap between the frozen pretrained image encoder and frozen LLMs. Despite having significantly fewer trainable parameters, it performs well on various multimodal tasks. LLaVA [8] employs a simple linear layer to project image features into the text embedding space and efficiently finetunes LLMs [23] for better performance. Building upon BLIP-2, MiniGPT-4 [10] collects a large-scale high-quality dataset of image-text pairs and achieves better language generation ability. VisionLLM [15] leverages the reasoning and parsing capacities of LLMs, producing strong performance on multiple fine-grained object-level and coarse-grained reasoning tasks.

> 💡 **批注**: Image-language 模型的两条技术路线：
> - **Q-Former 路线** (BLIP-2 → MiniGPT-4 → InstructBLIP): 用 learned queries 做 cross-attention，输出固定数量 tokens（32）。MA-LMM 沿用此路线。
> - **Linear projection 路线** (LLaVA): 直接线性映射，每图 256 tokens，更多但更简单。

---

**Video-language models.** Previous image-language models such as Flamingo [22] and BLIP-2 [7, 9] can also support video inputs. They simply flattened the spatio-temporal features into 1D sequences and then fed them into the language models for video inputs. However, these approaches can not effectively capture the temporal dynamics of videos. Based on this motivation, Video-LLaMA [12] enhances BLIP-2 structure by adding an additional video querying transformer to explicitly model the temporal relationship. Similarly, building on LLaVA [8], Video-ChatGPT [21] simply average pools the frame-level features across spatial and temporal dimensions to generate video-level representation. VideoChat [13] utilizes perception models to generate action and object annotations, which are then forwarded to LLMs for further reasoning. Despite the advancements, these models are primarily designed for short videos. Inspired by the Token Merging [24] which averages highly similar tokens to reduce the computation cost, we propose an extension of this idea to video data, specifically along the temporal axis. This extension aims to mitigate the challenges posed by extensive token numbers and computational cost associated with processing long video inputs. Several concurrent works [25–27] have also explored similar strategies of merging akin tokens for video inputs. Please refer to the supplementary material for more detailed discussions.

> 💡 **批注**: Video-language 方案总结：
> | 方法 | 时序建模 | 问题 |
> |------|---------|------|
> | BLIP-2/Flamingo | Flatten 1D | 无显式时序建模 |
> | Video-LLaMA | Extra video Q-Former | 训练参数多，不支持长视频 |
> | VideoChatGPT | Avg pool | 丢失时序信息 |
> | VideoChat | 外部感知模型 | 依赖外部标注 |
>
> MA-LMM 的 Token Merging 灵感来自 [24]（空间域 → 时间域）。

---

**Long-term video models.** Long-term video understanding methods focus on capturing long-range patterns in long videos, which typically exceed 30 seconds. To mitigate the computational demands of processing long videos, a prevalent approach involves using pre-extracted features, sidestepping the need for joint training of backbone architectures [28–32]. Alternatively, some research works aim to devise sparse video sampling methods [33, 34], reducing the number of input frames by only preserving salient video content. Other works like Vis4mer [35] and S5 [36] leverage the streamlined transformer decoder structure of S4 [37] to enable long-range temporal modeling with linear computation complexity. Inspired by the memory bank design [38–41], we propose to integrate the long-term memory bank with large multimodal models to enable efficient and effective long-term temporal modeling capabilities.

> 💡 **批注**: 长视频方法三条路线：
> 1. **Pre-extracted features** [28-32]: 预提取特征，不 end-to-end
> 2. **Sparse sampling** [33, 34]: 只采关键帧
> 3. **State-space models** (S4/S5/ViS4mer): 线性复杂度，但不是 LMM
>
> MA-LMM 的 memory bank 设计受 MeMViT [41] 启发，但关键区别：MeMViT 用 FIFO + learnable pooling，MA-LMM 用 **similarity-based merging**。

---

## 🔖 Section 总结

### 核心洞察
1. MA-LMM 建立在 BLIP-2/InstructBLIP 的 Q-Former 架构上，不引入额外模块
2. 与 MeMViT 的关系最密切（都是 memory bank + auto-regressive），但压缩策略不同
3. 与 TESTA/MovieChat/Chat-UniVi 是 concurrent work，都用 token merging 减少冗余
