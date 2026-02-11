[← 返回 README](../README.md)

# Abstract

## 📌 预览
MA-LMM 的核心 idea：不是一次塞更多帧，而是在线逐帧处理 + memory bank 存历史信息，解决 LLM 上下文长度和 GPU 显存限制。

---

With the success of large language models (LLMs), integrating the vision model into LLMs to build vision-language foundation models has gained much more interest recently. However, existing LLM-based large multimodal models (e.g., Video-LLaMA, VideoChat) can only take in a limited number of frames for short video understanding. In this study, we mainly focus on designing an efficient and effective model for long-term video understanding. Instead of trying to process more frames simultaneously like most existing work, we propose to process videos in an online manner and store past video information in a memory bank. This allows our model to reference historical video content for long-term analysis without exceeding LLMs' context length constraints or GPU memory limits. Our memory bank can be seamlessly integrated into current multimodal LLMs in an off-the-shelf manner. We conduct extensive experiments on various video understanding tasks, such as long-video understanding, video question answering, and video captioning, and our model can achieve state-of-the-art performances across multiple datasets.

> 💡 **Abstract 批读**:
> - **问题**: 现有 LMM 只能处理有限帧数（LLaMA context limit 2048，每帧 32/256 tokens），长视频不可行
> - **方案**: 在线处理 + memory bank（不是硬塞更多帧）
> - **关键卖点**: plug-and-play，可直接嵌入现有 LMM
> - **任务覆盖**: 长视频理解、VQA、Video Captioning
>
> 💡 **医学影像迁移思考**: 这个 "在线逐帧处理 + memory bank" 的范式天然适合多帧医学影像（如多切片 CT/MRI）。每个 slice 相当于一帧，memory bank 可以存储之前 slice 的信息用于跨切片推理。关键问题是：医学影像的冗余模式（相邻切片高度相似）是否与视频帧的冗余模式类似？

---

## 🔖 Section 总结

### 核心洞察
1. 长视频理解的瓶颈不是模型能力，而是 **LLM 上下文长度** 和 **GPU 显存**
2. 解决思路从 "处理更多帧" 转向 "在线处理 + 记忆存储"
3. Plug-and-play 设计意味着低迁移成本
