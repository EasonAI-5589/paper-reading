[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
三条线索：(1) 推理 MLLM 的进展；(2) 现有医学 benchmark 的局限；(3) 视频数据用于医学 benchmark 构建。

---

### Reasoning Multimodal Large Language Models

With advances in models and benchmarks, interest in the reasoning capabilities of MLLMs has grown (Wang et al., 2024; Xie et al., 2024; Chen et al., 2025; Deng et al., 2025). Recent MLLMs now support medical reasoning tasks like clinical decision-making, chain-of-thought generation, and diagnostic inference (AlSaad et al., 2024). Llava-Med (Li et al., 2023) and GPT-4V (OpenAI, 2023b) show generalist abilities in radiology and biomedical VQA but often lack interpretable reasoning. MedCoT (Wang et al., 2025) addresses this with a multi-expert prompting framework that improves rationale quality and accuracy. MedVLMR1 (Pan et al., 2025) applies reinforcement learning to encourage plausible rationales without ground truth, improving radiology QA. Med-Gemini (Saab et al., 2024) combines domain-adapted prompting with long-context modeling for complex cross-modal inference. These advancements in applying MLLMs to medical reasoning tasks underscore the critical need for rigorous benchmarks that effectively evaluate their reasoning capabilities.

> 💡 **推理 MLLM 进展**: LLaVA-Med → MedCoT（多专家 prompting）→ MedVLM-R1（RL 训练 rationale）→ Med-Gemini。趋势是从"能答"到"能推理"，但缺少评估多图推理的 benchmark。

---

### Multimodal Medical Benchmarks

Existing benchmarks for evaluating MLLMs in the medical domain remain limited in scope. Most notably, the majority focus on single-image question answering tasks. For example, VQA-RAD (Lau et al., 2018), VQA-Med-2019 (Ben Abacha et al., 2019), VQA-Med-2021 (Ben Abacha et al., 2021), and SLAKE (Liu et al., 2021) primarily target single-question VQA tasks within the radiology domain, while Path-VQA (He et al., 2020) is dedicated exclusively to pathology. With the rapid advancement of MLLMs, more generalized benchmarks such as PMC-VQA (Zhang et al., 2023), OmniMedVQA (Hu et al., 2024), and GMAI-MMBench (Chen et al., 2024) have been introduced to assess broader model capabilities across diverse medical fields. However, these benchmarks remain limited, as they primarily focus on single-image VQA tasks—falling short of reflecting the demands of real-world medical applications.

> 💡 **单图 benchmark 全景**: VQA-RAD/VQA-Med/SLAKE（放射科单图）→ PMC-VQA/OmniMedVQA/GMAI-MMBench（更广但仍单图）。

Recent efforts such as MMMU (H&M) (Yue et al., 2024a), MMMU-Pro (H&M) (Yue et al., 2024b), and MedXpertQA MM (Zuo et al., 2025) have incorporated multi-image VQA tasks. Nonetheless, their construction overlooks the critical need for clinical reasoning across multiple images—a core requirement in real-world diagnostic settings. Moreover, these VQA benchmarks lacks of ground-truth reasoning chains, making it difficult to determine whether the models are genuinely performing multi-image reasoning. We provide a comprehensive comparison of MEDFRAMEQA with existing benchmarks in Table 1.

> 💡 **多图 benchmark 的不足**: MMMU/MedXpertQA 有多图，但 (1) 图片间缺乏临床推理关联；(2) 没有 ground-truth reasoning chain。MedFrameQA 同时解决了这两个问题。

---

### Video Data For Medical Benchmarking

Recent studies have advanced the use of video data for medical dataset construction. Speech recognition models like Whisper (Radford et al., 2023) have made it easier to extract data from videos (Zellers et al., 2021; Zhang et al., 2025). Quilt-1M (Ikezogwo et al., 2023) collected one million paired image-text samples from histopathology YouTube videos. MedVidQA (Gupta et al., 2023) and NurViD (Hu et al., 2023) target instructional and nursing procedures. Cotaract-1K (Ghamsarian et al., 2024) consists of 1,000 videos of cataract surgeries conducted in the eye clinic from 2021 to 2023.

> 💡 **视频→医学数据的先例**: Quilt-1M 是最直接的灵感来源——从 YouTube 组织病理视频提取 100 万图-文对。MedFrameQA 沿用了这个思路，但进一步做到多帧合并和 VQA 生成。

Despite advancements in video dataset construction, limited attention has been paid to leveraging video data for benchmarking MLLMs in the medical domain. YouTube's rich medical content (Osman et al., 2022; Derakhshan et al., 2019) offers natural reasoning chains for multi-frame VQA evaluation. To this end, we utilize YouTube videos and design a VQA generation pipeline that automatically constructs multi-image VQA questions, aiming to assess the reasoning capabilities of MLLMs across complex multi-image scenarios.

> 💡 **研究空白**: 视频数据大多用于构建训练集，很少用于 benchmark。MedFrameQA 的创新在于用视频的时序连贯性构建评测集。

---

## 🔖 Section 总结

### 核心洞察
1. Related Work 梳理清晰，三条线索收束于 MedFrameQA 的定位
2. 关键 prior work: Quilt-1M（YouTube 视频→图-文对的方法论），MMMU/MedXpertQA（多图但不够的 benchmark）
3. 本文的差异化：时序连贯多图 + ground-truth reasoning chain
