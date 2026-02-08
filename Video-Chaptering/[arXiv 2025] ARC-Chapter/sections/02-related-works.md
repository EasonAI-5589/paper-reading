[← 返回 README](../README.md)

# 2 Related Works

## 📌 预览
三条线索梳理：Global Video Understanding → Temporal Segmentation (短视频) → Long-Form Video Structuring。核心 gap：现有方法要么只处理短视频，要么用单一模态、小数据集。

---

**Global Video Understanding.** Early video understanding [1; 7; 13; 23; 26; 33; 37; 41; 42; 49; 52; 53; 57] research primarily targeted global comprehension tasks, such as video question answering, video captioning, and video classification. These methods treat entire videos as holistic units, extracting global representations to predict semantic labels or generate summaries. While effective for short videos, they often fail to capture complex temporal dynamics and hierarchical structures of long-form content [24; 30].

> 💡 **Global 方法的局限**:
> 把整个视频当一个整体，只能产出 video-level 的标签/摘要。对于长视频，丢失了时间结构信息。

---

![Figure 2](../images/7d0a7673bdb7eb96f5e60643c9caa12daa61973ebd44332eaf9c3f1d85aad89b.jpg)
*Figure 2: 自动视频标注管线概览。从视频帧提取 visual captions (含 OCR)，从音频提取 ASR 转录，按时间对齐后合并为多模态转录文本。结合原始 chapter markers，由 LLM 生成结构化 chapter 和时间对齐的视频描述。*

> 💡 **Figure 2 批读**:
> - 管线的核心思路是"先提取再推理"：用专门工具（Whisper、Qwen2.5-VL）提取信息，再用 text-only LLM 做推理
> - 这比直接用 MLLM 处理视频要高效得多（成本和上下文长度都可控）
> - 注意：Figure 2 排版在 Related Work 页面，但内容属于 Section 3 的标注管线

---

**Temporal Segmentation for Short Videos.** To address the limitations of global approaches, recent works [14; 15; 17; 28; 30; 40; 47; 50; 56] have shifted towards modeling the temporal structure of videos. Datasets like ActivityNet Captions [19], Charades-STA [11], YouCook2 [55] and Breakfast [21] provide timestamped event annotations, enabling tasks such as temporal event localization, action segmentation, and dense video captioning. These approaches move beyond global representations to identify and describe fine-grained events and local temporal dependencies. However, most temporally-structured datasets [25; 48] are limited to short clips, typically under several minutes, and thus do not capture the challenges of ultra-long videos found in lectures, podcasts, or livestreams. The lack of large-scale, long-duration datasets with fine-grained temporal annotations remains a major bottleneck.

> 💡 **短视频 vs 长视频**:
> - 已有数据集（ActivityNet、YouCook2）通常在几分钟以内
> - 长视频（lecture、podcast）需要处理小时级内容，数据和方法都不够
> - 这里提到的 dense video captioning 是 chaptering 的前身任务

---

**Long-Form Video Structuring.** A few efforts [35; 45] have explored the structuring of hour-long videos. The VidChapters-7M dataset [45] provides a large-scale benchmark for video chaptering, with millions of videos and annotated chapter boundaries, better reflecting real-world scenarios such as vlogs, podcasts, and meetings where long-term temporal reasoning is essential.

Despite these advances, significant challenges remain. Existing chaptering models often rely on limited modalities, such as automatic speech recognition, are trained on small-scale datasets, and produce coarse, uninformative descriptions, which limits their scalability across diverse video domains. To address these issues, we propose a scalable, multimodal framework for long-form video chaptering, supported by a large-scale dataset with detailed chapter descriptions.

> 💡 **Related Work 总结**:
> - VidChapters-7M [45] 是之前最大的 chaptering 数据集，Chapter-Llama [35] 是之前 SOTA
> - 但 Chapter-Llama 只用 ASR（单模态），训练数据只有 ~20k，标注粗糙
> - ARC-Chapter 的定位：多模态 + 大规模 + 层级标注

---

## 🔖 Section 总结

### 核心洞察
1. Video chaptering 在技术路线上从 global → temporal → long-form 逐步演进
2. 之前的 long-form 方法（Chapter-Llama）的主要限制：单模态（ASR-only）、小规模训练、粗标注
3. ARC-Chapter 同时解决这三个问题
