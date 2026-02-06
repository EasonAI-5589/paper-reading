# 2. Related Works

## 📄 原文逐段解析

### 2.1 Global Video Understanding

> Early video understanding research primarily targeted global comprehension tasks, such as video question answering, video captioning, and video classification.
>
> ==早期：全局理解任务（VQA、Video Captioning、分类）==

> These methods treat entire videos as holistic units, extracting global representations to predict semantic labels or generate summaries.
>
> ==方法：将视频作为整体单元，提取全局表示==

> While effective for short videos, they often fail to capture complex temporal dynamics and hierarchical structures of long-form content.
>
> ==局限：适合短视频，无法捕捉长视频的复杂时序动态和层级结构==

**代表工作：**
- Phi-4-mini, Gemini, Seed1.5, LLaVA, DeepSeek-VL, NExT-GPT, ViSA, VideoLLaMA3, InternVL3

---

### 2.2 Temporal Segmentation for Short Videos

> Recent works have shifted towards modeling the temporal structure of videos.
>
> ==趋势转变：从全局理解 → 建模时序结构==

> Datasets like ActivityNet Captions, Charades-STA, YouCook2 and Breakfast provide timestamped event annotations, enabling tasks such as temporal event localization, action segmentation, and dense video captioning.
>
> ==关键数据集：ActivityNet Captions, Charades-STA, YouCook2, Breakfast==

> These approaches move beyond global representations to identify and describe fine-grained events and local temporal dependencies.
>
> ==进步：从全局表示 → 细粒度事件 + 局部时序依赖==

**但仍有局限：**

> However, most temporally-structured datasets are limited to short clips, typically under several minutes, and thus do not capture the challenges of ultra-long videos found in lectures, podcasts, or livestreams.
>
> ==问题：数据集仅限短片段（几分钟），无法覆盖讲座、播客、直播等超长视频==

> The lack of large-scale, long-duration datasets with fine-grained temporal annotations remains a major bottleneck.
>
> ==瓶颈：缺乏大规模、长时长、细粒度时序标注的数据集==

---

### 2.3 Long-Form Video Structuring

> A few efforts have explored the structuring of hour-long videos.
>
> ==少数工作探索了小时级长视频的结构化==

#### VidChapters-7M

> The VidChapters-7M dataset provides a large-scale benchmark for video chaptering, with millions of videos and annotated chapter boundaries, better reflecting real-world scenarios such as vlogs, podcasts, and meetings where long-term temporal reasoning is essential.
>
> ==VidChapters-7M：目前最大的章节数据集，百万级视频，更贴近真实场景==

#### 现有方法的问题

> Despite these advances, significant challenges remain:
> - Existing chaptering models often rely on **limited modalities** (such as ASR only)
> - Are trained on **small-scale datasets**
> - Produce **coarse, uninformative descriptions**
>
> which limits their scalability across diverse video domains.
>
> ==现有问题：模态单一、数据规模小、描述粗糙 → 泛化能力差==

---

## 💡 Key Takeaways

1. **演进路径**：全局理解 → 短视频时序分割 → 长视频结构化
2. **现有数据集局限**：短片段为主，缺乏长时长细粒度标注
3. **现有模型局限**：模态单一（仅 ASR）、规模小、描述粗
4. **ARC-Chapter 定位**：填补长视频结构化的空白

---

## 📊 Related Works 分类

| 类别 | 代表数据集 | 视频时长 | 任务 |
|------|-----------|----------|------|
| Global Understanding | - | 短 | VQA, Captioning, Classification |
| Short Video Segmentation | ActivityNet, YouCook2, Charades-STA | <5 min | Action Seg, Dense Captioning |
| Long-Form Structuring | **VidChapters-7M** | 10-60 min | **Video Chaptering** |

---

*[返回论文目录](../README.md)*
