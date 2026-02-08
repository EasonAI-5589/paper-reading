[← 返回 README](../README.md)

# A Related Work (Appendix)

## 📌 预览
回顾三大方向：MLLM 发展、MLLM 对齐、MLLM 评估。核心观点：开源 MLLM 缺乏系统对齐，现有对齐数据集规模和质量不足，安全/RM 评估 benchmark 稀缺。

---

**Multimodal large language models** have seen remarkable progress in recent years, with significant advancements in both performance and capabilities. Leveraging cutting-edge LLMs such as GPTs [56, 8], LLaMA [62, 63], Alpaca [60], Vicuna [14], and Mistral [28], MLLMs are increasingly demonstrating enhanced multimodal capabilities, especially through end-to-end training approaches. These advancements have been crucial in enabling models to handle a range of multimodal tasks, including image-text alignment, reasoning, and instruction following, while addressing challenges related to data fusion across different modalities. Recent open-source MLLMs such as Otter [31], mPLUG-Owl [69], LLaVA [43], Qwen-VL [5], Cambrian-1 [61], Mini-Gemini [39], MiniCPM-V 2.5 [26], DeepSeek-VL [47], SliME [79] and VITA [21, 22] have contributed to solving some of the most fundamental multimodal problems, such as improving vision-language alignment, reasoning, and following instructions. Some of the most notable open-source models, such as InternLM-XComposer-2.5 [77] and InternVL-2 [13], have exhibited impressive progress in multimodal understanding, closely competing with proprietary models across a range of multimodal benchmarks. However, despite these achievements, there is still a noticeable gap in security and alignment when compared to closed-source models. As highlighted by recent studies [81], most open-source MLLMs have not undergone rigorous, professional alignment processes, which has hindered their ability to effectively align with human preferences.

> 💡 **关键 gap**: 开源 MLLM 在能力上接近闭源模型，但在安全和对齐方面仍有明显差距。

---

**MLLM Alignment.** With the rapid development of MLLMs, various alignment algorithms have emerged, showcasing different application scenarios and optimization goals. For instance, in the image domain, Fact-RLHF [58] is the first multimodal RLHF algorithm, and more recently, LLaVA-CRITIC [67] has demonstrated strong potential with an iterative DPO strategy. These algorithms have shown significant impact on reducing hallucinations and improving conversational capabilities [80, 72], but they have not led to notable improvements in general capabilities. There have also been some preliminary explorations in the multi-image and video domains, such as MIA-DPO and PPLLaVA. However, alignment in image and video domains is still fragmented, with little research done under a unified framework.

We believe that the main limitation hindering the development of current alignment algorithms is the lack of a high-quality, multimodal alignment dataset. Few existing manually annotated MLLM alignment datasets are available, and most contain fewer than 10K samples [58, 72, 71], which is significantly smaller than large-scale alignment datasets in the LLM field. This small dataset size makes it difficult to cover multiple modalities and diverse task types. Furthermore, machine-annotated data faces challenges related to quality assurance.

> 💡 **对齐领域的核心瓶颈**: 不是算法不行，而是数据不够好、不够大。现有人工标注数据集不到 10K，远小于 LLM 领域。MM-RLHF 的 120K 是一个量级的提升。

---

**MLLM Evaluation.** With the development of MLLMs, a number of benchmarks have been built [18, 23]. For instance, MME [19] constructs a comprehensive evaluation benchmark that includes a total of 14 perception and cognition tasks. MMBench [44] contains over 3,000 multiple-choice questions covering 20 different ability dimensions. Seed-Bench [35] consists of 19,000 multiple-choice questions. MMT-Bench [70] scales up even further, including 31,325 QA pairs. MME-RealWorld [81] places greater emphasis on quality and difficulty. However, benchmarks specifically focused on reward models [36] and those dedicated to evaluating safety and robustness remain relatively scarce. To further promote comprehensive evaluation of MLLM alignment, this paper contributes two benchmarks: one for reward models and another more comprehensive safety benchmark.

> 💡 **评估 benchmark 的空白**: 通用能力 benchmark 很多，但 reward model benchmark 和 safety benchmark 稀缺——MM-RLHF 填补了这两个空白。

---

## 🔖 Section 总结

### 核心洞察
1. 开源 MLLM 能力强但对齐弱——alignment 是最大短板
2. 现有对齐数据集规模太小（<10K），限制了算法发展
3. Reward model 和 safety 评估 benchmark 稀缺——MM-RLHF 补充了这一空白
