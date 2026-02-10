[← 返回 README](../README.md)

# 5. Discussion

## 📌 预览
总结核心发现，讨论 8 个 limitations 和未来方向。

---

In this study, we evaluated the impact of context length on clinical prediction tasks across four models—Mamba, Llama, GPT, and Hyena—trained on longitudinal EHR data. We are the first to pretrain and release the full weights of these non-GPT architectures at the scale of millions of EHRs. With a context length of 16k tokens, Mamba achieved the highest average AUROC across 14 prediction tasks on the EHRSHOT benchmark, surpassing the prior state-of-the-art by +0.03 points. In addition to the best performance, Mamba also offers faster training, quicker inference, and the potential to support longer contexts (Gu & Dao, 2024). Notably, longer context versions of Mamba and Llama performed well in handling EHR-specific issues like token repetition due to copy-forwarding, irregular inter-token time intervals, and increased token complexity from disease progression. This improvement, however, wasn't universal, as Hyena's performance declined significantly beyond 4k tokens, underscoring the need to empirically validate each architecture for long context use.

> 💡 **Mamba 的综合优势**: 不仅 AUROC 最高，还有更快的训练/推理速度和更长上下文的潜力。对资源受限的医院来说，这是一个重要的实际考量。

---

**Limitations / Future Work:** While our findings highlight the potential for long-context models to successfully model EHR data, several limitations should be considered.

First, we did not evaluate transformer-based models at context lengths beyond 4k tokens due to limited computational resources. Running a vanilla 16k transformer takes roughly 16x more compute/memory than at a context length of 4k, which was a core motivator for the development of the subquadratic architectures evaluated in this work.

Second, model sizes were kept consistent across architectures to isolate the impact of context length. Preliminary findings suggest smaller Mamba models with 16k tokens perform well, which may reduce the need for larger models unsuitable for resource-constrained settings. Future work should quantify the impact of model size on performance.

Third, our evaluations focused on clinical risk prediction tasks, but broader clinical tasks (e.g., phenotyping, treatment selection) merit further consideration.

Fourth, our pretraining dataset was sourced from a single institution due to data privacy concerns, which may limit generalizability.

Fifth, we explored only three EHR-specific properties. Future research could extend this to more attributes of EHR data – e.g., partial observation due to underdiagnosis or miscoding (Pivovarov et al., 2014; Che et al., 2018), multimodal signals (Soenksen et al., 2022), and event-associated metadata (McDermott et al., 2023).

Sixth, we focused on the impact of these EHR-specific properties on downstream evaluations, but they may also have effects on pretraining convergence and stability, which we leave to future work.

Seventh, while the metrics we introduce offer a novel lens for examining EHR data, they are fairly simple and could be improved with additional context. For example, having our repetition metric distinguish between meaningful and non-meaningful repetition (e.g., a repeated lab test in an ICU stay is likely more informative than a repeated diagnosis code of a chronic condition like hypertension) could improve model performance in high-repetition settings. And for the irregularity metric, disease status may influence the regularity of time intervals between events (e.g. a cancer patient may exhibit more regular visits than a patient suffering from acute cardiovascular events), which future work could explore by stratifying results based on specific disease phenotypes.

Eighth, other promising transformer alternatives, such as linear attention models (Arora et al., 2024), hybrid architectures (Poli et al., 2023b; Lieber et al., 2024), and recurrent models (Peng et al., 2023a), should be explored in future work that builds upon the framework introduced here.

> 💡 **Limitations 总结（8 个）**:
> 1. ⚠️ Transformer 只到 4k（计算限制），无法公平对比 16k
> 2. 固定 120M 参数，没探索 size scaling
> 3. 只评估 risk prediction，没覆盖 phenotyping/treatment selection
> 4. 单机构数据（Stanford），泛化性存疑
> 5. 只分析了 3 个 EHR 属性，还有 underdiagnosis、多模态等
> 6. 没分析 EHR 属性对预训练收敛的影响
> 7. 指标太简单（如重复指标没区分"有意义的重复"vs"无意义的重复"）
> 8. 没测 linear attention、hybrid、recurrent 等其他亚二次架构
> 
> 最关键的是 1 和 4：如果能在 16k 上跑 Llama 可能表现更好；单机构数据限制了结论的普适性。

---

## 🔖 Section 总结

### 核心洞察
1. Mamba 是 EHR 长上下文的首选：性能最佳 + 效率最高
2. 需要对每种架构进行经验验证——Hyena 的失败说明不能想当然
3. 未来方向丰富：更大规模、更多架构、多机构、多模态、更精细的指标
