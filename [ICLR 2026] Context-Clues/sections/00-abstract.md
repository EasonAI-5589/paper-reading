[← 返回 README](../README.md)

# Abstract

## 📌 预览
论文核心主张：长上下文亚二次架构（Mamba）能显著提升 EHR 临床预测性能，同时 EHR 数据具有三个与自然语言不同的独特属性，影响长上下文模型的表现。

---

Foundation Models (FMs) trained on Electronic Health Records (EHRs) have achieved state-of-the-art results on numerous clinical prediction tasks. However, most existing EHR FMs have context windows of <1k tokens. This prevents them from modeling full patient EHRs which can exceed 10k's of events. Recent advancements in subquadratic long-context architectures (e.g., Mamba) offer a promising solution. However, their application to EHR data has not been well-studied.

> 💡 **问题定义**: 现有 EHR FM 受限于 <1k 的上下文窗口，但单个患者的 EHR 可以超过 10k 事件。这是一个经典的"记忆瓶颈"问题——模型只能看到患者历史的冰山一角。

We address this gap by presenting the first systematic evaluation of the effect of context length on modeling EHR data. We find that longer context models improve predictive performance – our Mamba-based model surpasses the prior state-of-the-art on 9/14 tasks on the EHRSHOT prediction benchmark.

> 💡 **核心发现 1**: Mamba-16k 在 EHRSHOT benchmark 14 个任务中 9 个超过先前 SOTA（CLMBR-t-base），说明更多上下文确实有帮助。

For clinical applications, however, model performance alone is insufficient – robustness to the unique properties of EHR is crucial. Thus, we also evaluate models across three previously underexplored properties of EHR data: (1) the prevalence of "copy-forwarded" diagnoses which creates artificial repetition of tokens within EHR sequences; (2) the irregular time intervals between EHR events which can lead to a wide range of timespans within a context window; and (3) the natural increase in disease complexity over time which makes later tokens in the EHR harder to predict than earlier ones.

> 💡 **核心发现 2 — EHR 三大独特属性**:
> 1. **Copy-forwarding**: 慢性病诊断在每次就诊时被重复记录（如高血压），导致 token 人为重复，占据上下文窗口
> 2. **Irregular time intervals**: 自然语言中 token 间隔恒为 1，但 EHR 中连续事件可能相隔几天到几年
> 3. **Disease progression**: 自然语言中后面的 token 因为有更多上下文通常更容易预测（perplexity 降低），但 EHR 中疾病随年龄增长变得更复杂，后面的 token 反而更难预测
> 
> 这三个属性对 agent memory 系统同样有启发意义：冗余信息、不规则时间跨度、信息复杂度随时间增长。

Stratifying our EHRSHOT results, we find that higher levels of each property correlate negatively with model performance (e.g., a 14% higher Brier loss when making predictions for the most versus least irregular patients), but that longer context models are more robust to more extreme levels of these properties. Our work highlights the potential for using long-context architectures to model EHR data, and offers a case study for identifying new challenges in modeling sequential data motivated by domains outside of natural language. We release our model checkpoints and code at: https://github.com/som-shahlab/long_context_clues

> 💡 **关键数字**: 最不规则 vs 最不规则患者的 Brier loss 差异 14%。长上下文模型对这些极端属性更鲁棒。

---

## 🔖 Section 总结

### 核心洞察
1. 长上下文（16k）+ 亚二次架构（Mamba）= EHR 临床预测 SOTA
2. EHR 数据与自然语言有本质区别：重复性、不规则性、疾病进展
3. 长上下文不仅提升性能，还提升对 EHR 特殊属性的鲁棒性
