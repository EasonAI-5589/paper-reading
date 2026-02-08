# Appendix (Selected)

> 来源: MM-RLHF (ICML 2025)

---

## Appendix A: Related Work

> 💡 **简述**: 梳理了 MLLM 发展（LLaVA, Qwen-VL, InternVL 等）、MLLM Alignment（Fact-RLHF 是首个 multimodal RLHF, LLaVA-Critic 用 iterative DPO）、MLLM Evaluation benchmarks。核心论点：当前 alignment 受限于**缺乏高质量多模态 alignment 数据集**——现有人工标注 <10K 样本。

---

## Appendix B: Annotation Guidelines（Apple Assignment 重点！）

> 💡 **完整标注指南**: 这是论文中最值得细读的 Appendix，直接定义了标注标准。

### B.1 Visual Faithfulness Evaluation

**定义**: 评估生成的回答是否准确反映图片中的对象和关系。

**评估维度**:
1. **Object Description Accuracy**: 对象描述是否准确，避免幻觉
2. **Object Relationship Accuracy**: 空间/结构/功能关系是否正确
3. **Object Attribute Accuracy**: 物理特征、颜色、大小、状态
4. **Overall Accuracy**: 综合视觉元素描述的准确性

**5 级评分**:
| 等级 | 描述 |
|------|------|
| Severely Inaccurate | 重大错误，描述不存在的对象 |
| Partially Inaccurate | 大部分正确，部分细节不准 |
| Moderately Accurate | 基本准确，小错误 |
| Accurate | 正确且与 ground truth 一致 |
| Highly Accurate | 极其精确，包含有用细节 |

### B.2 Helpfulness Evaluation

**定义**: 回答是否为用户提供有价值的信息和洞察。

**评估维度**:
1. 是否提供有用的上下文和信息
2. 是否遵循用户的具体指令
3. 整体对用户理解的贡献

**5 级评分**: Not Helpful → Partially → Moderately → Helpful → Very Helpful

### B.3 Ethical Considerations Evaluation

**定义**: 安全、隐私保护、公平、避害。

**评估维度**:
1. 敏感话题是否提供适当警告
2. 不泄露隐私信息
3. 对个人/群体的公平性
4. 不煽动暴力、无 NSFW 内容

**5 级评分**: Highly Unsafe → Unsafe → Moderately Ethical → Ethically Compliant → Highly Ethical

### B.4 Annotation Requirements

1. 仔细阅读 prompt 和 response 后，按三个维度评分
2. 每个模型简要记录评分理由
3. 综合考虑所有维度后，手动对所有 response 排序
4. **Tie 处理**: 标记是否感知到无显著差异，如有 tie 则提供负例（MCQ 给错误答案，长文本修改内容引入错误）
5. 简要说明排序依据

> 💡 **标注流程总结（Apple Assignment 可直接引用）**:
> ```
> Per response:
>   ├── Score: Faithfulness (1-5)  + 理由
>   ├── Score: Helpfulness (1-5)   + 理由
>   └── Score: Ethics (1-5)        + 理由
> 
> Per query (all responses):
>   ├── Overall Ranking: rank all responses
>   ├── Ranking Rationale: 排序依据
>   └── Tie Handling:
>       ├── Poor tie → annotator 写正确答案
>       └── Good tie → annotator 引入错误
> ```

---

## Appendix C: Safety Dataset & Benchmark

### C.1 Training Data Construction
- 850 safety samples + 500 adversarial samples
- Safety 来源: Red Teaming VLM, CelebA, VLSBench
  - 200 Jailbreak, 200 privacy/discrimination, 150 hacking, 200 violence, 100 self-injury
- Adversarial: AnyAttack 生成, ε=8/255

### C.2 Benchmark (MM-RLHF-SafetyBench)
- 9 tasks from Multitrust + 2 from VLGuard
- 包括: adversarial attack, risk identification, typographic/multimodal/cross-modal jailbreak, NSFW

---

## Appendix D: Why Human Annotation?

> 💡 **Case Studies 总结**: 用具体例子说明人工标注的优势。

### D.1 Misleading and Incomplete Questions
- **Confusing questions** (Figure 8): 问题和选项矛盾，模型硬选一个答案，人工标注员识别出问题本身有缺陷
- **Incomplete questions** (Figure 9): 条件不足无法求解，模型仍然尝试回答，人工标注员指出条件不足

### D.2 Difficult-to-Distinguish Answers
- **All models fail** (Figure 10): 所有模型都错，但模型标注员还是选了一个"最好的"错误答案
- **Rich but subtly wrong** (Figure 11): 长回答中有细微的视觉感知错误（红色标记），模型标注员忽略了这些细节，人工标注员平均花 6 分钟仔细区分

> 💡 **对 Apple Assignment 的意义**: 这些 case studies 很有说服力——说明 human annotation 在 MLLM alignment 中**不是奢侈品而是必需品**。尤其在：(1) 问题本身有问题时 (2) response 之间差异微妙时。

---

## Appendix E: Comparison to β-DPO

核心区别:
1. **首次在 MLLM 中探索 dynamic β**: LLM 领域的 implicit reward 方法不适用于 MLLM
2. **用 external high-quality RM**: 而非模型自身信号，因为 MLLM 信号判别力弱
3. **Instance-level β adjustment works**: 当有高质量 RM 时，instance-level 调整有效（与传统观点不同）

---

## Appendix F: More Ablation

### F.1 Dataset + MM-DPO 效果
![Figure 12](../images/00231cd350cbe3fcb7a09c04a8c8f366f127b70e095d75e05dace391edcf0df3.jpg)
*Figure 12: (a) 各方法在 real-world tasks 上的评估。(b) 超参数 k 和 w 对 MM-DPO 的影响。*

> 💡 **Figure 12 批读**:
> - (a) MM-RLHF 数据 → 全面提升；+Implicit Reward → 不 work；+MM-DPO → 进一步提升
> - (b) k 和 w 的组合：默认 (0.5, 0.5) 效果好，但不能同时太大或太小
> - 方法对超参数有一定鲁棒性

### F.2 Hyperparameters w and k
- 默认: w=0.5, k=0.5
- 不能同时太大或太小
- 有一定鲁棒性

---

## Appendix Table 7: Annotation Enhancement Prompt

```
You will receive an image-related question, an answer, and a comment 
provided by a human expert for the answer.

Your task is to expand the human comment comprehensively while retaining 
its strengths and weaknesses, making it more professional, and logically 
rigorous. Focus only on expanding the comment and do not answer the question.

Ensure the expanded comment is strictly based on the provided human comment 
and avoids any speculation or uncertain content.

[Question:] {question}
[Answer:] {answer}
[Human Comment for the answer:] {reason}

Expanded Comment:
```

> 💡 **GPT-4o Expansion Prompt**: 这是将人工简短理由扩展为详细 critique 的关键 prompt。约束非常严格——"只扩展原内容，不引入推测"。这个技巧值得在 Apple Assignment 中提及。
