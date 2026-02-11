[← 返回 README](../README.md)

# 5 Experiment

## 📌 预览
在 5 个 benchmark × 3 MAS × 3 LLM 上全面实验，G-Memory 一致提升性能（最高 +20.89%），token 消耗仅略增，消融确认 insight 和 interaction 两层都有贡献。

---

In this section, we conduct extensive experiments to answer: (RQ1) How does G-Memory perform compared to existing single/multi-agent memory architectures? (RQ2) Does G-Memory incur excessive resource overhead? (RQ3) How sensitive is G-Memory to its key components and parameters?

## 5.1 Experiment Setup

**Datasets and Benchmarks.** To thoroughly evaluate the effectiveness of G-Memory, we adopt five widely-adopted benchmarks across three domains: (1) Knowledge reasoning, including HotpotQA [76] and FEVER [77]; (2) Embodied action, including ALFWorld [78] and SciWorld [79]; (3) Game, namely PDDL [80]. Details on these benchmarks are in Appendix A.1.

**Baselines.** We select four representative single-agent memory baselines, including non-memory, Voyager [16], MemoryBank [36], and Generative Agents [19], as well as three multi-agent memory implementations from MetaGPT [21], ChatDev [46], and MacNet [47], denoted as MetaGPT-M, ChatDev-M, and MacNet-M, respectively. Details are in Appendix A.2.

**MAS and LLM Backbones.** We select three representative multi-agent frameworks to integrate with G-Memory and the baselines, including AutoGen [13], DyLAN [72], and MacNet [47]. For instantiating these MAS frameworks, we adopt two open-source LLMs, Qwen-2.5-7b and Qwen-2.5-14b, as well as one proprietary LLM, gpt-4o-mini.

**Parameter Configurations.** We implement the embedding function $\mathbf{v}(\cdot)$ in Equation (4) with ALL-MINILM-L6-V2 [81]. The number of the most relevant interaction graphs $M$ in Equation (7) is set among $\{2, 3, 4, 5\}$, and the number of relevant queries $k$ in Equation (4) is set among $\{1, 2\}$.

> 💡 **实验设置评价**:
> - 覆盖面很全：3 个领域 × 5 个 benchmark × 3 个 MAS × 3 个 LLM = 45 种组合
> - Baseline 选择合理：4 个单 Agent 记忆 + 3 个 MAS 记忆
> - 用了开源 LLM（Qwen-7b/14b），结果更有参考价值
> - 注意：没有比较 MemGen（可能是因为 MemGen 是单 Agent 系统，不直接适用于 MAS）

---

## 5.2 Main Results (RQ1)

> 💡 **主实验结果见 Table 1**（GPT-4o-mini）、**Table 2**（Qwen-7b）、**Table 3**（Qwen-14b），均在 Appendix 中。这里总结关键发现。

**Takeaway ❶: G-Memory consistently improves performance across all task domains and MAS frameworks.** As shown in Table 2, when integrated with AutoGen and MacNet (powered by Qwen-2.5-7b), G-Memory surpasses the best-performing single-/multi-agent memory baselines by an average of 6.8% and 5.5%, respectively. With the more capable Qwen-2.5-14b, the improvement is even more pronounced: in Table 3, G-Memory boosts MacNet's performance on ALFWorld from 58.21% to 79.10%, achieving a substantial 20.89% gain.

> 💡 **最惊人的结果**: MacNet + Qwen-14b 在 ALFWorld 上从 58.21% → 79.10%（+20.89%）。这说明 G-Memory 对较强 LLM 的提升更大——因为强 LLM 更能利用好记忆中的经验信息。

**Takeaway ❷: Multi-agent systems demand specialized memory designs.** A thorough examination of existing baselines reveals a surprising insight: most memory mechanisms fail to consistently benefit MAS settings. In Table 2, baselines such as Voyager and MemoryBank degrade AutoGen's performance on PDDL by as much as 4.17% and 1.34%, respectively. We attribute this to the inability of these methods to provide agent role-specific memory support, which is essential in the PDDL strategic game tasks, where effective division of labor is critical to success. Even MAS-oriented designs, such as ChatDev-M, result in a 2.32% performance drop when applied to MacNet+SciWorld. We attribute this to ChatDev-M's narrow memory scope—storing only the execution results of past queries, which provides limited utility in embodied action environments. These findings highlight the necessity of G-Memory's core characteristics: role-specific memory cues, abstracted high-level insights, and trajectory condensation—all of which are critical for effective memory in MAS.

> 💡 **重要发现**: 单 Agent 记忆搬到 MAS 可能**有害**！
> - Voyager 在 PDDL 上降了 4.17%——因为它没有 role-specific 记忆
> - ChatDev-M 在 SciWorld 上降了 2.32%——因为它只存最终结果
> - 这验证了论文的核心论点：MAS 需要专门设计的记忆机制

---

## 5.3 Cost Analysis (RQ2)

![Figure 3](../images/f9264cea116f1b048d24709e2522ccf469fecd2100c26164c64b684e0eae825b.jpg)
*Figure 3: Cost analysis of G-Memory. We showcase the performance versus the overall system token cost when combined with different memory architectures.*

> 💡 **Figure 3 批读**:
> - G-Memory 在 PDDL+AutoGen 上提升 10.32%，额外 token 仅 1.4×10⁶
> - MetaGPT-M 额外 2.2×10⁶ token 只换来 4.07% 提升
> - G-Memory 的 token 效率明显更高——归功于 graph sparsifier 的压缩能力

**Takeaway ❸: G-Memory achieves high-performing collective memory without excessive token consumption.** As depicted in Figure 3, G-Memory consistently delivers the highest performance improvement (10.32% ↑ over no-memory setting on PDDL + AutoGen) while maintaining a modest increase in token consumption (only 1.4×10⁶). In contrast, MetaGPT-M incurred an additional 2.2×10⁶ tokens for a mere 4.07% gain. This clearly demonstrates the token-efficiency of G-Memory.

---

## 5.4 Framework Analysis (RQ3)

**Sensitivity Analysis.** Regarding the hop expansion, as shown in Figure 4a, 1-hop expansion consistently yields the best or near-best performance across tasks, with peak accuracies of 85.82% (ALFWorld), 55.24% (PDDL) in AutoGen. In contrast, 2-hop and 3-hop settings often degrade performance, e.g., PDDL drops to 49.79% (2-hop). This suggests that excessive hop expansion may introduce irrelevant insights during memory upward traversal, impairing task-specific reasoning.

Similarly, Figure 4b shows that the optimal $k$ is among $\{1, 2\}$. Larger $k$ values (e.g., $k=5$) can significantly degrade the system performance, e.g., 7.71% ↓ on ALFWorld + AutoGen and 2.5% ↓ on FEVER + DyLAN, indicating that retrieving more queries may introduce task-irrelevant noise.

> 💡 **超参数敏感性**:
> - Hop expansion: **1-hop 最优**，多跳引入噪声（2-hop 在 PDDL 降 5.45%）
> - Query 数量 k: **k=1 或 2 最优**，k=5 在 ALFWorld 降 7.71%
> - 核心教训：**记忆检索宁精勿滥**——少量高质量记忆 > 大量噪声记忆

**Ablation Study.** Figure 4c presents an ablation of G-Memory by isolating the impact of the high-level insight module and fine-grained interactions.

![Figure 4](../images/69897fe04c46eb64b8fd5c5401bb0de6051e2328c75d13bdabd06ee8c4a9dc47.jpg)
*Figure 4: (a) Sensitivity analysis of the hop expansion; (b) Sensitivity analysis of the number of selected queries k; (c) Ablation study on two variants of G-Memory.*

> 💡 **Figure 4c 消融分析批读**:
> - 只有 Interaction（无 Insight）：AutoGen 降 4.47%，DyLAN 降 3.82%
> - 只有 Insight（无 Interaction）：AutoGen 降 3.95%，DyLAN 降 3.39%
> - **两者都有贡献，但 Interaction 贡献略大**（因为保留了更具体的操作级别信息）
> - 最佳配置是两者结合——高层策略 + 底层操作经验的互补

As shown, removing either part leads to a consistent performance drop. When only fine-grained interactions are enabled, the average scores drop by 4.47% ↓ for AutoGen and 3.82% ↓ for DyLAN compared to the full method. Conversely, enabling only insights leads to smaller drops of 3.95% and 3.39%. This indicates that while both components are contributive, interactions offer a slightly greater impact, likely due to their preserving more fine-grained, dialogue-level contextual grounding.

---

## 5.5 Case Study

![Figure 5](../images/d784ce0756ebd3ed820657102316fce9a6003b8bb5867d0cfa4d2d183c5b3d05.jpg)
*Figure 5: Case study of G-Memory.*

> 💡 **Figure 5 批读**:
> - **ALFWorld 案例**: "put clean cloth in countertop" → 检索到 "put clean egg in microwave"（都需要先清洗）→ 提供了 agent 交互片段（solver 试图直接放置被 ground agent 纠正）
> - **HotpotQA 案例**: 检索到 insight "avoid mistakenly referring to similarly named individuals" → 防止了命名混淆错误
> - 展示了 G-Memory 的两层记忆在实际中如何互补工作

Figure 5 illustrates concrete memory cues provided by G-Memory across diverse tasks. For example, in the ALFWorld+AutoGen setting, given the task query "put a clean cloth in countertop", G-Memory successfully retrieves a highly analogous historical query, "put a clean egg in microwave"—both requiring the object to be in a clean state. Alongside this, G-Memory surfaces a critical trajectory segment where the solver agent attempts to place the egg in the microwave before cleaning, prompting the ground agent to intervene. This collaborative trajectory offers actionable guidance for the current task. Moreover, the high-level insights retrieved by G-Memory prove equally valuable for task execution. In the context of HotpotQA's web search task, G-Memory retrieves an insight warning against "mistakenly referring", which helps prevent agents from incorrectly answering based on similarly named individuals. Overall, G-Memory provides effective multi-level memory support across varied domains, including embodied action, knowledge reasoning, and game environments.

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 最大 embodied 提升 | +20.89% (ALFWorld, MacNet+Qwen-14b) |
| 最大 QA 提升 | +10.12% (HotpotQA, AutoGen+Qwen-14b) |
| Token 额外开销 | ~1.4×10⁶ (vs MetaGPT-M 的 2.2×10⁶) |
| 最优 hop expansion | 1-hop |
| 最优 k | 1 或 2 |
| Insight 消融降幅 | -3.95% (AutoGen), -3.39% (DyLAN) |
| Interaction 消融降幅 | -4.47% (AutoGen), -3.82% (DyLAN) |

### 核心洞察
1. G-Memory 是**唯一在所有设置下都正向提升的记忆方案**——其他方案在某些设置下反而有害
2. **较强的 LLM backbone 从 G-Memory 获益更多**——说明记忆质量的上限取决于 LLM 的利用能力
3. Token 效率高是因为 graph sparsifier 的压缩作用——用少量精选记忆替代大量冗余上下文
4. Insight 和 Interaction 两层都不可或缺，但 Interaction 略更重要（提供具体操作指导）
