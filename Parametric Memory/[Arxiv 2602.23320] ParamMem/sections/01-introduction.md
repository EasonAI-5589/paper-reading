[← 返回 README](../README.md)

# 1-2. Introduction & Preliminaries

## 📌 预览
Reflexion 框架的局限 → Reflection 多样性与性能的正相关 → ParamMem 的设计动机。

---

## Reflection 的重复性问题

LLMs have exhibited remarkable progress in complex reasoning. A key insight driving recent advances is test-time scaling. Among these, reflection-based frameworks have proven effective: agents verbally reflect on task feedback and accumulate self-reflections in episodic memory.

However, self-reflection often produces **repetitive and inaccurate outputs**, which hinders effectiveness.

![Figure 1](../images/28674cc83de102a4f296b124d2b1ee95854792d39ccc6203c5e3f7d3485130ac.jpg)
*Figure 1: Strong positive correlation between reflective diversity and task performance (Pearson r=0.76).*

> 💡 **Figure 1 批读**: 5 个数据集上的实验一致表明：reflection 越多样（pairwise cosine distance 越大），任务成功率越高。这为 ParamMem 的设计提供了实证动机。

## 三种 Memory 类型

| Memory | 机制 | 代表方法 |
|--------|------|---------|
| **Episodic** | 存储当前任务的历次 reflection | Reflexion |
| **Cross-sample** | 检索相似任务的推理轨迹 | DoT-bank |
| **Parametric (ParamMem)** | 将跨样本模式编码到参数中 | **本文** |

## Reflexion Framework

At iteration $k$: $y_k \sim p_\theta(\cdot \mid x, r_{1:k-1})$

**ParamAgent**: $y_k \sim p_\theta(\cdot \mid x, r_{1:k-1}, r_k^g)$ where $r_k^g \sim p_\psi(\cdot \mid x)$

**ParamAgent-plus**: $y_k \sim p_\theta(\cdot \mid x, r_{1:k-1}, \text{RETRIEVE}(B, x), r_k^g)$

![Figure 2](../images/8cf7df6790d39082fba37f4ccaf581c6881fba9f063541c175c09fffeae13bec.jpg)
*Figure 2: Memory mechanism comparison across frameworks.*

> 💡 **Figure 2 批读**: ParamAgent-plus 是最完整的框架：episodic (当前经验) + cross-sample (检索历史) + parametric (生成式)。三种记忆互补——episodic 是精确回忆，cross-sample 是相似案例，parametric 是泛化模式。

---

## 🔖 Section 总结

### 核心洞察
1. **Diversity is key**: reflection 重复性是自我反思方法的瓶颈，多样性直接决定性能
2. **三种记忆互补**: episodic（精确但有限）+ cross-sample（相似但受检索质量限制）+ parametric（泛化但需训练）
3. **ParamMem 的独特价值**: 不靠 prompt engineering 或检索，而是通过参数化实现 implicit generalization
