# 3. Current Evaluation Framework

## 原文

The automatic evaluation framework proposed for ActivityNet Captions has been widely utilized for the DVC task.

**Definitions:**
- $\mathcal{G}$: set of reference captions
- $\mathcal{P}$: set of generated captions
- $g$: a reference caption
- $p$: a generated caption

Each caption has a proposal that indicates a time span of an event. The IoU between g and p is defined as:

$$\text{IoU}(g, p) = \max\left(0, \frac{\min(e(g), e(p)) - \max(s(g), s(p))}{\max(e(g), e(p)) - \min(s(g), s(p))}\right)$$

where $s(\cdot)$ and $e(\cdot)$ return the start and end time of the proposal.

A set of reference captions whose IoU exceeds threshold τ for p:

$$G_{p,\tau} = \{g \in \mathcal{G} | \text{IoU}(g, p) \geq \tau\}$$

When $G_{p,\tau} = \emptyset$, a random string is added as penalty.

**Final evaluation:**

$$E(\mathcal{G}, \mathcal{P}, \tau) = \frac{\sum_{p \in \mathcal{P}} \sum_{g \in G_{p,\tau}} f(g, p)}{\sum_{p \in \mathcal{P}} |G_{p,\tau}|}$$

where $f(\cdot, \cdot)$ is METEOR. The final score is usually averaged for τ = 0.9, 0.7, 0.5, 0.3.

---

## 理解与批注

### IoU 计算示意

```
Reference:  |----[g]------|
Generated:      |---[p]---|
                ↑ overlap ↑

IoU = overlap / union
```

### 评估流程

```
Step 1: 对每个生成的 p，找所有 IoU > τ 的参考 g
Step 2: 计算所有匹配对的 METEOR
Step 3: 平均

问题：一个 p 可以匹配多个 g！
```

### 示例

IoU 矩阵 (τ = 0.5):

|    | p1  | p2  | p3  | p4  | p5  |
|----|-----|-----|-----|-----|-----|
| g1 | 0.7✓| 0.1 | 0.4 | 0.9✓| 0.1 |
| g2 | 0.2 | 0.3 | 0.5✓| 0.4 | 0.5✓|
| g3 | 0.4 | 1.0✓| 0.3 | 0.7✓| 0.8✓|
| g4 | 0.8✓| 0.7✓| 0.6✓| 1.0✓| 0.1 |

匹配对：
- $G_{p_1} = \{g_1, g_4\}$ → 2 对
- $G_{p_2} = \{g_3, g_4\}$ → 2 对
- $G_{p_3} = \{g_2, g_4\}$ → 2 对
- $G_{p_4} = \{g_1, g_3, g_4\}$ → 3 对
- $G_{p_5} = \{g_2, g_3\}$ → 2 对

**总共 11 对**，计算 METEOR 并平均

> ⚠️ 问题：g4 被匹配了 4 次！这合理吗？
