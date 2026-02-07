# 3. Current Evaluation Framework

## 定义

- $\mathcal{G}$: 参考 caption 集合
- $\mathcal{P}$: 系统生成的 caption 集合
- $g$: 参考 caption
- $p$: 生成的 caption

## IoU 计算

每个 caption 有一个 proposal（时间跨度）。IoU 定义为：

$$\text{IoU}(g, p) = \max\left(0, \frac{\min(e(g), e(p)) - \max(s(g), s(p))}{\max(e(g), e(p)) - \min(s(g), s(p))}\right)$$

其中 $s(\cdot)$ 和 $e(\cdot)$ 分别返回开始和结束时间。

## 匹配规则

对于阈值 $\tau$，定义匹配集合：

$$G_{p,\tau} = \{g \in \mathcal{G} | \text{IoU}(g, p) \geq \tau\}$$

- 如果 $G_{p,\tau} = \emptyset$，添加随机字符串作为惩罚

## 评估公式

$$E(\mathcal{G}, \mathcal{P}, \tau) = \frac{\sum_{p \in \mathcal{P}} \sum_{g \in G_{p,\tau}} f(g, p)}{\sum_{p \in \mathcal{P}} |G_{p,\tau}|}$$

其中 $f(\cdot, \cdot)$ 是评测指标（如 METEOR）。

最终分数通常是 $\tau = 0.9, 0.7, 0.5, 0.3$ 的平均值。

## 示例

假设 IoU 矩阵如图：

|   | p1 | p2 | p3 | p4 | p5 |
|---|----|----|----|----|-----|
| g1 | 0.7 | 0.1 | 0.4 | 0.9 | 0.1 |
| g2 | 0.2 | 0.3 | 0.5 | 0.4 | 0.5 |
| g3 | 0.4 | 1.0 | 0.3 | 0.7 | 0.8 |
| g4 | 0.8 | 0.7 | 0.6 | 1.0 | 0.1 |

当 $\tau = 0.5$ 时：
- $G_{p_1} = \{g_1, g_4\}$
- $G_{p_2} = \{g_3, g_4\}$
- ...

产生 **11 个匹配对**，计算 METEOR 并平均。

> ⚠️ 问题：一个生成 caption 可以匹配多个参考，导致 **loose matching**
