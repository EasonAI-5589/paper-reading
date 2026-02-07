# 5. SODA: Story Oriented Dense Video Captioning Evaluation

## 5.1 Optimal Matching Using Dynamic Programming

### 问题建模

将匹配问题建模为**组合优化问题**：找到一对一匹配，最大化 IoU 之和，同时保持时序顺序。

### Cost 定义

$$C_{i,j} = \begin{cases} \text{IoU}(g_i, p_j) & \text{if } \text{IoU}(g_i, p_j) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

### 动态规划

先按时间顺序排序 caption（根据 proposal 开始时间）。

定义 $S[i][j]$：前 $i$ 个生成 caption 和前 $j$ 个参考 caption 的最优匹配分数。

**初始化**:
$$S[i][0] = 0 \quad (0 \leq i \leq |\mathcal{P}|)$$
$$S[0][j] = 0 \quad (0 \leq j \leq |\mathcal{G}|)$$

**递推**:
$$S[i][j] = \max \begin{cases} S[i-1][j] & \text{(跳过 } p_i \text{)} \\ S[i-1][j-1] + C_{i,j} & \text{(匹配 } p_i, g_j \text{)} \\ S[i][j-1] & \text{(跳过 } g_j \text{)} \end{cases}$$

### 示例

对于之前的 IoU 矩阵（$\tau = 0$），DP 找到最优匹配：
- $(g_1, p_1), (g_3, p_2), (g_4, p_4)$
- 最优分数：$0.7 + 1.0 + 1.0 = 2.7$

> ✅ 保持了时序顺序，避免了 loose matching

## 5.2 F-measure for Evaluating Video Story Description

### 动机

需要惩罚过多或过少的 caption，用 F-measure：

$$\text{F-measure}(\mathcal{G}, \mathcal{P}) = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Precision & Recall

基于最优匹配计算：

$$\text{Precision}(\mathcal{G}, \mathcal{P}) = \frac{\sum_{g \in \mathcal{G}} f(g, p_{a(g)})}{|\mathcal{P}|}$$

$$\text{Recall}(\mathcal{G}, \mathcal{P}) = \frac{\sum_{g \in \mathcal{G}} f(g, p_{a(g)})}{|\mathcal{G}|}$$

其中 $a(g)$ 是 $g$ 匹配到的生成 caption。

### 效果

| Caption 数量 | Precision | Recall | F-measure |
|-------------|-----------|--------|-----------|
| 太多 | 低 | 高 | 低 |
| 太少 | 高 | 低 | 低 |
| 刚好 | 适中 | 适中 | **高** |

## 5.3 IoU 加权 (SODA_c)

### 动机

当前框架中 IoU 只用于确定匹配，不直接影响分数。但高 IoU 的匹配应该更有价值。

### 改进的 Cost 函数

$$C_{i,j} = \text{IoU}(g_i, p_j) \times f(g_i, p_j)$$

> ✅ 即使 METEOR 分数高，如果 IoU 低，评估分数也会降低
