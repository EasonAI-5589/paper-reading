# 5. SODA Method

## 5.1 Optimal Matching Using Dynamic Programming

### 原文

To determine the matching between generated and reference captions, we regard the matching as a combinatorial optimization problem: finding one-to-one matching between the captions that maximizes the sum of the IoU by considering temporal ordering. Following the current evaluation framework, we also use the threshold τ for the matching; we define cost C_{i,j} between a reference caption g_i and a generated caption p_j based on the IoU as follows:

$$C_{i,j} = \begin{cases} \text{IoU}(g_i, p_j) & \text{if } \text{IoU}(g_i, p_j) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

Then, we sort the captions based on temporal ordering, that is, in the order of the beginning time of their proposals, by utilizing function s(·), and define S[i][j], which stores the maximum score of optimal matching between 1st to i-th generated captions and the 1st to j-th reference truth captions, as follows:

**Initialization:**
$$S[i][0] = 0 \quad (0 \leq i \leq |\mathcal{P}|)$$
$$S[0][j] = 0 \quad (0 \leq j \leq |\mathcal{G}|)$$

**Recurrence:**
$$S[i][j] = \max \begin{cases} S[i-1][j] & \text{(skip } p_i \text{)} \\ S[i-1][j-1] + C_{i,j} & \text{(match } p_i, g_j \text{)} \\ S[i][j-1] & \text{(skip } g_j \text{)} \end{cases}$$

Figure 4 shows an example process to obtain the optimal matching for the example given in Figure 1, with τ = 0. After filling out table S by dynamic programming, S[4][5] stores the optimal matching score, 2.7. Thus, we can obtain the optimal matching between g_k and p_ℓ by tracing the path, from [4,5] to [0,0]. In the example, the optimal matching is (g_1, p_1), (g_3, p_2), (g_4, p_4).

---

### 理解与批注

#### DP 状态定义
- `S[i][j]`: 前 i 个生成 caption 和前 j 个参考 caption 的最优匹配分数
- 先按开始时间排序，保证时序

#### 状态转移的三种情况
```
S[i][j] = max{
    S[i-1][j],           // 跳过 p_i（不匹配这个生成的）
    S[i-1][j-1] + C_i,j, // 匹配 (p_i, g_j)
    S[i][j-1]            // 跳过 g_j（这个 reference 没被匹配）
}
```

#### 示例
IoU 矩阵 (τ=0):

|    | p1  | p2  | p3  | p4  | p5  |
|----|-----|-----|-----|-----|-----|
| g1 | 0.7 | 0.1 | 0.4 | 0.9 | 0.1 |
| g2 | 0.2 | 0.3 | 0.5 | 0.4 | 0.5 |
| g3 | 0.4 | 1.0 | 0.3 | 0.7 | 0.8 |
| g4 | 0.8 | 0.7 | 0.6 | 1.0 | 0.1 |

**最优匹配**: (g1, p1), (g3, p2), (g4, p4)  
**最优分数**: 0.7 + 1.0 + 1.0 = 2.7

> 💡 关键：匹配保持时序！如果 g1 匹配 p1，g3 只能匹配 p2 之后的

---

## 5.2 F-measure for Evaluating Video Story Description

### 原文

To give a low score for too many or too few captions, the sum of METEOR scores should be normalized by considering the number of generated and reference captions. Thus, we propose an evaluation metric based on F-measure as follows:

$$\text{F-measure}(\mathcal{G}, \mathcal{P}) = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Here, Precision and Recall are defined on the basis of the optimal matching as follows:

$$\text{Precision}(\mathcal{G}, \mathcal{P}) = \frac{\sum_{g \in \mathcal{G}} f(g, p_{a(g)})}{|\mathcal{P}|}$$

$$\text{Recall}(\mathcal{G}, \mathcal{P}) = \frac{\sum_{g \in \mathcal{G}} f(g, p_{a(g)})}{|\mathcal{G}|}$$

When systems generate too many captions, Precision scores tend to be low, while Recall scores tend to be high. Thus, the systems cannot obtain good F-measure scores. When systems generate too few captions, they also cannot obtain good F-measure scores since they tend to receive good Precision scores but poor Recall scores.

---

### 理解与批注

#### 为什么需要 F-measure？

| Caption 数量 | Precision | Recall | F-measure |
|-------------|-----------|--------|-----------|
| 太多 (200+) | 低 ↓ | 高 ↑ | 低 ↓ |
| 太少 (1-2) | 高 ↑ | 低 ↓ | 低 ↓ |
| **刚好 (3-4)** | 适中 | 适中 | **高 ↑** |

> 💡 F-measure 同时惩罚过多和过少

#### 直觉理解
- **Precision**: 分母是 |P|（生成数量）→ 生成越多，分母越大，分数越低
- **Recall**: 分母是 |G|（参考数量）→ 生成越少，覆盖越少，分数越低
- **F-measure**: 两者的调和平均 → 只有数量匹配时才能同时高

---

## 5.3 IoU 加权 (SODA_c)

### 原文

In evaluating video story descriptions, the IoU plays an important role. Even if METEOR scores between generated and reference captions are perfect, they make no sense if the IoU between the captions is zero. However, in the current evaluation framework, the IoU is utilized only for determining the matching between the captions. Thus, the IoU does not directly affect the sum of METEOR scores. To reflect the IoU more directly to evaluation scores, we propose an alternative of the cost in Equation (4):

$$C_{i,j} = \text{IoU}(g_i, p_j) \times f(g_i, p_j)$$

By utilizing this cost, even if the METEOR score is high, the evaluation score can be lowered when the IoU score is low.

---

### 理解与批注

#### 为什么需要 IoU 加权？

原来的问题：
```
IoU = 0.1 (几乎不重叠)
METEOR = 1.0 (文字完全匹配)
结果: 分数很高 ❌
```

SODA_c 的解决：
```
Cost = IoU × METEOR = 0.1 × 1.0 = 0.1
结果: 分数很低 ✅
```

> 💡 SODA_c 是最终推荐的版本，同时考虑时间重叠和文本质量

#### 三个 SODA 变体总结

| 变体 | τ 设置 | Cost 函数 | 特点 |
|------|--------|----------|------|
| SODA (a) | 0.9, 0.7, 0.5, 0.3 平均 | IoU | 对 τ 敏感 |
| SODA (b) | 0 | IoU | 更敏感，无阈值 |
| **SODA (c)** | 0 | IoU × METEOR | **推荐**，综合考虑 |
