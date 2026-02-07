# 3. Method

> 来源: PyramidDrop (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: 实证研究 → PyramidDrop 设计 → 效率分析

---

### 3.1 Study of Visual Token Redundancy in LVLMs

实验设置：LLaVA-v1.5-7B (32 层)，在 layer 2/8/16/24 分别剪不同比例 token。

**关键发现**:
- Layer 2: 对剪枝极敏感（无论剪多少都掉性能）
- Layer 16: 保留 10% token 几乎无影响
- Layer 24: 性能与 image token 完全无关

Attention map 可视化：
- 浅层: attention 均匀分布在所有 image token 上
- 深层: attention 稀疏，集中在与问题相关的局部区域

> 💡 **批注**: 这个发现解释了为什么 FastV 在 Layer 2 就剪会丢性能 — 浅层 LLM 正在做全局理解，需要所有 token 参与。

---

### 3.2 PyramidDrop

![Figure 2](../images/e1109872f13c1ec849cba89f8fd2ce572a5d6831b0bbcf469a8fe56775bcf17f.jpg)
*Figure 2: PyramidDrop 概览。将 LLM 分成多个 stage，每个 stage 结束时丢弃部分 image tokens。*

> 💡 **Figure 2 批读**:
> ```
> 32 层 LLM 分成 4 个 stage (S=4):
> 
> Stage 1 (Layer 0-7):   保留 100% tokens (576)
> Stage 2 (Layer 8-15):  保留 50% tokens  (288)  ← 在 Layer 7 结束时剪
> Stage 3 (Layer 16-23): 保留 25% tokens  (144)
> Stage 4 (Layer 24-31): 保留 12.5% tokens (72)
>
> 每个 stage 的 token 数 = V₀ × λ^(s-1)
> ```

#### 剪枝依据

只用 **last instruction token** 和 image tokens 的 attention 做排序：

$$\text{similarity} = q_j^{t_I} \times (k_j^v)^T$$

> 💡 **批注**: 
> - 只用 last instruction token（而非所有 text token）做排序 — 比 SparseVLM 更简洁
> - 复用 self-attention 的 Q/K，不需要额外参数
> - 和 FlashAttention 兼容！这很重要，SparseVLM 需要提取完整 attention 矩阵

#### 效率分析

λ=0.5, S=4 时理论节省 53.2% 计算量：

$$\text{Cost} = \frac{1-\lambda^S}{S \cdot (1-\lambda)} \cdot c \cdot N \cdot L$$

> 💡 **批注**: PyramidDrop 的额外开销只有 S-1 次向量内积（last token vs image tokens），复杂度 O(n)，可忽略不计。

---

## 💡 Section 总结

### 方法对比
| 特性 | FastV | SparseVLM | PyramidDrop |
|------|-------|-----------|-------------|
| 剪枝位置 | Layer 2 固定 | 每层自适应 | 分 stage 渐进 |
| 剪枝依据 | attention score | text rater guided | last instruction token |
| 额外参数 | 无 | 无 | 无 |
| Training-free | ✅ | ✅ | ✅ (也可训练) |
| 训练加速 | ❌ | ❌ | ✅ |
| FlashAttn 兼容 | ✅ | ⚠️ 需要提取 A | ✅ |

### 核心洞察
1. **渐进式比一刀切更好** — 浅层信息重要，深层冗余
2. **用 last instruction token 做排序** — 比全部 text token 更简洁且有效
3. **可以同时加速训练** — 这是 PyramidDrop 独有的优势
