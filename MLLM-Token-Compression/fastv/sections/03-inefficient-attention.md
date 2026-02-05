# 3. Inefficient Visual Attention in VLLMs

> ==这是 FastV 最核心的发现章节！==

---

## 3.1 Preliminaries

### Token 类型分类

> The input tokens at each decoding step can be categorized into four distinct types:
> - **System prompt (sys)**: 控制 LLM 行为的通用消息
> - **Image tokens (img)**: 视觉编码器输出的线性化特征
> - **User instruction (ins)**: 用户的查询问题
> - **Output tokens (out)**: 自回归生成的输出
>
> ==四种 token 类型：system prompt / image tokens / user instruction / output tokens==

### 定义指标

**Attention Allocation (λ)**：某类 token 在某层收到的总 attention score
$$\lambda_{sys}^j = \sum_{i=1}^n \alpha_{sys}^{i,j}$$

**Attention Efficiency (ε)**：attention allocation / token 数量
$$\epsilon = \lambda / \text{token\_count}$$

> ==Attention Efficiency 反映每个 token 平均收到多少 attention==

---

## 3.2 Experiment Settings

- 从多个任务随机采样：Flickr30K, PCA-Bench, A-OKVQA, MMMU
- 收集每个 output token 在不同层的 attention score 分布
- 分析不同类型 token 的 attention allocation 和 efficiency

---

## 3.3 Results ⭐

### 核心发现 1：Attention 不平衡与层深度相关

> In shallow layer the attention allocation is relatively more balanced than in deep layers.
>
> ==浅层相对平衡，深层严重不平衡==

### 核心发现 2：Image tokens attention efficiency 最低

> Image tokens have the **lowest attention efficiency** in both shallow and deep layers. System prompt is of extremely high attention efficiency in deep layers, which is **472 times** that of image tokens, taking up **85%** total attention scores.
>
> ==Image tokens 效率最低！深层只有 system prompt 的 1/472==

### 量化数据

| 层深度 | Image Tokens | System Prompt | 比例 |
|--------|--------------|---------------|------|
| Shallow (Layer 1-2) | 基准 | 2x | 1:2 |
| Deep (Layer 3+) | 基准 | **472x** | **1:472** |

> ==在深层，image tokens 只获得 system prompt 的 0.21% attention！==

---

## 3.4 Insights ⭐

### 现象解释

> There are vertical strong lines (in the system prompt) that takes up most of attention scores. The existence of vertical strong line shows that there are some input tokens that **consistently received high attention** during the whole decoding process.
>
> ==Attention map 中出现"垂直强线"：某些 tokens 持续获得高 attention==

### 信息聚合机制

> A small portion of **anchor tokens** aggregate the information from all input tokens and the model much favors to attend to those anchor tokens in deep layers.
>
> ==少量 "anchor tokens" 聚合了所有信息，深层主要 attend 这些 anchor tokens==

### 解释

```
浅层（Layer 1-2）:
  Visual tokens → [Self-Attention] → 信息聚合到 "anchor tokens"
                                      (通常是 system prompt)

深层（Layer 3+）:
  Model → 主要 attend to anchor tokens
       → Visual tokens 不再被直接 attend
       → 因此 visual tokens 的 attention efficiency 极低
```

> ==Visual tokens 的信息在浅层已经被提取并聚合到 anchor tokens 上，深层不再需要直接访问 visual tokens==

---

## 💡 Key Takeaways

1. **发现 1**：Visual tokens 在深层 attention 极度稀疏（1:472）
2. **发现 2**：存在 "anchor tokens" 聚合机制
3. **启发**：既然深层不需要 visual tokens，为什么不剪掉？

---

*[返回论文目录](../README.md)*
