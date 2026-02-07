# 4. Analysis and Discussion

> 来源: VisionZip (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: 这一节是论文的精华——解释**为什么冗余存在**（4.1）、**为什么 VisionZip 比 text-relevant 方法好**（4.2）、**VisionZip 的部署优势**（4.3）。

---

### 4.1 Reasons of Redundancy in Visual Tokens

> 💡 **4.1 要点预览**: 冗余的根本原因是 Transformer 的 self-attention + softmax 会把信息 "快捷" 集中到少数 proxy token。

#### Visualization of the Redundancy

![Figure 5](../images/ddfa944fff06638f7e39b5804b5f6753c9dec9e10007543dd1bfbec39bc3fb0c.jpg)
*Figure 5: 不同层的 attention 分布。浅层分散，中间层突然集中，23 层（-2 层）达到峰值集中度。*

> 💡 **Figure 5 批读**:
> ```
> 层级变化:
> ├── 浅层 (1-10):  attention 均匀分布在整张图
> ├── 中间层 (10-15): attention 突然开始集中
> ├── 23 层 (-2 层):  attention 高度集中于少数 token ⭐
> └── 最后层 (24):    attention 又分散了
>     → 因为最后层要和 CLIP text branch 对齐 (contrastive loss)
>     → 所以 VLM 选 -2 层而不是最后层
> ```

#### Explanation

As layer depth increases, instead of aggregating knowledge from all tokens, the model tends to **"shortcut"** by concentrating information into a few proxy tokens. If a CLS token is present, knowledge may further concentrate into the CLS token.

> 💡 **批注**: 用大白话解释信息集中的机制：
> ```
> Softmax 的放大效应:
>
> softmax(z_i) = e^{z_i} / Σ e^{z_j}
>
> 其导数 = softmax(z_i) × (1 - softmax(z_i))
>
> 这意味着:
> ├── z 大 → 梯度大 → 训练时被进一步强化 → 更大
> └── z 小 → 梯度小 → 训练时几乎不更新 → 更小
>
> 结果: 马太效应！强者更强，弱者更弱
> → 少数 token 的 attention 越来越高
> → 多数 token 的 attention 越来越低
> → 信息全集中到少数 "代理 token" (proxy tokens)
> ```
> 
> 类似现象在 LLM 中叫 **"Attention Sink"** [52]，在语义分割中叫 **"Global Token"** [43]。

![Figure 6](../images/4421a27332ed66d080b38b5b2d471b4c82794869e1bb89e303eb1e971d27755e.jpg)
*Figure 6: (a) Softmax 导数的放大效应；(b) Feature misalignment — 关于"人"的信息不在人上面，而在路上的 proxy token。*

> 💡 **Figure 6 批读**:
> ```
> (a) Softmax 导数曲线:
> ├── z < -2: 梯度 ≈ 0 (被忽略的 token 永远被忽略)
> ├── z ≈ 0:  梯度最大 (transition zone)
> └── z > 2:  梯度呈指数增长 (dominant token 被强化)
>
> (b) Feature Misalignment 示例:
> ├── 图中有一个人和一辆出租车
> ├── 与 "人" 语义相关的 token 不在人身上！
> └── 信息被集中到马路上的一个 proxy token ⭐
>     → 这就是为什么 text-relevant 方法会选错 token
> ```

> 💡 **4.1 小结**:
> - 冗余的根因: Softmax 的马太效应 → 信息集中到少数 proxy token
> - Proxy token 的位置不一定在语义主体上（feature misalignment）
> - 这是 Transformer 架构的固有特性，不是 bug

---

### 4.2 Why VisionZip Outperforms Previous Work?

> 💡 **4.2 要点预览**: Text-relevant 方法（FastV, SparseVLM）因为 feature misalignment 会选错 token——选到的 "语义相关" token 实际信息量很少。

#### Text-Relevant Efficient VLM 的问题

FastV 和 SparseVLM 用 LLM 中 text-visual attention 来选择 token。看起来合理：选和问题最相关的 token。

> 💡 **批注**: 但问题在于 **feature misalignment**！
> ```
> 问: "What is the person doing?"
>
> Text-relevant 方法的思路:
> ├── LLM 注意到 "person" 这个词
> ├── 找与 person 语义最相关的 visual token
> └── 选中人身上的 token → 但这些 token 信息量很少！
>     因为 vision encoder 已经把人的信息集中到路上的 proxy token 了
>
> VisionZip 的思路:
> ├── 直接用 CLS attention 选 dominant token
> ├── Proxy token（信息量最大的）被优先选中
> └── 虽然位置不在人上面，但包含了人的信息 ⭐
> ```

#### 验证实验

**Table 5: Feature misalignment 定量验证 (TextVQA, SparseVLM 保留 64 tokens)**

| 实验 | 输入 | 输出 | Accuracy | Δ |
|------|------|------|----------|---|
| Baseline | 576 → 64 | SparseVLM 选 64 | 51.1 | - |
| Ex1: 去掉 top-50 dominant | 526 → 64 | SparseVLM 选 64 | 46.4 | -9.2% |
| Ex2: 只给 top-128 | 128 → 64 | SparseVLM 选 64 | 52.5 | +2.7% |

> 💡 **Table 5 批读**:
> ```
> Ex1: 先去掉 50 个 dominant token，再让 SparseVLM 选
> → 性能暴跌 9.2%！
> → 说明 dominant token 是信息的核心载体
>
> Ex2: 只给 VisionZip 选出的 top-128，再让 SparseVLM 选 64
> → 性能反而提升 2.7%！
> → 说明 VisionZip 预筛的 token 质量更高
> → SparseVLM 在高质量 token 池中选效果更好
>
> 结论: Text-relevant 方法选的"语义相关"token
>       ≠ 信息量最大的 token
>       因为 vision encoder 的 feature misalignment
> ```

> 💡 **4.2 小结**: VisionZip 胜出的根本原因不是方法更复杂，而是 **理解了 vision encoder 的信息分布规律**——信息在 dominant/proxy token 上，不在语义对应位置上。

---

### 4.3 The Advantage of the VisionZip

> 💡 **4.3 要点预览**: VisionZip 的三大实际优势——兼容量化、13B > 7B、多轮对话。

#### Easy to Deployment

| 配置 | Memory (Mb) | SQA Acc |
|------|-------------|---------|
| 7B-Full | 18,952 | 70.2 |
| 13B-Full | 36,721 | 73.5 |
| 13B-8bit + VisionZip | 16,632 | 70.8 |
| 13B-4bit + VisionZip | 10,176 | 70.3 |

> 💡 **批注**: 13B-4bit + VisionZip 只用 10GB 显存，性能和 7B-Full 持平！
> → 在消费级 GPU (如 3090 24G) 上就能跑 13B 模型
> → VisionZip + 量化 = 极致性价比部署方案

| 模型 | Time | TextVQA |
|------|------|---------|
| 7B | 1,714s | 61.3 |
| 13B | 2,516s | 64.3 |
| 13B + VisionZip | 1,246s | 62.2 |

> 💡 **批注**: 13B + VisionZip 比 7B 原版 **更快 (1246 vs 1714)** 且 **更好 (62.2 vs 61.3)**！这个结论对实际部署意义重大。

#### Advantage on Multi-turn Conversations

![Figure 7](../images/47bc85538f716f7db69d04628922c9fac38dd1443ef96896c76442dd368bd1d7.jpg)
*Figure 7: VisionZip vs text-relevant 方法在多轮对话中的对比。*

> 💡 **Figure 7 批读**:
> ```
> 多轮对话的问题:
>
> Text-relevant 方法 (FastV/SparseVLM):
> ├── Round 1: "What color is the car?" → 选了和 car 相关的 token
> ├── 这些 token 被存入 KV cache
> ├── Round 2: "What is the person doing?" → 新问题！
> └── 但 KV cache 里存的是 car 相关的 token → 答不好！
>
> VisionZip (text-agnostic):
> ├── 选的是 dominant token（信息最丰富的）
> ├── 不和任何特定问题绑定
> └── 对任何后续问题都能回答 ⭐
> ```
>
> **大白话**: Text-relevant 方法像是针对第一个问题定制了一副眼镜，换个问题就看不清了。VisionZip 是配了一副通用眼镜，虽然不针对任何问题，但什么都能看到。

> 💡 **4.3 小结**: VisionZip 的实际部署优势——兼容量化、让 13B 比 7B 更实用、多轮对话不掉分。

---

## 💡 Section 总结

### 核心洞察
1. **冗余原因**: Softmax 马太效应 → 信息集中到 proxy token → 大部分 token 冗余
2. **Feature misalignment**: Proxy token 位置 ≠ 语义主体位置 → text-relevant 方法选错 token
3. **VisionZip 胜出原因**: 直接选 dominant token（信息最集中的），不依赖 text-visual 语义对应
4. **部署优势**: 兼容量化、13B+VisionZip > 7B、多轮对话天然支持

### 与 text-aware 方法的核心对比
| 维度 | VisionZip | FastV/SparseVLM |
|------|-----------|-----------------|
| 选 token 依据 | Vision encoder attention | LLM text-visual attention |
| 是否受 misalignment 影响 | ❌ | ✅ (选到低信息量 token) |
| 多轮对话 | ✅ | ❌ (KV cache 绑定旧问题) |
| 量化兼容 | ✅ | 取决于实现 |
| 压缩位置 | Encoder 端 | LLM 内部 |
