[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 把多模态推理方向分成 **Explicit Reasoning**（显式 CoT、Think-with-Image）和 **Latent Reasoning**（训练式 / training-free / latent visual 注入）两大支，并指出现有 latent 方法都要么得训练、要么固定位置触发，给 DMLR 的「test-time + 动态」立 flag。

---

**Explicit Reasoning.** Many prior works have explored visual reasoning. Early approaches mainly relied on semantic CoT, where the model performs all inference in the text space after a one-time visual encoding [5, 4, 20, 21]. However, this separation between perception and reasoning often leads to misalignment and hallucination [6, 22, 23, 24, 22, 25]. To address these limitations, recent studies adopt a Thinking-with-Images paradigm, where the model can draw auxiliary elements [13, 26, 27], zoom or crop regions [11, 8, 28, 29], or generate intermediate visual cues [30, 10, 9], enabling it to reason directly over visual structures.

> 💡 **Explicit Reasoning 的两个子家族**:
>
> | 子家族 | 操作方式 | 代表 | DMLR 视角下的问题 |
> |---|---|---|---|
> | **Semantic CoT** | 图编码一次，然后纯文字推理 | KAM-CoT[5]、LLaVA-OV[4]、SFT-vs-RL[20]、VL-Rethinker[21] | 感知-推理分离，幻觉多 |
> | **Thinking-with-Images** | 推理时主动操作图：画、裁、缩 | Visual Sketchpad[27]、GRIT[11]、Pixel Reasoner[8]、DeepEyes[9]、DeepEyesV2[28]、ReFocus[10] | 工具调用不稳，inference 开销大 |
>
> 作者重点引用了 [22] (More thinking, less seeing) 和 [23] (Latent space steering) — 这两篇都在说"思考越多反而看得越糊"，正是 DMLR 用 confidence 监督的动机来源。

---

**Latent Reasoning.** Recently, an increasing number of studies have begun to shift reasoning from the explicit token space to the model's latent representation space. Some methods introduce dedicated training frameworks that optimize latent representations to support more effective internal reasoning [31, 14, 32, 33, 34, 35], while others propose training-free approaches that manipulate latent activations during inference to refine the reasoning process [15, 36, 37, 38, 39]. In addition, several recent works explore injecting visual information into the latent space [16, 17, 40, 41, 18], enabling models to iteratively operate over both latent semantic features and latent visual cues, thereby supporting a more flexible form of interleaved multimodal reasoning.

> 💡 **Latent Reasoning 三个子家族 + DMLR 的位置**:
>
> ```
> Latent Reasoning
> ├── 训练式 (Training-required)
> │   ├── CoCoNut [14] — 连续 latent CoT 训练
> │   ├── Fractional Reasoning [31] — latent steering vectors
> │   ├── ThinkAct [32] — VLA 的 latent planning
> │   ├── MILR [33] — 多模态生成的 test-time latent reasoning
> │   ├── Latent Reasoning as Vocab Superposition [34]
> │   └── Token Perception RL [35]
> ├── Training-free (✅ DMLR 同类)
> │   ├── LatentSeek [15] — instance-level latent policy gradient (DMLR 的直系前辈)
> │   ├── Soft Thinking [36] — 连续概念空间推理
> │   ├── Soft Tokens, Hard Truths [37]
> │   ├── LTPO [38] — test-time 思想策略优化
> │   └── Feature Steering for CoT [39]
> └── Latent visual 注入
>     ├── Latent Visual Reasoning [16]
>     ├── Machine Mental Imagery [17]
>     ├── Latent CoT for Visual Reasoning [40]
>     ├── ICoT [41] — Interleaved-modal CoT (✅ DMLR 主对照基线)
>     └── Multimodal CoCoNut [18]
> ```
>
> DMLR 同时落在 **「training-free」** 和 **「latent visual 注入」** 的交集——之前的工作里几乎没人同时占这两块。

> 💡 **暗藏的论证逻辑**: 作者其实在说"训练式的 latent reasoning 不灵活，training-free 的 LatentSeek 又没考虑视觉模态，所以我们做 training-free + 视觉感知"。引用 [16-18, 40, 41] 是为了表明"有人做过视觉 latent 注入，但他们都要训练"。

---

## 🔖 Section 总结

### 关键定位
- **DMLR ⊂ Latent Reasoning ∩ Training-free ∩ Multimodal**: 这是它的"窄门"，也是它的卖点。

### 最直接的对照基线（实验里出现的）
| Baseline | 范式 | 在表 1 中表现 |
|---|---|---|
| Vanilla | 模型直接答 | 起点 |
| Multimodal CoT [52] | Semantic CoT | 经常比 vanilla 还差（语言偏见） |
| CCoT [51] | Compositional CoT | 比 Multimodal CoT 略好 |
| ICoT [41] | Interleaved-modal CoT（视觉 latent 注入） | 在数学上有 gain，视觉接地有限 |
| **+DMLR (ours)** | training-free latent + DVI | 95%+ 任务最优 |

### 留给读者的关键问题
1. 训练式 latent reasoning 也许性能上限更高，DMLR 训练免费的代价是什么？
2. 既然 ICoT 也做 "latent + visual injection"，DMLR 的"动态"具体动在哪？（→ Section 4.2 的 DVI 算法详细回答）
