[← 返回 README](../README.md)

# 6–8. Related Work, Limitation & Conclusion

## 📌 预览

本节涵盖论文的后三节：6. Related Work 将 LaCT 定位在 TTT 方法和长上下文建模的研究脉络中，并与 InfiniAttention、Block-Recurrent Transformer 等最相关工作做对比；7. Limitation 诚实地指出了三个局限性：缺乏旋转不变性、任务覆盖有限、语言推理能力未验证；8. Conclusion 总结了 LaCT 的核心贡献并展望未来方向。

---

# 6 Related Work

Test-time training. Test-Time Training (TTT) [2] is an emerging concept in sequence modeling that extends the concept of recurrent states in RNNs to online-adapted neural network components. In TTT models, a subset of weights, termed "fast weights," are updated to learn in-context. Existing methods typically employ a self-supervised loss that encourages these fast weights to memorize key-value associations from in-context tokens, using variants of gradient descent for online adaptation. TTT [2, 4] has opened a vast design space for new recurrent model architectures. For instance, many recent works have developed novel test-time optimizers [5, 7] and online training objectives [45]. However, current TTT approaches often suffer from low hardware utilization and limited state sizes, and consequently have not yet demonstrated their full potential. Our work primarily addresses these challenges by advocating for a new paradigm of using extremely large online minibatch (chunk) sizes for updating the fast weights. This paradigm can achieve orders-of-magnitude higher hardware utilization without relying on error-prone custom kernel implementations. Furthermore, it enables efficient scaling of nonlinear state sizes and offers the flexibility to use diverse fast weight neural networks and optimizers, thereby accelerating research progress in this area.

> 💡 **批注：LaCT 在 TTT 研究脉络中的定位**
>
> TTT 领域的研究进展可以分为几个方向：
> - **Online 目标设计**：如何设计自监督损失使 fast weights 更好地存储上下文（[2, 4, 45]）
> - **Fast Weight 优化器**：如何高效地更新 fast weights（[5, 7] 探索了不同优化器）
> - **架构设计**：fast weight 网络的结构选择
>
> LaCT 的创新点不在于发明新的 TTT 目标或优化器，而是从**系统层面**重新审视实现效率，提出了大 chunk 范式。这是一个"做对已知事情"（TTT）而非"发明新事物"的贡献——正是这种方向使得 LaCT 可以轻松集成其他方向的进展（如 Muon 优化器、SwiGLU 架构）。
>
> **与 Parametric Memory 领域的联系**：LaCT 可以被看作是"参数化记忆（Parametric Memory）"的一种实现方式——将序列的历史信息存储在神经网络参数中，而不是显式的 key-value 缓存里。大 chunk 使得这种参数化记忆可以是"高容量非线性的"，真正发挥出参数化表示的优势。

---

Combining chunk attention with recurrence. Several recent models combine local chunk attention with linear recurrence, such as Gated Attention Unit (GAU) [46], MEGA [47], MEGALODON [48], and InfiniAttention [26]. Among these, InfiniAttention is conceptually closest to our work, as it incorporates recurrence at the chunk level using the delta rule—interpreted as an online linear regression objective from the perspective of Test-Time Training (TTT). However, this update rule is limited in expressivity. In contrast, we employ a significantly more expressive update mechanism derived from a more general TTT framework, and demonstrate the substantial gains this brings.

Block-Recurrent Transformer [49] also explores large chunk memory updates, where memory tokens act as recurrent states that can self-attend and cross-attend with input tokens during each chunk update via attention mechanisms. The Perceiver-style register-token attention baseline used in our novel view synthesis experiments (Sec. 5.1, Table 2) is conceptually similar to the Block-Recurrent Transformer in its use of register tokens for context compression. As shown in Figure 4, our method significantly outperforms this approach in both speed and quality, with a comparable state size.

> 💡 **批注：与 InfiniAttention 和 Block-Recurrent Transformer 的对比**
>
> **InfiniAttention [26]**：
> - 相似点：都在 chunk 级别更新记忆状态
> - 关键区别：InfiniAttention 用 delta rule（一种线性更新规则）更新线性记忆矩阵，表达能力受限
> - LaCT 的优势：用非线性 SwiGLU MLP 作为 fast weight，配合 Muon 优化器，表达能力远超线性状态
>
> **Block-Recurrent Transformer [49]**：
> - 相似点：都用 large chunk 更新记忆（register tokens）
> - 关键区别：用 attention mechanism（另一个 O(N²) 操作）来更新 register tokens，复杂度高
> - LaCT 的优势：用梯度下降更新 fast weights，更高效且支持非线性状态
>
> **MEGA [47] / MEGALODON [48]**：结合了 exponential moving average（EMA）的线性递推和 local attention，但 EMA 的记忆容量极低（d 维向量而非矩阵），无法与 LaCT 的 $O(d^2)$ 状态相比。

---

Novel view synthesis. Novel view synthesis (NVS) is a long-standing task at the intersection of computer vision, graphics, and computational photography, requiring algorithms to render images of a static scene from previously unobserved viewpoints. Optimization-based approaches, such as NeRF [50] and 3D Gaussian Splatting [9], have achieved significant breakthroughs. These methods optimize a set of parameterized graphics primitives (i.e., explicit or implicit representations of radiance fields) through differentiable volumetric rendering to minimize reconstruction loss on input images. After an optimization process typically lasting tens of minutes, these approaches can render novel views photorealistically, and the optimized parameters form a 3D representation of the input scene.

Recently, data-driven approaches [32, 29, 36, 51] have also shown promising results. These methods can either directly render novel views or predict 3D representations given input images. Although successful on simpler object datasets, these methods often struggle with densely sampled scenes (e.g., scenes with over 100 input images). Our experiments demonstrate that our large-chunk test-time training approach outperforms or achieves comparable performance to 3D Gaussian Splatting on challenging scene datasets with up to 128 input images with $960 \times 536$ resolution at challenging scene datasets.We hope our method will inspire further research into effectively scaling data-driven NVS methods to longer and more complex input sequences.

> 💡 **批注：LaCT 在 NVS 领域的贡献**
>
> NVS 领域有两个主要范式：
> - **优化方法**（NeRF、3DGS）：每个新场景都需要几分钟到小时的优化，但质量高
> - **前馈方法**（LRM、LVSM、LongLRM）：一次前向传播生成结果，但通常局限于少量输入视角（≤32 视角）
>
> LaCT 的贡献是**将前馈方法扩展到 128 视角（1M tokens）**，这是前馈 NVS 方法的新 frontier。特别是在场景数据集上超越 3DGS（一个需要按场景优化的方法），说明 LaCT 的压缩质量已经接近"显式 3D 重建"的水平。
>
> **快速推理的实际意义**：3DGS 需要几分钟优化才能渲染，LaCT 的推理只需一次前向传播（几秒内），在实时渲染或大规模场景处理中有显著优势。

---

Autoregressive video diffusion. Current state-of-the-art video generation is dominated by bidirectional diffusion transformers operating in latent space [52, 53, 54, 43]. These methods factorize the video distribution into a sequence of conditional distributions based on noise levels, following diffusion processes [55, 56] or flow matching [57], then use a diffusion transformer to jointly learn all the conditional distribution. Autoregressive video diffusion [58, 59, 60, 61, 62, 63] introduces an additional temporal dimension to this factorization, where the neural networks learns to model the conditional probability of the next chunks of videos at different noise levels, conditional on previous videos and noisier version of current video frames.

During training, some autoregressive methods employ teacher forcing, supervising the model on noisy video frames given previous clean context frames as condition [58, 59, 60], though this can lead to low token utilization, i.e. only a small portion of tokens get supervision. To improve token efficiency, other techniques such as progressive noise injection [61] or the use of frame-independent noises (sometimes in a diffusion-forcing style) [62, 64, 65] have been proposed. When applying our large-chunk design to autoregressive video generation, we format the input sequence with interleaved clean and noisy chunks (see Equation 12). This strategy achieves over $50 \%$ token utilization and integrates effectively with our large-chunk TTT implementation, by only changing a few lines to constrain fast-weights are only updated on clean frame chunks.

> 💡 **批注：LaCT 在视频扩散领域的创新点**
>
> 视频扩散的 autoregressive 方向面临两个核心挑战：
>
> 1. **长程时间一致性**：双向扩散 Transformer（如 Wan 2.1）天然处理整段视频，但无法生成无限长视频；自回归方法每次只生成几帧，但如何保持历史帧的长程一致性是难点。
>
> 2. **Token 利用率**：teacher forcing 格式（历史 clean + 当前 noisy）导致只有 noisy frames 有监督信号，clean frames 是无监督的条件输入，整体 token 利用率低。
>
> LaCT 的 interleaved noisy-clean 格式解决了这两个问题：
> - fast weight 存储历史 clean 帧的压缩记忆，解决长程一致性（无需全注意力覆盖历史）
> - 所有 noisy frames 都有监督信号，token 利用率 > 50%
>
> 这个贡献说明 LaCT 不仅是一个"替换 attention 的新架构"，更是一个使视频扩散走向真正无限长视频生成的路径。

---

# 7 Limitation

One limitation of our method is the absence of rotation invariance. Unlike softmax attention and linear attention, which remain invariant under uniform rotations of queries and keys (a property leveraged by relative positional encodings such as RoPE [42]), our SwiGLU and Linear Fast Weight components do not exhibit this property. The practical implications of this absence remain underexplored.

We conduct our experiments on three tasks. Although the tasks are diverse and cover different modalities, the effectiveness of our method would request of more tasks. For example, the novel-view synthesis task is essentially a 3D reconstruction with input pose information. The task of unposed reconstruction is more challenging and is not explored in this paper.

On the language modeling task, some key aspects are not explored due to computation limitation. These aspects include the reasoning capacity of our LaCT model and also the scalability regarding the parameter size. Previous papers showed that a main weakness of the state-based model (where LaCT belongs to) is its reasoning ability. However, the reasoning ability is only gained with certain amount of training compute thus it is beyond our budget.

Lastly, for the autoregressive video diffusion, it is hard to find a reliable and distinguishable metric to measure the model's scalability. It is in contrast to the language modeling with perplexity (i.e., log likelihood loss) and the novel-view synthesis with PSNR. We show the validation loss in our paper and it is a common choice in evaluating the scalability of video generation. This is a general problem for the video generation evaluation and is not specific to our paper.

> 💡 **批注：三大局限性的深度分析**
>
> 作者非常诚实地列举了三个限制，这对于正确理解 LaCT 的适用边界很重要：
>
> **1. 缺乏旋转不变性**：
> - RoPE 等相对位置编码之所以有效，是因为 softmax attention 和线性 attention 在 Q/K 旋转下保持不变，使得位置编码可以编码相对位置关系
> - SwiGLU Fast Weight 的非线性激活打破了这种旋转不变性，意味着 LaCT 对 token 表示的旋转变换不鲁棒
> - 实际影响：在语言模型中，模型可能对 token 的绝对坐标而非相对位置更敏感；实验中使用了 RoPE base=1M 来缓解这个问题，但理论保证更弱
>
> **2. 任务覆盖有限**：
> - 三个任务虽然多样，但都有其特殊性（NVS 需要 pose 信息、语言模型已知结构、视频有时序结构）
> - 无 pose 的 3D 重建、代码理解、数学推理等任务未验证
>
> **3. 语言模型推理能力未测试**：
> - 这是状态空间/记忆模型（SSM/TTT）领域的通病——已有研究表明 SSM 在推理密集型任务（如 ARC、数学）上弱于 Transformer
> - LaCT 的非线性大状态是否能缓解这个问题，需要更大计算预算的实验

---

# 8 Conclusion

We presented LaCT, a novel model architecture that integrates large-chunk test-time training for capturing long context with window attention for modeling local structure. We validated LaCT across three diverse tasks spanning different modalities—novel view synthesis, language modeling, and autoregressive video diffusion—and demonstrate its effectiveness by achieving superior or competitive performance when compared to state-of-the-art baselines. LaCT achieves high GPU efficiency even with native PyTorch implementation with dozens of lines of code and supports efficient scaling up of the state size and more flexible designs in test-time training models and optimizers. By open-sourcing the code and weights, we hope that LaCT can advocate future research explorations into more performant architectures for long-context modeling.

> 💡 **批注：结论与未来展望**
>
> LaCT 的核心贡献可以用一个公式概括：
>
> $$\text{LaCT} = \underbrace{\text{大 Chunk TTT}}_{\text{长程记忆 + 高效率}} + \underbrace{\text{Window Attention}}_{\text{局部结构}} + \underbrace{\text{SwiGLU Fast Weight}}_{\text{非线性容量}} + \underbrace{\text{Muon 优化器}}_{\text{稳定更新}}$$
>
> **开源的意义**：作者承诺开源代码和权重，这与"民主化研究"的理念一致——纯 PyTorch 实现 + 开源代码，使任何研究者都可以在此基础上快速探索新的 fast weight 架构、优化器或任务适配。
>
> **LaCT 作为研究范式的价值**：
> - **验证了"大 chunk"不是退步而是进步**：打破了"per-token 更新 = 更好的上下文学习"的传统假设
> - **开辟了 fast weight 扩展的新路径**：状态大小可以达到模型参数量的 40%，这个扩展空间以前因效率原因从未被探索
> - **跨模态统一框架**：同一套代码适配图像集合、文本序列、视频序列，降低了多模态长上下文建模的门槛
>
> **对 Parametric Memory 领域的启示**：LaCT 证明了参数化记忆（把历史信息存进神经网络参数而非 KV 缓存）在足够大的状态容量下，可以与显式记忆方法竞争甚至超越，这为"记忆即网络参数"的研究方向提供了强有力的实证支持。

---

## 🔖 Section 总结

### 关键相关工作速查
| 方法 | 与 LaCT 的关系 |
|-----|-------------|
| TTT [2] | LaCT 的基础框架，LaCT 提出大 chunk 范式 |
| Mamba [12] / GLA [13] / DeltaNet [15] | 线性 per-token 递推，LaCT 的主要竞争对手 |
| InfiniAttention [26] | 最接近的相关工作，但用线性 delta rule，表达能力弱 |
| Block-Recurrent Transformer [49] | 也用大 chunk 记忆，但用 attention 更新（复杂度高）|
| GAU [25] | SWA 与 TTT 共享 QKV 的灵感来源 |
| LVSM [29] / GS-LRM [32] | NVS baseline 和数据处理参考 |
| Wan 2.1 [43] | 视频扩散微调的基础模型 |
| Muon [8] | fast weight 更新的高效优化器 |

### 核心洞察
1. **LaCT 是"提高 TTT 天花板"而非"发明新机制"**：通过解决效率瓶颈，释放了 TTT 框架本身的潜力，使得此前因效率原因未被探索的大状态非线性 TTT 成为可能。
2. **旋转不变性是值得关注的开放问题**：SwiGLU Fast Weight 无法使用 RoPE 等相对位置编码的理论性质，这可能限制其在需要精确位置感知的任务上的表现。
3. **推理能力是 SSM/TTT 类方法的共同挑战**：LaCT 属于状态空间模型家族，已知这类方法在推理密集型任务上偏弱，这是未来需要解决的关键问题。
4. **视频生成的评估指标缺失是领域性问题**：不特定于 LaCT，但值得关注——缺乏可靠定量指标使得视频生成模型的扩展性难以系统评估。
5. **开源+纯 PyTorch 实现是方法普及的关键**：去除 CUDA kernel 依赖，配合开源代码，使 LaCT 有望成为长上下文序列建模的新基础工具。
