[← 返回 README](../README.md)

# 4 Experimental Setup

## 📌 预览
这一节回答两个问题：A2C2 在什么任务上测，correction head 具体做成什么样。Kinetix 用来测高动态与延迟鲁棒性，LIBERO Spatial 用来测标准 manipulation 和 long-horizon 退化；两者一起构成“低维高动态”到“多模态 VLA”两端的验证。

---

## 4.1 Benchmark and Datasets

We use the two simulation environments, Kinetix and LIBERO Spatial, for the experiments. Kinetix is first used for evaluating the performance under highly dynamic manipulation and locomotion tasks. Secondly, we used the LIBERO Spatial benchmark to evaluate the performance as a standard benchmark of robot manipulation. Especially, because Shukor et al. (2025) reports that long-horizon significantly degrades performance in LIBERO Spatial, making the task a natural choice for evaluating robustness under long horizons.

### 4.1.1 Kinetix

We used Kinetix, which provides demonstrations across 12 highly dynamic tasks (see Appendix A.1). It includes environments ranging from locomotion and grasping to game-like settings. Importantly for our setting, Kinetix contains highly dynamic environments where delayed or inconsistent action generation quickly leads to failure. This makes it a natural testbed for studying the limitations of action chunking and for benchmarking inference-time algorithms such as RTC, which aim to preserve responsiveness and continuity under latency.

Unlike quasi-static benchmarks, Kinetix environments employ torque- and force-based actuation, making asynchronous inference crucial. Kinetix consists of 12 tasks without language input. One million steps of data were collected by using an expert model. Following RTC experiments, we first train expert policies using RPO (Rahman & Xue, 2022) and a binary success reward. For each environment, a 1-million transition dataset is generated with the expert policy.

> 💡 **为什么先上 Kinetix**: 如果任务本身是 quasi-static，很多 stale action 问题不会立刻暴露。Kinetix 的价值就在于它对时序错位极其敏感。

### 4.1.2 LIBERO

LIBERO is a benchmark suite designed to study lifelong robot learning with a focus on knowledge transfer across tasks (Liu et al., 2023). They offer several task suites and datasets. In this work, we specifically use the LIBERO Spatial dataset, which emphasizes spatial reasoning in manipulation tasks as a widely used benchmark for robot manipulation.

For benchmarking 3D robot manipulation, we used the LIBERO Spatial benchmark, which provides 432 episodes and 52,970 frames across 10 tasks. The dataset consists of multimodal input, including top and wrist RGB images (`256 × 256`), an 8-dimensional state, and language instructions.

> 💡 **为什么还要测 LIBERO**: 这能证明 A2C2 不是只适合无语言、低维状态的 toy setting，而是也能接到真正的 VLA 输入形态上。

---

## 4.2 Model Training

In Kinetix, we used a flow-matching policy as the base model, following prior work on RTC (Black et al., 2025). The correction head network is a 3-layer multilayer perceptron (MLP). The input layer receives the concatenation of the state vector (2722-dim), the base action (6-dim), and the 2-dimensional sinusoidal positional feature. We did not use language instructions or latent representations from base policies, as the model was trained and evaluated separately for each task. Hidden layers have 512 units each with ReLU activation (Nair & Hinton, 2010) and layer normalization (Ba et al., 2016). The output layer produces a 6-dimensional residual vector, which is added elementwise to the base action. The total parameter count is 0.31M. Figure 3 shows the implementation detail for the Kinetix experiment.

![Figure 3](../images/afe58ccd4c8fde9892c30b7a451cc9b14b16d0fb9da42d447bf41930fe4789ac.jpg)
*Figure 3: Kinetix 中的 correction head 是一个很小的 MLP，输入当前状态、base action 和 chunk 位置编码，输出 residual action。*

> 💡 **Figure 3 批读**:
> - Kinetix 版本里，计算 $\Delta a_{t+k}$ 的结构就是一个很小的 $3$ 层 MLP。
> - 输入端直接拼接三类信息：当前状态 $s_{t+k}$、原始动作 $a_{t+k}^{\mathrm{base}}$ 和时间位置特征 $\tau_k$；这里没有额外的视觉、语言或 latent 融合模块。
> - 中间层只是标准的全连接变换，因此这个版本本质上是在测试：仅靠当前状态与 base action，是否已经足以恢复 step-level reactivity。
> - 输出端给出一个与 action 维度相同的残差向量 $\Delta a_{t+k}$，再与 $a_{t+k}^{\mathrm{base}}$ 逐元素相加得到执行动作。
> - 这张图强调的是最小实现：作者有意把 correction head 压到 $0.31\mathrm{M}$ 参数，说明 A2C2 的核心并不依赖复杂结构，而在于“每步做残差修正”这一机制本身。
> - 这里作者刻意用最小设计验证观点。如果只靠 `state + base action + time feature` 就能显著提分，说明问题确实出在 step-level reactivity，而不是必须上更复杂的 world model。

For LIBERO Spatial, we adopted SmolVLA (Shukor et al., 2025) (450M parameters) as the base, since it provides competitive performance among VLA models. The correction head consists of a transformer encoder and a lightweight MLP. Visual observations (top and wrist cameras) are encoded into 512-dimensional tokens using a ResNet-18 (He et al., 2016) pretrained on ImageNet (Deng et al., 2009). Language instructions are embedded by the smolVLM encoder provided in the base policy. The base action, latent features of the base policy, and the sinusoidal time embedding are also projected into 512-dim tokens. All tokens are concatenated and processed by a 6-layer transformer encoder. The pooled embedding, along with the base action and state vector, is passed through a 3-layer MLP (hidden size 512) to predict the residual action. The number of total parameters is 32M. Figure 4 shows the implementation detail for the LIBERO experiment.

![Figure 4](../images/9921238921874492fbd5cc718d66ab81b5d87fd13e2df094564369fa6fb63fe9.jpg)
*Figure 4: LIBERO 中的 correction head 更复杂，需要消费视觉 token、语言特征、state、base action 和时间嵌入。*

> 💡 **Figure 4 批读**:
> - LIBERO 版本里，计算 $\Delta a_{t+k}$ 的结构不再是单纯的 MLP，而是“transformer encoder $+$ lightweight MLP”的两段式结构。
> - 输入端先把视觉观测、语言特征、base latent、base action 和时间特征都投影成 token，再交给 transformer encoder 做多模态融合；这一步负责回答“当前视觉局面和任务语义下，这个 base action 是否需要改”。
> - transformer 的 pooled embedding 随后与 $a_{t+k}^{\mathrm{base}}$ 和状态向量一起送入一个 $3$ 层 MLP，最终输出残差 $\Delta a_{t+k}$。
> - 因此，Kinetix 版本更像是状态驱动的局部修正器，而 LIBERO 版本已经是多模态条件化的 step-level correction head。

We also release the source code for both Kinetix and LIBERO experiments. See Appendix A.3 for the details.

## 🔖 Section 总结

### 核心洞察
1. **Benchmark 设计有明确分工**: Kinetix 负责放大延迟和动态性问题，LIBERO Spatial 负责验证多模态 VLA 场景下是否仍然成立。
2. **两套 correction head 是按输入复杂度扩展的**: Kinetix 用最小 MLP 验证思想，LIBERO 再把视觉、语言和 latent 一起接进来。
3. **作者始终坚持“小头补反馈”**: 不管在哪个 benchmark 上，correction head 都被刻意控制在显著小于 base model 的规模。
