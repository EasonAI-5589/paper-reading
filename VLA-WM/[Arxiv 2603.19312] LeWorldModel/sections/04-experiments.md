[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览

实验覆盖两大板块：① Latent Planning 性能（4 个环境 + 消融）→ ② 物理理解评估（Probing + Violation-of-Expectation）。核心结论：LeWM 在 PushT 和 Reacher 上超越所有 baseline，在 OGBench-Cube 上略低于 DINO-WM，规划速度快 48 倍。

---

## 4.1 实验设置

### 环境

| 环境 | 类型 | 描述 | 数据量 |
|------|------|------|--------|
| **Two-Room** | 2D 导航 | 两房间穿门导航 | 10k episodes, ~92 steps/ep |
| **PushT** | 2D 操作 | 推 T 形方块到目标位置 | 20k expert episodes, ~196 steps/ep |
| **OGBench-Cube** | 3D 操作 | 机械臂抓取放置方块 | 10k episodes, 200 steps/ep |
| **Reacher** | 2D 运动 | 两关节臂到达目标位置 | 10k episodes, 200 steps/ep |

### Baselines

| 方法 | 类型 | 说明 |
|------|------|------|
| **PLDM** | 端到端 JEPA | VICReg 7 项损失，最接近的对比 |
| **DINO-WM** | 冻结编码器 JEPA | DINOv2 特征，可选加本体感知 |
| **GCBC** | 行为克隆 | 目标条件的监督学习 |
| **GCIQL / GCIVL** | 离线 RL | 目标条件的 Q-learning / Value Learning |
| **Random** | — | 随机动作 |

> 💡 **所有方法超参数固定不变**（跨环境不调参），这是一个严格的评估设置。LeWM 的优势正是在于只有 1 个超参数，固定后到处都 work。

---

## 4.2 Planning Performance（Figure 6）

| 环境 | LeWM | PLDM | DINO-WM | DINO-WM+prop | GCBC | GCIQL | GCIVL |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Two-Room** | 87% | 97% | 100% | 100% | 100% | 100% | 100% |
| **Reacher** | **86%** | 78% | 79% | — | — | — | — |
| **PushT** | **96%** | 78% | 74% | 92% | 75% | 20% | 33% |
| **OGBench-Cube** | 74% | 65% | **86%** | — | 84% | 64% | 56% |

> 💡 **批读 Figure 6**:
>
> **LeWM 胜出的场景（PushT, Reacher）**:
> - PushT: 96% vs PLDM 78%（+18%），vs DINO-WM 74%（+22%）
> - 甚至超过 DINO-WM+proprioception（92%）——仅用像素就超过了"像素+本体感知"！
> - 说明 LeWM 的端到端训练能学到比预训练编码器更好的 task-relevant 表征
>
> **LeWM 稍弱的场景（Two-Room, OGBench-Cube）**:
> - Two-Room: 87% vs 其他大多 100%。作者分析是因为 SIGReg 在低复杂度环境中强制高维高斯分布，与环境的低内在维度不匹配
> - OGBench-Cube: 74% vs DINO-WM 86%。3D 视觉复杂度更高，端到端训练的 tiny encoder 信息容量有限
>
> **总体**: 在更 challenging 的任务上 LeWM 更强，在简单/高视觉复杂度任务上略弱。

### 规划速度（Figure 3）

| 方法 | 规划时间 | 加速比 |
|------|:---:|:---:|
| **LeWM** | **0.98s** | **1×** |
| DINO-WM | 47s | 0.02× |

> 💡 **48× 加速的来源**: LeWM 编码器输出 ~200× 更少的 tokens（1 个 [CLS] vs DINO-WM 的 patch tokens），导致 predictor rollout 极快。

### 固定计算量对比（Figure 3 右两图）

在同等 FLOPs 下：
- PushT: LeWM **90%** vs DINO-WM **13%**
- OGBench-Cube: LeWM **74%** vs DINO-WM **48%**

> 💡 **公平对比的关键**: 给 DINO-WM 和 LeWM 同样的计算预算（FLOPs），LeWM 大幅胜出。DINO-WM 的高性能很大程度上靠"暴力搜索"（更多 CEM 迭代），而非更好的世界模型。

---

## 4.3 消融实验（Appendix G）

### SIGReg 超参数鲁棒性

| 消融 | 结论 |
|------|------|
| 投影数 M（64~1024） | 几乎无影响 |
| 积分节点数（4~32） | 几乎无影响 |
| $\lambda$（0.01~0.5） | $\lambda \in [0.01, 0.2]$ 稳定在 80%+，$\lambda=0.5$ 时退化 |
| Embedding 维度（8~384） | <184 时退化，之后饱和 |
| Encoder 架构（ViT vs ResNet-18） | 两者竞争力接近（96% vs 94%） |
| Predictor 大小（Tiny/Small/Base） | ViT-S 最优（96%），Tiny 和 Base 略差 |
| Predictor Dropout（0~0.5） | 0.1 最优（96%），0.0 仅 78%——dropout 是关键！ |
| 加 Decoder loss | 不改善甚至略降（96% → 86%） |

> 💡 **消融的核心发现**:
>
> 1. **λ 是真正唯一需要调的超参数**，而且搜索空间只有一维
> 2. **Predictor dropout 很关键**: 0% → 78%, 10% → 96%，没有 dropout 性能大幅退化
> 3. **不需要重建损失**: 加了 decoder loss 反而变差，说明 JEPA 训练目标本身就足够
> 4. **对架构不敏感**: ViT 和 ResNet 都 work

### 训练稳定性（Figures 18-19）

- **LeWM**: 两项损失平滑单调收敛。预测损失稳步下降，SIGReg 初期快速下降后稳定
- **PLDM**: 七项损失噪声大、非单调。多项损失之间梯度竞争

> 💡 损失数量少 → 训练信号清晰 → 优化更稳定。这是"简单即正义"的最好例证。

---

## 4.4 物理理解评估

### Latent Probing（Table 1 — PushT）

| 物理量 | LeWM (Linear r) | PLDM (Linear r) | DINO-WM (Linear r) |
|--------|:---:|:---:|:---:|
| Agent Location | **0.974** | 0.955 | 0.977 |
| Block Location | **0.986** | 0.938 | **0.997** |
| Block Angle | **0.902** | 0.745 | **0.979** |

> 💡 LeWM 在所有物理量上显著优于 PLDM，与 DINO-WM 接近。DINO-WM 在某些量上更好可能是因为 DINOv2 在 ~124M 图像上预训练，视觉先验更强。但 LeWM 仅用 15M 参数 + 任务数据就能达到接近水平，性价比极高。

### Latent 解码可视化（Figure 8）

训练过程中的 decoder（**不参与训练！**）可以从 192 维 [CLS] token 重建出完整场景，说明 latent space 保留了充足的视觉信息。

### Temporal Straightening（Figure 17，涌现现象）

> LeWM achieves higher temporal straightness than PLDM, despite PLDM employing a dedicated temporal smoothness regularization term.

> 💡 **有趣的涌现**: LeWM 没有任何鼓励"时间平滑"的正则化项，但 latent 轨迹自然变得"直"了。PLDM 专门加了 temporal smoothness loss，反而没 LeWM 直。这暗示 SIGReg 的各向同性高斯约束隐式引导了时间结构。

### Violation-of-Expectation（Figure 10）

三种轨迹：正常 / 视觉扰动（颜色变化） / 物理扰动（传送）

| 环境 | 物理扰动检测 | 视觉扰动检测 |
|------|:---:|:---:|
| Two-Room | ✅ 显著 | ✅ |
| PushT | ✅ 显著 | 弱 |
| OGBench-Cube | ✅ 显著 | 不显著 |

> 💡 LeWM 对**物理违规**（传送）的检测远比**视觉变化**（颜色）敏感——这正是物理理解的表现。模型学到了"物体不应该瞬移"的物理先验。

---

## 💡 实验总结：与 Fast-WAM 的呼应

谢赛宁建议两篇一起看的原因：

| | Fast-WAM | LeWM |
|---|---|---|
| **核心问题** | WAM 的增益来自训练还是推理？ | JEPA 如何稳定端到端训练？ |
| **关键发现** | 训练时视频建模 > 推理时未来想象 | 两项损失 > 七项损失（简单即正义） |
| **共同指向** | 好的训练目标比复杂推理更重要 | 好的正则化比复杂 heuristics 更重要 |
| **技术路线** | Diffusion-based WAM（6B 参数） | JEPA（15M 参数） |
| **实际影响** | 190ms 实时推理 | 48× 规划加速 |

**共同启示**: 具身智能的世界模型不需要暴力堆大模型，关键在于**训练目标的设计**和**表征学习的质量**。
