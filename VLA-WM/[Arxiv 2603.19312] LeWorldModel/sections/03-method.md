[← 返回 README](../README.md)

# 3. Method: LeWorldModel

## 📌 预览

方法分两大部分：① 学习 latent 世界模型（离线训练）→ ② Latent 规划（推理时 MPC）。训练部分的核心是两项损失：MSE 预测 + SIGReg 防坍缩。

---

## 3.1 Learning the Latent World Model

### 离线数据集

> We consider a fully offline and reward-free setting. LeWorldModel is trained solely from unannotated trajectories of observations and actions, without access to reward signals or task specifications.

训练数据：长度 T 的轨迹，包含原始像素观测 $o_{1:T}$ 和动作 $a_{1:T}$。

> 💡 **完全离线 + 无奖励**: 数据可以来自任何行为策略（专家或探索性的），只要覆盖环境动力学。这是最通用的设置。

### 模型架构

两个组件：

$$\text{Encoder:} \quad z_t = \text{enc}_\theta(o_t)$$
$$\text{Predictor:} \quad \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t)$$

| 组件 | 架构 | 参数 |
|------|------|------|
| **Encoder** | ViT-Tiny | ~5M params, patch=14, 12 layers, 3 heads, hidden=192 |
| **Predictor** | ViT-S (Transformer) | ~10M params, 6 layers, 16 heads, 10% dropout |
| **总计** | — | **~15M params** |

**Encoder 细节**:
- 输出 = [CLS] token embedding + 1-layer MLP projector（带 Batch Normalization）
- BN 后的 projector 是必要的，因为 ViT 最后一层的 LayerNorm 会阻止 SIGReg 正常工作

**Predictor 细节**:
- 动作通过 **Adaptive Layer Normalization (AdaLN)** 注入每一层
- AdaLN 参数初始化为零 → 训练初期动作条件的影响渐进式增加（稳定训练）
- 输入 N 帧历史表征，因果掩码防止看到未来
- Predictor 后也有 projector

> 💡 **15M 参数的设计哲学**: 用 ViT-Tiny 编码器（不是 ViT-Base 或 ViT-Large），目标就是"小而美"。对比 DINO-WM 需要冻结一个 ~300M 的 DINOv2，LeWM 的整个模型比 DINO-WM 的编码器还小 20 倍。

---

### 训练目标

#### 损失 1: 预测损失（Next-Embedding Prediction）

$$\mathcal{L}_{\text{pred}} \triangleq \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \quad \hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t) \tag{1}$$

> 💡 标准的 teacher-forcing MSE 损失。通过这个损失，encoder 被激励学习一个**对 predictor 来说可预测的表征**。但单靠这个损失 → 表征坍缩（把一切编码成常量）。

#### 损失 2: SIGReg 防坍缩正则化

$$\text{SIGReg}(Z) \triangleq \frac{1}{M} \sum_{m=1}^{M} T(h^{(m)}) \tag{2}$$

其中 $Z \in \mathbb{R}^{N \times B \times d}$ 是 latent embeddings，$h^{(m)} = Zu^{(m)}$ 是沿随机方向 $u^{(m)}$ 的投影，$T(\cdot)$ 是 Epps-Pulley 正态性检验统计量。

**原理**（Cramér-Wold 定理）：
- 如果所有一维投影都匹配标准正态 → 联合分布匹配各向同性高斯
- SIGReg → 0 当且仅当 $P_Z \to \mathcal{N}(0, I)$

> 💡 **SIGReg 的优雅之处**:
>
> 1. **数学上有保证**: Cramér-Wold 定理给出了严格的防坍缩保证，不是启发式
> 2. **高维友好**: 不直接在高维空间做正态性检验（不可行），而是投影到 M 个随机一维方向分别检验
> 3. **实现简单**: 只需要随机投影 + 一维统计检验 + 求平均

#### 总损失

$$\mathcal{L}_{\text{LeWM}} \triangleq \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z) \tag{3}$$

| 超参数 | 默认值 | 敏感度 |
|--------|--------|--------|
| $\lambda$（SIGReg 权重） | 0.1 | $\lambda \in [0.01, 0.2]$ 时性能稳定（>80%） |
| $M$（投影数） | 1024 | 几乎无影响 |

> 💡 **对比 PLDM 的 7 项损失**:
>
> ```
> PLDM: L_pred + α·L_var + β·L_cov + γ·L_time-sim + ζ·L_time-var + ν·L_time-cov + μ·L_IDM
> LeWM: L_pred + λ·SIGReg
> ```
>
> PLDM 的 6 个超参数需要联合搜索（O(n⁶)），LeWM 的 1 个超参数可以二分搜索（O(log n)）。这是从实用角度最重要的贡献之一。

### 伪代码（Algorithm 1）

```python
def LeWorldModel(obs, actions, lambd=0.1):
    emb = encoder(obs)           # (B, T, D)
    next_emb = predictor(emb, actions)  # (B, T, D)
    
    # 预测损失
    pred_loss = F.mse_loss(emb[:, 1:], next_emb[:, :-1])
    
    # 逐步 SIGReg（防坍缩）
    sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))
    
    return pred_loss + lambd * sigreg_loss
```

> 💡 **全部训练逻辑在 10 行代码内**。不需要 stop-gradient、EMA、目标网络、动量更新——所有参数联合端到端优化。

---

## 3.2 Latent Planning

### 推理流程（Figure 4）

```
1. 编码初始观测和目标观测: z₁ = enc(o₁), z_g = enc(o_g)
2. 初始化随机动作序列
3. Predictor 在 latent space 展开到 horizon H
4. 终端代价: C(ẑ_H) = ‖ẑ_H - z_g‖²
5. CEM (Cross-Entropy Method) 迭代优化动作序列
6. MPC: 执行前 K 个动作 → 从新观测重新规划
```

### CEM 参数

| 参数 | 值 |
|------|-----|
| 候选动作序列数 | 300 |
| 优化迭代步数 | 30 (PushT) / 10 (其他) |
| Elite 数量 | 30 |
| 规划 horizon | 5 步（= 25 环境步，因 frame-skip=5） |
| MPC 策略 | 执行全部 5 步后重新规划 |

> 💡 **为什么 LeWM 规划快 48×**:
>
> - LeWM 编码器只产生 1 个 [CLS] token（192 维）
> - DINO-WM 用 DINOv2 patch embeddings → ~200× 更多 tokens
> - token 数少 → predictor rollout 更快 → CEM 的 300×30 次 rollout 全部在 1 秒内完成
