# ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy |
| **作者** | Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, Dongbin Zhao |
| **发布时间** | 2025-02 |
| **arXiv** | [2502.05450](https://arxiv.org/abs/2502.05450) |
| **GitHub** | https://github.com/cccedric/conrft |
| **项目主页** | https://cccedric.github.io/conrft/ |

---

## Section 1: Motivation & Problem Definition

### 1.1 研究问题定义

#### 核心任务
> 如何高效地用强化学习微调 VLA 模型，使其从有限且不一致的人类示教中学习，同时保证安全探索？

#### 问题范畴
- VLA 模型的 RL 微调
- 样本效率优化
- 安全探索机制

#### 现有方法的问题

##### 问题1：纯 BC 微调的局限
- **现有做法**: 使用 Behavior Cloning 微调 VLA
- **局限**: 无法超越示教质量，需要大量高质量数据
- **理想状态**: 从少量示教中学习并持续改进

##### 问题2：RL 微调的挑战
- **现有做法**: 直接用 RL 微调 VLA
- **局限**:
  - 探索不安全（机器人可能损坏）
  - 样本效率低
  - 训练不稳定
- **理想状态**: 安全、高效、稳定的 RL 微调

### 1.2 本文方法与核心创新

#### 总体方案
ConRFT 使用两阶段训练：离线预训练 + 在线人机协作微调。

#### 关键创新点

| 创新 | 说明 | 解决的问题 |
|------|------|------------|
| **Consistency Policy** | 用 Consistency Model 替代 Diffusion | 单步生成动作，加速推理 |
| **Cal-ConRFT** | BC + Q-learning 离线预训练 | 从有限示教中提取策略 |
| **HIL-ConRFT** | 人机协作在线微调 | 安全探索 + 样本效率 |

---

## Section 2: Related Work

| 工作 | 核心方法 | 优势 | 局限 |
|------|----------|------|------|
| Diffusion Policy | DDPM 生成动作 | ✓ 多模态动作 | ✗ 推理慢（多步采样） |
| Consistency Policy | 单步生成 | ✓ 快速推理 | ✗ 未与 RL 结合 |
| SERL | RL + 人类干预 | ✓ 安全探索 | ✗ 不针对 VLA |
| OpenVLA | 开源 VLA | ✓ 通用 | ✗ 只支持 BC |

---

## Section 3: Method

### 3.0 Preliminary: Consistency Model

Consistency Model 是 Diffusion Model 的加速版本：

$$
f_\theta(x_t, t) = c_{skip}(t) \cdot x_t + c_{out}(t) \cdot F_\theta(x_t, t)
$$

**核心思想**: 学习直接映射 $x_t \rightarrow x_0$，无需迭代采样。

**关键约束** (Self-Consistency):
$$
f_\theta(x_t, t) = f_\theta(x_{t'}, t'), \quad \forall t, t' \in [0, T]
$$

### 3.1 Consistency Policy for VLA

#### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Consistency Policy                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Observation → Octo Encoder → obs_enc                       │
│                                                             │
│  Noise → Sample σ from Karras Schedule                      │
│       ↓                                                     │
│  x_t = x_0 + σ · ε   (加噪)                                  │
│       ↓                                                     │
│  ┌───────────────────────────────────────────┐              │
│  │  Base Network (MLP)                       │              │
│  │  input: concat(x_t, obs_enc, t_embed)    │              │
│  │  output: denoised action                  │              │
│  └───────────────────────────────────────────┘              │
│       ↓                                                     │
│  action = c_skip · x_t + c_out · network_output             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Karras Noise Schedule

$$
\sigma_i = \left( \sigma_{max}^{1/\rho} + \frac{i}{n-1}(\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho}) \right)^\rho
$$

### 3.2 Cal-ConRFT (Offline Stage)

#### 核心思想
> 结合 BC 和 Q-learning，从有限示教中学习策略同时稳定价值估计。

#### 损失函数

$$
\mathcal{L}_{offline} = \mathcal{L}_{consistency} + \lambda \cdot \mathcal{L}_{Q}
$$

其中：
- $\mathcal{L}_{consistency}$: Consistency Model 重建损失
- $\mathcal{L}_{Q}$: Q-learning 损失（用于策略改进）

#### 训练流程

```
1. 收集少量人类示教 (10-20 demos)
2. 训练 Reward Classifier（用于在线阶段）
3. Cal-ConRFT 离线预训练:
   - BC: 学习模仿示教
   - Q-learning: 学习价值函数，为改进做准备
```

### 3.3 HIL-ConRFT (Online Stage)

#### 核心思想
> 人机协作在线微调：人类在危险时刻干预，策略从干预中学习。

#### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                   HIL-ConRFT Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌─────────────────────────────┐  │
│  │   Actor Thread  │←────→│      Robot Environment      │  │
│  │  (Policy执行)    │      │   (Franka + RealSense)      │  │
│  └────────┬────────┘      └─────────────────────────────┘  │
│           │                                                 │
│           │ Transitions                                     │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │  Replay Buffer  │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │  Learner Thread │                                        │
│  │  (Policy更新)    │                                        │
│  └─────────────────┘                                        │
│                                                             │
│  Human Intervention: SpaceMouse 接管控制                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 人类干预机制

| 情况 | 人类操作 | 系统响应 |
|------|----------|----------|
| 策略正常执行 | 无操作 | 继续执行 |
| 即将碰撞/失败 | 按住 SpaceMouse | 接管控制 |
| 完成干预 | 松开 SpaceMouse | 策略恢复执行 |

**干预数据处理**: 人类干预的轨迹被标记为高质量示教，加入 Replay Buffer。

### 3.4 Reward Classifier

用于在线阶段判断任务是否成功：

```python
# 训练 Reward Classifier
1. 收集成功/失败图像 (hold spacebar for positive)
2. 训练二分类器
3. 在线阶段用于计算 reward
```

---

## Section 4: Experiments

### 4.1 实验设置

| 配置 | 取值 |
|------|------|
| **机器人** | Franka Emika Panda |
| **相机** | Intel RealSense |
| **VLA 基线** | Octo |
| **任务数** | 8 个真实任务 |
| **示教数量** | 10-20 个/任务 |

### 4.2 主要结果

| 方法 | 成功率 | Episode 长度 |
|------|--------|-------------|
| BC (Octo) | 39.4% | baseline |
| Diffusion Policy | 65.2% | - |
| **ConRFT** | **96.3%** | 1.9x 更短 |

**关键发现**:
- 相比 BC，成功率提升 **144%**
- 只需 **45-90 分钟** 在线微调
- Episode 长度缩短 **1.9x**（执行更高效）

### 4.3 消融实验

| 配置 | 成功率 | 结论 |
|------|--------|------|
| 完整 ConRFT | 96.3% | baseline |
| 无 Cal-ConRFT | 78.1% | 离线预训练重要 |
| 无 Q-learning | 82.4% | Q-learning 提升策略质量 |
| 无人类干预 | 71.2% | HIL 对安全和效率关键 |

---

## Section 5: Takeaways

### 核心贡献
1. **首个 Consistency Policy + RL 的 VLA 微调方法**
2. **两阶段训练**: 离线提取 + 在线改进
3. **人机协作**: 安全探索 + 高效学习
4. **实际验证**: 8 个真实任务，96.3% 成功率

### 与其他工作的联系
- **Consistency Model**: 用于加速动作生成
- **SERL**: 借鉴人类干预机制
- **Octo**: 作为 VLA 基座模型

### 对实验室工作的启发
1. **Consistency Policy 比 Diffusion Policy 更适合 RL**（单步采样）
2. **少量示教 + RL 微调 > 大量示教 BC**
3. **人机协作是安全 RL 的关键**

---

## 代码要点

### 核心文件结构
```
conrft/
├── serl_launcher/
│   ├── agents/
│   │   └── conrft_agent.py    # ConRFT Agent
│   ├── networks/
│   │   └── consistency_policy.py  # Consistency Policy
│   └── envs/
│       └── franka_env.py      # Franka 环境
└── examples/
    └── run_conrft.sh          # 训练脚本
```

### 训练流程
```bash
# Step 1: 训练 Reward Classifier
python train_reward_classifier.py

# Step 2: 录制示教
python record_demos_octo.py --num_demos 20

# Step 3: Cal-ConRFT 离线预训练
bash run_learner_conrft_pretrain.sh

# Step 4: HIL-ConRFT 在线微调
bash run_actor_conrft.sh &
bash run_learner_conrft.sh
```

---

## BibTeX

```bibtex
@article{chen2025conrft,
  title={ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy},
  author={Chen, Yuhui and Tian, Shuai and Liu, Shugao and Zhou, Yingting and Li, Haoran and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2502.05450},
  year={2025}
}
```
