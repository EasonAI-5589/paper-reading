[← 返回 README](../README.md)

# 5. Experiments

## 实验设置

- **平台**：DROID（Franka Panda + Robotiq Gripper，1个腕部相机 + 2个第三人称相机）
- **数据集**：DROID 95k 轨迹，564 场景（训练）；2% holdout 作验证集
- **对比基线**：WPE（单视角）、IRASim（单视角）、Ctrl-World-Single-View（消融）
- **评估方式**：256 条 10 秒随机采样视频，每轮接受 15步动作块，自回归预测 10 轮

---

## 5.1 世界模型质量（Table 1）

### 量化结果

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ |
|------|-------|-------|---------|------|------|
| WPE-Single | 20.33 | 0.772 | 0.131 | 25.50 | 156.4 |
| WPE-Multi | 21.17 | 0.774 | 0.117 | 26.46 | 147.1 |
| IRASim-Single | 21.36 | 0.793 | 0.110 | 23.47 | 138.1 |
| IRASim-Multi | 20.21 | 0.828 | 0.091 | 25.00 | 165.4 |
| Ctrl-World-Single | 21.27 | - | - | - | 127.5 |
| **Ctrl-World (full)** | **23.56** | **0.828** | **0.091** | **25.00** | **97.4** |

> 💡 关键数字：Ctrl-World FVD = 97.4，比 IRASim-Single 138.1 低 **29%**

### 关键发现

1. **IRASim-Multiview 的 FVD 反而更差（165.4 > 138.1）**：说明简单地把多视角拼在一起不行，需要 Ctrl-World 这样专门设计的联合预测机制
2. **Ctrl-World 在 PSNR 上也领先最多**（23.56 vs 21.36），说明像素级重建质量也更好

---

## 5.2 消融实验（Table 2）

### 各组件的贡献

| 组件 | Third FVD | Wrist FVD |
|------|----------|----------|
| 完整 Ctrl-World | **97.4** | **127.1** |
| 去掉 memory | 105.5 | 133.1 |
| 去掉 frame-level cond | 122.7 | 179.1 |
| 去掉 joint pred | - | 158.1 |

**结论：**
- 帧级动作条件对腕部相机最重要（FVD 127 → 179，+41%）
- Memory 对长时一致性有稳定贡献
- 多视角联合预测对腕部相机有效（FVD 127 → 158）

---

## 5.3 策略评估（Policy Evaluation）

### 设置

在自建 DROID 平台上（**新相机位置，非训练分布**），测试三个策略：π0、π0-FAST、π0.5

7 类任务：Pick-and-Place、Towel-Folding、Drawer、Wipe-Table、Close-Laptop、Pull-tissue、Stack

### 结论

**指令跟随相关性：slope = 0.87**（世界模型评估 vs 真实环境）

> 💡 这意味着：Ctrl-World 能**准确排列**不同策略的性能，可以替代部分真实 rollout

**局限：**
- 低级执行成功率（success rate）的估计相关性略低（slope = 0.81）
- 精细物理交互（碰撞、旋转等）建模不精确
- 策略在真实环境重试失败，但世界模型有时无法模拟这种行为

---

## 5.4 策略改进（Policy Improvement）

### 任务类型

4 类下游任务（unseen objects + novel instructions）：
- **Spatial**："pick the object in the top left corner"
- **Shape**："pick the smaller red block"
- **Towel-Direction**："fold the towel from right to left"
- **New Object**："pick the glove and place in box"

### 结果（Figure 9）

| 任务类型 | Base Policy | Finetuned |
|---------|------------|-----------|
| Spatial | 0.44 | 0.57 |
| Shape | 0.25 | 0.75 |
| Towel-Dir | 0.29 | 0.80 |
| New Object | 0.39 | 0.91 |
| **Average** | **0.38** | **0.83** |

**提升 +44.7%**

### Pipeline 细节

- 每任务生成 400 条 synthetic rollout
- 保留 25-50 条成功轨迹（人工打分）
- finetune π0.5 2000 步
- 多样性：LLM 指令改写 + arm 初始位置随机重置

---

## 💡 批读注解

### 最值得关注的实验：Policy Improvement

这个结果是整篇论文最 impressive 的：

- Shape 理解：25% → 75%（+200% 相对提升！）
- New Object：39% → 91%（+133%）

为什么效果这么好？关键是**合成数据对准了分布**：
- 基础 π0.5 训练数据里没有这些特定指令/物体
- World model 在想象空间里"模拟"了这些场景的成功轨迹
- SFT 后 policy 学会了这些新 skill

### 一个重要局限

Ctrl-World 无法改善**低级执行成功率**（比如之前见过的任务的精度）。原因：
- 世界模型本身的物理建模精度有限
- 它能改善"知道该做什么"（instruction following），但改善不了"能不能执行好"（execution quality）

这正好说明 Ctrl-World 适合的 scenario：**让 policy 学会新指令/新物体**，而不是提升已有技能的执行精度。

### 跟我们的项目关系

我们在 LIBERO 上做的 evaluation（PSNR/SSIM/FVD 指标）对应的就是 Section 5.1 的框架。等 Ctrl-World 训完后，正式跑全量 evaluation 的结果可以直接对标这个 Table 1 的格式。
