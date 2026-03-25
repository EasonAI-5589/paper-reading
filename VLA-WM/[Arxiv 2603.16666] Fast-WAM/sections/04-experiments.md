[← 返回 README](../README.md)

# 4. Experiments

## 📌 预览

实验围绕三个层次展开：① 与 SOTA 方法的整体对比 → ② 控制变量实验（全文核心）→ ③ 真实世界性能 + 效率分析。每个层次都在回答同一个问题：训练时视频建模 vs 推理时未来想象，哪个更重要？

---

## 4.1 Implementation Details

| 参数 | 值 |
|------|-----|
| Video 骨干 | Wan2.2-5B（预训练 DiT + T5 text encoder + Video VAE） |
| Action Expert | 与 Video DiT 同架构，隐层 $d_a=1024$，约 1B 参数 |
| 总模型 | ~6B 参数 |
| 动作 horizon | h = 32 |
| 视频帧数 | 9 帧/chunk（4× 时间下采样） |
| 多相机处理 | 多相机图像拼接为单张图像，再输入 VAE |
| 噪声调度 | logit-normal（训练 + 推理，与 Wan2.2 一致） |
| 推理去噪步数 | 10 步 |
| CFG scale | 1.0 |
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01, cosine annealing |
| 精度 | Mixed precision + gradient clip 1.0 |
| 测试 GPU | NVIDIA RTX 5090D V2 32GB |
| Fast-WAM-IDM 噪声增强 | $p=0.5$ 对 GT 视频 tokens 加噪 |

---

## 4.2 Experiment Setup

### Benchmark 1: LIBERO

| 设置 | 值 |
|------|-----|
| 子集 | Spatial, Object, Goal, Long（4 个） |
| 每子集数据 | 500 demos, 10 tasks |
| 训练步数 | 20k steps |
| 评估 | 40 tasks × 50 seeds = 2000 trials |

### Benchmark 2: RoboTwin 2.0

| 设置 | 值 |
|------|-----|
| 任务数 | 50+ 双臂操作任务（bimanual） |
| 训练数据 | 2,500 clean demos + 25,000 randomized demos |
| 训练步数 | 30k steps |
| 评估 | 100 trials/task, Clean + Randomized |

### Real-World: 毛巾折叠

| 设置 | 值 |
|------|-----|
| 平台 | Galaxea R1 Lite |
| 训练数据 | 60 小时遥操作 |
| 训练步数 | 30k steps |
| 评估指标 | **成功率 + 平均完成时间** |

> 💡 **为什么评估完成时间**: 成功率衡量"能不能做到"，完成时间衡量"做得好不好"。反复试错也可能最终成功，但高效执行才说明策略真正学到了好的运动规划。

---

## 4.3 Main Results

### 4.3.1 RoboTwin 整体对比（Table 1）

| Method | Embodied PT. | Clean | Rand. | Average |
|--------|:---:|:---:|:---:|:---:|
| π0 | ✅ | 65.92 | 58.40 | 62.2 |
| π0.5 | ✅ | 82.74 | 76.76 | 79.8 |
| Motus | ✅ | 88.66 | 87.02 | 87.8 |
| Motus from WAN2.2 | ❌ | 77.56 | 77.00 | 77.3 |
| LingBot-VA | ✅ | 92.90 | 91.50 | 92.2 |
| LingBot-VA from WAN2.2 | ❌ | 80.60 | – | 80.6 |
| **Fast-WAM (Ours)** | **❌** | **91.88** | **91.78** | **91.8** |

> 💡 **批读 Table 1 (上半)**:
> - Fast-WAM **无 embodied pretraining** 就达到 91.8%，超过所有无预训练基线（Motus 77.3%, LingBot-VA 80.6%），差距巨大（+11~14%）
> - 甚至逼近**有**预训练的 LingBot-VA（92.2%），超过有预训练的 Motus（87.8%）和 π0.5（79.8%）
> - **数据效率极高**: 视频联合训练提供的表征质量足以替代 embodied pretraining

### 4.3.2 ⭐ RoboTwin 控制变量对比（Table 1 下半 — 全文最重要的实验）

| Variant | Clean | Rand. | Average |
|---------|:---:|:---:|:---:|
| **Fast-WAM** | **91.88** | **91.78** | **91.8** |
| Fast-WAM-Joint | 90.84 | 90.32 | 90.6 |
| Fast-WAM-IDM | 91.16 | 91.34 | 91.3 |
| **Fast-WAM w.o. video co-train** | **82.76** | **84.80** | **83.8** ⬇️ |

> 💡 **批读 — 全文最重要的定量结果**:
>
> ```
> 有视频训练的三个变体: 90.6% ~ 91.8%（差距仅 1.2%）
>                      ↕ 差距 8.0%!
> 无视频训练:            83.8%
> ```
>
> **推理时是否生成未来**: 影响 < 1.2%（噪声级别）
> **训练时是否有视频目标**: 影响 ~8%（实质性差距）
>
> 结论：**训练时视频建模的价值远大于推理时未来想象**。

---

### 4.3.3 LIBERO 整体对比（Table 2）

| Method | Embodied PT. | Spatial | Object | Goal | Long | Average |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenVLA | ✅ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | ✅ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | ✅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA | ✅ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | ✅ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| **Fast-WAM** | **❌** | **98.2** | **100.0** | **97.0** | **95.2** | **97.6** |

### LIBERO 控制变量

| Variant | Spatial | Object | Goal | Long | Average |
|---------|:---:|:---:|:---:|:---:|:---:|
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Fast-WAM-Joint | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| Fast-WAM-IDM | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| **Fast-WAM w.o. video co-train** | **89.2** | **99.2** | **95.4** | **90.0** | **93.5** ⬇️ |

> 💡 **批读 LIBERO 控制变量**:
>
> ```
> 有视频训练: 97.6% ~ 98.5%（差距 <1%）
>          ↕ 差距 4.1%
> 无视频训练: 93.5%
> ```
>
> 趋势与 RoboTwin 完全一致。特别注意 **Spatial（89.2%）和 Long（90.0%）** 两个子集退化最严重，说明视频建模对**空间推理**和**长时域任务**尤其关键——这些恰恰是需要物理动力学理解的场景。

---

### 4.3.4 真实世界 + 效率分析（Figure 4）

#### 成功率 vs 完成时间（Figure 4 左）

| Method | 成功率 (估) | 平均完成时间 |
|--------|:---:|:---:|
| π0.5 (pretrained) | ~95% | ~120s |
| Fast-WAM-IDM | ~90% | ~175s |
| Fast-WAM | ~85% | ~145s |
| Fast-WAM-Joint | ~80% | ~160s |
| π0.5 w.o. pretrain | ~60% | ~200s |
| **Fast-WAM w.o. video co-train** | **~10%** ⬇️⬇️ | **~235s** |

> 💡 **真实世界结果最有说服力**:
> - 去掉视频联合训练 → 成功率从 ~85% **暴跌到 ~10%**！这不是小退化，是彻底失效
> - 三个有视频训练的 Fast-WAM 变体之间差距不大（80%-90%），远小于有无视频训练的差距
> - Fast-WAM 的完成时间（~145s）优于 IDM（~175s）和 Joint（~160s），说明高效推理也有助于更流畅的闭环控制

#### 推理时延（Figure 4 右）

| Method | Latency |
|--------|:---:|
| π0.5 | 180 ms |
| **Fast-WAM** | **190 ms** |
| Fast-WAM w.o. video co-train | 190 ms |
| Fast-WAM-Joint | 580 ms |
| **Fast-WAM-IDM** | **810 ms** |

> 💡 **时延分析**:
> - Fast-WAM 与 VLA（π0.5）时延相当（190 vs 180ms），满足实时闭环控制要求（<200ms）
> - Fast-WAM-IDM 是 Fast-WAM 的 **4.3×**（810ms vs 190ms）——完全不满足实时性
> - Fast-WAM-Joint 580ms 也超出实时范围
> - Fast-WAM 是**唯一同时满足 WAM 级表现和 VLA 级时延**的方案

---

## 💡 实验如何佐证论文核心观点（三级递进）

论文核心 claim: **WAM 的价值主要在训练时视频建模，不在推理时未来想象。**

### Level 1: 去掉推理时未来想象 → 性能几乎不变

| Benchmark | Fast-WAM | Joint | IDM | 最大差距 |
|-----------|:---:|:---:|:---:|:---:|
| RoboTwin | 91.8 | 90.6 | 91.3 | 1.2% |
| LIBERO | 97.6 | 98.5 | 98.0 | 0.9% |
| Real-world | ~85% | ~80% | ~90% | ~10% |

→ **推理时未来想象不是必需的**

### Level 2: 去掉训练时视频建模 → 性能断崖式下跌

| Benchmark | 有视频训练 | 无视频训练 | 差距 |
|-----------|:---:|:---:|:---:|
| RoboTwin | 91.8% | 83.8% | -8.0% |
| LIBERO | 97.6% | 93.5% | -4.1% |
| Real-world | ~85% | ~10% | **-75%** |

→ **训练时视频建模是核心贡献**

### Level 3: 不需要 embodied pretraining 也能达到 SOTA

| 对比 | 无 PT Fast-WAM | 有 PT 基线 |
|------|:---:|:---:|
| vs Motus (PT) | 91.8% | 87.8% (**Fast-WAM 更高**) |
| vs LingBot-VA (PT) | 91.8% | 92.2% (接近) |
| vs π0.5 (PT) | 91.8% | 79.8% (**Fast-WAM 更高**) |

→ **视频联合训练的表征学习效果可以替代 embodied pretraining**
