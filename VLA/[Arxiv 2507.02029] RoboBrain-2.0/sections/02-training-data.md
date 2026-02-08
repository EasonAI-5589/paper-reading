# 3. Training Data

> 来源: RoboBrain 2.0 Technical Report

---

## 📄 原文

> 💡 **Section 概览**: 数据是 RoboBrain 系列的核心竞争力。v2 的数据分三大类：通用VQA、空间数据、时序数据。

---

![Figure 4](../images/b38f5604013974ce27e5ef9ea43fa84c8219f8609d0b9f858c1c4d9ebe7f5eed.jpg)
*Figure 4: 训练数据分布*

> 💡 **Figure 4 批读**:
> ```
> 三大数据类别:
> ├── General MLLM VQA: 873K
> │   ├── LLaVA-665K → 531K (过滤后)
> │   └── LRV-400K → 342K (GPT-4 合成)
> │
> ├── Spatial Data: ~2.5M+
> │   ├── Visual Grounding: 86K conversations (LVIS 152K images)
> │   ├── Object Pointing: 190K QA (Pixmo) + 347K (RoboPoint)
> │   ├── Affordance: 561K QA (PACO-LVIS) + 320K (RoboPoint spatial)
> │   ├── Spatial Understanding: 826K samples
> │   │   ├── 2D web images (OpenImage → pseudo-3D scene graphs)
> │   │   ├── 3D scene videos (MMScan, 3RScan, ScanQA, SQA3D, SpaceR)
> │   │   └── 3D embodied videos (CA-1M → 100K frames)
> │   └── Spatial Referring: 802K samples (RefSpatial)
> │
> └── Temporal Data: ~1.1M+
>     ├── Ego-View Planning: 50K (EgoPlan-IT)
>     ├── ShareRobot Planning: 1M QA (from v1)
>     ├── AgiBot Planning: 9.1K QA (19 tasks, 109K frames)
>     ├── Multi-Robot Planning: 44.1K (DeepSeek-V3 生成, 1659 种协作任务)
>     └── Close-Loop Interaction: OTA trajectories (AI2Thor, 120 indoor envs)
> ```

---

### 3.1 General MLLM VQA (873K)

> 💡 **数据工程细节**:
> - LLaVA-665K: 去掉 bbox 相关 QA，合并同图多 QA，截断 >2048 token → 531K
> - LRV-400K: GPT-4 在 Visual Genome 稠密标注上生成 16 种任务 → 342K
> - 纯文本对话单独 batch，吞吐提升 25%

### 3.2 Spatial Data

> 💡 **空间数据的构建 pipeline 是本文最重要的技术贡献之一**:

**Spatial Understanding (826K)** — 这是新增的核心数据：
```
2D web images → pseudo-3D pipeline:
├── RAM (物体类别预测)
├── GroundingDINO (2D bbox)
├── Qwen2.5-VL (层级化 caption: "cup" → "第三个杯子从左数")
├── UniDepth V2 + WildeCamera (深度 + 相机内参 → 3D 点云)
├── SAM 2.1 (实例分割)
└── 构建 scene graph → 模板 + LLM (QwQ) 生成 QA pairs
```

> 💡 **关键创新**: 用 pseudo-3D 方法从 2D 图片构建 3D scene graph，覆盖 31 种空间概念
> （前人数据集通常只有 ~15 种）。这个 pipeline 很值得学习。

**Spatial Referring (802K)**:
- 和 grounding/pointing 的区别：targeting 单一无歧义目标（对应 pick-and-place 场景）
- 从 scene graph 采样 caption-point pairs + top-down occupancy maps

### 3.3 Temporal Data

> 💡 **时序数据是 v2 相比 v1 最大的新增**:

**Multi-Robot Planning (44K)**: 
- 3 种环境: 家庭、超市、餐厅
- 1,659 种协作任务 → DeepSeek-V3 生成 44,142 样本
- 基于 RoboOS 的 scene graph + robot specs + tool lists

**Close-Loop Interaction**:
- AI2Thor 模拟器，120 种室内环境，4000+ 交互物体
- Observation-Thought-Action (OTA) 轨迹
- GPT-4o 生成详细的 thought process（情境分析、空间推理、自我反思、验证）
- 随机注入失败事件 → 鲁棒性训练

> 💡 **Close-Loop 是关键差异化能力** — v1 完全没有 closed-loop interaction，
> v2 通过模拟失败 + 思维链来学习在线纠错。

---

## 💡 Section 总结

### 数据量对比
| 类别 | v1 | v2 |
|------|----|----|
| General VQA | ~9M (LCS+Image+SI+OV) | 873K (精选高质量) |
| Robot Planning | 1M (ShareRobot) + 800K (RoboVQA) | 1M (ShareRobot) + 50K (Ego) + 9K (AgiBot) + 44K (Multi-Robot) |
| Spatial | 6K affordance + 6K trajectory | **2.5M+** (pointing/grounding/affordance/understanding/referring) |
| Temporal | ❌ | **OTA + close-loop + multi-agent** |
| **Total** | ~12M | ~5M+ (更精，质量优先) |

### 核心洞察
1. **v2 的数据策略变了**: v1 堆量（12M），v2 重质量+多样性（5M 但覆盖更多任务）
2. **Pseudo-3D pipeline 是核心贡献** — 从 2D 图片构建 3D scene graph 的全自动方法
3. **Multi-Robot + Close-Loop 是差异化数据** — 市场上几乎没有其他模型有这类训练数据
4. **大量使用 GPT-4o/DeepSeek-V3 生成数据** — 降低人工标注成本
