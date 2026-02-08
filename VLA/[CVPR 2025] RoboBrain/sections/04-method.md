# 4. RoboBrain Model

> 来源: RoboBrain (CVPR 2025)

---

## 📄 原文

> 💡 **Section 概览**: 模型架构和训练策略。架构本身就是 LLaVA + LoRA，创新在于多阶段训练的设计。

---

### 4.1 Model Architecture

![Figure 4](../images/d4ccb5c16f629f5fd3b1ad40fcba407524d10a9458a47821bcde41adcc24bc8e.jpg)
*Figure 4: RoboBrain Pipeline — 基础模型 + A-LoRA (affordance) + T-LoRA (trajectory)*

> 💡 **Figure 4 批读**:
> ```
> 模型架构:
> ├── Vision Encoder: SigLIP (so400m-patch14-384)
> │   ├── 27 hidden layers
> │   ├── 14×14 patch → 729 tokens/image
> │   └── 支持动态分辨率: max 384×{6×6} = 2304×2304
> │
> ├── Projector: 2-layer MLP (gelu)
> │   └── 将 visual tokens 映射到 LLM 语义空间
> │
> ├── LLM: Qwen2.5-7B-Instruct
> │   ├── 28 hidden layers
> │   └── 支持 128K token context
> │
> ├── A-LoRA: LoRA (rank=64) → Affordance
> │   └── 输出: bounding box {l_x, l_y, r_x, r_y}
> │
> └── T-LoRA: LoRA (rank=64) → Trajectory
>     └── 输出: waypoints [(x_1,y_1), ..., (x_N,y_N)]
> ```

> 💡 **架构评价**:
> - 这就是标准的 LLaVA-OneVision 架构，没有结构创新
> - A-LoRA 和 T-LoRA 的设计很实用：LoRA 参数只有 28M，不影响基础模型的 planning 能力
> - 用不同 LoRA 做不同任务（affordance vs trajectory）是个简洁的设计
> - 但 affordance 用 bounding box 表示比较粗糙，不如 segmentation mask 精细

**Foundational Model for Planning**:
- SigLIP encoder → MLP projector → Qwen2.5-7B
- 标准自回归生成：给 image/video + text instruction → 生成 text response

**A-LoRA for Affordance**:
- Affordance = 物体可操作区域，用 bounding box 表示
- 格式: {l_x, l_y, r_x, r_y}（左上角 + 右下角坐标）

**T-LoRA for Trajectory**:
- Trajectory = 2D visual trace（末端执行器的运动路径）
- 格式: {(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)}

---

### 4.2 Training

> 💡 **训练策略是本文的另一个重要贡献**:

![Table 1](../images/af3ab8e1859238e44fd23140bc7acbd084743b6dc3225c123edb5d8ce21be939.jpg)
*Table 1: 各训练阶段详细配置*

> 💡 **Table 1 批读**:
> ```
> Phase 1: 通用 OneVision 训练（打基础）
> ├── Stage 1: Projector 对齐 (LCS-558K, 只训 Projector 17M 参数)
> ├── Stage 1.5: 全模型 (4M image-text, 增强多模态理解)
> └── Stage 2: 全模型 (3.2M SI + 1.6M OV, 指令跟随 + 视频理解)
> 
> Phase 2: 机器人训练（核心）
> ├── Stage 3: 全模型 (3M = 1.3M robotic + 1.7M general)
> │   ├── RoboVQA-800K
> │   ├── ScanView-318K (MMScan, 3RScan, ScanQA, SQA3D)
> │   ├── ShareRobot-200K (子集)
> │   └── 1.7M general data (防遗忘)
> │
> └── Stage 4: LoRA 微调
>     ├── A-LoRA: 10K affordance data (28M params)
>     └── T-LoRA: 400K trajectory data (28M params)
> 
> 训练资源: 16~22 × 8×A800 GPU
> ```

> 💡 **关键设计决策**:
> 1. **Robot:General = 4:6** — 消融实验验证的最佳比例（Table 7）
> 2. **防止灾难性遗忘** — Stage 3 混入 1.7M 通用数据
> 3. **LoRA 做具体任务** — Affordance/Trajectory 数据量小，全模型微调会过拟合
> 4. **Trajectory data 400K vs Affordance 10K** — 轨迹数据多 40 倍，说明轨迹更难学

> 💡 **批注 - 为什么 T-LoRA 训练数据是 400K 而 Affordance 只有 10K？**
> - Affordance 标注数据只有 6K 图片，加上其他开源 affordance 数据也就 10K
> - Trajectory 数据可以从演示视频中自动提取 waypoints，更容易扩展
> - 这也暗示了 affordance 标注是个瓶颈，未来可以用自动标注来扩展

---

## 💡 Section 总结

### 架构速查
| 组件 | 具体模型 | 参数量 |
|------|----------|--------|
| Vision Encoder | SigLIP so400m-patch14-384 | ~400M |
| Projector | 2-layer MLP (gelu) | 17M |
| LLM | Qwen2.5-7B-Instruct | ~7B |
| A-LoRA | rank=64, FFN layers | 28M |
| T-LoRA | rank=64, FFN layers | 28M |
| **Total** | | **~8B** |

### 核心洞察
1. **模型架构没创新**，就是 LLaVA + LoRA，但训练策略设计得很仔细
2. **4 阶段 + 2 阶段 LoRA = 6 阶段训练**，需要大量 GPU 资源（最多 22×8 A800）
3. **数据配比是关键** — robot:general = 4:6 是个有参考价值的经验
4. **LoRA 分离 affordance 和 trajectory** — 简洁有效，两个任务不互相干扰
