# LaST₀: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model

> **arXiv:** [2601.05248](https://arxiv.org/abs/2601.05248)
> **项目主页:** https://vla-last0.github.io/
> **机构:** 北京大学 / 北航人形机器人创新中心 / CUHK / Simplexity Robotics
> **通讯作者:** 张珊航（北京大学）
> **批读状态:** ✅ 完整批读（2026-04-02）

---

## 一句话总结

用**隐式时空 CoT**（而不是显式文本/图像 CoT）让 VLA 在 act 之前高效推理，同时通过 **Mixture-of-Transformers 双系统架构**实现低频推理 + 高频执行的解耦。

---

## 核心问题

现有显式 CoT VLA 方法（生成文本推理链或预测未来图像）有两个根本瓶颈：

1. **推理延迟高**：自回归生成文本/像素开销大，限制了控制频率（CoT-VLA 只有 1.1 Hz）
2. **表征空间受限**：自然语言无法精确表达低层物理属性（位姿、力、空间结构）

---

## 方法概览

### 核心组件

#### 1. Latent Spatio-Temporal CoT (LaST CoT)
- 对每个未来关键帧 k，从三种模态提取 latent token：
  - **视觉 latent** zᵥ：SigLIP-Large 编码未来 RGB 帧（语义）
  - **点云 latent** zₚ：Uni3D 编码未来点云（3D 几何）—— **仅训练时用，推理时不需要**
  - **本体感知 latent** zₛ：action tokenizer 编码机器人状态
- 三种 token 按时间顺序交错排列：`[z¹ᵥ, z¹ₚ, z¹ₛ, z²ᵥ, z²ₚ, z²ₛ, ...]`
- 每种模态只用 **1 个 token**（average pooling 压缩），高效且够用
- 监督方式：**cosine similarity loss**（连续 latent 回归，不是 softmax）

#### 2. Mixture-of-Transformers (MoT) 双系统架构
- 基座：**Janus-Pro（DeepSeek-LLM 1.5B）**，24层 decoder-only transformer
- 两个 expert 共享 self-attention，但 FFN/投影矩阵/LayerNorm 各自独立：
  - **慢推理 expert**：低频（κ步一次），自回归生成 LaST CoT latent
  - **快执行 expert**：高频（每步），基于最新 latent + 当前观测，用 Flow Matching 生成 action
- 异步频率：κ ∈ {2, 4, 8}，训练时随机混合比例（1:1, 1:2, 1:4），推理时选 1:4

### 训练流程
1. **大规模预训练**：400K+ 轨迹（Open-X，DROID，ROBOMIND 等）
2. **SFT**：联合优化两个 expert
   - 慢 expert：ℒ_latent（cosine loss）
   - 快 expert：ℒ_flow（Flow Matching loss）
   - 混合频率训练提升鲁棒性

---

## 实验结果

### 仿真（RLBench，10 任务）
| 方法 | 参数量 | 成功率 | 推理速度 |
|------|--------|--------|----------|
| LaST₀ | 3.3B | **82%** | **15.4 Hz** |
| HybridVLA | 7B | 74% | - |
| π₀.₅ | 3B | 65% | 13.8 Hz |
| CoT-VLA | - | - | 1.1 Hz |
| CogACT | 7B | 61% | - |

### 真机（3类平台，各 10 任务）
- Franka 单/双臂：**72%** vs π₀.₅ 59%（+13%）
- AgileX 移动操作：+14%
- 天工人形手：+14%
- 长时序任务（连续 3 次）：0.66→0.47→0.33 vs π₀.₅ 0.47→0.20→0.07

---

## 消融实验关键发现

1. **模态缺一不可**：单独用图像/点云/本体感知 latent 分别是 74/76/75%，三者合一是 82%
2. **1 个 token 就够**：每种模态 1 token 已达最优，增多无收益（说明 latent 信息密度高）
3. **时序覆盖 4 步最佳**：0→1→2→4 步依次从 68%→74%→78%→82%，超过 4 步无收益
4. **混合频率训练最好**：单一比例约 75-79%，混合策略达 82%

---

## 局限性

- 点云数据需要深度传感器（推理时不需要，但训练时要采集）
- 作者未在 HuggingFace 上开源权重（截至批读时）
- 在极高 DoF 或高速任务上的泛化性还未充分验证

---

## 与 WAM 的关系（Last-WAM 背景）

LaST₀ 的 latent CoT 本质上是在**预测未来世界状态的 latent 表征**（视觉+几何+本体），这与 WAM（World Action Model）的世界模型思路有天然的联结：

- WAM（DreamZero）在**像素空间**预测未来帧 → 开销大
- LaST₀ 在**latent 空间**预测未来状态 → 高效
- **Last-WAM 猜想**：把 LaST₀ 的 latent 时空表征作为 WAM 的世界模型 backbone，兼顾效率与物理感知能力

---

## 参考文献关联

- Janus-Pro（基座 VLM）：Chen et al. 2025, arXiv 2501.17811
- DreamZero（WAM 概念来源）：arXiv 2602.15922
- CoT-VLA（显式 CoT 基线）：Zhao et al. 2025
- π₀.₅（Flow Matching VLA 基线）：Intelligence et al. 2025
