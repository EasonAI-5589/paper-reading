[← 返回 README](../README.md)

# 1. Introduction

## 核心论点

现有通用 VLA 策略（π0、π0.5 等）在陌生场景下容易失效，但评估和改进代价极高：
- **评估**：需要大量真实环境 rollout，成本高、慢
- **改进**：失败后需要重新收集专家 demo，难以规模化

**痛点：** "缺乏一个快速、廉价的反馈驱动机制来精炼通用模型"

---

## 现有 World Model 的三个局限

> 💡 这三条是 Ctrl-World 设计动机的直接来源

| 局限 | 问题 | Ctrl-World 的解法 |
|------|------|------------------|
| **只有单第三人称视角** | 严重的部分可观测性，导致幻觉（如物体没碰就"瞬移"到夹爪里）。而且现代 VLA 通常需要第三人称 + 腕部相机双输入 | **多视角联合预测** |
| **缺乏细粒度动作控制** | 预训练视频模型只接受文本/图像条件，无法精确跟随高频动作序列 | **帧级动作条件（Frame-level Action Conditioning）** |
| **长时序一致性差** | 预测误差积累，长时序下场景漂移、物体错位 | **姿态条件记忆检索（Memory Retrieval）** |

---

## Ctrl-World 的定位

Ctrl-World 的目标：把**预训练的被动视频生成模型（SVD）**改造成**策略兼容的可交互仿真器**。

```
预训练视频模型 (SVD)
    ↓ 三个核心改造
1. 多视角联合预测
2. 帧级动作条件
3. 记忆检索
    ↓
Ctrl-World：可控多视角 World Model
    ↓ 两个应用
策略评估（Policy Evaluation）+ 策略改进（Policy Improvement）
```

---

## 与相关工作的差异

| 方法类别 | 代表工作 | 与 Ctrl-World 区别 |
|---------|---------|-------------------|
| 视频生成做数据增强 | Gen2Act, etc. | 生成假 action label，非 action-conditioned |
| 视频模型直接当 policy | UniSim, etc. | 从视频预测 action，不是预测未来观测 |
| Action-conditioned WM | WPE, IRASim | 单视角，缺乏长时一致性，不兼容现代 VLA |
| **Ctrl-World** | **本文** | 多视角 + 帧级控制 + 记忆检索，专为 policy-in-the-loop 设计 |

---

## 💡 批读注解

**为什么"单视角"是个大问题：**

想象一下机械臂抓东西：第三人称相机能看到"手往哪儿去"，但看不清"夹爪是不是真的捏住了"——这需要腕部相机。单视角 world model 在接触发生时容易产生幻觉（看起来抓到了，其实没有）。Ctrl-World 通过联合预测腕部相机解决了这个问题。

**这篇论文跟我们项目的关系：**
- Ctrl-World 是我们 Latent-Act WAM 项目的 **baseline**
- 我们在 LIBERO 上 finetune 的正是 Ctrl-World（SVD backbone 版本）
- 未来可能的 research direction：把 SVD backbone 换成 Wan2.1，看看效果
