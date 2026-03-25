[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览

Introduction 分四步递进：① WAM 的动机（建模物理世界演化）→ ② 现有 WAM 的问题（两个增益来源纠缠不清）→ ③ Fast-WAM 的解耦思路 → ④ 控制变量实验设计。

---

## 1.1 WAM 的动机

> Building general-purpose embodied agents requires policies that can not only map visual observations to actions, but also reason about how the physical world evolves under interaction. This has motivated growing interest in World Action Models (WAMs), which combine future visual prediction and action modeling in a unified framework.

> 💡 WAM 相比 VLA 的核心价值主张：不仅做 obs→action 映射，还要建模物理世界如何在交互下演变。这个出发点是合理的——物理理解对机器人操作很重要。

---

## 1.2 现有 WAM 的问题

> Most existing WAMs follow an imagine-then-execute paradigm: they first generate future observations, then predict actions conditioned on the imagined future. While intuitive, this design incurs substantial test-time latency due to iterative video denoising.

> The effectiveness of WAMs may stem from **two distinct sources**: (1) the video prediction objective during **training**, which may help the model acquire stronger physical priors and action-conditioned representations, and (2) explicit future generation during **inference**, which may provide additional foresight for action prediction. Existing WAM systems typically **entangle** these two factors.

> 💡 **全文最核心的 insight 在这里**: 之前所有 WAM 都把两个因素绑在一起——训练时用视频预测目标，推理时也生成未来视频。从来没人问过：如果只保留训练时的视频目标，推理时不生成未来，还行不行？
>
> 这就像"鸡汤煮面条"——面条好吃是因为鸡汤（训练视频目标提供的表征），还是因为最后撒的葱花（推理时生成的未来观测）？

---

## 1.3 Fast-WAM 的解耦思路

> Based on this perspective, we propose Fast-WAM, a WAM architecture that preserves video co-training during training but skips future prediction at test time. Instead of using a pretrained video generation model to iteratively synthesize future frames during inference, Fast-WAM repurposes a pretrained video Diffusion Transformer (DiT) as a **single-pass world encoder** for action generation.

> 💡 **关键设计选择**: 把视频 DiT 从"生成器"变成"编码器"。训练时它学习生成未来视频（获得世界理解），推理时它只做一次前向传播提取表征。这个转换非常优雅。

---

## 1.4 控制变量实验设计

> To study our central question in a controlled way, we instantiate Fast-WAM into variants that mirror representative imagine-then-execute WAM designs.

三种变体对应三种范式：

| 变体 | 对应范式 | 代表工作 |
|------|---------|---------|
| **(A) Fast-WAM-Joint** | 联合去噪（视频+动作一起） | WAM [4], Motus [5] |
| **(B) Fast-WAM-IDM** | 先生成视频再预测动作 | LingBot-VA [3], ViDAR [7] |
| **(C) Fast-WAM w.o. video co-train** | 去掉视频训练目标，保留架构不变 | （控制组） |

> 💡 **实验设计非常精巧**: 三个变体共享同一框架（骨干、tokenization、训练配方），只改变一个因素。这是真正的 controlled comparison，比之前各自为政的 WAM 论文之间横向比较更有说服力。
>
> - Fast-WAM vs Joint/IDM → 隔离"推理时未来想象"的贡献
> - Fast-WAM vs w.o. video co-train → 隔离"训练时视频建模"的贡献

---

## 1.5 三个贡献

1. **提出并研究基本问题**: WAM 的增益来自训练时视频建模还是推理时未来想象？
2. **提出 Fast-WAM**: 保留视频联合训练但消除推理时未来预测，实现实时推理
3. **控制变量实验**: 证明大部分增益来自视频联合训练目标本身，推理时未来生成的贡献远小于预期

> 💡 **对 Introduction 的整体评价**: 问题提得好——简单、基本、重要，但此前被忽视。方法设计直接服务于回答这个问题。逻辑链路清晰，没有多余的叙述。
