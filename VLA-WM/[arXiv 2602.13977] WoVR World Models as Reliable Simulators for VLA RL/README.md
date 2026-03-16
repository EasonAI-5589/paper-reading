# WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL

**作者**: Zhennan Jiang\*, Shangqing Zhou\*, Yutong Jiang, Zefang Huang, Mingjie Wei, Yuhui Chen, Tianxing Zhou, Zhen Guo, Hao Lin, Quanlu Zhang, Yu Wang, Haoran Li†, Chao Yu†, Dongbin Zhao  
**机构**: Tsinghua University / Institute of Automation CAS / Zhongguancun Academy / Infinigence AI  
**年份**: 2026 | **arXiv**: [2602.13977](https://arxiv.org/abs/2602.13977)  
**链接**: [PDF](https://arxiv.org/pdf/2602.13977) · [HuggingFace](https://huggingface.co/Collections/RLinf/wovr) · [GitHub](https://github.com/RLinf/RLinf) · [paper.pdf](paper.pdf)

---

## 一句话总结
用 world model（视频生成模型）替代物理模拟器，给 VLA 做 RL 后训练。核心问题是 world model 会 hallucinate（幻觉），WoVR 用三层机制解决。
World model 做 VLA RL 的根本障碍是 hallucination（闭环误差积累腐蚀优化信号），WoVR 用三层机制显式控制：① 稳定的 action-conditioned world model（Wan 5B + dual-channel + first-frame anchoring）；② KIR 缩短有效误差深度；③ PACE 维持 policy-model 分布对齐。LIBERO 平均 +29.3 pp，真实机器人 +30.0 pp。

---

## 核心贡献

1. **Hallucination 视角**：首次明确将 world model RL 的可靠性问题定义为「hallucination problem」，区分 autoregressive error accumulation 和 policy-induced distribution shift 两个根源
2. **Stabilized World Model**：Wan 2.2-TI2V-5B + dual-channel action injection（AdaLN + cross-attention）+ first-frame anchoring + noisy context，实现 23 FPS 高速推理，全面超越 OpenSora/EVAC/Cosmos-Predict2
3. **KIR（Keyframe-Initialized Rollouts）**：从任务关键状态/failure state 附近初始化 rollout，缩短有效误差深度；配合 masked GRPO（屏蔽 post-success 步骤 + length normalization）
4. **PACE（Policy-Aligned Co-Evolution）**：低频一次性 world model 对齐，解决 policy 更新导致的 distribution shift，消融证明贡献 -10.5 pp

---

## 📖 批读导航

| Section | 文件 | 内容 |
|---------|------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 | 问题定位 + 三层机制 + 关键数字 |
| [01 - Introduction](sections/01-introduction.md) | 引言 | Hallucination 定义 + 两个根源 + Figure 1 + 三层解决方案 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作 | VLA RL / World Model / World Model as Simulator |
| [03 - Preliminary](sections/03-preliminary.md) | 预备知识 | MDP → WM-MDP 替换 + Figure 2 |
| [04 - Method](sections/04-method.md) | 方法 | 4.1 World Model + 4.2 KIR+GRPO + 4.3 PACE |
| [05 - Experiments](sections/05-experiments.md) | 实验 | Q1 WM质量 + Q2 LIBERO policy + Q3 真实机器人 |
| [06 - Ablation](sections/06-ablation.md) | 消融 | 6.1 WM 机制 + 6.2 Policy 机制 |
| [07 - Conclusion & Appendix](sections/07-conclusion-appendix.md) | 结论 + 附录 | 总结 + 局限 + VLAW对比 + GPU策略 |

---

## 关键数字

| 指标 | 数值 |
|------|------|
| LIBERO 平均 base → WoVR | 39.9% → **69.2%**（+29.3 pp） |
| LIBERO-Long（最难） | 13.7% → **35.8%**（+22.1 pp）|
| LIBERO-Object（最亮眼） | 36.3% → **82.0%**（+45.7 pp） |
| 真实机器人（avg） | 61.7% → **91.7%**（+30.0 pp） |
| World model 推理速度 | **23 FPS**（vs OpenSora 7 FPS） |
| FVD（512步，vs OpenSora） | **68.011** vs 89.391 |
| WMPO 在 LIBERO-Long 提升 | **0 pp**（WoVR +22.1 pp）|
| PACE 消融贡献 | -10.5 pp（w/o PACE 降到 0.710） |

---

## 📊 Citation Landscape

> 数据来源：[Semantic Scholar](https://www.semanticscholar.org/paper/search?q=WoVR+World+Models+Reliable+Simulators) · [Connected Papers](https://www.connectedpapers.com/main/2602.13977)
> ⚠️ 新论文（2026-02），被引次数可能尚未统计

### 核心参考文献分组

#### VLA / Robot Policy
| 论文 | 年份 | 说明 |
|------|------|------|
| OpenVLA-OFT [1] | 2025 | WoVR 的 base VLA model |
| π₀ [2] | 2024 | Flow-matching VLA |
| π₀.₅ [3] | 2025 | Open-world VLA |
| LIBERO [59] | 2023 | 实验 benchmark |
| HiL-SERL [16] | 2025 | Reward classifier 设计参考 |

#### World Model / Video Generation
| 论文 | 年份 | 说明 |
|------|------|------|
| Wan 2.2-TI2V-5B [20] | 2025 | WoVR world model backbone |
| OpenSora 2.0 [19] | 2025 | WMPO 的 backbone，WoVR 的对比 baseline |
| Ctrl-World [46] | 2025 | VLAW 的 world model（Guo et al.，也是本领域重要工作） |
| Diffusion Forcing [53] | 2024 | Noisy context 的灵感来源 |
| EVAC [42] | 2025 | World model 对比 baseline |

#### RL / Policy Optimization
| 论文 | 年份 | 说明 |
|------|------|------|
| PPO [12] | 2017 | 标准 on-policy RL |
| GRPO / DeepSeek-R1 [14] | 2025 | WoVR 采用的 policy gradient 算法 |
| WMPO [22] | 2025 | 最直接的竞品（world model + RL） |
| World-Env [21] | 2025 | 另一个 world model simulator 工作 |
| RLinf [24] | 2025 | WoVR 的系统基础 |

### 同期相关工作（同为 2602.xxxxx，2026-02 爆发）
| 论文 | arXiv | 简述 |
|------|-------|------|
| VLAW | 2602.12063 | World model fine-tune + filtered BC，DROID 真实机器人 |
| World-VLA-Loop | 2602.06508 | VLA + World Model 闭环学习 |
| Beyond Imitation | 2602.12628 | RL-based sim-real co-training for VLA |
| World-Gymnast | 2602.02454 | World model 内做 RL 训练机器人 |
| RISE | 2602.11075 | Compositional world model self-improving |
| GigaBrain-0.5M* | 2602.12099 | World model-based RL for VLA |

> 💡 **2026-02 是 World Model + VLA RL 的爆发月**：至少 6 篇同期独立工作，说明这是整个 community 同时攻克的方向，竞争激烈。WoVR 的系统性最强（三层机制 + 详细消融），VLAW 的真实机器人实验最有说服力。
