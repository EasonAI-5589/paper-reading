# π*0.6: a VLA That Learns From Experience (RECAP)

**作者**: Physical Intelligence (50+ contributors)
**会议**: ICLR 2026
**链接**: [Blog](https://pi.website/blog/pistar06)

## 一句话总结
RECAP 通过 advantage conditioning 让 VLA 从真实世界部署经验中进行 offline RL 自我提升，在叠衣服、做咖啡、装箱子等复杂任务上 throughput 翻倍、failure rate 减半。

## 核心贡献
1. **RECAP 方法**: 一种通用的 VLA RL 训练框架，整合 demonstrations + autonomous rollouts + expert interventions
2. **Advantage conditioning for policy extraction**: 比 PPO/AWR 更适合 flow matching VLA，把 RL 转化为 conditional supervised learning
3. **π*0.6 模型**: 基于 π0.6 (Gemma 3 4B + 860M action expert) + advantage indicator，首个通过 RL 显著提升的大规模 VLA
4. **真实世界验证**: 咖啡机连续运行 13 小时、叠衣服 2+ 小时、工厂装箱，达到实际可用水平

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要：RECAP 方法概述 |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 + Figure 1-2 |
| [02 - Related Work](sections/02-related-work.md) | 五条相关工作线梳理 |
| [03 - Preliminaries](sections/03-preliminaries.md) | RL 基础 + Regularized RL 理论 |
| [04 - Method (RECAP)](sections/04-method.md) | 核心方法：Distributional VF + Advantage Conditioning |
| [05 - Implementation](sections/05-implementation.md) | π*0.6 架构、Reward 设计、训练流程 |
| [06 - Experiments](sections/06-experiments.md) | 三大任务评估 + Ablations |
| [07 - Discussion](sections/07-discussion.md) | 未来方向：自动化、探索、online RL |
| [08 - Appendix](sections/08-appendix.md) | 推导细节 + 超参数 + 数据量 |

## 关键数字

| 指标 | 数值 |
|------|------|
| VLA backbone | Gemma 3 4B |
| Action expert | 860M params (flow matching) |
| Value function | 670M params (distributional, 201 bins) |
| Control freq | 50 Hz |
| Throughput 提升 | ~2× (hardest tasks) |
| Failure rate 降低 | ~2× |
| 咖啡连续运行 | 13 hours |
| 叠衣服连续运行 | 2+ hours (new home) |
| Targeted failure removal | 97% success (1200 trajectories) |

## 方法速览

```
Pre-training (offline RL on diverse demo data):
  1. Train distributional VF → predict steps-to-success
  2. Compute advantages → binarize (threshold = 30th percentile)
  3. Train VLA with "Advantage: positive/negative" text conditioning

Post-training (per task):
  4. SFT on task demos (I=True for all)
  5. Deploy → collect autonomous + intervention data
  6. Retrain VF → recompute advantages → retrain VLA
  7. Repeat 5-6 for 1-2 iterations
```
