[← 返回 README](../README.md)

# 6. Discussion

## 📌 预览
作者坦诚讨论了三个主要局限：推理延迟、rollout 数据需求、搜索深度。这些是 Cosmos Policy 走向实际部署的关键瓶颈。

---

We presented Cosmos Policy, a state-of-the-art robot policy fine-tuned from the Cosmos-Predict2 video foundation model that demonstrates strong performance in LIBERO, RoboCasa, and ALOHA robot environments. We also show that incorporating policy rollout data to refine world model and value function predictions enables effective model-based planning. Limitations and future work:

We observe substantially lower inference speed when using model-based planning (e.g., around 5 seconds to produce one action chunk), which may limit applicability to dynamic tasks. How to speed up the search is an important direction for future study. In addition, effective planning requires substantial rollout data to achieve accurate predictions beyond the demonstration distribution. Learning from fewer rollouts would increase the accessibility of our approach. Lastly, we focus on best-of-N planning with one layer in the search tree; extending the world model's prediction horizon and planning to greater depths could potentially lead to more effective search.

> 💡 **三大局限详解**:
> 
> **1. 推理延迟 (~5s)**
> - Direct policy: 0.61s (5步) 或 0.95s (10步) → 可接受
> - Planning: 4.9s (8 GPU 并行) → 对反应式任务太慢
> - 未来方向：模型蒸馏、更高效的搜索算法、减少去噪步数（1步去噪已达 66.4%）
> 
> **2. Rollout 数据需求**
> - 需要 648 个 rollouts 才能有效训练 planning model
> - 对于新环境/新任务，需要先部署 base policy 收集数据 → 额外成本
> - 未来方向：从更少的 rollouts 学习（few-shot world model adaptation）
> 
> **3. 搜索深度限制**
> - 当前只做 best-of-N（搜索树深度 = 1）
> - 没有多步前瞻（multi-step lookahead）
> - 未来方向：扩展 world model 预测 horizon，做更深层的 tree search（类似 MCTS）

> 💡 **我的补充思考**:
> - **视频模型 vs VLM 的互补性**：Cosmos Policy 展示了视频模型在 low-level control 上的优势，但 VLM 在语义理解和泛化上仍有优势。未来可能的方向是结合两者？
> - **Scaling 问题**：当前用 2B 模型，如果用更大的视频模型（如 Cosmos-Predict2-7B）会不会更好？论文没有讨论 scaling behavior
> - **与 RL 的结合**：当前用 imitation learning + rollout fine-tuning，但没有做 online RL。rollout 数据的价值说明了 on-policy data 的重要性，未来是否可以闭环迭代？
> - **跨任务泛化**：ALOHA 实验是 multi-task 的，但 LIBERO 和 RoboCasa 是 per-suite 训练的。大规模跨任务泛化能力还需要验证

---

## 7 REPRODUCIBILITY STATEMENT

We release model checkpoints, training data, and code (including training and evaluation scripts) on our project website. Further training and evaluation details are provided in Appendix A.2 and A.3, respectively.

> 💡 **开源承诺**: 代码、模型、数据全部开源 → https://research.nvidia.com/labs/dir/cosmos-policy/
> 这对复现和后续研究非常有价值。

---

## ACKNOWLEDGMENTS

We thank Yu-Wei Chao, Lars Ankile, Alexander Swerdlow, Max Li, and the anonymous reviewers for their constructive comments which have helped improve various elements of this paper. We also thank Dan Blick, Sophia Huang, Mohammad Harrim, Yuzhu Dong, and Pooya Jannaty for their assistance with the OSS release of this project. This work was completed during an internship at NVIDIA in collaboration with Stanford University, and the first author was partially supported by the Robotics and AI Institute.

> 💡 **背景**: 第一作者 Moo Jin Kim 是在 NVIDIA 实习期间完成的，与 Stanford 合作。之前他在 Stanford 做了 OpenVLA 和 OpenVLA-OFT，现在转向视频模型路线。这条从 VLA → Video Policy 的研究轨迹本身就很有意思。

---

## 🔖 Section 总结

### 核心洞察
1. **推理延迟**是最大的实际限制，但 1-step denoising 的尝试（66.4% vs 67.1%）展示了加速的可能性
2. **Planning 需要 rollout 数据** → 这是一个 chicken-and-egg 问题（需要 policy 先跑才有数据来改进 planning）
3. **单层搜索**是当前的简化选择，更深的搜索可能带来更大提升
4. 开源全套代码/模型/数据 → 后续研究基础扎实
