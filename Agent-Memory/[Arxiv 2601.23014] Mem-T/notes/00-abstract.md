[← 返回 README](../README.md)

# Mem-T: Densifying Rewards for Long-Horizon Memory Agents

Yanwei Yue¹˙*, Guibin Zhang²˙*, Boci Peng¹, Xuanbo Fan¹, Jiaxin Guo¹, Qiankun Li³†, Yan Zhang¹†

¹Peking University ²National University of Singapore ³Nanyang Technological University

ywyue25@stu.pku.edu.cn, guibinz@u.nus.edu

\* Equal Contribution † Corresponding Author

🔗 [GitHub](https://github.com/yanweiyue/Mem-T) | [HuggingFace](https://huggingface.co/EdwinYue/Mem-T-4B)

---

## Abstract

Memory agents, which depart from predefined memory-processing pipelines by endogenously managing the processing, storage, and retrieval of memories, have garnered increasing attention for their autonomy and adaptability. However, existing training paradigms remain constrained: agents often traverse long-horizon sequences of memory operations before receiving sparse and delayed rewards, which hinders truly end-to-end optimization of memory management policies. To address this limitation, we introduce Mem-T, an autonomous memory agent that interfaces with a lightweight hierarchical memory database to perform dynamic updates and multi-turn retrieval over streaming inputs. To effectively train long-horizon memory management capabilities, we further propose MoT-GRPO, a tree-guided reinforcement learning framework that transforms sparse terminal feedback into dense, step-wise supervision via memory operation tree backpropagation and hindsight credit assignment, thereby enabling the joint optimization of memory construction and retrieval. Extensive experiments demonstrate that Mem-T is ● high-performing, surpassing frameworks such as A-Mem and Mem0 by up to $14.92\%$, and ⊗ economical, operating on a favorable accuracy-efficiency Pareto frontier and reducing inference tokens per query by $\sim 24.45\%$ relative to GAM without sacrificing performance.

> 💡 **批注**: Mem-T 的两大卖点很清晰：(1) 层次化记忆架构 + 自主管理；(2) MoT-GRPO 把稀疏终端奖励变成密集逐步信号。核心痛点是 long-horizon memory operation 中的 temporal credit assignment — 这在 RL for agents 领域是真正的 open problem。用 tree 做 rollout 然后 backprop reward 是一个很聪明的中间方案，比 MCTS 轻但比 flat GRPO 信息量更大。
