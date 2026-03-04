[← 返回 README](../README.md)

# Abstract

## 📌 预览
π0.5 是基于 π0 的 VLA 模型，通过异构数据协同训练实现开放世界泛化，能在从未见过的家庭环境中执行长时域灵巧操作任务。

---

In order for robots to be useful, they must perform practically relevant tasks in the real world, outside of the lab. While vision-language-action (VLA) models have demonstrated impressive results for end-to-end robot control, it remains an open question how far such models can generalize in the wild. We describe $\pi _ { 0 . 5 }$ , a new model based on $\pi _ { 0 }$ that uses co-training on heterogeneous tasks to enable broad generalization. $\pi _ { 0 . 5 }$ uses data from multiple robots, high-level semantic prediction, web data, and other sources to enable broadly generalizable realworld robotic manipulation. Our system uses a combination of co-training and hybrid multi-modal examples that combine image observations, language commands, object detections, semantic subtask prediction, and low-level actions. Our experiments show that this kind of knowledge transfer is essential for effective generalization, and we demonstrate for the first time that an end-to-end learning-enabled robotic system can perform longhorizon and dexterous manipulation skills, such as cleaning a kitchen or bedroom, in entirely new homes.

> 💡 **Abstract 批读**:
> - **核心问题**: VLA 模型能否在真实世界中泛化？
> - **方法**: 基于 π0，通过异构数据协同训练（co-training）——多机器人数据、高层语义预测、网络数据等
> - **关键创新**: 混合多模态训练样本（图像 + 语言 + 检测 + 子任务预测 + 低层动作）
> - **核心结果**: 首次证明端到端学习的机器人系统能在**全新家庭**中执行长时域灵巧操作（清洁厨房/卧室）
> - **关键词**: co-training, heterogeneous data, open-world generalization, mobile manipulation

---

## 🔖 Section 总结

### 核心洞察
1. **泛化的关键不是规模，而是数据多样性** — 97.6% 的训练数据不来自目标移动操作平台
2. **异构知识迁移** — 从其他机器人、语义预测、网络数据中迁移知识
3. **首次里程碑** — 端到端学习系统在全新家庭中完成 10-15 分钟的长时域操作任务
