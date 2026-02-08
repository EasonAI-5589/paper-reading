[← 返回 README](../README.md)

# Abstract

## 📌 预览
π₀ 论文的摘要：提出基于预训练 VLM + flow matching 的通用机器人策略，在多种灵巧操作任务上展示了强大能力。

---

Robot learning holds tremendous promise to unlock the full potential of flexible, general, and dexterous robot systems, as well as to address some of the deepest questions in artificial intelligence. However, bringing robot learning to the level of generality required for effective real-world systems faces major obstacles in terms of data, generalization, and robustness. In this paper, we discuss how generalist robot policies (i.e., robot foundation models) can address these challenges, and how we can design effective generalist robot policies for complex and highly dexterous tasks. We propose a novel flow matching architecture built on top of a pre-trained vision-language model (VLM) to inherit Internet-scale semantic knowledge. We then discuss how this model can be trained on a large and diverse dataset from multiple dexterous robot platforms, including single-arm robots, dual-arm robots, and mobile manipulators. We evaluate our model in terms of its ability to perform tasks via direct prompting, follow language instructions from people and from a high-level VLM policy, and its ability to acquire new skills via fine-tuning. Our results cover a wide variety of tasks, such as laundry folding, table cleaning, and assembling boxes.

> 💡 **Abstract 批读**:
> - **核心方法**: 预训练 VLM + flow matching → VLA 模型
> - **数据**: 多平台（单臂、双臂、移动操作）的大规模灵巧操作数据
> - **评估维度**: (1) 直接 prompting (2) 语言指令跟随 (3) fine-tuning 学新技能
> - **代表性任务**: 叠衣服、清理桌子、组装箱子
> - **关键词**: generalist robot policy, robot foundation model, flow matching, VLM, dexterous manipulation

---

## 🔖 Section 总结

### 核心洞察
1. 机器人学习面临三大瓶颈：**数据、泛化、鲁棒性** → 通用机器人策略（foundation model）是解决方案
2. 架构创新：VLM backbone + flow matching（而非自回归离散化）→ 支持连续动作分布
3. 训练策略：大规模多样化预训练 + 高质量 fine-tuning
