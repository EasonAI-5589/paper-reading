[← 返回 README](../README.md)

# VII. Discussion, Limitations, and Future Work

## 📌 预览
总结 π₀ 的贡献，讨论局限性和未来方向。

---

We presented a framework for training a robot foundation model, which we refer to as $\pi _ { 0 }$ , that consists of pre-training on highly diverse data, followed by either out-of-box evaluation or fine-tuning to complex downstream tasks.

Our empirical evaluation studies tasks that combine dexterity, generalization, and temporally extended multi-stage behaviors. Our model incorporates Internet-scale vision-language model (VLM) pre-training with flow matching for representing complex high-frequency action chunks. Our pre-training mixture consists of 10,000 hours of dexterous manipulation data from 7 different robot configurations and 68 tasks, in addition to large amounts of previously collected robot manipulation data from OXE [10], DROID [23], and Bridge [52]. To our knowledge, this represents the largest pre-training mixture ever used for a robot manipulation model. Our fine-tuning experiments include over 20 tasks, where we show that our model outperforms a variety of baselines, including prior VLA models [24] and models designed specifically for dexterous manipulation [57, 9]. We also examine how our post-training recipe can enable highly complex tasks, such as folding multiple articles of clothing from arbitrary initial configurations or assembling boxes.

> 💡 **批注**: π₀ 成果总结：
> - **最大规模**预训练混合数据（10,000 小时 + OXE/DROID/Bridge）
> - **20+ 下游任务**验证，全面超越 baseline
> - 展示了前所未有的**灵巧操作复杂度**

---

Our framework broadly resembles the training procedures employed for large language models, which typically consist of pre-training a base model on very large datasets scraped from the web, followed by a post-training procedure that aims to "align" the model to enable it to follow instructions and perform user commands. It is generally recognized that most of the "knowledge" in such models is acquired in the pre-training phase, while the post-training phase serves to tell the model how it should leverage that knowledge to fulfill user commands. Our experiments imply that an analogous phenomenon might take place with robot foundation models, where pre-trained models have some zero-shot capabilities, but complex tasks like laundry folding require fine-tuning with high-quality data. Training on only this high-quality data results in a brittle model that does not reliably recover from mistakes, while running the pre-trained model in zero shot does not always exhibit the fluent strategies demonstrated in the post-training data.

> 💡 **批注 — LLM 类比的深层含义**:
> - Pre-training = 知识获取（物理世界的"常识"）
> - Post-training = 行为对齐（如何优雅地完成任务）
> - 两者互补：pre-training 提供恢复能力，post-training 提供策略流畅性
> - 这可能是机器人领域的 "GPT moment"

---

We hope that our results will serve as a stepping stone toward general and broadly applicable robot foundation models. Our experiments suggest that such models may soon be a reality, but there are a number of limitations and ample room for future work. First, our experiments do not yet provide a comprehensive understanding of how the pre-training datasets should be composed: we combined all data available to us, but understanding what type of data is more helpful to add and how it should be weighted remains an open problem. Not all tasks in our evaluation work reliably, and it remains unclear how to predict how much and what kind of data is needed to attain near-perfect performance. Finally, it remains to be seen how much positive transfer there is in combining highly diverse data, particularly from different tasks and different robots: although our results suggest that universal pre-trained robot foundation models might become a reality, it is left for future work to understand whether this universality extends to much more distinct domains, such as autonomous driving, navigation, and legged locomotion.

> 💡 **局限性总结**:
> 1. **数据配比不明确**: 混合了所有可用数据，但最优配比未知
> 2. **不是所有任务都可靠**: 性能因任务而异
> 3. **数据需求不可预测**: 不知道需要多少/什么类型的数据才能达到近完美性能
> 4. **跨域迁移未验证**: 是否能推广到自动驾驶、导航、足式运动？
>
> 💡 **未来方向**:
> - 数据 scaling law for robotics（类似 Chinchilla for LLM）
> - 更广泛的 embodiment（不止操作臂）
> - 更系统的预训练数据配比研究

---

## 🔖 Section 总结

### 核心洞察
1. π₀ 验证了 **LLM 范式（pre-training + post-training）在机器人领域的可行性**
2. 知识在预训练中获取，行为在 post-training 中对齐 — 与 LLM 的发现一致
3. 主要局限在于对**数据配比和 scaling 规律**的理解不足
4. 通用性的边界未知：操作 → 驾驶/导航/运动 的迁移是开放问题
