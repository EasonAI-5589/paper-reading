[← 返回 README](../README.md)

# VI. Discussion and Future Work

## 📌 预览
总结 π0.5 的成果和局限性，提出未来方向：更复杂的提示、更丰富的上下文/记忆、更广泛的数据源探索。

---

We described $\pi_{0.5}$, a co-trained model that builds on the $\pi_0$ VLA to integrate a variety of data sources and enable generalization to new environments. The $\pi_{0.5}$ VLA can control mobile manipulators to perform tasks in homes that were never seen in the training data, cleaning kitchens and bedrooms, making beds, hanging towels, and performing other multi-stage and dexterous behaviors. $\pi_{0.5}$ is trained on about 400 hours of mobile manipulation data, but includes a much larger amount of data from other robots, including non-mobile manipulators in diverse environments and data collected under laboratory conditions. It is also co-trained jointly with data from the web, as well as high-level prediction data for outputting language commands based on robot observations. The generalization capabilities of $\pi_{0.5}$ demonstrate that this co-training recipe facilitates effective transfer, enabling highly generalizable control of a mobile manipulator with only a medium-sized mobile manipulation dataset.

> 💡 **成果总结**:
> - 400 小时移动操作数据 + 大量异构数据 → 全新家庭中的广泛泛化
> - co-training 是关键：不是靠堆积目标域数据，而是靠知识迁移

---

$\pi_{0.5}$ is not without its limitations. While our VLA exhibits broad generalization, it still makes mistakes. Some environments present persistent challenges (e.g., unfamiliar handles on drawers, or cabinets that are physically hard for the robot to open), some behaviors present challenges with partial observability (e.g., the robot arm occluding a spill that should be wiped), and in some cases the high-level subtask inference is easily distracted (e.g., closing and opening a drawer multiple times while putting away items). Addressing these challenges with better co-training, transfer, and larger datasets is a promising direction for future work.

> 💡 **三类局限**:
> 1. **物理挑战**: 陌生门把手、难开的柜子 → 需要更多操作多样性
> 2. **部分可观测**: 机械臂遮挡目标 → 需要更丰富的上下文/记忆
> 3. **高层推理分心**: 反复开关抽屉 → 需要更好的任务跟踪

---

Other future work directions could address the technical constraints of our method. While $\pi_{0.5}$ can perform a variety of behaviors to clean up kitchens and bedrooms, it processes relatively simple prompts. The complexity of the prompts that the model can accommodate is determined by the training data, and more complex preferences and instructions could be incorporated by producing more intricate and diverse annotations, either with human labelers or synthetically. The model also uses a relatively modest context, and incorporating richer context and memory could make the model significantly more capable in settings with more partial observability, such as tasks that require navigating between different rooms or remembering where objects are stored. More broadly, $\pi_{0.5}$ explores a particular combination of heterogeneous data sources, but the specific sources of data can be explored even more broadly. For instance, the ability of our system to learn from verbal instructions provides a powerful new supervision modality, and future work could explore this and other ways that people can provide robots with additional contextual knowledge. We hope that our work will serve as a foundation for a new generation of VLAs that exhibit broad generalization to diverse real-world environments.

> 💡 **未来方向**:
> 1. **更复杂的提示**: 当前提示简单（如"清洁厨房"），可通过更丰富的标注支持复杂偏好
> 2. **更长的上下文和记忆**: 跨房间导航、记住物品位置
> 3. **更广泛的数据源**: 语言指令是一个强大的新监督模态
> 4. **合成标注**: 用 AI 生成更多样的训练标注
>
> **个人评价**: 这些局限恰好指向了后续工作 π0.6/Hi Robot 等的方向

---

## 🔖 Section 总结

### 核心洞察
1. π0.5 证明了 co-training recipe 可以用**中等规模**的目标域数据实现广泛泛化
2. 主要局限来自物理多样性不足、部分可观测性、高层推理的稳定性
3. 语言指令作为监督模态的潜力巨大，值得进一步探索
