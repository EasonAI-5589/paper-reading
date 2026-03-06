# V. CONCLUSION

We presented Multi-Scale Embodied Memory (MEM), an approach for equipping VLAs with long-horizon memory. MEM enables VLAs to perform long-horizon tasks that require tens of minutes of memory, while obeying real-world latency constraints. MEM also enables in-context adaptation of manipulation strategies and, when combined with the π₀.₆ VLA, achieves state-of-the-art performance across a wide range of manipulation tasks. To achieve this, MEM introduces a mixed-modal memory architecture that combines short-horizon, video-based memory, with a long-horizon, language-based memory mechanism. We believe that MEM is only the first step towards building robot policies that can effectively manage very long-horizon memory. Future work can explore how we can scale memory to last beyond the horizon of a single episode, to span weeks, months, or years of deployment, and allow us to build robots that learn continually at deployment time.

> 💡 **总结**：MEM = 短时视频记忆 + 长时语言记忆，赋予 VLA 15 分钟级别的任务处理能力，同时不牺牲推理速度和操作精度。
>
> 💡 **Future Work 的方向非常有野心**：从单 episode 的记忆 → 跨 episode 的持续学习（周、月、年）。如果实现，机器人就能真正"越用越好"——在部署中持续积累经验。这基本上是通向 AGI-level 机器人的路线图。
>
> 💡 **论文局限性**（作者没有明确提，但值得注意）：
> 1. **Language memory 的信息瓶颈**：纯文本无法描述所有空间信息（比如精确的物体位姿），所以 15 分钟前的精细空间记忆仍然会丢失
> 2. **LLM 标注的质量上限**：语言记忆的训练数据由 LLM 自动生成，标注质量受限于 LLM 的能力
> 3. **评估偏向厨房场景**：大部分任务都在厨房环境中，泛化到其他领域还需验证
> 4. **计算资源门槛**：基于 π₀.₆ + Gemma 3-4B，需要 H100 级别的 GPU，不太适合小型实验室复现
> 5. **没有与 MemoryVLA [38] 等同期工作做直接对比**
