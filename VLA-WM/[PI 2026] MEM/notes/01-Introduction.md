# I. INTRODUCTION

Efficiently and effectively endowing robotic policies with memory requires multiple levels of abstraction. While in principle we could simply encode the entire sequence of past observations into the context of the policy, this becomes intractable for long tasks, necessitating either very short sequences or significant subsampling. However, in practical settings, the representation required for long- and short-term memory is likely to be very different. For example, the robot might need to remember a recent observation to handle occlusions, and it might need to remember that it has already added one of the ingredients when cooking a meal. But these memories are fundamentally different: the former might require storing a few images over a short time period, the latter might require long-term memory but only a few bits of information.

> 💡 **问题定义精准**：作者用一个非常直觉的例子说明了问题——记住"手臂遮挡前看到的物体"需要高保真的图像记忆（几秒内），而记住"已经加过盐了"只需要语义级别的信息（可以跨越几分钟）。这两种记忆本质上需要不同的表征。

> 📝 **为什么长任务很难——计算量爆炸问题**：
>
> 把 9000 张图塞进模型里是不可能的——显存爆炸、推理太慢。所以之前的方法只能二选一：
> - **"非常短的序列"**：只看最近几秒的帧，长期记忆直接丢掉 ❌
> - **"重要的子采样"**（subsampling）：9000 帧里每 100 帧取 1 帧，但这样精细动作的信息（比如手滑了）就看不到了 ❌
>
> 两种方法都是妥协。MEM 的做法是：**用视频编码器压缩短时帧**（保留细节），**用语言记录长期语义**（极度压缩但不丢重要信息），绕开这个两难困境。

An effective memory architecture for robot policies should use multiple modalities to represent memories at these different levels of abstraction. For short-horizon memory, dense image-based memory is well-suited to resolve occlusions and allows the robot to quickly adapt its manipulation strategy, e.g., by changing the grasp after failing to pick up an object. For long-horizon memory, we often only need to keep track of events at a semantic level, such as which ingredient has already been added to a dish. In this case, a language-based representation provides much better compression than raw observations, and allows us to store high-level memories over long time periods.

> 💡 **设计哲学**：短时记忆 = 视频（稠密、高保真），长时记忆 = 语言（压缩、语义级）。这个 factorization 非常合理，而且与人类记忆系统有一定类比——我们也不会记住过去一小时的每一帧画面，而是记住"做了什么"。

Based on these observations, we introduce Multi-Scale Embodied Memory (MEM), a system for equipping policies with multi-modal, long-horizon memory. MEM combines two key ingredients to make long-horizon memory tractable. First, we use a video encoder architecture to effectively encode multiple seconds of dense image-based memory into a compact representation. Second, we introduce a language-based memory mechanism in which the policy keeps track of semantic events in a compressed language format. This memory system can not only accommodate very long horizon tasks, but also enables a variety of new capabilities by leveraging the short-term memory, such as in-context adaptation to correct mistakes, and resilience to partial observability and self-occlusion.

> 💡 **两个关键组件**：
> 1. **Video Encoder**：把多秒的图像序列压缩成紧凑表示（短时记忆）
> 2. **Language Memory**：用自然语言追踪语义事件（长时记忆）
>
> 这样设计的好处是两者各司其职，不会互相拖累。

To evaluate MEM, we integrate it into the π₀.₆ model [34], a generalist VLA trained on a diverse mixture of robot, vision-language, and video data. We show that the resulting policy achieves state-of-the-art performance across a wide range of complex manipulation tasks. We also show that MEM enables our policy to solve long-horizon tasks like cleaning up a whole kitchen or preparing a grilled cheese sandwich, which require keeping track of memories for up to fifteen minutes.

> 💡 **基于 π₀.₆**：这是 Physical Intelligence 最新的 VLA backbone（基于 Gemma 3-4B）。MEM 是在这个强大的 base model 上叠加记忆能力。
>
> 💡 **任务难度很高**：清理整个厨房、做烤奶酪三明治——这些都是需要多步骤、长时间的真实世界任务，远超之前 VLA 的演示范围。

In summary, we introduce a system for multi-scale, long-horizon memory for robot policies. By effectively representing short- and long-horizon memory via video and language representations respectively, MEM allows robot policies to keep track of memories across tens of minutes, without sacrificing runtime constraints. To implement MEM, we use an efficient video encoder architecture and a language-based memory system, and demonstrate their effectiveness through state-of-the-art performance across diverse robot tasks. With MEM, we enable robot policies to perform complex tasks like cleaning up whole kitchens, which can span up to fifteen minutes.

> 💡 **不牺牲推理速度**：这是工程上的关键约束。机器人控制要求低延迟（几百毫秒内），所以不能简单地把所有历史帧都扔进 transformer。MEM 的设计在记忆能力和推理延迟之间取得了很好的平衡。
