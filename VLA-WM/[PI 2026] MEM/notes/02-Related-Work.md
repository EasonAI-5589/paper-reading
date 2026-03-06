# II. RELATED WORK

**Vision language action models with memory.** Recent works have demonstrated that learned robot control policies, trained on large amounts of diverse robot experience, can lead to generalizable manipulation, e.g., in unseen environments [19, 18, 42, 45, 30, 51]. Vision language action models (VLAs) [15, 8, 32, 31, 13, 22, 5, 48, 53, 2, 40, 33, 43, 49, 4, 50, 44] are a popular approach for training such generalist policies, in which a pre-trained vision-language model is finetuned with robot experience. While many of today's state-of-the-art VLAs are trained without memory and act purely based on the current observation of the environment [22, 18, 19, 42, 45, 30], a growing number of works have explored adding memory to policy training since it is a core requirement for solving a wide range of real-world tasks.

> 💡 **现状**：大部分 SOTA VLA（包括 OpenVLA、π₀、π₀.₅、Gemini Robotics 等）都是**无记忆**的——只看当前帧做决策。这篇论文正是要补上这个短板。

While early works explore architectures with recurrent memory modules [36, 27], more recent works that use transformer-based architectures simply pass a dense history of prior observations into the policy [37, 23, 31]; computational and latency constraints make it challenging to scale such approaches to support very long-horizon memory. Some works have explored latent memory architectures [38, 17, 12], but only evaluated on short-horizon memory tasks. Others have proposed various heuristics to compress memory information, e.g., by relying on purely proprioceptive memory [52], 2D point tracks [54, 9], by only retaining keyframes from prior timesteps [39, 47, 28], or by representing memory in natural language [25].

> 💡 **现有方案的困境**：
> | 方案 | 问题 |
> |------|------|
> | 稠密历史帧 | 计算量爆炸，延迟不可接受 |
> | 本体感受记忆 | 丢失环境视觉信息 |
> | 2D 点轨迹 | 丢失精确的空间信息（如抓取角度） |
> | 关键帧 | 需要激进稀疏化，丢失动态信息 |
> | 纯语言记忆 | 丢失精细的空间/视觉信息 |
>
> 没有一种单一模态能同时满足所有记忆需求——这正是 MEM 多模态方案的动机。

A challenge for all these approaches is that it is hard to find a one-modality-fits-all solution for robot memory, and each individual representation will result in a compromise on capabilities: proprioception, point traces, or natural language alone, for example, lose precise spatial information about grasp angle and height that is necessary to correct a slipped grasp. Keyframes, on the other hand, need to aggressively sparsify the observation history to make inference computationally feasible in long-horizon tasks, but may lose the ability to estimate environment and robot dynamics, which often requires more densely sampled observations.

In contrast to these prior works, we introduce a multi-modal memory system that combines short-horizon, dense vision-based memory with long-horizon, language-based memory. This allows us to resolve partial observability, perform in-context adaptation, and solve long-horizon tasks, all without sacrificing efficiency. Finally, an orthogonal line of work proposes approaches for mitigating causal confusion [11, 54] in which a policy with memory erroneously learns to copy over prior actions, e.g., by introducing auxiliary objectives [46]. While similar approaches could be combined with our memory system, our experiments suggest that we can achieve high policy performance without such objectives, which may be attributed to our large-scale and diverse training data.

> 💡 **Causal confusion**：这是给 policy 加记忆时的经典陷阱——模型可能学会"抄作业"（直接复制之前的 action），而不是真正利用记忆信息。MEM 声称大规模多样化训练数据能自然缓解这个问题，不需要额外的辅助目标。这个说法有一定道理，但也值得后续验证。

**Long-Context Models.** Outside of robotics, a large body of work has explored the training of long-context language and vision-language models [26, 41, 29]. In particular, in the context of video processing, there is a rich literature on designing efficient encoders for long video inputs [1, 3, 24]. Our work leverages similar ideas to [3, 1] for using sparse attention operations to efficiently process video inputs. Yet, in the context of robotics, latency constraints imposed by the real world require additional efficiency considerations: processing more than ten minutes of high-frequency, multi-frame video within a latency budget of a few hundred milliseconds is challenging, even on modern hardware. Thus, our work combines a short-horizon, video-based memory architecture with a significantly more compressed, language-based memory architecture for effective long-horizon context.

> 💡 **与 NLP/CV 长上下文工作的区别**：NLP 里做长上下文可以慢慢处理（几秒甚至几十秒），但机器人控制要求几百毫秒内出结果。所以不能简单把长视频理解的方法搬过来用，需要做 latency-aware 的设计。这也是为什么要把长时记忆压缩到语言而不是保留原始视频。
