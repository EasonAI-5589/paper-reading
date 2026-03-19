[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
Introduction 从 AI 系统越来越直接控制物理世界出发，引出实时性的核心矛盾：模型越大越强，但也越慢。然后分析 action chunking 的不足（仍有延迟问题），引出 RTC 的 inpainting 思路。

---

As AI systems have become more capable, they have also interacted more and more directly with their environment. Whether they're executing terminal commands [45], playing Pokémon on livestream [20], or browsing the web on your behalf [65], recent advances—driven primarily by large-scale deep learning—have enabled these systems to increasingly control, rather than merely process, the vast heterogeneity of the outside world. Embodied agents, where machine learning models directly control real, physical constructs, are perhaps the quintessential example. The same advances fueling agentic language and vision models are also making great strides in physical intelligence on platforms ranging from humanoid robots [4] to autonomous cars [60].

> 💡 **开场批读**: 作者先把 embodied AI 放在一个更大的叙事里——从执行命令、玩游戏到浏览网页，AI 正在从"处理信息"转向"控制世界"。引用了 Gemini 玩 Pokémon [20] 和 BrowseComp [65] 来说明这个趋势。Physical Intelligence 嘛，名字就说明了方向。

---

Cyber-physical systems, unlike chatbots and image generators, always operate in real time. While a robot is "thinking", the world around it evolves according to physical laws. Thus, delays between inputs and outputs have a tangible impact on performance. For a language model, the difference between fast and slow generation is a satisfied or annoyed user; for a robot action model, on the other hand, it could be the difference between a robot handing you a hot coffee or spilling it in your lap.

> 💡 **核心矛盾**: 这段话精确地指出了 embodied AI 和 chatbot 的本质区别——**物理世界不会等你**。LLM 慢一点用户只是不耐烦，但机器人慢一点可能就把咖啡洒了。这是整篇论文的动机。

---

Unfortunately, the effectiveness of modern large-scale machine learning comes with high latency as an unavoidable side effect. Large language models (LLMs), vision-language models (VLMs), and vision-language-action models (VLAs)—the last referring to a class of models designed for visuomotor control—have billions of parameters [8, 30, 5, 4, 58]. These models are not only slow to run, but also require heavy-duty hardware that is difficult to attach to edge devices such as mobile robots, adding even more overhead for remote inference. Edge hardware will improve over time, but as robot datasets grow in size, so will the best VLAs [28].

> 💡 **延迟不可避免**: 这里引用了 scaling law [28] 来论证：硬件会进步，但模型也会跟着变大。所以**延迟问题不会自己消失**，需要从算法层面解决。这个论点很有说服力——你不能指望等 GPU 变快来解决问题。

---

Thus, applying large models to real-time control problems effectively will require some form of asynchronicity: that is, a model must think about its future actions while executing a previous one. Action chunking [68, 33, 11], where a model outputs and executes a sequence of multiple actions for each inference call, presents a partial solution. Although action chunking has already achieved many state-of-the-art results in dexterous manipulation [5, 4, 58], it still suffers from the latency problem. Chunking sacrifices the reactivity of a system to external stimuli and also introduces discontinuities in the transition points between chunks, as adjacent chunks may jump between different modes (or "strategies") from the learned action distribution. Such anomalies are especially harmful to learning-based systems, as they produce a distribution shift in dynamics that the model is likely not equipped to handle. Naive smoothing strategies, such as averaging multiple predictions together [68], are not guaranteed to produce valid actions and may only make matters worse (e.g., see Figure 2).

> 💡 **Action chunking 的两难**: 这段分析得很到位：
> - **Execution horizon 长** → 不够灵活，对新信息反应慢
> - **Execution horizon 短** → chunk 边界频繁切换，容易 **mode-jumping**（上一个 chunk 想走左边，下一个想走右边）
> - **Temporal Ensembling** 看似能平滑，但**平均多个有效 action 不一定还是有效 action**（想想多模态分布的均值）
> 
> 关键概念 **mode-jumping**: 这在 multi-modal 策略中是个严重问题。比如绕障碍物可以走左边也可以走右边，两个连续 chunk 如果选了不同模式，中间的突变就是 OOD 的。

---

A good real-time system must produce a consistent and continuous control signal, incorporating the latest observations without perturbing the environment's natural dynamics or the model's ability to produce correct actions. In this work, we present real-time chunking (RTC), which poses asynchronous action chunking as an inpainting problem. Our algorithm generates the next action chunk while executing the previous one, freezing the actions that are guaranteed to be executed (due to inference delay) and "inpainting" the rest. It is applicable to any diffusion- [22] or flow-based [36] VLA, and operates purely at inference time, requiring no changes to existing training recipes.

> 💡 **RTC 核心思想**: "asynchronous action chunking as an inpainting problem"——把异步执行看成 inpainting 问题。已经被执行（或必将被执行）的 action 就是"已知像素"，需要生成的新 action 就是"被 mask 掉的区域"。这个类比非常自然，因为 diffusion/flow 模型本身就擅长 inpainting。

---

Our contributions are as follows. First, we present a novel system for asynchronous, real-time inference of action chunking diffusion- or flow-based policies for continuous control. Since standard simulation benchmarks are quasi-static—and have mostly been saturated with pseudo open-loop inference strategies [11]—we devise a new benchmark based on the Kinetix simulator [43] consisting of 12 highly dynamic manipulation and locomotion tasks. In the real world, we evaluate RTC on 6 challenging bimanual manipulation tasks using the π₀.₅ VLA [24] as the base policy. Across both simulation and the real world, we demonstrate that RTC is fast and performant; it is uniquely robust to inference latency, even in highly precise tasks such as lighting a match (Figure 1), and it achieves greatly improved task throughput on all real tasks.

> 💡 **贡献总结**:
> 1. **RTC 算法本身**: 推理时 inpainting 框架，适用于 diffusion/flow VLA
> 2. **新 benchmark**: 现有仿真 benchmark 太简单（quasi-static），被 open-loop 策略就能搞定了。Kinetix 的 12 个高动态任务更能检验实时性
> 3. **大规模真实实验**: 6 个任务 × 多种延迟配置 × 10 trials = 480 episodes，28 小时纯机器人执行时间。这个实验量在 VLA 领域算很大了。

---

## 🔖 Section 总结

### 核心洞察
1. **物理世界不等你** — 延迟对 embodied AI 的影响是致命的，不同于 NLP/CV
2. **Action chunking 是把双刃剑** — 解决了时序一致性，但引入了 chunk 边界不连续
3. **Temporal Ensembling 不靠谱** — 对多模态分布取平均可能产生无效 action
4. **RTC 的定位: 推理时算法** — 不改训练、不改模型，利用 flow/diffusion 的 inpainting 能力
