[← 返回 README](../README.md)

# II. Related Work

## 📌 预览
Related Work 从三个维度定位 π₀：(1) VLA 模型，(2) 扩散/flow matching 用于动作生成，(3) 大规模机器人学习。

---

Our work builds on recently proposed methods in large-scale robot learning, as well as multimodal language models. Our work is most closely related to recently proposed vision-language action (VLA) models, which use pre-trained VLMs that are fine-tuned for robot control [7, 24, 55]. Such models employ autoregressive discretization to represent actions in a manner analogous to text tokens. In contrast, our model employs a novel design that fine-tunes a VLM to produce actions via flow matching [32, 28], a variant of diffusion [20, 46]. This allows us to handle high-frequency action chunks [57] (up to $50 ~ \mathrm{Hz}$,) and highly dexterous tasks, which we show pose a major challenge for prior autoregressive VLAs [7]. This resembles a number of recent works on diffusion models for action generation [9, 60]. In contrast to these works, our model uses a pre-trained VLM backbone [5]. Our contribution is also fundamentally integrative, focusing on a framework for robot foundation models, including not only the model architecture itself but also a pre-training recipe, pre-training and post-training phases, and a range of real-world experiments.

> 💡 **批注**: π₀ vs 现有 VLA 的关键区别：
> | | RT-2 / OpenVLA | Diffusion Policy | **π₀** |
> |---|---|---|---|
> | 动作表示 | 自回归离散化 | 扩散 (连续) | Flow matching (连续) |
> | VLM backbone | ✅ | ❌ | ✅ |
> | Action chunking | ❌ | ✅ | ✅ |
> | 高频控制 | ❌ (低频) | ✅ | ✅ (50Hz) |
> 
> π₀ = VLM backbone + flow matching + action chunking，集三家之长。

---

Outside of robot control, many models have been proposed that combine pre-trained language models with diffusion [40, 41, 14], including models that specifically hybridize diffusion and autoregressive large language models [19, 29, 59]. Such models are typically concerned with image generation, but our action generation model builds on a number of previously proposed concepts. Like Zhou et al. [59], we train our model via a diffusion-style (flow matching) loss applied on individual sequence elements, in lieu of the standard cross-entropy loss for decoder-only transformers. Like Liu et al. [29], we use a separate set of weights for the tokens corresponding to diffusion. Incorporating these concepts into a VLA model, we introduce what to our knowledge is the first flow matching VLA that produces high-frequency action chunks for dexterous control.

> 💡 **批注**: 
> - **Transfusion [59]**: 在 transformer 的不同 token 上用不同 loss（cross-entropy vs diffusion）→ π₀ 借鉴此思路
> - **Playground v3 [29]**: 对扩散 token 使用独立权重 → π₀ 的 action expert 借鉴此设计
> - π₀ 的创新 = 把这些图像生成领域的技术引入 VLA

---

Our work also builds on a rich history of prior works on large-scale robot learning. Early work in this area often utilized self-supervised or autonomous data collection [26, 22, 8], providing a tractable data source for simple tasks such as grasping [18, 37] or pushing [56], but without the complexity of more dexterous behaviors. More recently, a number of high-quality datasets have been collected for robot control that enable broad generalization [23, 10, 52, 33, 34, 43, 13, 6], but typically for simpler tasks that consist of object relocation and rudimentary furniture manipulation (e.g., drawer opening) [31, 15]. More dexterous tasks have been studied at a smaller scale, typically with 10s or 100s of training trajectories [57], equivalent to 10 or less hours. Since one of our aims is to study complex and dexterous behaviors, we utilize a much larger dataset, with about 10,000 hours of demonstrations, complemented by the open-source OXE dataset [10]. To our knowledge, this represents by far the largest robot learning experiment in terms of the amount of robot data. At this scale, we show that a more sophisticated pre-training/post-training recipe is highly effective — analogously to the recipes used for large language models, a pre-training phase endows our model with a broad base of knowledge, which is then refined in a post-training phase with higher-quality curated data to achieve the desired behavior.

> 💡 **批注**: 数据规模的演进：
> - 早期：自监督抓取/推动 → 简单任务
> - 近期：OXE、DROID、Bridge 等 → 更广泛但仍简单（拿放、开抽屉）
> - ACT 等灵巧操作 → 小规模（10s-100s 轨迹，≤10 小时）
> - **π₀: ~10,000 小时** → 迄今最大规模的机器人学习实验

---

The complexity of the tasks we illustrate goes significantly beyond prior work. While recent work has illustrated a number of more complex and dexterous behaviors, such as tying shoelaces [58] or cooking shrimp [17], we show that our framework can learn very long tasks, sometimes tens of minutes in length, for behaviors that combine both physical dexterity and combinatorial complexity. For example, our laundry folding task requires the robot to manipulate a variety of clothing items that can start in any configuration, and fold multiple items in sequence. Our table bussing task requires discerning the class of novel objects (trash or dishes). We show that a single cross-embodiment model can be used as the base model for these tasks. To our knowledge, our work demonstrates the longest dexterous tasks in the end-to-end robot learning literature.

> 💡 **批注**: 
> - π₀ 的任务复杂度远超前作：**数十分钟长**的多阶段任务
> - 任务兼具**物理灵巧性**（折叠变形物体）和**组合复杂性**（识别不同物品类别）
> - 声称是端到端机器人学习文献中**最长的灵巧任务**

---

## 🔖 Section 总结

### 核心洞察
1. π₀ 处于 VLA + diffusion/flow matching + 大规模机器人学习的交汇点
2. 对比 RT-2/OpenVLA（自回归离散化）→ flow matching 支持高频连续动作
3. 对比 Diffusion Policy（无 VLM）→ VLM backbone 提供语义理解
4. 数据规模质的飞跃：10,000 小时 vs 之前的 ≤10 小时灵巧操作数据
5. 任务复杂度质的飞跃：数十分钟多阶段任务 vs 之前的简单拿放
