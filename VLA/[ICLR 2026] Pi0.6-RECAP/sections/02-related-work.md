[← 返回 README](../README.md)

# II. Related Work

## 📌 预览
Related Work 梳理了五条线：(1) intervention-based 改进 (2) RL for manipulation (3) VLA 的 RL fine-tuning (4) value function + end-to-end RL for VLAs (5) reward/advantage conditioning 方法。RECAP 的核心区别：end-to-end offline RL + flow matching VLA + advantage conditioning（不需要 on-policy PPO）。

---

Policies trained with imitation learning are known to suffer from compounding errors [7] and, at best, can only be as performant as the demonstration data. The goal of this work is to improve the reliability and speed of vision-language-action policies by going beyond imitation learning from offline demonstrations. Prior works have used online interventions to improve robotic manipulation policies [8–11]. We adopt a form of such interventions, called human-gated DAgger [10, 12]. In contrast to these works, our method uses both expert interventions and fully autonomous experience, resulting in an RL-based framework that integrates multiple data sources. There is a large body of work on using RL for autonomous improvement of robotic manipulation policies [13–21], including methods using diffusion-based policies [22–24], in multi-task settings [25, 26], and using pre-trained multi-task policies [27–29]. Unlike these works, we study how to scale real-world RL to large VLA policies for long-horizon, fine-grained manipulation tasks.

> 💡 **第一段批读**:
> - Imitation learning 的根本问题：compounding error + 性能上限 = demo 质量
> - 两条路线的整合：intervention（DAgger 系列）+ autonomous RL experience
> - RECAP 的定位：不只是 intervention，也不只是 RL，而是二者结合

---

Many recent works have studied how to improve a base VLA model through RL. Several works directly apply the proximal policy optimization (PPO) algorithm and variations thereof to VLA fine-tuning [30–34], yielding approaches that are difficult to extend to real-world RL in an efficient and scalable fashion. Another line of research has explored RL fine-tuning on top of pre-trained VLA models, where RL either trains a residual policy [35, 36], fine-tunes an action head network [37], selects or refines actions proposed by the VLA [38–40], or optimizes a policy acting in the noise space of a diffusion-based VLA [41]. Some of these works have also explored ways to distill the learned behavior back into the VLA for end-to-end iterative improvement [35, 36, 38, 42].

> 💡 **VLA + RL 方法分类**:
> - **Direct PPO**: [30-34] 直接对 VLA 做 PPO，但不好 scale 到真实世界
> - **Residual RL**: [35,36] 训练残差 policy，不改 VLA 本身
> - **Action head RL**: [37] 只微调 action head
> - **Action selection/refinement**: [38-40] VLA 提 proposal，RL 做选择
> - **Noise space RL**: [41] 在 diffusion noise space 做优化
> - RECAP 的区别：**end-to-end 训练整个 VLA**，用 advantage conditioning 避免 policy gradient 的复杂性

---

These prior works generally use discrete actions or simple Gaussian continuous action distributions. A critical distinction is that we train an entire VLA end-to-end using (iterated) offline RL, with an expressive flow matching VLA model. This is made possible by a simple and scalable advantage-conditioned policy extraction method, which removes much of the complexity of using policy gradient style objectives with large VLA models. In our comparisons, we show that this significantly outperforms a more traditional policy gradient based extraction scheme.

> 💡 **关键区分**:
> - 先前工作：discrete actions 或 simple Gaussian → 容易做 policy gradient
> - RECAP：flow matching VLA（没有 tractable log-likelihood）→ 不能直接用 PPO
> - 解决方案：advantage conditioning = 把 RL 问题转化为 conditional supervised learning

---

More closely related to RECAP in terms of methodology, a number of prior works have integrated value functions and end-to-end RL training of VLAs on real robots [43–46]. For example, Huang et al. [43] apply calibrated Q-learning to an offline demonstration dataset for grasping tasks, without an online improvement phase. Zhang et al. [44] use direct preference optimization (DPO) to optimize pick-and-place skills from human preferences, using online rollouts from a VLA. Finally, Zhai et al. [45], Ghasemipour et al. [46] use PPO and REINFORCE respectively with time-to-completion value functions to train VLAs for tasks like moving a bowl, unfolding a mat, and pushing objects on a table. In contrast to these prior works, we describe an iterated offline RL framework for VLAs with multiple advantages. First, our method supports high-capacity diffusion and flow-based VLAs, unlike the discrete-action models studied in prior works. Second, we avoid the need for on-policy PPO or REINFORCE by using an advantage conditioning strategy for policy extraction, which can utilize all prior (off-policy or offline) data. Lastly, our evaluation consists of complex, dexterous, and temporally extended tasks, where our method increases throughput by about $2\times$ while handling deformable objects, liquids, and multi-stage tasks.

> 💡 **最相关工作对比**:
> | 工作 | 方法 | 任务复杂度 | 局限 |
> |------|------|-----------|------|
> | Huang et al. [43] | Calibrated Q-learning | Grasping | 无 online 改进 |
> | Zhang et al. [44] | DPO | Pick-and-place | 简单任务 |
> | Zhai et al. [45] | PPO + time-to-completion VF | Bowl/mat/push | 简单任务 |
> | Ghasemipour et al. [46] | REINFORCE | 同上 | On-policy 需求 |
> | **RECAP** | Advantage conditioning | 咖啡/叠衣/装箱 | — |
>
> RECAP 三大优势：(1) 支持 flow matching (2) 不需要 on-policy (3) 复杂 long-horizon 任务

---

Prior works have explored the idea of conditioning the policy on rewards, values, and advantages [47–56], including methods that use classifier-free guidance [4]. We extend this approach to pre-train and fine-tune a large-scale generalist VLA policy [5], incorporating a variety of data sources (including demonstrations, interventions, and autonomous policy roll-outs) to learn real robotic manipulation tasks. Recent research has also studied how to effectively train multi-task, language-conditioned reward functions [57–63] and value functions [45, 64, 65]. Building on these works, we also train a language-conditioned distributional value function, which allows us to estimate state-action advantages for our advantage-conditioned VLA training framework.

> 💡 **两条技术线索的融合**:
> - **Reward/advantage conditioning**: Decision Transformer 系列 [47-56] + CFGRL [4]
> - **Language-conditioned value functions**: [57-65]
> - RECAP = 两者结合：language-conditioned distributional VF + advantage-conditioned VLA

---

## 🔖 Section 总结

### 核心洞察
1. RECAP 在方法论上最接近 CFGRL [4]，但将其扩展到大规模 VLA + 真实世界
2. 与 PPO-based VLA RL 相比，advantage conditioning 更 scalable（不需要 log-likelihood）
3. 与先前 VLA RL 工作相比，任务复杂度提升了一个量级（5-15 分钟 long-horizon vs 简单 pick-and-place）
