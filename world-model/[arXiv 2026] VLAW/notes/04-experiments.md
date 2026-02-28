# Experiments

---

## 5.1. Experimental Settings

**Setups and Tasks.** We conduct experiments on the DROID platform (Khazatsky et al., 2024). In the DROID setup, a Franka Panda arm is equipped with a Robotiq gripper. Observations are captured using two third-person cameras and one wrist-mounted camera, as illustrated in Figure 4. We evaluate our method on five categories of contact-rich tasks, described below.

- **Stacking**: Four colored blocks are randomly placed on the table. The robot receives: "stack block A on block B," where A, B ∈ {red, green, blue, yellow}.
- **Open Book**: A book is randomly placed on the table. Evaluate across four different books. Instruction: "open the book cover."
- **Erase Marks**: One to three marker drawings on a whiteboard. Instruction: "erase all marks using a tissue."
- **Scooping**: Use a scoop to transfer snacks into a bowl. Both randomly placed. Instruction: "transfer some A to the bowl," where A ∈ {peanuts, candies, almonds}.
- **Drawing**: Draw a complete circle on a whiteboard using a marker.

> 💡 **任务选择的用心**：5个任务涵盖了接触丰富操作的主要难点：
> - **刚体接触**：Stacking（方块堆叠，需要精确放置）
> - **铰接物体**：Open Book（翻书，柔性约束）
> - **可变形/擦拭**：Erase Marks（纸巾擦白板，接触面变化大）
> - **颗粒物体**：Scooping（铲花生/糖果，颗粒力学）
> - **工具使用**：Drawing（用笔画圆，精细控制）
>
> 这些任务在传统仿真里都很难建模，展示了世界模型方案相对于基于仿真的方案的优势。

**Base Models and Hyperparameters.** We use π₀.₅ (Intelligence et al., 2025b) as the base VLA model and Ctrl-World (Guo et al., 2025a) as the base world model. For each task category, we collect 25 expert demonstrations and finetune π₀.₅ on this data to warm-start the policy, which serves as our base policy. The reward model is initialized from Qwen3-VL-4B-Instruct (Team, 2025a).

In each iteration, we roll out 50 trajectories per task category in the real world. We finetune the world model for 50K training steps using these rollout trajectories. We then generate 500 synthetic trajectories per task using the updated world model. The reward model is additionally finetuned using rollout data from the first iteration. The policy is updated with 2k steps with batch size 256. We perform a total of two iterations.

> 💡 **资源预算**：
> - 真实 rollout：50/任务/迭代 × 5 任务 × 2 迭代 = **500 个真实轨迹** 总量
> - 合成 rollout：500/任务/迭代 × 5 任务 × 2 迭代 = **5000 个合成轨迹**
> - 10:1 的合成/真实比例，这正是世界模型方案的价值所在
>
> 💡 **训练量**：世界模型 50K steps，策略 2K steps（batch 256）。策略更新步数相当少，说明 flow-matching SFT 在有好数据时收敛很快。

---

## 5.2. Can we learn an accurate action-conditioned world model for contact-rich tasks?

**Action replay inside the world model.** We evaluate the fidelity of the learned world model by replaying real-world action sequences inside the world model. Specifically, we randomly select a starting frame from a real-world trajectory and auto-regressively feed a 5-second sequence of recorded action chunks.

We compare against two baselines: the original pretrained world model and a model finetuned only on expert demonstration data.

We use two categories of metrics:

- **(1) Video distance metrics**: PSNR, SSIM (pixel-level); LPIPS, FID, FVD (perceptual/distributional)
- **(2) Interaction event confusion matrix**: Filter clips involving object interactions, classify each as success/failure, compare predicted vs. real outcomes.

### Table 1: Video Quality + Event Confusion Matrix

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ | TP↑ | FN↓ | TN↑ | FP↓ |
|--------|-------|-------|--------|------|------|-----|-----|-----|-----|
| Pretrained Ctrl-World | 16.32 | 0.634 | 0.347 | 41.03 | 225.13 | - | - | - | - |
| + Expert Rollout | 19.87 | 0.748 | 0.189 | 12.76 | 99.98 | 28 | 2 | 9 | **11** |
| + Expert + Online Rollout | **21.77** | **0.784** | **0.136** | **9.58** | **64.12** | 26 | 4 | **19** | **1** |

> 💡 **Table 1 的关键发现**：
>
> **视频质量全面提升**：Online rollout 微调后 PSNR +5.45, SSIM +0.15, FVD 从 225→64，说明世界模型对目标任务的建模精度大幅提高。
>
> **最重要的发现在 confusion matrix**：
> - 只用 Expert Rollout：TP=28, FP=**11** → 世界模型过度乐观，11 个失败案例被预测为成功
> - 加入 Online Rollout：TP=26, FP=**1** → FP 从 11 降到 1！过度乐观偏差被有效消除
> - 代价是 FN 从 2→4（漏掉了一些成功），但这个 tradeoff 是值得的——假阳性比假阴性危害更大（会给策略错误的"奖励信号"）
>
> 💡 **为什么 confusion matrix 比 PSNR 更重要**：视频看起来漂亮（高 PSNR）但物理结果预测错误，对 policy learning 来说完全没用。这个评估设计很赞——直接测试"世界模型能不能正确预测接触结果"。

**Policy-in-the-loop rollout.** We further evaluate the world model by rolling out the policy directly inside the learned model. The post-trained world model maintains high visual fidelity and physical plausibility even for long-horizon rollouts of up to 20 seconds. Example rollouts are shown in Figure 5.

> 💡 **Figure 5 展示了长期 rollout 稳定性**：20 秒（20 个 action step）的自回归生成不崩溃，这对于 video diffusion model 来说并不容易。Ctrl-World 使用的 diffusion forcing 技术可能在这里发挥了作用（缓解 compounding error）。
>
> 💡 **Figure 6 是 ablation 的定性支撑**：同一初始帧 + 同一动作序列，三种世界模型的预测对比：
> - Pretrained：太糊，物理不准
> - +Expert Only：过度乐观（本该失败的预测成了成功）
> - +Online Rollout：与真实结果一致

---

## 5.3. Can world model generated data improve VLA policy performance?

**Baselines:**

- **(1) Filtered BC**: Filter successful trajectories from real-world rollouts, perform SFT. Same rollout budget (50/category) for fair comparison.
- **(2) DSRL** (Wagenmaker et al., 2025): Improve π₀.₅ by optimizing its noise space through online exploration. Same rollout budget.

### Table 2: Detailed Success Rates

| Method | Stacking | Wiping | Open Book | Scooping | Drawing | Mean |
|--------|----------|--------|-----------|----------|---------|------|
| Base model | 0.62 | 0.46 | 0.56 | 0.44 | 0.22 | 0.460 |
| DSRL | 0.70 | 0.40 | 0.50 | 0.60 | 0.30 | 0.500 |
| Filtered BC-1 | 0.80 | 0.62 | 0.72 | 0.64 | 0.46 | 0.648 |
| Filtered BC-2 | 0.88 | 0.76 | 0.82 | 0.74 | 0.56 | 0.752 |
| Ours-1 | 0.80 | 0.72 | 0.80 | 0.72 | 0.68 | 0.744 |
| **Ours-2** | **0.92** | **0.86** | **0.86** | **0.92** | **0.78** | **0.868** |

> 💡 **结果分析**：
>
> **DSRL 几乎没用**：Mean 只从 0.460→0.500（+4%），在 Wiping 和 Open Book 上甚至下降了。原因分析：
> 1. Multi-task 设定下 RL 优化更难
> 2. DSRL 在 noise space 做优化不更新模型参数，表达能力受限
> 3. 说明"在 flow-matching 策略上直接做 RL"确实很难
>
> **Filtered BC 是强 baseline**：BC-2 达到 0.752，说明仅用真实成功轨迹做 SFT 就能大幅提升。这也是 π₀.₆* 的核心思路。
>
> **VLAW 的优势在第二轮迭代显现**：
> - Ours-1 (0.744) vs Filtered BC-1 (0.648)：+9.6%，世界模型合成数据的直接贡献
> - Ours-2 (0.868) vs Filtered BC-2 (0.752)：+11.6%，差距在扩大
> - 迭代效果累积，说明世界模型也在持续改进
>
> 💡 **Drawing 任务最难也改善最大**：Base=0.22 → Ours-2=0.78（+56%），而 Filtered BC-2 只到 0.56。画圆需要精细连续控制，世界模型能生成更多"画成功了"的轨迹用于训练。
>
> 💡 **Scooping 翻倍**：Base=0.44 → Ours-2=0.92（+48%），颗粒物体铲取是非常难的物理交互任务。

**Ablations (Figure 9):**

> 💡 **消融实验结论**：
> - 减少合成数据量（500→250）：性能下降 → 数据量重要
> - 去掉真实成功轨迹只用合成数据：进一步下降 → 真实数据不可替代
> - 说明合成数据是补充而非替代，两者互补

**Large-scale rollout visualizations (Figure 8):**

> 💡 **Figure 8 的直观解释**：从同一个初始帧出发，世界模型并行生成 15 个不同轨迹（编号 0-14）。真实世界里机器人失败了（GT 列），但在想象中的 15 个轨迹里有一些成功了——这些成功轨迹就可以用来教策略"如何纠正错误"。
>
> 这本质上是 **hindsight experience replay 的世界模型版**——把失败变成学习机会。

**Reward model analysis:**

> 💡 **Table 3 (Appendix C) 的关键数据**：
> - Direct Yes/No：FP=8/40（20% 假阳性率）
> - Threshold P(yes)>0.8：FP=2/40（5% 假阳性率），但 FN=12 vs 7
> - 精度从 65% 提升到 83%，代价是召回从 81% 降到 45%
> - 在这个场景下是正确的 tradeoff：少量但干净的正样本 > 大量但噪声的正样本
