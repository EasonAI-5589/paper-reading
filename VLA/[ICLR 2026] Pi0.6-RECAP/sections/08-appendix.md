[← 返回 README](../README.md)

# Appendix

## 📌 预览
附录包含：贡献者列表(A)、额外 VF 可视化(B)、log-likelihood 推导(C)、PPO 实现细节(D)、CFG 推理(E)、算法超参数细节(F)。

---

## A. Contributions

Data collection and operations. Michael Equi, Chelsea Finn, Lachy Groom, Hunter Hancock, Karol Hausman, Rowan Jen, Liyiming Ke, Marinda Lamb, Vishnu Mano, Suraj Nair, Charvi Sharma, Laura Smith, Will Stoeckle, Anna Walling, Blake Williams.

Annotation and supplemental data. Chelsea Finn, Catherine Glossop, Hunter Hancock, Brian Ichter, Rowan Jen, Liyiming Ke, Chandra Kuchi, Karl Pertsch, Laura Smith, Will Stoeckle, Quan Vuong, Anna Walling.

Policy training and research. Ashwin Balakrishna, Kevin Black, Danny Driess, Michael Equi, Yunhao Fang, Chelsea Finn, Catherine Glossop, Karol Hausman, Gashon Hussein, Brian Ichter, Liyiming Ke, Sergey Levine, Yao Lu, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Alex Swerdlow, Marcel Torne, Quan Vuong, Lili Yu, Zhiyuan Zhou.

Policy infrastructure. Kevin Black, Karan Dhabalia, Danny Driess, Michael Equi, Liyiming Ke, Adrian Li-Bell, Suraj Nair, Allen Z. Ren, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Alex Swerdlow, Haohuan Wang, Ury Zhilinsky, Zhiyuan Zhou.

Robot hardware. Ali Amin, Raichelle Aniceto, Grace Connors, Adnan Esmail, Thomas Godden, Ivan Goryachev, Tim Jones, Ben Katz, Devin LeBlanc, Mohith Mothukuri, Sukwon Yoo.

Robot infrastructure. Ken Conley, James Darpinian, Jared DiCarlo, Karol Hausman, Szymon Jakubczak, James Tanner.

Writing and illustration. Kevin Black, Danny Driess, Michael Equi, Chelsea Finn, Hunter Hancock, Karol Hausman, Brian Ichter, Liyiming Ke, Sergey Levine, Suraj Nair, Allen Z. Ren, Laura Smith, Jost Tobias Springenberg, Zhiyuan Zhou.

> 💡 **团队规模**: Physical Intelligence 全公司级别的项目，50+ 贡献者，涵盖数据收集、标注、训练、基础设施、硬件、写作等

---

## B. Additional Value Function Visualization

![Figure 13](../images/dcf7f12aa7456b849d2923961baeb5aa75407a2660a0e377ec8713b2bfcc40b9.jpg)
*Fig. 13: Additional visualization of value function on five different tasks. Red parts highlight places where value drops, green parts highlight places where value increases, and yellow parts highlight oscillating value regions. Images show the corresponding frames and descriptions of the episode.*

> 💡 **Figure 13 批读**:
> - 5 个不同任务的 VF 可视化：espresso、box assembly、hang towel、attach hook 等
> - VF 能跨任务工作：检测进展（绿）、错误（红）、犹豫（黄）
> - 说明 multi-task distributional VF 的泛化能力

---

## C. Computing the log-likelihood for policy improvement

> 💡 **Appendix C 要点**: 推导 Eq. 5 的完整过程。核心思路：
> 1. 分解 likelihood = autoregressive (sub-task + discrete actions) × flow matching (continuous actions)
> 2. Flow matching 没有 closed-form likelihood → 用 one-step Gaussian approximation
> 3. 参考 [80] 得到 ELBO-style lower bound
> 4. 最终 loss = discrete CE + flow matching MSE（加权和）

---

## D. PPO implementation

> 💡 **Appendix D 要点**: PPO baseline 的实现细节
> - 用 SPO [83] 的约束替代标准 PPO clipping
> - 分别对 autoregressive 和 flow-matching 部分施加 trust region
> - Flow matching 部分的 trust region 难以 enforce → PPO 效果差的原因之一

---

## E. Using CFG for test-time policy improvement with β > 1

> 💡 **Appendix E 要点**: 推理时的 CFG
> - 用 conditional + unconditional model 的梯度差做 guidance
> - β 控制 guidance 强度（β=1 直接用 conditional，β>1 更 aggressive）
> - 实践中 β ∈ [1.5, 2.5] 比较好；太大会推到 action distribution 边界 → 动作过于激进
> - β 和训练时的 $\epsilon_\ell$ 都能 sharpen 分布，但作用时机不同

---

## F. Additional algorithm details

**Advantage Estimation:**
- Post-training: N=50 lookahead for n-step advantage
- Pre-training: N=T (full episode, higher variance but cheaper — single VF call)

**Advantage conditioning dropout:** 30% probability

**Advantage threshold $\epsilon_\ell$:**
- Pre-training: ~30% of demo data has positive advantage
- Fine-tuning: ~40% of eval rollouts have positive advantage
- Exception: T-shirt & shorts task uses ~10% (only top performance is positive)

**Dataset composition per task:**
| 任务 | Demo 数据 | Autonomous | Corrections |
|------|----------|------------|-------------|
| Laundry (simple) | — | 300/iter × 4 robots | 无 |
| Laundry (diverse) | — | 450 eval | 287 correction |
| Laundry (failure removal) | — | ~1000 autonomous | 280+378 corrections |
| Box assembly | 600 demo | 600 autonomous/iter | 360 corrections/iter |
| Cafe | — | 414 autonomous | 429 corrections |

> 💡 **数据量并不大**: 每轮几百条 trajectories 就够，说明 RECAP 的 sample efficiency 不错

---

## 🔖 Section 总结

### 核心洞察
1. Flow matching 的 log-likelihood 只能用 lower bound 近似，这是 PPO 不好用的根本原因
2. CFG (β>1) 提供额外推理时提升，但需要谨慎调参
3. 数据量适中（几百条/轮），RECAP 的 sample efficiency 合理
