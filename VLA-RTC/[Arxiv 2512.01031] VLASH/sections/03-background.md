[← 返回 README](../README.md)

# 3 Background

## 📌 预览
Background 把后文所有符号先定下来：$H$ 是 prediction horizon，$K$ 是 execution horizon，$\Delta$ 是以控制步计的推理延迟。真正要记住的只有一句话：异步推理时，动作生成区间和动作落地区间并不重合。

---

Action chunking policy. We consider an action chunking policy $\pi _ { \theta } ( A _ { t } \mid o _ { t } , s _ { t } )$ [16, 31, 42], where $o _ { t }$ is the environment observation (e.g., image, multi-view visual input), $s _ { t }$ is the robot state (e.g., joint positions, gripper state), and $t$ is the controller timestep. At each timestep $t$ , the policy generates a chunk of future actions

> 💡 **符号定义**: $H$ 是一次预测多远，$K$ 是一次先执行多少，$\Delta$ 是推理延迟多少个控制步。后面所有关于异步稳定性的问题，本质上都来自这三个量之间的相对关系。

---

$$
A _ { t } = [ a _ { t } , a _ { t + 1 } , \ldots , a _ { t + H - 1 } ] ,
$$

where $H$ is the number of actions in the chunk. We refer to $H$ as the prediction horizon.

> 💡 **第一层定义**: $A_t$ 不是单步动作，而是从 $a_t$ 开始的一整段未来动作；$H$ 对应 prediction horizon，也就是模型一次向前规划的长度。接下来作者会再引入 $K$，把“预测多少”和“实际先执行多少”区分开。

---

$$
I _ { t } ^ { \mathrm { p r e d } } = \left[ t , t + K \right)
$$

as the time interval where the first $K$ actions from the action chunk $A _ { t }$ are planned to be executed. During actual execution, however, the $K$ actions from $A _ { t }$ will start being applied later due to inference latency [4, 31].

> 💡 **第二层定义**: 实际系统里通常不会把 $H$ 个动作全执行完，而是只先执行前 $K$ 个动作，再重新推理，所以 $K$ 是 execution horizon。于是模型在时刻 $t$ 生成 $A_t$ 时，默认对应的是 prediction interval $I_t^{\mathrm{pred}} = [t, t+K)$。

---

Let $\Delta > 0$ be the inference latency measured in control steps. Then the $K$ actions from $A _ { t }$ are actually executed on the robot over the execution interval

> 💡 **关键转折**: 真正的问题从这里开始。作者把推理延迟写成 $\Delta$ 个控制步，而不是毫秒，这样后面讨论错位时就不依赖具体控制频率，而是直接落在控制时序本身。

$$
I _ { t } ^ { \mathrm { e x e c } } = [ t + \Delta , t + \Delta + K ) .
$$

---

Asynchronous inference and interval misalignment. With asynchronous inference, the robot continues executing the previous action chunk while $\pi _ { \theta }$ computes $A _ { t }$ in the background. As illustrated in Fig. 2, when $\Delta \mathit { \Theta } > 0$ , the action chunk $A _ { t }$ is planned for the prediction interval $I _ { t } ^ { \mathrm { p r e d } } = \left[ t , t + K \right)$ but actually executed over the shifted execution interval $I _ { t } ^ { \mathrm { e x e c } } = \left[ t + \Delta , t + \Delta + K \right)$ . Intuitively, the actions in $A _ { t }$ are not wrong for the original prediction interval $[ t , t + K )$ . However, under asynchronous inference, by the time they are executed, the environment and robot state have changed, so the same action sequence is applied to a different state and scene, leading to unstable and discontinuous behavior [4, 29].

> 💡 **核心结论**: $A_t$ 这串动作本身并不是“预测错了”，它对原本的 prediction interval $[t, t+K)$ 是合理的；问题在于异步执行时，它真正落在了右移 $\Delta$ 步的 execution interval $[t+\Delta, t+\Delta+K)$ 上。于是同样的动作被施加到了一个已经变化过的 robot state 和 environment 上，这就是 lag、抖动和不连续的根源。

---

## 🔖 Section 总结

### 核心洞察
1. $H$ 决定一次预测多远，$K$ 决定一次先执行多少，$\Delta$ 决定时序错位有多严重。
2. 异步（async）控制的本质不仅仅是“重叠计算与执行以变快”，更是“预测区间和执行区间发生了物理分离”。
3. **对实时控制的意义**: 明确这三个变量，是分析所有异步策略的基石；任何改善实时控制表现的方法，最终都在试图填平这 $\Delta$ 步带来的信息鸿沟。
