[← 返回 README](../README.md)

# 3. Problem Formulation

## 符号定义

现代通用策略 π 的输入输出：

```
输入：ot = [It¹, ..., Itⁿ, qt]
         ↑                ↑
     n 个相机视角     机器人 arm pose（EEF 位姿）

指令：l（语言）

输出（动作块）：
at+1, at+2, ..., at+H ~ π(·|ot, l)
```

---

## World Model 的目标

给定当前观测 `ot` 和策略输出的动作块 `At = [at+1, ..., at+H]`，预测未来多视角观测：

```
ot+1, ..., ot+H ~ W(·|ot, At)
```

然后把最终预测 `ot+H` 送回策略 π，产生下一个动作块，实现**自回归的闭环 rollout**：

```
ot → π → At → W → ot+H → π → At+H → W → ...
```

---

## 关键约束

World Model W 必须满足：
1. **多视角**：输出和输入格式匹配现代 VLA（第三人称 + 腕部）
2. **可控性**：精确跟随每一步动作（fine-grained action control）
3. **长时一致性**：多步自回归不发散

---

## 💡 批读注解

**action 到底是什么格式？（重要！我们项目的相关问题）**

论文在 Section 4.1 中提到：
> "We also transform each action sequence into Cartesian-space robot arm poses [a′t+1:t+H]"

即：policy 输出的 action 会被转换为**笛卡尔空间的 EEF pose**（绝对位姿）再喂进去。

这跟我们分析的代码一致：Ctrl-World 实际上是 **state-conditioned**（用 EEF pose），而不是真正的 delta action-conditioned。

**这意味着：**
- 跟 VLA policy 联动时，需要把 policy 的 delta action 积分成绝对 EEF pose 序列
- 这是 Ctrl-World 在工程上的一个隐藏约束
