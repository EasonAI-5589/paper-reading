[← 返回 README](../README.md)

# Appendix（附录）

## 📌 预览
附录覆盖六部分：A 额外实验设置、B 数据集详情、C 自动标注 Pipeline（★ 最重要）、D 补充实验、E Human-in-Loop 详细说明、F 未来工作。

---

## Appendix C：自动标注 Pipeline（★ 与数据生成任务直接相关）

### 整体流程

```
GT 轨迹 + 关键帧图像
    ↓
[已知] GT 子任务列表 + GT Sketch 坐标（bbox/arrow/point）
    ↓
Step 1：Temporal Reasoning（Figure 8）
    ↓
Step 2：Spatial Reasoning（Figure 9）
    ↓
训练数据：图像 + CoT 推理链 + 子任务描述 + Sketch 坐标 JSON
```

**关键设计思路：LLM 的角色是"反向构造推理链"，不是"自主规划"**

> GT 答案（子任务/Sketch 坐标）事先给定为 LLM 的"隐藏知识"，LLM 只需要构造合理的推理过程解释为什么这个答案正确——答案一定对，推理链质量也高。
>
> 对比自主规划：让 LLM 自己猜下一步可能猜错，生成的训练数据质量不稳定。

---

### Figure 8：Temporal Reasoning Prompt

**目的**：生成时序推理链（我做了什么 / 现在要做什么）

**SYSTEM_PROMPT 做了什么**：
1. 角色设定：你是机器人标注专家，处于任务执行中途
2. 隐藏知识注入：GT 完整计划 + 前一个子任务 + 当前子任务已告知
3. 规定推理步骤：Acknowledge Instruction → Analyze Scene → State Progress → Declare Current Action
4. 输出格式约束：必须用 `<think></think>` + `<answer>{current_subtask}</answer>`
5. 重要约束：有一张带 Visual Prompt 的参考图作为输入，但不能提及它的存在

**USER_PROMPT 做了什么**：
- 传入 3 张场景图 + 1 张带 Visual Prompt 的参考图
- 模板变量：`{previous_reasoning_output}` / `{all_subtasks_ordered}` / `{previous_subtask}` / `{current_subtask}`

**输出示例**：
```xml
<think>
  场景描述 + 我做完了什么 + 现在要做什么
</think>
<answer>当前子任务描述</answer>
```

---

### Figure 9：Spatial Reasoning Prompt

**目的**：生成空间推理链（为什么这样标 Sketch）

**SYSTEM_PROMPT 做了什么**：
1. 角色设定：机器人视觉专家，专门生成 Visual Prompt 标注
2. **隐藏知识注入**：GT Sketch 坐标（bbox/arrow/point 的具体像素坐标）已给定
3. 关键规则："你的目标不是发明新的标注，而是构造推理链解释为什么 GT 坐标是最优的"
4. 四步推理：解构子任务 → 分析场景 → 逐一解释每个 Sketch → 综合推理
5. Few-shot 示例（EXAMPLE）：完整展示输入输出格式

**USER_PROMPT 做了什么**：
- 传入 1 张场景图像
- 传入：子任务描述 + GT Sketch 坐标 JSON
- 输出：`<think>` 推理解释 + `<answer>` 原样输出 GT 坐标 JSON

**输出示例（倒茶任务）**：
```xml
<think>
  解构子任务"pour tea"：action=倒, target=茶壶, destination=杯子
  分析场景：茶壶在左上方，杯子在右下方
  point [167,56] = 壶嘴位置（出水起点）
  rotation_z_ccw [155,49] = 绕Z轴旋转让壶嘴对准
  star_point [170,74] = 杯口（目标落点）
  jagged_arrow [167,54 → 171,76] = 倒茶轨迹
</think>
<answer>{"point": [[167,56]], "rotation_z_ccw": [[155,49]], "star_point": [[170,74]], "jagged_arrow": [[167,54,171,76]]}</answer>
```

---

### Figure 8 vs Figure 9 对比

| | Figure 8 | Figure 9 |
|--|---------|---------|
| **目的** | 时序推理链（子任务进度） | 空间推理链（Sketch 标注理由） |
| **隐藏知识** | GT 子任务列表 | GT Sketch 坐标 JSON |
| **图像输入** | 3张场景图 + 1张参考图 | 1张场景图 |
| **输出 answer** | 当前子任务描述（文字） | GT 坐标 JSON（原样输出） |

---

## Appendix D.3：Human vs Automated Annotation

| 方法 | Stack | Hang | Empty | A2B-L | A2B-R | 平均 |
|------|-------|------|-------|-------|-------|------|
| 自动化 | 21.5% | 21.0% | 23.0% | 32.0% | 23.0% | 24.1% |
| 人工 | 34.5% | 25.0% | 28.0% | 43.0% | 28.0% | 31.7% |
| 差距 | +13.0 | +4.0 | +5.0 | +11.0 | +5.0 | **+7.6%** |

> 💡 **批注**：自动化 pipeline 平均差人工 7.6%，可行但有提升空间。这是我们这周 pipeline 设计要努力缩小的差距。

---

## Appendix D.1：Sub-task vs Task-level Sketching

逐子任务生成 Sketch 比整任务一次性生成效果好：

| 方法 | 平均成功率 |
|------|-----------|
| Task-Level（一次性生成整个任务的 Sketch）| 20.2% |
| Sub-Task（逐步生成，本文方法）| 31.7% |

> 💡 **批注**：任务越长，逐子任务方法的优势越大（Stack Blocks +25%）。一次性规划引入噪声，逐步规划可以聚焦当前子目标。

---

## Appendix D.4：推理速度

在单张 RTX 4090 上：
- Sketch 生成时间：每个子任务 3.5~6.4 秒额外开销
- 动作执行时间：与 π₀ 基本持平

总时间归一化后（SR/Time），Action-Sketcher 效率更高：Stack Blocks 是 π₀ 的 5.9 倍。

---

## Appendix E：Human-in-Loop 详细说明

两种干预方式（Figure 7）：

**Sketch Correction（纠错）**：
- 执行前人工检查 Sketch，发现不准确 → 调整坐标/关键点/箭头
- 工具：PyQt 标注界面
- 代价：每个子任务仅需 3-5 秒审查

**Intent Supervision（意图监督）**：
- 即使 Sketch 空间上准确，人也可以修改来改变机器人意图
- 用于注入个人偏好或安全约束
- 让机器人行为可以通过"画图"来实时控制，而不是调参数

---

## 对本周 Pipeline 任务的直接启示

1. **GT 坐标获取**：从仿真器读出 3D 轨迹关键点 → 用相机矩阵投影到 2D 得到 Sketch 坐标
2. **LLM 标注策略**：参考 Figure 8/9 的 prompt 设计，把 GT 坐标作为隐藏知识注入 → 让 LLM 反向构造推理链
3. **逐子任务**：不要一次性生成整个任务的 Sketch，逐步生成效果更好
4. **自动化差距**：论文自动 pipeline 比人工差 7.6%，我们的目标是缩小这个差距
