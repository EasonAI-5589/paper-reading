[← 返回 README](../README.md)

# 6 Experiments

## 📌 预览
离线模拟评估框架 + 自定义数据集 + 三维度评估（意图解析、参数恢复、执行成功率）+ 深入消融实验。

---

## 6.1 系统实现

- 核心 LLM: **GPT-4**（指令遵循能力 + 实时响应延迟）
- 模块化 RAG 框架集成
- 离线模拟而非真实链上部署（原因：Gas 费高昂 + 需要控制变量）

## 6.2 数据集构建

| 维度 | 详情 |
|------|------|
| 任务总数 | 35 个 Web3 任务 |
| 每任务变体 | 5 条自然语言指令 |
| 任务分类 | Query（只读查询）+ Operation（状态变更操作） |
| 标注内容 | ground-truth 意图 + 参数 + 可执行动作计划 |
| 错误模拟 | `insufficient_funds`, `approval_required`, `slippage_too_high` 等 |

每个实例扩展为：
- 有序 API 调用序列 + 步骤间依赖关系（如先验证地址再查余额）
- 模拟 API 响应（成功 + 失败场景）

> 💡 **批读**: 数据集规模偏小（35 任务 × 5 变体 = 175 条），但覆盖了 Web3 的核心操作场景。错误注入设计是亮点，能评估 Agent 的鲁棒性。

## 6.3 Intent and Parameter Retrieve 评估

### 指标
- **IPA (Intent Parsing Accuracy)**: 操作类型精确匹配率
- **PRA (Parameter Retrieve Accuracy)**: 参数正确提取/推断率

### 结果 (Table 1)

| 系统 | IPA | PRA |
|------|-----|-----|
| GPT-4 (zero-shot) | 83.3% | 37.1% |
| GPT-4 w/o RAG | 88.9% | 65.4% |
| **Web3Agent (ours)** | **93.9%** | **89.6%** |

> 💡 **批读**:
> - IPA 差距不大（83.3% → 93.9%），说明意图识别对通用 LLM 来说不难
> - PRA 差距巨大（37.1% → 89.6%），这是 RAG 的核心价值 — 通用 LLM 容易幻觉参数（默认 "ETH" 或 "mainnet"）
> - 73% 的意图分类失败源于关键参数缺失，说明 **意图识别和参数恢复是耦合的**

## 6.4 消融实验

### 模块消融 (Table 2)

| 系统变体 | Query Step SR | Query Task SR | Op Step SR | Op Task SR |
|---------|:---:|:---:|:---:|:---:|
| w/o Instruction Chain | 76.6% | 80.9% | **15.2%** ⬇️ | **48.3%** ⬇️ |
| w/o Previous Action Desc | 85.2% | 70.1% | 52.5% | 59.4% |
| w/o Calibration | 89.6% | 86.8% | 78.3% | 71.5% |
| **Full Web3Agent** | **96.8%** | **94.1%** | **91.2%** | **80.3%** |

关键发现：
- **Instruction Chain 最关键**: 去掉后 Op Task SR 暴跌至 48.3%，模型无法生成连贯的多步计划
- **Previous Action Description 次之**: 去掉后模型无法理解先前 API 调用的语义结果
- **Calibration 影响最小**: 但在不可逆链上操作中其价值更大

### RAG 消融 (Table 3)

| 系统变体 | Query Step SR | Query Task SR | Op Step SR | Op Task SR |
|---------|:---:|:---:|:---:|:---:|
| w/o Operation Chunks | **51.4%** ⬇️ | **63.3%** ⬇️ | **10.1%** ⬇️ | **25.6%** ⬇️ |
| w/o Error Chunks | 92.3% | 87.9% | 81.4% | 74.7% |
| **Full Web3Agent** | **96.8%** | **94.1%** | **91.2%** | **80.3%** |

关键发现：
- **Operation Chunks 是 RAG 的灵魂**: 去掉后 Op Task SR 从 80.3% → 25.6%，LLM 开始幻觉不存在的步骤和参数
- **Error Chunks 对操作任务更重要**: 帮助解释链特定的失败信号（insufficient_funds 等）并引导回退策略

> 💡 **批读**: 消融实验设计得很好，清楚展示了各模块的贡献。最大的发现是：**没有结构化的操作知识（Operation Chunks），LLM 在 Web3 多步任务上几乎无法工作**（Op Task SR 25.6%）。这说明纯 LLM 推理在专业领域的局限性。

---

## 🔖 Section 总结

### 核心洞察
1. **RAG > Prompt Engineering > Zero-shot**: 参数恢复是拉开差距的关键维度
2. **Instruction Chain 是多步任务执行的命脉**: 没有它，Operation 任务成功率从 80.3% 暴跌至 48.3%
3. **Operation Chunks 是 RAG 的核心**: 去掉后 Agent 的多步推理能力近乎崩溃
4. Calibration 在模拟环境中影响较小，但在真实链上环境中的价值可能被低估
