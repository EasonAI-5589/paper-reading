[← 返回 README](../README.md)

# 4 System Overview and Methodology

## 📌 预览
Web3Agent 的核心方法论：六模块流水线架构 + 领域专用 RAG + 形式化定义。这是论文最重要的技术章节。

---

## 整体架构（Figure 4）

用户输入 → **Intent Extraction** → **Instruction Chains** → **Action Prediction**（含 Previous Action Description + RAG）→ **Controllable Calibration** → **Executor** → 区块链网络

关键设计原则：
- **单 LLM 架构**: 一个通用 LLM (GPT-4) 完成所有推理任务
- **RAG 作为轻量知识路由器**: 不依赖多 Agent 或多 LLM
- **模块间 JSON 通信**: 确保透明性、模块化协调、鲁棒错误恢复

## 4.1 形式化定义

### Definition 1: Action（动作）
$$a = f(o, v)$$
- $o$: 目标对象（合约地址、Token 地址、交易端点）
- $v$: 关联值（如 Swap 金额）
- $f \in \{approve, swap, stake, transfer, input\}$
- 只有 `input` 类型需要具体值 $v$

### Definition 2: API State（API 状态）
$$s_i = \{status, data, error\}$$
- 区块链端点返回的结构化响应
- 包含钱包余额、授权状态、池信息、Gas 估算、错误码

### Definition 3: Candidate Set（候选动作集）
动作预测时的有界搜索空间，包含：
- Continue（继续下一步）
- Retry（重试当前步）
- Go Back（回滚重做）
- Interrupt（终止工作流）

> 💡 **批读**: 候选集的设计很巧妙 — 通过限制 LLM 的输出空间来减少幻觉。这比开放式生成可靠得多，尤其在高风险金融场景中。

## 4.2 领域专用 RAG（Figure 5）

三类语义分区 Chunk Store：

| Chunk 类型 | 内容 | 服务模块 |
|-----------|------|---------|
| **Operation Chunk** | 标准化工作流（如 SwapTokens、StakeAssets） | Instruction Chains Generator |
| **API Chunk** | API 端点参数结构和功能描述 | Action Prediction |
| **Error Chunk** | 已知失败码和恢复策略 | Action Prediction |

检索策略：
- 使用向量嵌入进行语义相似度搜索
- 元数据路由过滤：只检索与当前模块相关的 chunk 类型
- 参考 RePlug 和 DSPy 的最佳实践，但针对 Web3 语义定制

> 💡 **批读**: "按模块路由检索"是关键创新 — Instruction Chains Generator 只用 Operation chunk，Action Prediction 只用 API + Error chunk。避免了通用 RAG 中检索噪声污染 prompt 的问题。

## 4.3 Chatbot and Intent Extraction

- 解析自然语言为结构化任务表示
- 支持多轮对话消歧（如缺少链名时追问）
- 实体识别 + 语义模式提取 → 操作类型、资产符号、金额、目标网络

示例：
- 输入: "Swap 1000 $USDC to $ETH on Ethereum"
- 输出: `{operation: swap, fromToken: USDC, toToken: ETH, amount: 1000, chain: Ethereum}`

## 4.4 Instruction Chains Generator（Figure 6）

- 根据解析的意图，从 Operation Chunk Store 检索匹配的规范工作流
- 解析为显式指令链（instruction chain）
- 步骤示例：余额检查 → 授权处理 → 报价获取 → 交易构建 → 广播

> 💡 **批读**: 这个模块本质上是用 RAG 替代了硬编码规则。传统做法是 if-else 匹配操作类型，这里用语义检索来泛化 — 可以处理新的 Token 对、新链，只要 chunk store 中有类似的工作流。

## 4.5 Previous Action Description Generator

递归生成执行历史的自然语言摘要：
$$f_i = z(a_i, r_i, f_{i-1})$$

- 将 (动作, API 响应, 历史摘要) 递归压缩为人类可读的进度描述
- 为下游 Action Prediction 提供上下文
- 抽象掉底层 API 语义，保留决策相关信号

示例：`approve_USDC → success` → "Successfully approved 1000 USDC for swapping on Ethereum."

## 4.6 Action Prediction（Figure 7）

决策核心，输入 prompt 包含六部分：
1. 任务描述（用户意图）
2. 当前指令步骤（来自 instruction chain）
3. 历史动作序列
4. 自然语言历史摘要（来自 Previous Action Description）
5. API 响应（含成功输出和错误返回）
6. 候选动作集（有界搜索空间）

错误处理：当 Executor 返回错误时，结合状态码和上下文决定 Retry / Go Back / Skip / Interrupt。

## 4.7 Controllable Calibration

执行前安全门：
1. **逻辑一致性检查**: 动作是否符合指令链、是否尊重已执行步骤的顺序
2. **上下文可行性检查**: 动作在当前链上状态下是否可执行（如授权是否完成、余额是否充足）

通过 → Executor；不通过 → 返回 Action Prediction 重新预测

> 💡 **批读**: 这是高风险场景的关键保障。链上交易不可逆，Calibration 相当于一个 "double-check" 层。消融实验显示去掉它影响相对较小（Op Task SR 从 80.3% → 71.5%），但在真实链上环境中其价值会更大。

## 4.8 Executor

- 纯执行模块，不涉及 LLM 推理
- 将验证通过的动作编码为标准化 API 请求
- 返回结果归一化为统一 schema（status, data, error）
- 支持 dry-run / 模拟调用

---

## 🔖 Section 总结

### 核心洞察
1. **单 LLM + 模块化 RAG** 架构比多 Agent 系统更简洁，模块间通过 JSON 解耦
2. **候选动作集** 约束 LLM 输出空间，显著降低幻觉风险
3. **三类 chunk 分区路由** 是 RAG 在专业领域应用的有效范式
4. **递归历史摘要** 解决了长序列推理中的上下文维护问题
5. **Calibration 双重验证** 在不可逆操作场景中至关重要
