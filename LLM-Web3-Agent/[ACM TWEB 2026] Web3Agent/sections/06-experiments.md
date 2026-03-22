[← 返回 README](../README.md)

# 6 Experiment

## 📌 预览
实验采用离线模拟框架（避免链上 gas 费用），评估三个维度：(1) Intent + Parameter 识别准确率；(2) 多步执行成功率；(3) 模块 ablation study。数据集覆盖 35 种 Web3 任务，每种生成 5 条自然语言指令。核心结果：Web3Agent 在 IPA 93.9%、PRA 89.6%、Query Task SR 94.1%、Operation Task SR 80.3%。

---

To rigorously evaluate Web3Agent's ability to autonomously perform complex blockchain operations, we adopt an offline simulation framework rather than direct deployment on live blockchains. Although Web3Agent has demonstrated operational feasibility in real on-chain environments, executing transactions on-chain incurs substantial gas fees, limiting scalability. Moreover, comprehensive testing requires control over both user-level variables—such as wallet balances, token types, and swap amounts—and blockchain-level dynamics, including slippage and gas prices, which are difficult to manipulate in live settings. Offline simulation thus provides a cost-efficient and controllable environment for evaluating robustness under diverse and edge-case scenarios.

> 💡 **离线模拟 vs 真实部署**: 采用离线模拟是一个务实的决策——链上操作的 gas 费用确实是实验的障碍。但这引入了一个根本性问题：**模拟环境能多大程度代表真实链上行为？** 模拟中的 API 响应是预定义的，不存在真正的网络延迟、MEV 攻击、gas 价格波动等真实世界问题。作者在 Section 8.3 也承认了这个局限性，提出未来需要 testnet 和 mainnet 评估。不过对于一个探索性系统论文来说，离线评估的可控性和可重复性是合理的选择。

Building upon this framework, we conduct a comprehensive set of experiments to assess Web3Agent's performance across multiple dimensions of execution competence. Our evaluation spans the full task pipeline—from natural language understanding and intent recognition, to structured parameter retrieve, to the generation and execution of multi-step API plans. Specifically, we organize our evaluation into three components: (1) dataset construction that captures realistic Web3 usage patterns and linguistic variation; (2) intent and parameter retrieve to test language understanding capabilities; and (3) execution-level evaluation, including a detailed ablation study to isolate the contributions of key reasoning modules such as instruction chaining, calibration logic, and retrieval-augmented planning.

## 6.1 System Implementation

To implement Web3Agent, we deploy a modular RAG framework integrated with GPT-4 as the core reasoning engine. GPT-4 is selected for its superior instruction-following capabilities and real-time response latency, making it suitable for multi-turn interactions in dynamic Web3 environments.

> 💡 **实现细节不足**: 6.1 非常简短——只说了用 GPT-4，没有提到具体使用哪个版本（GPT-4-turbo? GPT-4o?）、向量数据库用什么（Pinecone? Chroma? FAISS?）、embedding 的维度、chunk 的数量和大小、top-k 检索的 k 值等关键实现细节。对于系统论文来说，这些细节对可复现性至关重要。

## 6.2 Dataset Construction

To systematically evaluate Web3Agent's ability across multiple stages of task execution—from intent recognition to multi-step API planning—we construct a comprehensive offline evaluation dataset. This dataset is designed to reflect realistic Web3 usage scenarios, incorporating diverse user expressions, parameter sparsity, and task complexity. All instances are annotated with ground-truth to support quantitative performance measurement.

The dataset covers a total of 35 representative Web3 tasks, as detailed in Section 3, and is divided into two categories: Query Tasks, which involve stateless, read-only operations such as retrieving token prices or checking approval status; and On-Chain Operation Tasks, which consist of multi-step, state-changing actions like token swaps, asset staking, and cross-chain bridging. This taxonomy ensures comprehensive coverage of user intents typically encountered in decentralized applications.

For each task, we generate five natural language instructions using GPT-4 to simulate variation in real-world user inputs. These samples cover a range of communication styles: fully specified formal requests (e.g., "Swap 100 USDC to ETH on Arbitrum"), informal dialogue-style prompts (e.g., "can u help me move 100 usdc to eth?"), and under-specified expressions with missing information (e.g., "Could you help me buy some eth?"). To systematically create sparse inputs, we apply a parameter dropout strategy: for each task, four of the five samples randomly omit each key parameter (such as token, chain, or amount) with a 30% probability. An example annotation is shown in Listing 1, illustrating a swap request with the amount field omitted. This challenges the agent's ability to infer or recover missing values from context and external tools.

> 💡 **数据集设计的优缺点**: 用 GPT-4 生成数据集然后用 GPT-4 来评估——这是一个潜在的"自我评价"偏差。模型可能更容易理解自己生成的语句风格。更理想的做法是收集真实用户的自然语言输入，或至少用不同的模型（如 Claude、Gemini）来生成评测数据。不过 30% 的 parameter dropout 策略是一个很好的设计——它测试了系统在信息不完整时的推理能力。35 种任务 × 5 条指令 = 175 条测试样本，规模不算大但覆盖了主要场景。

Each natural language instruction is manually annotated with a structured representation. This includes the intended operation type (one of the 35 task types), a complete set of ground-truth parameters, and a list of missing parameters specific to each input. These annotations provide a unified reference for evaluating both intent recognition and parameter retrieve performance.

```json
{
"utterance": "Could you help me swap some USDC to ETH on Arbitrum?",
"operation": "swap_tokens",
"paramsGroundtruth": {
"fromTokenAddress": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
"ToTokenAddress": "0xEeeeeEeeeEeeeEeeeEEEEEEEEEEEEEEEEEE",
"amount": "100",
"slippage": "0.5",
"userWalletAddress": "0x1234abcd5678ef90123456789abcdef012345678",
"chainIndex": "arbitrum"
},
"params_present": {
"fromTokenAddress": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
"ToTokenAddress": "0xEeeeeEeeeEeeeEeeeEEEEEEEEEEEEEEEEEE",
"slippage": "0.5",
"userWalletAddress": "0x1234abcd5678ef90123456789abcdef012345678",
"chainIndex": "arbitrum"
},
"params MISSING": [
"amount"
]
}
```

*Listing 1. Example of Annotated Swap Operation Input with Amount Omitted.*

To facilitate step-level and task-level execution analysis, each structured instance is further expanded into an executable action plan. This plan specifies the ordered API calls required to complete the task, along with the logical dependencies among steps (for example, verifying the validity of an address before querying its token balance, as shown in Listing 2). In addition, we simulate realistic API responses—including both success and failure cases—by injecting common error codes such as insufficient_funds, approval_required, and slippage_too_high. These simulated environments allow us to evaluate not just whether the system can generate the correct steps but also whether it can robustly respond to dynamic blockchain state and error handling conditions.

```json
{
    "task_id": "query_usdc_balance",
    "goal": "Query the USDC balance of address 0xABC on Ethereum",
    "steps": [
        "step_id": "step1 Validate_address",
        "api": "GET /api/v5/wallet/pre-transaction/validate-address",
        "input": {
            "chainIndex": "1",
            "address": "0xABCDEF1234567890abcdef1234567890abcdef12"
       },
        "expected_output": {
            "code": "#",
            "data": [
                "addressType": "1",
                "hitBlacklist": false
            ]
        }
    ],
    {
        "step_id": "step2_query_usdc_balance",
        "api": "POST /api/v5/wallet/asset-token-balances-by-address",
        "depends_on": "step1Validate_address",
        "condition": "addressType == '1' and hitBlacklist == false",
        "input": {
            "address": "0xABCDEF1234567890abcdef1234567890abcdef12",
            "tokenAddresses": [
                "chainIndex": "1",
                "tokenizerAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
            ]
        }
    }
}
```

![Listing 2 Figure](../images/bb76ab57eefd27df8835c4eb3f0fd47d557fcb3ba0bdfcd7615d535088420a11.jpg)

*Listing 2. Case: Query USDC Balance for a Valid Address.*

> 💡 **Action Plan 的结构设计**: Listing 2 展示了 step 间的依赖关系（depends_on）和条件执行（condition），这种设计类似于 DAG（有向无环图）执行引擎。condition 字段（如 `addressType == '1' and hitBlacklist == false`）允许系统在地址无效或被列入黑名单时跳过后续步骤。错误注入策略（insufficient_funds、approval_required、slippage_too_high）测试了系统的异常处理能力——这在真实 Web3 场景中极为重要。

## 6.3 Intent and Parameter Retrieve Evaluation

We evaluate Web3Agent's language understanding performance using the dataset described in Section 6.2. Each natural language instruction is paired with ground-truth annotations for both the task intent and its corresponding structured parameters. Given an input utterance, the agent is expected to output:

— An operation type (e.g., SwapToken, BridgeAsset).

— A structured parameter set that recovers all necessary fields, even if some were missing in the original input.

**Metrics.** As shown in Table 1, we adopt two standard metrics for this task: (1) Intent Parsing Accuracy (IPA): The percentage of utterances for which the predicted operation type exactly matches the ground truth. (2) Parameter Retrieve Accuracy (PRA): The percentage of required parameters correctly extracted or inferred in the output, averaged across all inputs. We compare the following systems:

<table><tr><td>System</td><td>IPA</td><td>PRA</td></tr><tr><td>GPT-4 (zero-shot)</td><td>83.3%</td><td>37.1%</td></tr><tr><td>GPT-4 w/o RAG</td><td>88.9%</td><td>65.4%</td></tr><tr><td>Web3Agent (ours)</td><td>93.9%</td><td>89.6%</td></tr></table>

*Table 1. Intent and Parameter Retrieve Accuracy Comparison*

Overall, Web3Agent significantly outperforms both GPT-4 baselines, achieving the highest accuracy in both intent parsing (93.9%) and parameter retrieval (89.6%). This demonstrates the effectiveness of combining structured prompts with RAG for handling complex Web3 tasks.

> 💡 **PRA 的巨大提升**: 最值得关注的数字是 PRA 从 37.1%（zero-shot）到 89.6%（Web3Agent）——52.5 个百分点的提升。这说明 GPT-4 虽然能理解用户意图（IPA 83.3% 已经不错），但在提取 Web3 特定参数（如 token 地址、链索引、滑点设置）方面严重依赖外部知识。RAG 从 operation chunks 中检索标准参数规范，使得系统能够填补用户输入中缺失的参数。这也解释了为什么 Web3Agent 在 ablation 中去掉 operation chunks 后性能暴跌——这些 chunks 不仅用于规划，也用于参数补全。

Breaking down the results, we observe that Intent Parsing Accuracy (IPA) remains relatively stable across models, ranging from 83.3% in the zero-shot setting to 93.9% with Web3Agent. This modest improvement suggests that most user queries contain explicit intent cues that even general-purpose LLMs can identify.

In contrast, PRA exhibits a much larger gap. GPT-4 (zero-shot) achieves only 37.1% PRA due to frequent omissions and hallucinations of parameters (e.g., defaulting to "ETH" or "mainnet"), while GPT-4 with prompt engineering (w/o RAG) improves moderately to 65.4%. Our Web3Agent, equipped with task-specific retrieval, substantially outperforms both, achieving 89.6% PRA by grounding each input against operation-specific argument schemas.

Further error analysis reveals that in 73% of intent classification failures, the input was missing one or more critical parameters (e.g., chain, token, or action verb), which led to misclassification—such as confusing SwapToken with BridgeAsset. This confirms that intent recognition is tightly coupled with parameter recovery, and thus motivates their joint evaluation as a unified metric of task understanding.

> 💡 **Intent-Parameter 耦合**: 73% 的意图分类错误与参数缺失相关——例如当用户没有指定链时，swap 和 bridge 在语义上确实很难区分。这个发现很有价值，说明**意图理解不能脱离参数恢复单独评估**。但反过来看，这也意味着如果 Web3Agent 的 RAG 检索出了错误的 operation chunk，整个下游流程都会出错。

## 6.4 Ablation Study

To better understand the contribution of individual components in Web3Agent's reasoning and planning pipeline, we conduct an ablation study focused on execution accuracy. We evaluate the model variants under two dimensions: (1) task type—query-based vs. on-chain operations, and (2) architectural modules—such as instruction guidance, prior action memory, calibration logic, and retrieval augmentation.

**Metrics.** We adopt two execution-focused metrics:

— Step Success Rate (Step SR): A step is considered successful if the system generates the correct API call (endpoint, method, parameters) and handles the API response appropriately, including retrying or adjusting when necessary [6].

— Task Success Rate (Task SR): A task is successful if all required steps are executed correctly in sequence, with appropriate condition handling, resulting in a valid final action plan [44].

<table><tr><td>System Variant</td><td>Task Type</td><td>Step SR</td><td>Task SR</td></tr><tr><td rowspan="2">w/o Instruction Chain Generator</td><td>Query</td><td>76.6%</td><td>80.9%</td></tr><tr><td>Operation</td><td>15.2%</td><td>48.3%</td></tr><tr><td rowspan="2">w/o Previous Action Description</td><td>Query</td><td>85.2%</td><td>70.1%</td></tr><tr><td>Operation</td><td>52.5%</td><td>59.4%</td></tr><tr><td rowspan="2">w/o Calibration</td><td>Query</td><td>89.6%</td><td>86.8%</td></tr><tr><td>Operation</td><td>78.3%</td><td>71.5%</td></tr><tr><td rowspan="2">Full Web3Agent</td><td>Query</td><td>96.8%</td><td>94.1%</td></tr><tr><td>Operation</td><td>91.2%</td><td>80.3%</td></tr></table>

*Table 2. Step and Task Success Rate by System Variant and Task Type*

Tables 2 and 3 reports the step-level success rate (Step SR) and task-level success rate (Task SR) across query and operation tasks under different module ablations. Overall, the Full Web3Agent configuration achieves the best performance across all dimensions, with a Step SR of 96.8% and Task SR of 94.1% for query tasks, and 91.2% / 80.3% respectively for operations. These results confirm the effectiveness of combining instruction chaining, feedback-aware planning, and calibration in supporting robust end-to-end execution.

Removing the Instruction Chain module leads to the most significant performance degradation, especially for operation tasks (Step SR drops to 15.2%, Task SR to 48.3%), as the model fails to generate coherent multi-step execution plans. For query tasks, task success remains relatively high (80.9%) due to the simplicity of single-step API calls. However, the Step SR drops sharply to 76.6%, as our evaluation strictly compares the generated API sequence against the ground-truth. For example, retrieving token prices typically requires a prior token validation step. Without instruction chaining, such validation is often skipped, resulting in an incomplete execution sequence despite the task outcome being correct. This discrepancy highlights that Step SR measures procedural alignment rather than functional success.

> 💡 **Ablation 结果的关键发现**: 模块重要性排序非常清晰：Instruction Chain > Previous Action Description > Calibration。Operation 任务的 Step SR 从 91.2% 降到 15.2%（去掉 Instruction Chain 后）是本文最有冲击力的数字——说明没有结构化的步骤指导，LLM 几乎无法完成多步链上操作。同时 Step SR 和 Task SR 的差异也值得关注：去掉 Instruction Chain 后，Query 的 Task SR (80.9%) 远高于 Step SR (76.6%)——这说明系统虽然跳过了一些中间步骤，但最终结果可能仍然正确。这种"过程不正确但结果正确"的情况在实际应用中可能被接受，但在安全敏感场景下是不可取的。

Removing the Previous Action Description module causes moderate performance drops, particularly for operation tasks (Step SR: 52.5%, Task SR: 59.4%). The model struggles to interpret the semantic results of prior API calls, leading to incorrect follow-ups and execution breaks or inconsistency in parameters.

Removing the Calibration module results in comparatively minor impact among other modules. While Step SR and Task SR for operations decline slightly to 78.3% and 71.5%, most plans remain functionally sound. This suggests that while calibration improves logical consistency, it is less critical for task-level success than instruction chaining or previous action interpretation.

**Fine-grained RAG Ablation.** To further understand the impact of RAG, we decompose it into three subcomponents and disable each independently:

<table><tr><td>System Variant</td><td>Task Type</td><td>Step SR</td><td>Task SR</td></tr><tr><td rowspan="2">w/o operation chunks</td><td>Query</td><td>51.4%</td><td>63.3%</td></tr><tr><td>Operation</td><td>10.1%</td><td>25.6%</td></tr><tr><td rowspan="2">w/o error chunks</td><td>Query</td><td>92.3%</td><td>87.9%</td></tr><tr><td>Operation</td><td>81.4%</td><td>74.7%</td></tr><tr><td rowspan="2">Full Web3Agent</td><td>Query</td><td>96.8%</td><td>94.1%</td></tr><tr><td>Operation</td><td>91.2%</td><td>80.3%</td></tr></table>

*Table 3. Step and Task Success Rate by System Variant and Task Type*

Removing Operation Chunks led to a notable decline in Web3Agent's Task SR. Specifically, when the prompt lacked structured references to the expected sequence of steps, the agent became more prone to reasoning errors and exhibited increased hallucination behavior. This included generating non-existent steps, fabricating parameter fields, or producing ill-formed API calls that deviated from the intended operation structure. This phenomenon is consistent with known limitations of large language models when tasked with multi-step procedural reasoning. Without grounded priors or constraints, it often over-generalize based on previous distribution heuristics. While the omission of execution order information significantly impacted task-level accuracy, its effect on step-level correctness was comparatively limited. However, even at the step level, the absence of structured parameter descriptions, such as those found in the operation chunk, led to a drop in Step SR. Without explicit references to required fields, types, or constraints, the agent frequently hallucinated parameter names or omitted critical inputs.

> 💡 **Operation Chunks 是系统的基石**: 去掉 operation chunks 的影响（Operation Task SR: 25.6%）甚至比去掉 Instruction Chain 模块本身（48.3%）还严重！这说明 Instruction Chain 模块的有效性**高度依赖** operation chunks 提供的标准化工作流。没有这些 chunks，LLM 会"幻觉"出不存在的步骤和参数——这在区块链环境中可能导致灾难性后果（调用错误的合约地址、发送到错误的链等）。Table 3 没有单独测试去掉 API chunks 的影响，这是一个遗漏。

Removing the Error Chunk module results in a moderate performance drop, particularly on operation tasks. While query performance remains relatively stable (Step SR: 92.3%, Task SR: 87.9%), the accuracy for operation tasks declines noticeably (Step SR: 81.4%, Task SR: 74.7%). This confirms that error chunks play a crucial role in interpreting blockchain-specific failure signals (e.g., insufficient_funds, approval_required, slippage_too_high) and guiding fallback or corrective actions. Compared to the system without calibration, the slightly higher step-level accuracy suggests that GPT-4 is able to implicitly compensate for the absence of explicit error-handling logic.

> 💡 **Error Chunks 的中等重要性**: Error chunks 的影响（Operation Task SR: 74.7% vs 80.3%）比 operation chunks 小得多，说明 GPT-4 对常见错误码有一定的内置理解能力。但对于 Web3 特有的错误（如 slippage_too_high、approval_required）仍需要外部知识来正确处理。有趣的是，去掉 error chunks 的影响与去掉 calibration 模块的影响相近——两者都是"锦上添花"而非"不可或缺"的组件。
