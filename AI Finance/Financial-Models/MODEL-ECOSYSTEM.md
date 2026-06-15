# Financial Reasoning Model Ecosystem

本页记录 FinChain 之外、截至 2026-06-15 可用于继续研究的金融推理模型、训练数据和奖励模型。重点不是罗列所有 Hugging Face 上传，而是判断哪些工作具备论文依据、官方资产和可复现价值。

## 结论先行

1. **开放资产最完整的新基线是 ODA-Fin-RL-8B**：Qwen3-8B 基座，公开 SFT 模型、RL 模型、318K SFT 数据和 12K RL 数据，均为 Apache-2.0。
2. **最值得形成方法创新的是 Fin-PRM**：金融推理开始从 outcome reward 转向 step-level process reward；模型与 4,969 条中文过程监督数据均已公开。
3. **Fin-R1 仍是最成熟的 FinChain 金融专项对照**：权重和训练代码公开，FinChain ChainEval 为 58.14，但完整训练数据开放程度弱于 ODA-Fin。
4. **FEVO 给出了更完整的训练配方**：CPT → SFT → RL，说明只在通用 instruct 模型上直接做 RL 不够；但尚未检索到官方公开权重或 FEVO-Train。
5. **Agentar-Fin-R1 强调可信数据和合规评测**：公开了 100K CoT 数据与 Finova benchmark，但尚未检索到 8B/32B 官方模型权重。
6. **StockLLM + FinSeer 是最直接可复现的金融 RAG 组合**：1.2B 生成模型与 109M 检索器的官方权重均已公开。
7. **Trade-R1 把过程奖励推进到随机市场**：不只看最终收益，而是验证检索证据、推理链和决策的一致性；但尚未检索到官方权重或代码。
8. **Open-FinLLMs 与 FinTral 补齐多模态训练谱系**：两者都覆盖表格或图像，但当前官方资产完整度不足，不宜直接作为首选复现基线。

## 模型与资产矩阵

| 工作 | 基座 / 规模 | 训练路线 | 模型权重 | 训练数据 | 代码 / 训练实现 | 当前用途 |
|------|-------------|----------|----------|----------|------|----------|
| [ODA-Fin-RL-8B](./%5BarXiv%202603.07223%5D%20ODA-Fin-RL-8B/) | Qwen3-8B | CoT SFT + GRPO | [🔗 SFT](https://huggingface.co/OpenDataArena/ODA-Fin-SFT-8B) / [🔗 RL](https://huggingface.co/OpenDataArena/ODA-Fin-RL-8B) | 318K SFT + 12K RL，开放 | [论文配置](https://arxiv.org/abs/2603.07223)：全参 SFT 16×A100；GRPO 8×A100；未发布代码 | 首选开放资产基线 |
| [Fin-PRM](./%5BarXiv%202508.15202%5D%20Fin-PRM/) | Qwen3 系 PRM | step + trajectory reward | [🔗](https://huggingface.co/DianJin/DianJin-Fin-PRM) | 4,969 条中文 PRM 数据，开放 | [🔗 Code](https://github.com/aliyun/qwen-dianjin/tree/master/DianJin-PRM)：TRL + VeRL | 过程监督、BoN、RL 奖励 |
| [Fin-R1](./%5BarXiv%202503.16252%5D%20FinR1/) | Qwen2.5-7B-Instruct | SFT + GRPO | [🔗](https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1) | 论文描述 60,091 条；完整开放性有限 | [🔗 Project](https://github.com/SUFE-AIFLM-Lab/Fin-R1)：仅 README/图片/报告，无训练脚本 | FinChain 金融专项强基线 |
| [Fin-o1](./%5BarXiv%202502.08127%5D%20Fino1/) | Llama-3.1 / Qwen2.5 / Qwen3，8B/14B | CoT SFT + PPO/DPO/GRPO | [🔗 8B](https://huggingface.co/TheFinAI/Fino1-8B) / [🔗 14B](https://huggingface.co/TheFinAI/Fino1-14B) / [🔗 Qwen3](https://huggingface.co/TheFinAI/Fin-o1-8B) | FinCoT 开放 | [论文配置](https://arxiv.org/abs/2502.08127)：比较 PPO、DPO、GRPO；未核实独立代码入口 | 比较基座迁移和数据复用 |
| [DianJin-R1](./%5BarXiv%202504.15716%5D%20DianJin-R1/) | Qwen2.5 7B/32B | 结构化 SFT + GRPO | [🔗 7B](https://huggingface.co/DianJin/DianJin-R1-7B) / [🔗 32B](https://huggingface.co/DianJin/DianJin-R1-32B) | DianJin-R1-Data 开放 | [🔗 Code](https://github.com/aliyun/qwen-dianjin/tree/master/DianJin-R1)：LLaMA-Factory + VeRL | 中英文金融与合规推理 |
| [RKEFino1](./%5BarXiv%202506.05700%5D%20RKEFino1/) | Qwen2.5-14B | 监管知识增强 SFT | [🔗](https://huggingface.co/YanAdjeNole/RKEFino1-14B) | 未见完整官方训练集 | [论文配置](https://arxiv.org/abs/2506.05700)：基于 Fino1，加入 XBRL/CDM/MOF；未发布代码 | 合规、XBRL、数值 NER |
| [Qwen-Open-Finance-R-8B](./%5BarXiv%202511.08621%5D%20Qwen-Open-Finance-R-8B/) | Qwen3-8B | 多语言金融指令微调 | [🔗](https://huggingface.co/DragonLLM/Qwen-Open-Finance-R-8B) | 未公开完整训练集 | [模型合集](https://huggingface.co/collections/DragonLLM/llm-open-finance)：未发布训练代码 | 多语言候选基线 |
| [FEVO](./%5BarXiv%202507.06057%5D%20FEVO/) | Qwen2.5-32B | CPT + SFT + RL | - | 未检索到 | [论文配置](https://arxiv.org/abs/2507.06057)：CPT→SFT→RL；未发布代码 | 方法与消融参照 |
| [Agentar-Fin-R1](./%5BarXiv%202507.16802%5D%20Agentar-Fin-R1/) | Qwen3 8B/32B | 可信数据 + 两阶段训练 | - | DeepFinance-100K 开放 | [🔗 Finova](https://github.com/antgroup/Finova) 仅评测；训练代码未发布 | 可信推理与合规评测参照 |
| [StockLLM + FinSeer](./%5BarXiv%202502.05878%5D%20StockLLM-FinSeer/) | Llama 1.2B + BERT 109M | 专用检索 + RAG 预测 | [🔗 LLM](https://huggingface.co/TheFinAI/StockLLM) / [🔗 Retriever](https://huggingface.co/TheFinAI/FinSeer) | 论文构建预测数据 | [论文配置](https://arxiv.org/abs/2502.05878)：生成模型微调 + 对比式 retriever；未发布完整代码 | 金融 RAG 首选小型基线 |
| [Trading-R1](./%5BarXiv%202509.11420%5D%20Trading-R1/) | 未公开 | SFT + 三阶段 curriculum RL | - | Tauric-TR1-DB 未公开 | [🔗 Project](https://github.com/TauricResearch/Trading-R1)：仍标注 releasing soon | 交易推理与风险奖励参照 |
| [Trade-R1](./%5BarXiv%202601.03948%5D%20Trade-R1/) | 未公开 | RAG 过程验证 + FSR/DSR | - | 未检索到 | [论文配置](https://arxiv.org/abs/2601.03948)：FSR/DSR；未发布代码 | 随机市场过程奖励参照 |
| [Open-FinLLMs](./%5BarXiv%202408.11878%5D%20Open-FinLLMs/) | FinLLaMA / FinLLaVA | CPT + instruct + multimodal | [🔗 Base](https://huggingface.co/TheFinAI/FinLLaMA) / [🔗 Instruct](https://huggingface.co/TheFinAI/FinLLaMA-instruct) | 52B token / 573K / 1.43M | [论文配置](https://arxiv.org/abs/2408.11878)：CPT→IFT→多模态；未核实完整代码 | 多模态训练谱系 |
| [FinTral](./%5BarXiv%202402.10986%5D%20FinTral/) | Mistral-7B | CPT + SFT + RLAIF/DPO + tools | - | 未检索到完整发布 | [🔗 Project](https://github.com/UBC-NLP/fintral)：内容不完整 | 多模态与工具方法参照 |

模型权重列中的 `-` 表示截至本页更新时间，在论文、作者 Hugging Face 组织和官方项目入口中未找到对应官方权重，不代表未来不会发布。

“代码 / 训练实现”列区分三种开放程度：`Code` 是官方实现仓库，`Project` 是尚不完整的项目页，`论文配置` 只表示论文公开了训练方法或算力配置，并不代表代码已发布。ODA-Fin-RL 当前仅确认公开了 [Data & Model collection](https://huggingface.co/collections/OpenDataArena/oda-finance)。

## 公开仓库实现基础

这些工作几乎都不是从零实现分布式大模型训练框架。真正的论文工程通常是在成熟框架上增加金融数据处理、提示模板、奖励函数和评测代码。

| 仓库 | 实现性质 | 上游基础 | 作者主要新增内容 | 可复现判断 |
|------|----------|----------|------------------|------------|
| [DianJin-R1](https://github.com/aliyun/qwen-dianjin/tree/master/DianJin-R1) | 完整项目适配 | SFT 使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，GRPO 使用 [VeRL](https://github.com/volcengine/verl)，另依赖 DeepSpeed、vLLM、EvalScope | 金融数据预处理、SFT/GRPO 启动脚本、格式与答案奖励、FSDP checkpoint 合并、CFLUE/FinQA 评测 | 当前最接近可直接复现的训练仓库 |
| [Fin-PRM](https://github.com/aliyun/qwen-dianjin/tree/master/DianJin-PRM) | 完整项目适配 | PRM SFT 基于 [TRL](https://github.com/huggingface/trl)，下游 GRPO 主要基于 [VeRL](https://github.com/volcengine/verl) | step/trajectory reward 数据与训练器、Fin-PRM 奖励接入、BoN、在线 RL 和离线数据筛选脚本 | 最适合改造成 FinChain-aware PRM |
| [Fin-R1](https://github.com/SUFE-AIFLM-Lab/Fin-R1) | 论文/模型展示仓库 | README 描述 Qwen2.5-7B-Instruct、SFT、GRPO 和 vLLM 推理 | 公开模型、数据构建说明、训练方法和推理命令 | 当前没有训练源码，不能仅凭仓库复现训练 |
| [Finova](https://github.com/antgroup/Finova) | Benchmark 评测仓库 | Python 评测脚本与模型 API/推理接口 | 数据、`run_scripts`、任务评测与结果汇总 | 能复现评测，不能训练 Agentar-Fin-R1 |
| [Trading-R1](https://github.com/TauricResearch/Trading-R1) | 占位项目 | - | 目前只有 README 和论文链接 | 不可复现 |
| [FinTral](https://github.com/UBC-NLP/fintral) | 占位项目 | - | 目前只有 README | 不可复现 |

### 对我们的工程选择

如果继续 ODA-Fin-RL / FinChain 路线，没有必要自写分布式训练底座。更稳妥的组合是：

1. 使用 LLaMA-Factory 或 TRL 完成 SFT/PRM 训练。
2. 使用 VeRL 完成 GRPO，并参考 DianJin 的奖励函数和 checkpoint 合并脚本。
3. 自己实现 FinChain 数据转换、Python trace verifier、step-level reward 和 ChainEval 评测。
4. 把 ODA-Fin 的开放数据与 checkpoint 作为起点，而不是复刻其未公开的训练工程。

## 可直接使用的数据

| 数据集 | 规模 | 语言 | 用途 | 许可 / 状态 |
|--------|------|------|------|-------------|
| [ODA-Fin-SFT-318k](https://huggingface.co/datasets/OpenDataArena/ODA-Fin-SFT-318k) | 318K | 中英 | 高质量 CoT SFT | Apache-2.0 |
| [ODA-Fin-RL-12k](https://huggingface.co/datasets/OpenDataArena/ODA-Fin-RL-12k) | 12K | 中英 | hard-but-verifiable GRPO | Apache-2.0 |
| [DianJin-Fin-PRM-Data](https://huggingface.co/datasets/DianJin/DianJin-Fin-PRM-Data) | 4,969 | 中文 | step/trajectory PRM | Apache-2.0 |
| [FinCoT](https://huggingface.co/datasets/TheFinAI/FinCoT) | 1K–10K | 英文 | 金融 CoT SFT | 来源数据许可需逐项核对 |
| [DianJin-R1-Data](https://huggingface.co/datasets/DianJin/DianJin-R1-Data) | 10K–100K | 中英/合规 | 结构化推理 SFT/RL | MIT |
| [Agentar-DeepFinance-100K](https://huggingface.co/datasets/antgroup/Agentar-DeepFinance-100K) | 100K 级 | 中文金融为主 | 多视角 CoT 蒸馏 | 模型卡未标明标准许可，使用前核对 |

## 训练范式演进

```text
金融语料适配
└── CPT / domain pretraining
    └── FEVO-C32B

结果监督推理
└── CoT SFT → GRPO / outcome reward
    ├── Fin-o1
    ├── Fin-R1
    ├── DianJin-R1
    └── ODA-Fin-RL-8B

过程监督推理
└── step-level + trajectory-level reward
    └── Fin-PRM

可信与合规增强
├── RKEFino1: regulation knowledge
└── Agentar-Fin-R1: trustworthy synthesis + compliance benchmark

检索与随机决策
├── StockLLM + FinSeer: historical-sequence RAG
├── Trading-R1: curriculum RL for executable trades
└── Trade-R1: evidence-reasoning-decision verification

多模态金融模型
├── FinTral: text + number + table + image + tools
└── Open-FinLLMs: FinLLaMA + FinLLaVA
```

## 对下一步工作的启示

### 方向 A：FinChain-aware Process Reward Model

FinChain 提供可执行 Python trace 和 gold steps，Fin-PRM 提供过程奖励建模范式。可以把符号执行正确性、步骤对齐和金融逻辑评价结合成新的 PRM 数据与训练目标。相比再次只做 answer reward GRPO，这条路线的方法差异更明确。

### 方向 B：Data-centric 可验证难度课程

ODA-Fin 表明数据难度和可验证性选择比单纯扩大数据更重要。可以用 FinChain 的模板、执行 trace 和 ChainEval 构造 difficulty/verifiability-aware curriculum，对比随机采样、仅困难采样和可验证困难采样。

### 方向 C：基座与领域知识解耦实验

FinChain 中 Qwen2.5-7B-Instruct 强于多数金融模型，而 FEVO 强调先扩充领域知识再做推理训练。可以在相同数据、相同 RL 配置下比较 Qwen2.5、Qwen3、Qwen-Open-Finance-R 和 ODA-Fin-SFT，分离基座能力、金融知识与推理后训练的贡献。

### 方向 D：从确定性验证扩展到随机决策

FinChain 能用 gold steps 和 Python 执行结果提供强确定性监督，Trade-R1 则用证据、推理链和决策的一致性过滤噪声市场收益。可以先在 FinChain 上训练可执行 PRM，再把语义一致性和风险调整收益加入奖励，测试过程验证能否迁移到资产选择或交易任务。

### 方向 E：检索器是否比继续增大模型更有效

StockLLM + FinSeer 提供了低成本对照：固定小型生成模型，只改变随机检索、通用 embedding 检索和金融专用 retriever。对 FinChain 可进一步比较检索题目模板、公式、相似执行 trace 和法规证据的收益。

## 建议的复现顺序

1. 在 FinChain 上复现 Qwen2.5-7B-Instruct、Fin-R1 和 ODA-Fin-RL-8B。
2. 统一输出模板与推理预算，避免 ChainEval 被格式差异污染。
3. 用 Fin-PRM 做 Best-of-N reranking，先验证过程奖励是否提升 ChainEval。
4. 再训练小规模 FinChain-aware PRM 或 GRPO，比较 outcome-only 与 process-aware reward。
5. 用 StockLLM/FinSeer 式专用检索器建立 RAG 对照，再决定是否进入 Trade-R1 式随机决策训练。

## 来源

- [ODA-Fin paper](https://arxiv.org/abs/2603.07223)
- [Fin-PRM paper](https://arxiv.org/abs/2508.15202)
- [FEVO paper](https://arxiv.org/abs/2507.06057)
- [Agentar-Fin-R1 paper](https://arxiv.org/abs/2507.16802)
- [Hugging Face: OpenDataArena](https://huggingface.co/OpenDataArena)
- [Hugging Face: DianJin](https://huggingface.co/DianJin)
- [Hugging Face: TheFinAI](https://huggingface.co/TheFinAI)
- [Trade-R1 paper](https://arxiv.org/abs/2601.03948)
- [Trading-R1 paper](https://arxiv.org/abs/2509.11420)
- [StockLLM + FinSeer paper](https://arxiv.org/abs/2502.05878)
- [Open-FinLLMs paper](https://arxiv.org/abs/2408.11878)
- [FinTral paper](https://arxiv.org/abs/2402.10986)

> 更新时间：2026-06-15。模型下载量、访问门槛和开放状态可能变化，开展实验前需再次核对。
