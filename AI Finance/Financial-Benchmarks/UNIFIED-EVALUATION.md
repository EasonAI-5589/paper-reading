# Unified Financial Evaluation

截至 2026-06-15，金融领域没有一个像 VLMEvalKit 那样覆盖主流金融 benchmark、统一模型接口、推理和结果汇总的事实标准。最接近的是 [FinBen / PIXIU](https://github.com/The-FinAI/PIXIU)，但 FinEval、Finova、FinChain 和较新的可靠性、多模态任务仍需要各自的 runner 或额外 adapter。

## 现有工具

| 工具 | 基础框架 | 已覆盖或适合的任务 | 模型后端 | 局限 |
|------|----------|--------------------|----------|------|
| [FinBen / PIXIU](https://github.com/The-FinAI/PIXIU) | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | FPB、FOMC、Headlines、FinQA、TAT-QA，以及 IE、分类、风险、预测等 30+ 数据集 | Hugging Face、本地服务、商业 API | 不含 FinChain、Finova、FinanceIQ、ConvFinQA 等完整新评测；部分生成任务有定制解析 |
| [FinEval](https://github.com/SUFE-AIFLM-Lab/FinEval) | 自有 Python runner | 中文金融知识、行业、安全、Agent | 自定义模型 adapter | 公开数据主要是初始学术知识部分；不是通用金融 suite |
| [Finova](https://github.com/antgroup/Finova) | 自有 Python/Shell runner | Agent、复杂推理、安全合规 | API 或本地推理适配 | 只覆盖 Finova，自定义工具调用和实体指标 |
| [FinChain](https://github.com/mbzuai-nlp/finchain) | 自有生成与 ChainEval | 答案准确率、步骤对齐和可执行 trace | 需按项目格式生成结果 | ChainEval 无法直接用普通 exact-match 代替 |
| [Lighteval](https://github.com/huggingface/lighteval) | 通用评测框架 | 适合新增 Hugging Face dataset、自定义 prompt 和 metric | Transformers、vLLM、SGLang、API | 现有金融任务少，需要重新移植 FinBen 任务 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 通用评测框架 | 适合分类、选择题、QA、生成和自定义 task group | HF、vLLM、SGLang、OpenAI-compatible API | Agent、代码执行和 ChainEval 仍需自定义 evaluator |

## 推荐方案

以 `lm-evaluation-harness + vLLM` 为统一主干最省工程量，因为 FinBen 已经基于 lm-eval 实现了一批金融任务。不要重写 benchmark 原始评分器，而是在统一推理层之上保留专用 evaluator。

```text
LLaMA-Factory SFT / VeRL GRPO checkpoint
                    │
                    ▼
             vLLM OpenAI API
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
 lm-eval tasks   agent adapters   executable evaluators
 FPB/FOMC/HL     Finova           FinChain/FinanceReasoning
 FinQA/TAT-QA    FinEval Agent     Python trace/ChainEval
 FinanceIQ
       └────────────┼─────────────┘
                    ▼
          unified JSONL + summary table
```

## 第一阶段覆盖

| Task group | Benchmark | 接入方式 |
|------------|-----------|----------|
| `finance_general` | FinEval、FinanceIQ | lm-eval YAML/Python task；选择题 accuracy |
| `finance_sentiment` | FPB、FOMC、Headlines | 直接复用 FinBen task；weighted F1 |
| `finance_numerical` | FinQA、TAT-QA、ConvFinQA | 复用 FinBen 前两项，新增 ConvFinQA adapter；答案解析 accuracy |
| `finance_agent` | Finova | 调用官方 runner，共享 vLLM endpoint |
| `finance_verifiable` | FinChain | 调用官方生成格式和 ChainEval；保存答案、步骤和执行 trace |

## 统一结果格式

每条结果至少保存以下字段，避免不同 benchmark 只能留下一个平均分：

```json
{
  "model": "checkpoint-path",
  "benchmark": "finchain",
  "subset": "portfolio_management",
  "sample_id": "...",
  "prompt_version": "zero-shot-cot-v1",
  "prediction": "...",
  "reference": "...",
  "metrics": {"accuracy": 1, "chain_eval": 0.83},
  "generation": {"temperature": 0, "max_tokens": 4096},
  "checkpoint_step": 1200
}
```

## 与训练流水线衔接

1. LLaMA-Factory 导出 SFT checkpoint 后，先跑 `finance_general + finance_sentiment + finance_numerical`。
2. VeRL 每个关键 GRPO checkpoint 只跑小型 development subset，控制评测成本。
3. 最终 checkpoint 再跑 Finova、FinChain 和完整 benchmark suite。
4. 同时报告分组成绩，不将 accuracy、weighted F1 和 ChainEval 直接平均成一个缺乏含义的总分。

## 判断

短期不需要再造完整评测框架。先在 lm-eval 上建立金融 task group，并用 adapter 调官方 Finova/FinChain evaluator，就能形成一个文本金融模型版本的 VLMEvalKit。后续加入 FinMMEval 等多模态任务时，再考虑接入 lmms-eval 或 VLMEvalKit。
