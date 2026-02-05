# 5. Evaluation

## 5.1 Benchmarks

> The evaluation benchmarks for MLLM token compression primarily span **image and video understanding tasks**.
>
> ==评估基准：图像理解 + 视频理解==

### Image Understanding Benchmarks

| Benchmark | 任务类型 |
|-----------|---------|
| VQAv2, GQA | 通用视觉问答 |
| TextVQA, DocVQA | OCR 和文档理解 |
| MMBench, POPE | 多模态理解评估 |
| ScienceQA | 科学推理 |

### Video Understanding Benchmarks

| Benchmark | 任务类型 |
|-----------|---------|
| MSVD-QA, MSRVTT-QA | 短视频问答 |
| ActivityNet-QA | 动作识别 |
| Video-MME, EgoSchema | 长视频理解 |
| MovieChat-1K | 小时级超长视频 |
| Charades-STA | 时间定位 (Temporal Grounding) |

---

## 5.2 Metrics

> Evaluation considers two perspectives: **effectiveness** (downstream task performance) and **efficiency** (computational cost).
>
> ==两个维度：效果 + 效率==

### 5.2.1 Effectiveness

| 指标 | 说明 |
|------|------|
| **Accuracy** | 模型预测与 ground-truth 匹配率 |
| **GPT-Score** | 开放式任务（如 caption）的 GPT 评分 |

### 5.2.2 Efficiency

| 指标 | 说明 | 备注 |
|------|------|------|
| **Token Retention Count/Ratio** | 压缩后保留的 token 数量/比例 | 相同压缩率不保证相同延迟（取决于压缩位置） |
| **Prefilling/Decoding FLOPs** | 前向传播的浮点运算量 | 硬件无关的理论成本 |
| **Prefilling/Decoding Latency** | 实际墙钟时间 | 依赖具体硬件和实现 |
| **Memory Usage** | 推理时峰值内存占用 | 对资源受限设备尤其关键 |

> **注意**: Token compression can reduce memory for attention KV caches and intermediate representations, but the reduction is highly dependent on **how** compression is implemented.
>
> ==内存节省程度取决于具体实现方式==
