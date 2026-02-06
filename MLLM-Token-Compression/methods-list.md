# MLLM Token Compression 方法汇总

> 整理自 Survey: "Towards Efficient MLLMs: A Survey on Token Compression"
> 用于 STAR-Pro Related Work 参考

---

## 📊 按重要性排序 (基础架构 / 高引用)

| 方法 | arXiv | 位置 | 核心思路 | GitHub |
|------|-------|------|----------|--------|
| **[ICML 2023] BLIP-2 (Q-Former)** | 2301.12597 | Projector | 可学习 queries + cross-attention | ✅ Salesforce |
| **[NeurIPS 2022] Flamingo** | 2204.14198 | LLM | Gated XATTN-DENSE layers | - DeepMind |
| **[NeurIPS 2023] LLaVA** | 2304.08485 | - | 简单 MLP projector，不压缩 | ✅ 基准模型 |
| **[ECCV 2024 Oral] FastV** | 2403.06764 | LLM Prefilling | 第 2 层后剪枝 50% visual tokens | ✅ pkunlp-icler (552⭐) |
| **[CVPR 2025] PyramidDrop** | 2410.17247 | LLM Prefilling | 渐进式多阶段剪枝 | ✅ Cooperx521 (141⭐) |
| **[CVPR 2025] VisionZip** | 2412.04467 | Outside-VE | 重要性 + 代表性约束 | ✅ dvlab-research (389⭐) |
| **[IJCV 2025] TokenPacker** | 2407.02392 | Projector | 粗到细策略，压缩 75-89% | ✅ CircleRadon (276⭐) |
| **[Arxiv 2510.02912] HoloV** | 2510.02912 | Outside-VE | 平衡语义连接性，解决 attention bias | ✅ obananas/HoloV (56⭐) |
| **[Arxiv 2404.16821] InternVL 1.5** | 2404.16821 | Projector | Pixel Shuffle 压缩 | ✅ OpenGVLab |
| **[Arxiv 2409.12191] Qwen2-VL** | 2409.12191 | Projector | Pixel Shuffle + 动态分辨率 | ✅ Alibaba |

---

## 🔬 按压缩位置分类

### 1. Vision Encoder (Inside) - 早期压缩

| 方法 | arXiv | 思路 | GitHub |
|------|-------|------|--------|
| **[COLING 2025] TRIM** | 2409.10994 | CLIP metric 计算 text-visual 相似度筛选 | ✅ FreedomIntelligence/TRIM (20⭐) |
| **[Arxiv 2503.11549] SAINT** | 2503.11549 | Similarity-Aware，CLS attention + similarity merging/dropping，75% 压缩 | ✅ ArmenJeddi/saint (42⭐) |
| **[ICCV 2025] VisPruner** | 2412.01818 | CLS attention 评估重要性，91% FLOPs 降低 | ✅ Theia-4869/VisPruner (68⭐) |
| **[Arxiv 2508.00553] HiPrune** | 2508.00553 | Training-free 层次化 attention 剪枝，99.3% 精度 @ 33.3% tokens | ✅ Danielement321/HiPrune |
| **[ICCV 2025] VFlowOpt** | 2508.05211 | Visual Information Flow 指导优化，90% prune + 89% KV-Cache 降低 | ✅ sihany077/VFlowOpt (10⭐) |
| **[Arxiv 2507.15428] EgoPrune** | 2507.15428 | Training-free，自我中心视频先验，两阶段压缩 | - |
| **[ICCV 2025] METEOR** | 2507.20842 | 多编码器协作剪枝 (Multi-Encoder) | ✅ YuchenLiu98/METEOR (5⭐) |

### 2. Vision Encoder (Outside) - 即插即用

| 方法 | arXiv | 思路 | GitHub |
|------|-------|------|--------|
| **[CVPR 2025] VisionZip** | 2412.04467 | 重要性 + 代表性约束 | ✅ dvlab (353⭐) |
| **[Arxiv 2506.07138] LLaVA-STF** | 2506.07138 | MBTF + STF，跨层特征拼接生成摘要 | ✅ visresearch/LLaVA-STF (29⭐) |
| **[Arxiv 2510.02912] HoloV** | 2510.02912 | 解决 attention bias，保留全局上下文 | ✅ obananas/HoloV (56⭐) |
| **[Arxiv 2410.07278] PAR** | 2410.07278 | Prompt-Aware，Query 解析 → 实体+动作 → 重新加权，83% FLOPs 降低 | - |
| **[Arxiv 2504.00654] QG-VTC** | 2504.00654 | 问题-视觉相似度指导，MLLM-based VQA | - |
| **[Arxiv 2508.17807] AttDebias** | 2508.17807 | Attention debiasing，解决 VLM attention bias 问题 | ✅ intcomp/attention-bias |
| **[Arxiv 2410.04417] SparseVLM** | 2410.04417 | Text-guided training-free，自注意力矩阵选择相关文本 token 评分视觉 token | ✅ Gumpest/SparseVLM |
| **[Arxiv 2503.10501] TokenCarve** | 2503.10501 | Information-preserving，训练无关，减缓信息损失率 | ✅ ShawnTan86/TokenCarve (23⭐) |
| **[ACM MM 2025] VISA** | 2508.17857 | 图摘要 (Graph Summarization)，组级 token 选择 + 聚合 | ✅ mobiushy/VISA (1⭐) |

### 3. Projector - 自然融合点

| 方法 | arXiv | 思路 | GitHub |
|------|-------|------|--------|
| **[ICML 2023] Q-Former (BLIP-2)** | 2301.12597 | 可学习 queries + cross-attention | ✅ Salesforce |
| **[Arxiv 2308.12966] Qwen-VL** | 2308.12966 | 单层 cross-attention | ✅ Alibaba |
| **[CVPR 2024] Honeybee** | 2312.06742 | C-Abstractor / D-Abstractor | ✅ khanrc (464⭐) |
| **[NeurIPS 2024] MQT** | 2405.19315 | Matryoshka 可变 query 数量 | ✅ gordonhu608/MQT-LLaVA (123⭐) |
| **[Arxiv 2409.09564] TG-LLaVA** | 2409.09564 | Text-guided Visual Feature Optimization + Learnable Latent Embeddings | ✅ AIDC-AI/TG-LLaVA |
| **[Arxiv 2501.03895] LLaVA-Mini** | 2501.03895 | Modality-Pre Fusion，压缩至 1 token | ✅ ictnlp/LLaVA-Mini (561⭐) |
| **[IJCV 2025] TokenPacker** | 2407.02392 | 粗到细策略 75-89% 压缩 | ✅ CircleRadon |
| **[Arxiv 2402.03766] MobileVLM V2** | 2402.03766 | LDP (Pooling) | ✅ |
| **[Arxiv 2405.20985] DeCo** | 2405.20985 | 2D Adaptive Pooling 下采样 | ✅ yaolinli |
| **[CVPR 2024] PLLaVA** | 2404.16994 | Adaptive Pooling | ✅ magic-research (677⭐) |
| **[Arxiv 2506.03990] DynTok** | 2506.03990 | 自适应分组+组内合并，低信息密度区域高压缩 | - Kuaishou |
| **[CVPR 2025] LLaVA-Scissor** | 2506.21862 | SCC 图分割 | ✅ HumanMLLM/LLaVA-Scissor (118⭐) |
| **[CVPR 2025] DivPrune** | 2503.02175 | Max-Min Diversity Problem (MMDP)，最大化 token 多样性 | ✅ vbdi/divprune (65⭐) |
| **[CVPR 2025] PACT** | 2504.08966 | Pruning + Distance Bounded Density Peak Clustering | ✅ orailix/PACT (55⭐) |
| **[CVPR 2025] PVC** | 2412.09613 | Progressive Visual Token Compression，统一图像+视频处理 | ✅ OpenGVLab/PVC (51⭐) |
| **[Arxiv 2403.15388] LLaVA-PruMerge** | 2403.15388 | CLS + spatial token 相似度筛选+合并，14x 压缩 | ✅ llava-prumerge |

### 4. LLM Prefilling - 参数最多

| 方法 | arXiv | 思路 | GitHub |
|------|-------|------|--------|
| **[ECCV 2024 Oral] FastV** | 2403.06764 | Attention score 排序剪枝 | ✅ pkunlp-icler (504⭐) |
| **[CVPR 2025] PyramidDrop** | 2410.17247 | 渐进式多阶段剪枝 | ✅ Cooperx521 |
| **[Arxiv 2411.10803] MustDrop** | 2411.10803 | 三阶段压缩 (VE+prefill+decode) | ✅ liuting20 (37⭐) |
| **[ICCV 2025] Feather** | 2412.13180 | 去除 RoPE 解决位置偏差，ensemble criteria | ✅ markendo/FEATHER (9⭐) |
| **[ICCV 2025] p-MoD** | 2412.04449 | Mixture-of-Depths，PRD 策略控制保留率 | ✅ MCG-NJU/p-MoD (43⭐) |
| **[Arxiv 2508.01548] GlimpsePrune** | 2508.01548 | VIP 预测器，单次前向动态剪枝，92.6% token 压缩 | ✅ HVision-NKU/GlimpsePrune (89⭐) |
| **[Arxiv 2501.14204] DyRate** | 2501.14204 | Attention 分布训练线性分类器，预测最优剪枝率 | - |
| **[Arxiv 2412.00447] ATP-LLaVA** | 2412.00447 | MLP 双头预测阈值，自适应剪枝率 | ✅ yxxxb/ATP-LLaVA |
| **[NeurIPS 2024] LLaVolta** | 2406.20092 | Visual Context Compressor，渐进式训练 | ✅ Beckschen/LLaVolta (65⭐) |
| **[Arxiv 2410.06169] YOPO** | 2410.06169 | You Only Prune Once，视觉计算冗余，sample-agnostic | ✅ ZhangAIPI/YOPO_MLLM_Pruning (105⭐) |
| **[AAAI 2026] FiCoCo** | 2411.17686 | Filter → Correlate → Compress | ✅ kawhiiiileo/FiCoCo (61⭐) |
| **[AAAI 2026] GlobalCom2** | 2501.05179 | Thumbnail 引导全局压缩，90% tokens 压缩保持 90%+ 性能 | ✅ xuyang-liu16/GlobalCom2 (38⭐) |
| **[ICCV 2025] FrameFusion** | 2501.01986 | 跨帧 cosine similarity 合并，70% token 压缩，1.6-3.6x 加速 | ✅ thu-nics/FrameFusion (68⭐) |
| **[Arxiv 2505.21334] HoliTom** | 2505.21334 | 全局冗余感知时序分割 + spatial-temporal merging，90%+ 压缩 | ✅ cokeshao/HoliTom (70⭐) |
| **[CVPR 2025] HLII (HICom)** | 2503.16036 | Hybrid-Level Instruction Injection，local+global 级别指令注入 | ✅ lntzm/HICom (19⭐) |
| **[NeurIPS 2022] Flamingo** | 2204.14198 | Gated XATTN-DENSE | - DeepMind |
| **[Arxiv 2408.04840] mPLUG-Owl3** | 2408.04840 | 文本自注意力 + 跨模态 attention | ✅ X-PLUG/mPLUG-Owl (2539⭐) |
| **[Arxiv 2505.17020] CrossLMM** | 2505.17020 | 双向 cross-attention (V2T + T2V)，解耦长视频序列 | ✅ shilinyan99/CrossLMM (25⭐) |
| **[CVPR 2025] VoCo-LLaMA** | 2406.12275 | 单个 Vision Compression token | ✅ Yxxxb (204⭐) |
| **[AAAI 2026] QuoTA** | 2503.08689 | Query-oriented Token Assignment，CoT 解耦查询指导 token 分配，长视频理解 | ✅ MAC-AutoML/QuoTA (77⭐) |
| **[ACL 2025 Findings] PruneVid** | 2412.16117 | Training-free 时空 token 合并 + LLM attention 选择性剪枝，视频理解优化 | ✅ Visual-AI/PruneVid (66⭐) |

### 5. LLM Decoding (KV Cache)

| 方法 | arXiv | 思路 | GitHub |
|------|-------|------|--------|
| **[EMNLP 2024] LOOK-M** | 2406.18139 | 累积 attention 估计重要性，text-prior 压缩 | ✅ SUSTechBruce (104⭐) |
| **[Arxiv 2411.10803] MustDrop** | 2411.10803 | 三阶段压缩 (VE+prefill+decode) | ✅ liuting20 |
| **[Arxiv 2506.05344] SparseMM** | 2506.05344 | Visual heads 稀疏性，非对称 KV 预算 | ✅ CR400AF-A/SparseMM (81⭐) |
| **[CVPR 2025] DyCoke** | 2411.15024 | 跨帧 token merging + 动态 KV 压缩 | ✅ KD-TAO/DyCoke (98⭐) |
| **[Arxiv 2409.14485] Video-XL** | 2409.14485 | Hour-scale video 理解，Visual Context Latent Summarization | ✅ VectorSpaceLab/Video-XL (611⭐) |
| **[Arxiv 2503.18478] Video-XL-Pro** | 2503.18478 | Reconstructive Token Compression for Extremely Long Video | ✅ VectorSpaceLab/Video-XL (611⭐) |

### 6. Hybrid - 多模块协同

| 方法 | arXiv | 思路 | GitHub |
|------|-------|------|--------|
| **[ICML 2024] CrossGET** | 2305.17455 | 双向引导，跨模态 token 动态合并 | ✅ sdc17/CrossGET |
| **[ECCV 2024] LLaMA-VID** | 2311.17043 | 每帧压缩为 2 tokens | ✅ dvlab-research |

---

## 🎯 与 STAR-Pro 最相关的方法

| 方法 | 相关度 | 原因 |
|------|--------|------|
| **[ECCV 2024 Oral] FastV** | ⭐⭐⭐ | LLM 阶段 attention-based 剪枝，STAR-Pro 直接对比 |
| **[CVPR 2025] PyramidDrop** | ⭐⭐⭐ | 渐进式剪枝，与 STAR-Pro Progressive stage 类似 |
| **[Arxiv 2510.02912] HoloV** | ⭐⭐⭐ | 解决 attention bias，支持 STAR-Pro 的 inconsistency 发现 |
| **[ICCV 2025] Feather** | ⭐⭐⭐ | 去除 RoPE 解决 positional bias，支持 STAR-Pro |
| **[CVPR 2025] VisionZip** | ⭐⭐ | Outside-VE 压缩 |
| **[IJCV 2025] TokenPacker** | ⭐⭐ | 粗到细策略 |
| **[Arxiv 2411.10803] MustDrop** | ⭐⭐ | 三阶段渐进压缩 |

---

## 📝 TODO

- [x] 补充 arXiv 链接 ✅ 基本完成
- [ ] 补充 citation 数量 (Google Scholar)
- [ ] 补充 GitHub stars
- [ ] 按时间排序（需要确认具体发布日期）

### 暂无公开 GitHub 的方法
- PAR (2410.07278) - 无公开仓库
- QG-VTC (2504.00654) - 无公开仓库
- DynTok (2506.03990) - Kuaishou 内部工作
- DyRate (2501.14204) - 无公开仓库
- EgoPrune (2507.15428) - 无公开仓库
- AdaptInfer (2508.06084) - ICLR 2026 under review，暂无公开

---

## 🆕 新发现方法 (2025年新增)

| 方法 | arXiv | 位置 | 核心思路 | GitHub |
|------|-------|------|----------|--------|
| **[NeurIPS 2025] CDPruner** | 2506.10967 | Outside-VE | 最大化条件多样性 (Conditional Diversity)，training-free | ✅ Theia-4869/CDPruner |
| **[ICCV 2025] Skip-Vision** | 2503.21817 | LLM | 自适应跳层，减少 35% 训练时间 + 75% FLOPs + 45% 延迟 | - (SJTU, 暂无公开) |
| **[NAACL 2025 Findings] LVPruning** | 2501.13652 | LLM | Cross-attention 计算 vision-language 交互重要性，无需修改原模型 | - (暂无公开) |
| **[AAAI 2026] LFTR** | 2501.17391 | Hybrid | Learning-Free Token Reduction，时空维度压缩 | - (AAAI 接收，代码待公开) |
| **[Arxiv 2504.00502] ShortV** | 2504.00502 | LLM | Layer Contribution 指标识别无效层并冻结视觉 tokens，training-free | ✅ icip-cas/ShortV |
| **[Arxiv 2508.06084] AdaptInfer** | 2508.06084 | LLM Prefilling | Plug-and-play 动态文本引导剪枝，利用推理时内部信号 | - (暂无公开) |
| **[Arxiv 2508.06038] Fourier-VLM** | 2508.06038 | Outside-VE | 频域 (Frequency Domain) 压缩视觉 tokens，DCT 系数筛选 | - (ShanghaiTech, 暂无公开) |
| **[Arxiv 2512.18747] IPCV** | 2512.18747 | VE | Information-Preserving Compression，超越 training-free SOTA | ✅ Perkzi/IPCV |
| **[Arxiv 2410.06169] YOPO** | 2410.06169 | LLM | You Only Prune Once，视觉计算冗余分析，sample-agnostic 剪枝 | ✅ ZhangAIPI/YOPO_MLLM_Pruning |
| **[Arxiv 2505.22654] VScan** | 2505.22654 | Hybrid | Rethinking Visual Token Reduction，全局+局部扫描 + LLM 中间层剪枝 | ✅ Tencent/SelfEvolvingAgent/VScan |
| **[Arxiv 2510.16753] ELMM** | 2510.16753 | Projector | Multi-view Visual Token Compressor (MVTC)，多视图自适应压缩 | - (待确认) |
| **[Arxiv 2508.18227] GM-Skip** | 2508.18227 | LLM | Metric-Guided Block Skipping，基于任务指标的自适应层跳过 | - (暂无公开) |
| **[Arxiv 2509.25584] Skip-It?** | 2509.25584 | LLM | 层跳过的理论条件分析，提供 VLM 层跳过的数学框架 | - (理论分析) |
| **[Arxiv 2510.18269] StreamingTOM** | 2510.18269 | Hybrid | Training-free plug-and-play，Causal Temporal Reduction + 4-bit KV，15.7x KV 压缩，2x TTFT 加速 | 🌐 [Project](https://yige24.github.io/StreamingTOM) |
| **[ICCV 2025] STTM** | 2507.07990 | LLM | Multi-Granular Spatio-Temporal Token Merging，Training-Free 视频加速 | ✅ HYUNJS/STTM |
| **[Arxiv 2512.00891] STC** | 2512.00891 | Hybrid | Streaming Token Compression (STC-Cacher + STC-Pruner)，plug-and-play 层次化压缩 | - (待公开) |
| **[IJCAI 2025] DToMA** | [PDF](https://ijcai.org/proceedings/2025/0258.pdf) | Hybrid | Training-free Dynamic Token Manipulation，长视频理解 | - (无公开代码) |
| **[ACL 2025 Findings] RedundancyLens** | 2501.19036 | LLM | 揭示 decoder-only MLLM 视觉 token 处理冗余，training-free 加速 | ✅ L-Hugh/RedundancyLens |
| **[Arxiv 2601.13879] V-Skip** | 2601.13879 | LLM | Visual-Anchored Information Bottleneck (VA-IB)，CoT 压缩，Dual-Path Gating，2.9x 加速，DocVQA 超 30% | - (待公开) |
| **[Arxiv 2602.00946] ConsensusDrop** | 2602.00946 | LLM | 融合视觉+跨模态显著性，99.7%/99.2%/95.2%/90.5% 不同压缩率保持 | - (待公开) |
| **[Arxiv 2602.02951] Nüwa** | 2602.02951 | Hybrid | 两阶段框架：VE后群体智能保留空间锚点 + LLM内 text-guided pruning，VG任务提升47% | - (待公开) |
| **[Arxiv 2601.21531] VTC-Robustness** | 2601.21531 | (分析) | 首次研究 VTC 对抗鲁棒性，提出 CAGE (Compression-AliGnEd attack)，压缩感知安全评估 | - (分析性论文) |
| **[Arxiv 2602.04804] OmniSIFT** | 2602.04804 | Hybrid | 模态非对称压缩 (Modality-Asymmetric)，视频空间+时间冗余剪枝 → 视觉锚点 → 音频 token 选择，Omni-modal LLM 专用 | - (待公开) |
| **[Arxiv 2602.03815] DualSpeed** | 2602.03815 | Training | 解决 VTP 训练-推理不匹配，Fast-Slow 框架：快速分支剪枝训练 + 慢速分支保持完整 tokens | - (待公开) |
| **[Arxiv 2601.22069] VTC-R1** | 2601.22069 | Hybrid | 将推理链渲染成图像作为"光学记忆" (Optical Memory)，training-free 长上下文推理加速 | - (待公开) |
| **[Arxiv 2512.15649] VTCBench** | 2512.15649 | (Benchmark) | Vision-Text Compression 评估基准，测试 VLM 长文本理解能力 | - (评估框架) |
| **[Arxiv 2602.01785] CodeOCR** | 2602.01785 | (应用) | 代码图像理解，视觉表示实现 8x token 压缩，语法高亮等视觉线索利用 | - (分析性论文) |
| **[Arxiv 2501.02268] G-Prune** | 2501.02268 | Outside-VE | 图视角，将 visual tokens 视为节点，基于语义相似度构建连接，保留前景+背景关键 tokens | ✅ jytmelon/G-Prune |
| **[Arxiv 2505.18757] ToDRE** | 2505.18757 | Hybrid | Token Diversity + Task Relevance 两阶段剪枝，greedy max-sum 多样化选择 + LLM 内任务相关性过滤 | - (暂无公开) |
| **[Arxiv 2506.13166] GreedyPrune** | 2506.13166 | Outside-VE | Greedy 搜索保留关键 visual token 集合，优化视角剪枝 | - (暂无公开) |
| **[Arxiv 2512.12560] StreamingAssistant** | 2512.12560 | LLM | 在线视频理解专用，高效 visual token 剪枝加速流式视频处理 | - (暂无公开) |
| **[Arxiv 2509.23663] HIVTP** | 2509.23663 | LLM | Training-free 层次化剪枝，中间层重要性评分指导 | - (暂无公开) |
| **[Arxiv 2509.15704] PTP** | 2509.15704 | LLM | Training-free Pyramid Token Pruning，Region+Token+Instruction 三级重要性，高分辨率 LVLM 专用 | - (暂无公开) |
| **[Arxiv 2509.12159] EfficientUICoder** | 2509.12159 | Hybrid | 输入输出双重压缩，UI 代码生成专用，55-60% 压缩率，44.9% 计算降低 | ✅ WebPAI/EfficientUICoder |
| **[Arxiv 2512.09927] Token Expand-Merge** | 2512.09927 | Hybrid | Training-free VLA 专用，早期剪枝 + action-guided merging，具身智能场景 | - (待公开) |
| **[Arxiv 2411.03312] InferenceOptimalVLM** | 2411.03312 | (分析) | 研究 visual tokens vs LLM 参数最优权衡 Scaling Laws，发现单 token 可达最优推理效率 | - (分析性论文) |
| **[Arxiv 2511.12280] D3ToM** | 2511.12280 | Projector | Decider-Guided Dynamic Token Merging，针对 Diffusion MLLMs，单 transformer 层即插即用模块，动态 merge ratio | - (待公开) |

---

## 📚 参考资源

| 资源 | 链接 | 说明 |
|------|------|------|
| **Awesome-Token-Compress** | [daixiangzi/Awesome-Token-Compress](https://github.com/daixiangzi/Awesome-Token-Compress) | ViT & VLM Token 压缩论文列表 |
| **MLLM-Token-Compression** | [yaolinli/MLLM-Token-Compression](https://github.com/yaolinli/MLLM-Token-Compression) | Survey 官方仓库 |
| **Awesome-Multimodal-Token-Compression** | [cokeshao/Awesome-Multimodal-Token-Compression](https://github.com/cokeshao/Awesome-Multimodal-Token-Compression) | 多模态 Token 压缩 |
| **Awesome-Token-Merge-for-MLLMs** | [JinXins/Awesome-Token-Merge-for-MLLMs](https://github.com/JinXins/Awesome-Token-Merge-for-MLLMs) | Token Merge/Reduce/Resample/Drop 论文汇总 |
| **Awesome-Collection-Token-Reduction** | [ZLKong/Awesome-Collection-Token-Reduction](https://github.com/ZLKong/Awesome-Collection-Token-Reduction) | Token Reduction 技术集合 (ML/AI) |
| **Benchmark Evaluation (DART)** | [arxiv 2510.07143](https://arxiv.org/abs/2510.07143) | VTC 方法评估框架，测试压缩方法真实性能 |
| **Token Reduction Survey** | [arxiv 2505.18227](https://arxiv.org/abs/2505.18227) | "Token Reduction Should Go Beyond Efficiency" 综述 |
| **Compression & Intelligence** | [arxiv 2601.20742](https://arxiv.org/abs/2601.20742) | 视觉编码与 Token 技术统一理论 |

---

*Last updated: 2026-02-06 19:45 CST (2号机添加论文来源标识)*

### 📊 统计
- **总方法数**: 100+ 篇论文
- **有 arXiv 链接**: ✅ 100%
- **有 GitHub 仓库**: ~85% (部分方法暂无公开代码)
- **暂无公开代码**: PAR, QG-VTC, DynTok, DyRate, EgoPrune, AdaptInfer, GM-Skip, Fourier-VLM, Skip-Vision, LVPruning, LFTR, StreamingTOM, STC, ELMM, DToMA, V-Skip, ConsensusDrop, Nüwa, HIVTP, PTP, OmniSIFT, DualSpeed, Token Expand-Merge, ToDRE, GreedyPrune, StreamingAssistant
