[← 返回 README](../README.md)

# Abstract

## 📌 预览
ToDRE 提出两阶段、training-free 的 visual token pruning 框架：Stage 1 用 greedy max-sum diversification 在 embedding space 选出多样性最大的 token 子集；Stage 2 利用 information migration 现象在 LLM decoder 的后半段移除所有 visual token。90% pruning 下保持 95.0% 性能，2.6× 加速。

---

_Visual token pruning aims to compress and prune redundant_
_visual tokens which play a critical role in efficient inference_
_with large vision-language models (LVLMs)._ _However, most_
_existing_ _work_ _estimates_ _visual_ _redundancy_ _using_ _a_ _single_
_metric, such as cross-modal attention or visual token simi-_
_larity._ _We show that visual token diversity and task-specific_
_token relevance are two crucial yet orthogonal factors that_
_complement each other in conveying useful information and_
_should therefore be treated separately for more effective vi-_
_sual token pruning._ _Building upon this insight, we design_
**TODRE** _, a two-stage and training-free framework that in-_
_corporates_ _**To**_ _ken_ _**D**_ _iversity and task_ _**RE**_ _levance for effective_
_token compression and efficient LVLM inference._ _Instead of_
_pruning redundant tokens, we introduce a greedy max-sum_
_diversification_ _algorithm_ _that_ _selects_ _and_ _retains_ _a_ _subset_
_of diverse and representative visual tokens after the vision_
_encoder._ _On top of that, ToDRE leverages an "information_
_migration" mechanism to eliminate task-irrelevant visual to-_
_kens within certain decoder layers of large language model_
_(LLM)_ _to_ _further_ _improve_ _token_ _pruning_ _and_ _LVLM_ _infer-_
_ence._ _Extensive experiments show that ToDRE prunes 90%_
_of visual tokens after the vision encoder as well as all visual_
_tokens_ _in_ _certain_ _LLM_ _decoder_ _layers,_ _leading_ _to_ _a_ _2.6×_
_speed-up_ _in_ _total_ _inference_ _time_ _while_ _maintaining_ _95.0%_
_model performance plus excellent model compatibility._

> 💡 **批注**:
> - **核心卖点**: 不是用单一指标（attention 或 similarity）来衡量冗余，而是拆成两个正交维度：**token diversity**（intra-modal）和 **task relevance**（cross-modal）
> - **Stage 1**: Greedy max-sum diversification → 保留最多样的 token 子集（在 embedding space，vision encoder 之后、LLM 之前）
> - **Stage 2**: Information migration → 在 LLM decoder 后半段 cross-modal attention 衰减时，直接移除所有 visual token
> - **关键数字**: 90% pruning，95.0% 性能保持，2.6× 加速
> - **优势**: Training-free + plug-and-play + architecture-agnostic

---

## 🔖 Section 总结

### 核心洞察
1. Visual token diversity 和 task relevance 是两个正交因素，应分别处理
2. 两阶段框架：diversity-driven selection + relevance-driven reduction
3. 90% pruning 仍保持 95% 性能，说明 visual token 冗余度极高
