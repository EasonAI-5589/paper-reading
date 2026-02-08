[← 返回 README](../README.md)

# Abstract

## 📌 预览
SCOPE 提出联合建模 saliency 和 coverage 的 visual token pruning 策略，解决现有方法只关注显著性而忽视语义完整性的问题。

---

Multimodal Large Language Models (MLLMs) typically process a large number of visual tokens, leading to considerable computational overhead, even though many of these tokens are redundant. Existing visual token pruning methods primarily focus on selecting the most salient tokens based on attention scores, resulting in the semantic incompleteness of the selected tokens. In this paper, we propose a novel visual token pruning strategy, called Saliency-Coverage Oriented token Pruning for Efficient MLLMs (SCOPE), to jointly model both the saliency and coverage of the selected visual tokens to better preserve semantic completeness. Specifically, we introduce a set-coverage for a given set of selected tokens, computed based on the token relationships. We then define a token-coverage gain for each unselected token, quantifying how much additional coverage would be obtained by including it. By integrating the saliency score into the token-coverage gain, we propose our SCOPE score and iteratively select the token with the highest SCOPE score. We conduct extensive experiments on multiple vision-language understanding benchmarks using the LLaVA-1.5 and LLaVA-Next models. Experimental results demonstrate that our method consistently outperforms prior approaches. Our code is available at https://github.com/kinredon/SCOPE.

> 💡 **Abstract 批读**:
> - **问题**: 现有 visual token pruning 方法只看 saliency（attention score），导致语义不完整
> - **方案**: SCOPE = Saliency + Coverage 联合建模
> - **核心机制**: 定义 set-coverage → token-coverage gain → 整合 saliency score → 迭代选择最高 SCOPE score 的 token
> - **特点**: Training-free，即插即用
> - **验证**: LLaVA-1.5 和 LLaVA-Next，多个 VL benchmark
> - **关键词**: submodular optimization 的味道——贪心迭代选 marginal gain 最大的元素

---

## 🔖 Section 总结

### 核心洞察
1. 现有方法的本质缺陷：saliency-only 选择会导致 token 集中在少数 salient 区域，语义覆盖不全
2. SCOPE 的创新：将 coverage（语义覆盖度）引入 token 选择，类似 submodular function maximization
3. 无需训练，直接应用于现有 MLLM
