## **Don’t Just Chase “Highlighted Tokens” in MLLMs:** **Revisiting Visual Holistic Context Retention**

**Xin Zou** [1] _[,]_ [2] **, Di Lu** [1] _[,][†]_ **, Yizhou Wang** [1] **, Yibo Yan** [1] _[,]_ [2] **, Yuanhuiyi Lyu** [1] _[,]_ [2] **,**
**Xu Zheng** [1] _[,]_ [3] **, Linfeng Zhang** [4] **, Xuming Hu** [1] _[,]_ [2] _[∗]_

1 The Hong Kong University of Science and Technology (Guangzhou)
2 The Hong Kong University of Science and Technology
3 INSAIT, Sofia University “St. Kliment Ohridski”
4 Shanghai Jiao Tong University

[https://github.com/obananas/HoloV](https://github.com/obananas/HoloV)


**Abstract**


Despite their powerful capabilities, Multimodal Large Language Models (MLLMs)
suffer from considerable computational overhead due to their reliance on massive
visual tokens. Recent studies have explored token pruning to alleviate this problem,
which typically uses text-vision cross-attention or [CLS] attention to assess and
discard redundant visual tokens. In this work, we identify a critical limitation of
such attention-first pruning approaches, _i.e._, they tend to preserve semantically
similar tokens, resulting in pronounced performance drops under high pruning
ratios. To this end, we propose HoloV, a simple yet effective, plug-and-play
visual token pruning framework for efficient inference. Distinct from previous
attention-first schemes, HoloV rethinks token retention from a holistic perspective.
By adaptively distributing the pruning budget across different spatial crops, HoloV
ensures that the retained tokens capture the global visual context rather than isolated
salient features. This strategy minimizes representational collapse and maintains
task-relevant information even under aggressive pruning. Experimental results
demonstrate that our HoloV achieves superior performance across various tasks,
MLLM architectures, and pruning ratios compared to SOTA methods. For instance,
LLaVA1.5 equipped with HoloV preserves 95.8% of the original performance after
pruning 88.9% of visual tokens, achieving superior efficiency-accuracy trade-offs.


**1** **Introduction**


Multimodal Large Language Models (MLLMs) have demonstrated outstanding capabilities [80, 12]
in tasks such as image captioning [35, 59, 14], visual question answering [24, 97, 36], and video
understanding [34, 62, 77]. However, these models [43, 76, 38] typically require converting visual
inputs into long sequence representations ( _i.e._, visual tokens), which increases the computational
complexity and cost of inference [95], especially for high-resolution images [41] and multi-frame
videos [55], where redundant visual information further exacerbates the computational overhead.


To address this challenge, researchers have introduced token pruning strategies [49, 13, 96, 85] that
aim to retain the highlighted visual tokens as well as prune others for accelerating MLLM’s inference.
These methods typically define importance criteria for tokens, such as attention scores [13, 19] or
gradient information [57, 56], to quantify the significance of visual tokens, and less important tokens
are pruned during the inference phase, which balances speed and performance, but with limitations.


_∗_ Corresponding author, _†_ Equal contribution


39th Conference on Neural Information Processing Systems (NeurIPS 2025).


![](images/paper.pdf-1-2.png)

0.25 0.50 0.75 0.95
pruning ratio



65


60


55


50



![](images/paper.pdf-1-0.png)





![](images/paper.pdf-1-3.png)



![](images/paper.pdf-1-1.png)





90

80

70

60

50

40











0.25 0.50 0.75 0.95
pruning ratio



0.25 0.50 0.75 0.95
pruning ratio



60

55

50

45

40

35



80

75

70

65

60

55



0.25 0.50 0.75 0.95
pruning ratio



Figure 2: Relationship between performance and pruning ratios of different baseline methods. As the
token pruning ratio grows, the performance of these attention-first strategies degrades dramatically,
while HoloV maintains the substantial performance even at 90% and 95% of the pruning ratios.



As shown in Fig. 1, FastV [13] is an intuitive solution
that ranks visual tokens based on attention distributions across different layers, and then prunes the bottom _R_ % of tokens based on the computational budget,
thus reducing visual token redundancy. Subsequently,
more work has followed this paradigm [89, 96, 4], designing different strategies to prune redundant visual
tokens via cross-modal ( _i.e._, text-vision) attention
from LLMs. Besides, there are vision-centric pruning methods [75, 25, 92, 64, 86] ( _e.g._, FasterVLM

[91]) that presume those visual tokens with low correlation to the [CLS] token in ViT [17], or those exhibit
duplicated features tokens [20] to be redundant.



![](images/paper.pdf-1-4.png)

![](images/paper.pdf-1-5.png)

Figure 1: Snapshots of FastV and our HoloV.

































Reserved

tokens


FastV


Reserved

tokens


HoloV



Although these pruning methods can recognize the inefficiency of visual tokens in MLLMs, they
are not consistently effective. As shown in Fig. 2, the performance decreases significantly as the
pruning ratio increases. In our argument, this occurs because these approaches implicitly assume that
_visual tokens with high attention correspond to higher informativeness_, which disregards the spatialsemantic relations of the visual scene, _i.e._, they tend to retain tokens from localized salient regions
where attention is drawn to, rather than those conducive to holistic semantic comprehension. Thus, at
a high pruning ratio, such methods would only retain homologous tokens with higher scores. In a
complex scene with multiple objects, retaining only "highlighted tokens" may sever relative positional
and semantic connectivity information or lose key tokens associated with the subject, leading to a
dramatic performance degradation. Besides, the attention mechanism introduces systematic biases

[78, 79], _i.e._, the position encoding mechanism of transformer-based MLLMs may introduce spatial
priors, those in upper and lower areas visual tokens usually being assigned higher attention weights
as shown in Fig. 3 right. This bias can distort the semantic contributions of the visual scene, leading
the model to produce incorrect or logically contradictory inferences, or even hallucinations [98, 101].
Drawing inspiration from the above discussion, we raise the following question: _“How to locate and_
_preserve those not highlighted but critical to visual holistic understanding tokens?”_


Cognitive science research suggests that the human visual system forms a complete semantic understanding by integrating local features with global scene cues [68, 2, 61] ( _e.g._, background textures and
spatial layouts). In MLLMs, we analyzed the text-mapping relationships of different visual tokens
through the strategy in [58]. As shown in Fig. 3 left, the objects in a scene could be represented by a





![](images/paper.pdf-1-6.png)

![](images/paper.pdf-1-7.png)

![](images/paper.pdf-1-8.png)

![](images/paper.pdf-1-9.png)


|Col1|“clouds”<br>“hills”, “peaks”<br>“jack”, “standing”<br>“Rod”, “ski”<br>“boot”, “ski”<br>“snow”, “tracks”|“clouds”|
|---|---|---|
||||
|||“snow”, “tracks”|



![](images/paper.pdf-1-16.png)



![](images/paper.pdf-1-10.png)

![](images/paper.pdf-1-11.png)

![](images/paper.pdf-1-12.png)

![](images/paper.pdf-1-13.png)

![](images/paper.pdf-1-14.png)

![](images/paper.pdf-1-15.png)

Figure 3: LEFT - Examples of textual semantics corresponding to visual tokens from scattered crops.
RIGHT - Sparsification visualization examples of FastV, where retention ratios are tagged in the pics.


2


small number of scattered tokens, and the semantic relationships between those tokens from different
regions facilitate the overall understanding, _e.g._, _“snow”, “ski”, “hills”_ are kind of self-explanatory.
Motivated by this insight, we propose HoloV, which explicitly balances overall semantic connectivity
and contextual attention during visual token pruning, addressing the critical limitation of redundancy
in attention-first strategies. Our analysis demonstrates the importance of preserving visual holistic
context, offering a new perspective on efficient visual token pruning in MLLMs. Through extensive
experiments on diverse benchmarks and MLLM architectures, we demonstrate that HoloV consistently surpasses existing state-of-the-art token pruning approaches, achieving up to 88.9% token
reduction while preserving about 96% of the original performance. Besides, HoloV is model-agnostic
and easily integrable into a wide range of MLLMs, making it well-suited for practical deployment.


**2** **Related Work**


**2.1** **MLLMs and Their Challenges**


The recent remarkable success of Large Language Models (LLMs) [60, 93, 70, 18, 54] has spurred
the trend of applying their strong capabilities to multimodal comprehension tasks, fostering the
development of MLLMs [1, 67]. Leveraging open-source LLMs such as LLaMA families [70, 71, 18],
MLLMs [6, 46, 47] have demonstrated enhanced adaptability across a range of visual understanding
tasks, leading to a more profound ability to interpret the world. While this empowers LLMs with
the capability of visual perception, the incorporation of lengthy visual tokens significantly escalates
the computational burdens. Moreover, studies have shown that existing MLLMs still suffer from
certain visual deficiencies [69, 32] and some hallucinations [29, 28]. Some work mitigates these
issues by increasing the resolution of input images or videos [53, 84], but this further exacerbates
the computational overhead. For example, LLaVA-1.5 [48] encodes a 336-resolution image into
576 visual tokens, while LLaVA-NeXT [47] doubles the resolution and generates 2,880 tokens.
LLaVA-OneVision [37] represents an image using 7,290 visual tokens, and Video-LLaVA [44] faces
even higher costs, as it must process numerous visual tokens from multiple frames during inference.
These visual tokens occupy a large portion of the context window of their LLMs. In this work, we
conducted experiments and analysis on these representative models to verify HoloV’s applicability.


**2.2** **Visual Redundancy Identification**


In MLLMs, visual redundancy identification facilitates the distillation of visual tokens with high
informativeness for faster inference. There are two main research directions: a) Vision-centric
strategies analyze the image’s structure and feature distribution to discard less relevant visual tokens

[13, 75]. Existing approaches include spatial-similarity clustering ( _e.g._, TokenLearner [63]), dynamic
pruning based on attention scores [25, 87, 82], and using information bottleneck or entropy metrics
during the prefilling stage to estimate background redundancy. b) Instruction-centric strategies
typically use cross-modal attention analysis or gradient accumulation to identify redundant tokens

[49, 99, 66]. Tokens with low attention or negligible gradient impact are deemed redundant [26].
Building on this, some studies explore learned importance scoring, training a lightweight end-to-end
model to predict each patch’s “instruction relevance,” enabling even finer-grained pruning [31, 73, 89].
As the existence of language bias in LLM may cause hallucinations, we use a vision-centric scheme.


**2.3** **Visual Token Compression and Pruning**


The inclusion of visual information in MLLMs introduces long token sequences, leading to high
computation and memory costs. For example, mini-Gemini-HD [41] generates 2880 tokens from
high-definition images, creating inference bottlenecks. To address this, research has focused on token
compression and pruning techniques in Vision Transformers [10] and MLLMs [27]. Methods like
LLaMA-VID [40] and DeCo [88] address this by modifying models and adding training, which
increases computational costs. ToMe [11] reduces tokens without training but disrupts early crossmodal interactions [81]. LLaVA-PruMerge [64] selectively retains key tokens while merging less
critical ones based on key similarity. FasterVLM [91] utilizes [CLS] attention scores from the visual
encoder to re-rank and retain top visual tokens. FastV [13] and SparseVLM [96] focus on token
selection using attention scores or cross-modal guidance, but overlook the role of token duplication
and lack Flash-Attention [16, 15]. Our proposed HoloV maintains hard acceleration compatibility
( _e.g._, Flash-Attention), and effectively retains visual holistic context during aggressive pruning.


3


**3** **Preliminary and Motivation**


**3.1** **Preliminary**


**Architecture of MLLMs** . Given an MLLM _M_ [MLLM] _θ_ parameterized by _θ_, with a general architecture
consisting of a text embedding layer, a vision encoder, a vision-text interface module, a text decoder
consisting of _L_ number of transformer layers, and an affine layer which predicts the distribution of
the next token. For an image-grounded text generation task, given a textual query _x_ and an input
image _v_, _M_ [MLLM] _θ_ first extracts vision features of _v_ by the vision encoder, and then converts them into
visual tokens _zv_ by MLP or Q-Former [74] modules. Aligned vision tokens _zv_ are concatenated with
the query _x_ as input to the text decoder, and finally decoded into a textual response _y_ autoregressive,
which is formulated as: _yt_ _∼_ _pθ_ ( _·|v, x, y<t_ ) _∝_ _softmax_ ( _fθ_ ( _·|v, x, y<t_ )), where _yt_ indicates the _t_ _[th]_

|Attentio<br>okens in|n mechanism. Cons<br>MLLMs, many studi<br>cy of visual tokens.<br>[5] to perform comp<br>is the dimension of|Col3|
|---|---|---|
|edundan<br>ttention<br>where _dk_|edundan<br>ttention<br>where _dk_|edundan<br>ttention<br>where _dk_|
|edundan<br>ttention<br>where _dk_|edundan<br>ttention<br>where _dk_|ension of|



by flat backgrounds or repetitive textures. Their spatial proximity leads these tokens to capture
overlapping features, making it hard to distinguish those not highlighted yet informative tokens.


Figure 4: LEFT - Distribution map of visual token attention. RIGHT - Visualization cases of FastV and
HoloV. HoloV retains contextual tokens with rich semantics, while FastV contains much redundancy.


**Positional Bias** . To further investigate attention-based token pruning methods, we take FastV as an
example and visualize the distribution of the retained visual tokens. As illustrated in Fig. 4 right, the
attention scores for image tokens present a consistent pattern: tokens located at the beginning and
end of the sequence tend to have higher attention and are thus more likely to be preserved during
pruning, leading to a positional bias. We extend our analysis by conducting statistics on samples
from the text-based VQA task using the VQA V2 [23] dataset. Notably, even though these samples
originate from a different task, the attention distributions of image tokens at the same layer remain
highly similar, revealing recurring patterns. While the overall shape of the distributions varies slightly
across layers, the set of tokens receiving relatively high attention remains stable. We suggest that
this phenomenon occurs because all visual tokens are processed with text tokens in the same manner
during decoding, leading to positional bias of text shift to the visual modality, _e.g._, boundary positions
of text usually imply important information, but for images, targets are mostly located in the center.



![](images/paper.pdf-3-0.png)

![](images/paper.pdf-3-1.png)

![](images/paper.pdf-3-2.png)

![](images/paper.pdf-3-4.png)

![](images/paper.pdf-3-5.png)

![](images/paper.pdf-3-6.png)

![](images/paper.pdf-3-7.png)

![](images/paper.pdf-3-8.png)

![](images/paper.pdf-3-10.png)

![](images/paper.pdf-3-12.png)

![](images/paper.pdf-3-13.png)

![](images/paper.pdf-3-14.png)

**Attention Dispersion** . In addition to positional bias, we further analyze
the phenomenon of attention dispersion, i.e., a small subset of similar
tokens receives the majority of attention, while most tokens are assigned
low attention scores [91]. Specifically, we compute the cumulative
distribution of visual tokens sorted by their attention scores, as shown
in Fig. 5. The curves of last-token attention [13] and equi last attn
with identical position embedding are noticeably less steep than that
for [CLS] attention. It is evident that compared to [CLS] attention,
text-vision attention tends to be dispersed over more visual tokens, _e.g._,
the top 20% of visual tokens account for only 40% of the total attention.


4



![](images/paper.pdf-3-11.png)

100


80


60


40


20


0



![](images/paper.pdf-3-3.png)

![](images/paper.pdf-3-9.png)

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||CLS] A|ttn|
|||<br> <br>|<br>Last At<br>Equi La|<br> tn<br> st Attn|


0 20 40 60 80 100
Visual Token Proportion (%)



Figure 5: Cumulative distribution of different attentions.


**3.3** **Holistic Context Trumps Local Duplicates**


Based on our previous analysis, attention-first token pruning methods suffer from over-localization
due to positional bias and attention dispersion, _i.e._, over-reliance on attention scores disrupts spatialsemantic relationships, _e.g._, breaking occlusion hierarchies in multi-object interactions. Thus, our
key insight is that visual token importance should be evaluated through global contextual cohesion,
_i.e._, jointly considers holistic context and local saliency rather than isolated attention magnitudes.






















































|100 FastV Random<br>holistic context retention strategy, i.e., pruning visual tokens through (%)<br>96<br>random masks to retain visual information from different regions. As 95 92 9193 9192 92 92 Performance<br>shown in Fig. 6 up, compared with FastV, this random strategy out- 90 89<br>performs on more than half of the benchmarks, which demonstrates 85 85<br>the significance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed, MMBench MME MM-Vet VQAText POPE<br>possibly because random pruning discards some salient fine-grained 100 99 Global only Local only 99 (%)<br>information. This result also suggests that local saliency is indispens- 98<br>98 97 Performance<br>able, especially for densely packed elements within small regions. 96 96<br>96 95<br>In addition, we conducted an exploratory experiment to investigate 94 93<br>92 92<br>how holistic context contributes to visual understanding in MLLMs. 92<br>Specifically, we use the global thumbnail and multiple local crops MMBench MME MM-Vet VQAText POPE<br>as visual input separately [47], and evaluate performance on the two Figure 6: UP - FastV v.s. Ran<br>settings against various benchmarks. As shown in Fig. 6 down, with dom strategy. DOWN - Perfor<br>only the global thumbnail yields strong results on general visual mance comparison of the thumb<br>nail and local crops as inputs.<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fine-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fine-grained visual information for semantic understanding.<br>4 Methodology<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>4.1 HoloV Framework<br>To address the pivotal question raised in Sec. 1 for effective and effci ient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>[CLS] [CLS]<br>... ... ...|FastV Random|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|96<br><br> <br>|
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]|92|92|9|1<br>91<br>92<br>~~93~~<br>92<br>92|1<br>91<br>92<br>~~93~~<br>92<br>92|1<br>91<br>92<br>~~93~~<br>92<br>92|1<br>91<br>92<br>~~93~~<br>92<br>92|1<br>91<br>92<br>~~93~~<br>92<br>92|1<br>91<br>92<br>~~93~~<br>92<br>92|1<br>91<br>92<br>~~93~~<br>92<br>92|1<br>91<br>92<br>~~93~~<br>92<br>92|
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||||85<br>~~89~~|85<br>~~89~~|||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||MB|ench|MME|MM|-Vet|VQAT|ext|PO|PE||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||99||Globa|l on|ly|Loca|l on|ly|~~99~~||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]|||9<br>|8<br>|||||~~97~~<br>|||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]|||96|~~95~~|96|||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||||3||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||92|92<br>|||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||MB<br> 6<br>ra<br> co<br> d l<br>          ra<br>             er<br>  PE<br>          m<br>           t<br>|ench <br>: U<br>teg<br> mp<br>  oca<br>          l vi<br>             al p<br>   [4<br>          ant<br>           he<br>      the<br>          s wi<br>    te o<br>            man<br>     ual<br>         dec<br>   e id|MME<br>P - <br>y.<br><br> aris<br>  l cro<br>           sual<br>              erc<br>2],<br>          ic u<br>            holi<br>|MM<br> F<br>DO<br> on<br>   ps<br>            u<br>              ep<br> w<br>           n<br>            st<br>|-Vet<br>ast<br>W<br>  of<br>    a<br>            nd<br>              tio<br> hic<br>           de<br>            ic<br>|VQAT<br>V v<br>N <br>   th<br>    s in<br>            ers<br>              n t<br> h s<br>           rsta<br>             co<br>|ext<br>.s.<br>- P<br>   e t<br>     pu<br>            ta<br>               as<br>  ug<br>           n<br>             nte<br>        de<br>             LM<br>        H<br>             te<br>       in<br>          tte<br>      V|PO<br> <br>e<br>    hu<br>     ts<br>            nd<br>               ks<br>  g<br>           di<br>             x<br>|PE<br>Ra<br>rf<br>    m<br>     .<br>            in<br>                b<br>  es<br>           ng<br>             t<br>|n<br>or<br>    b<br>            g<br>                u<br>  ts<br>           .<br>              o<br>         u<br>              n<br>        V<br>             s.<br>        e<br>          n|
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|LL<br>           thi<br>     n h<br>            tic<br>      tok<br>         ent<br>    ea|M<br>           n t<br>      o<br>             co<br>      en<br>         ral<br>     of|d<br>            he<br>      w<br>             m<br>       p<br>         iz<br>      H|co<br>             L<br>       our<br>             ple<br>       run<br>         e a<br>      olo|co<br>             L<br>       our<br>             ple<br>       run<br>         e a<br>      olo|r,<br>             .<br>        o<br>             n<br>       g,<br>          n<br>      .|o<br>              A<br>        lo<br>             es<br>        w<br>          tio|o<br>              A<br>        lo<br>             es<br>        w<br>          tio|
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|LL<br>           thi<br>     n h<br>            tic<br>      tok<br>         ent<br>    ea|M<br>           n t<br>      o<br>             co<br>      en<br>         ral<br>     of|d<br>            he<br>      w<br>             m<br>       p<br>         iz<br>      H||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|LL<br>           thi<br>     n h<br>            tic<br>      tok<br>         ent<br>    ea|M<br>           n t<br>      o<br>             co<br>      en<br>         ral<br>     of|d<br>            he<br>      w<br>             m<br>       p<br>         iz<br>      H||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|LL<br>           thi<br>     n h<br>            tic<br>      tok<br>         ent<br>    ea|M<br>           n t<br>      o<br>             co<br>      en<br>         ral<br>     of|d<br>            he<br>      w<br>             m<br>       p<br>         iz<br>      H||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|e<br>          n<br>    ra<br>            e<br>     s<br>        o<br>   or|LL<br>           thi<br>     n h<br>            tic<br>      tok<br>         ent<br>    ea|M<br>           n t<br>      o<br>             co<br>      en<br>         ral<br>     of|d<br>            he<br>      w<br>             m<br>       p<br>         iz<br>      H||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||||||||
|MMBench MME<br>MM-Vet VQAText<br>POPE<br>85<br>90<br>95<br>100<br>Performance (%)<br>92<br>91<br>91<br>92<br>85<br>96<br>~~93~~<br>92<br>~~89~~<br>92<br>~~ FastV~~<br>~~ Random~~<br>MMBench MME<br>MM-Vet VQAText<br>POPE<br>92<br>94<br>96<br>98<br>100<br>Performance (%)<br>99<br>98<br>96<br>92<br>~~97~~<br>96<br>~~95~~<br>92<br>93<br>~~99~~<br>Global only<br>Local only<br>Figure 6: UP - FastV v.s. Ran<br>dom strategy.<br>DOWN - Perfor<br>mance comparison of the thumb<br>nail and local crops as inputs.<br>        <br>holistic context retention strategy,_ i.e._, pruning visual tokens through<br>random masks to retain visual information from different regions. As<br>shown in Fig. 6 up, compared with FastV, this random strategy out-<br>performs on more than half of the benchmarks, which demonstrates<br>the signifcance of preserving holistic context for visual understand-<br>ing. On the VQA text dataset, however, the random strategy failed,<br>possibly because random pruning discards some salient fne-grained<br>information. This result also suggests that local saliency is indispens-<br>able, especially for densely packed elements within small regions.<br>In addition, we conducted an exploratory experiment to investigate<br>how holistic context contributes to visual understanding in MLLMs.<br>Specifcally, we use the global thumbnail and multiple local crops<br>as visual input separately [47], and evaluate performance on the two<br>settings against various benchmarks. As shown in Fig. 6 down, with<br>only the global thumbnail yields strong results on general visual<br>perception benchmarks such as MMBench [51], MME [21], and<br>MM-Vet [90], highlighting the inherent role of holistic context in guiding general visual understanding<br>On the contrary, using only local crops leads to poor performance in these general perception tasks bu<br>excels in fne-grained perception benchmarks such as TextVQA [65] and POPE [42], which suggests<br>that local duplicated saliency can offer fne-grained visual information for semantic understanding.<br>**4**<br>**Methodology**<br>Building on the above analysis, we propose HoloV, which better preserves the holistic context o<br>images for visual understanding. By removing redundant visual tokens before the LLM decoder, ou<br>approach could make MLLMs inference faster than methods that prune tokens within the LLM. An<br>overview of our approach is depicted in Fig. 7. In what follows, we elaborate on how our HoloV<br>guides overall visual token compression under a high pruning ratio to keep semantic completeness.<br>**4.1**<br>**HoloV Framework**<br>To address the pivotal question raised in Sec. 1 for effective and effcient visual token pruning, we<br>propose HoloV framework, which leverages crop-wise adaptive allocation to decentralize attention<br>over those non-highlighted but heterogeneous tokens. Fig. 7 illustrates the core idea of HoloV.<br>...<br>[CLS]<br>...<br>...<br>[CLS]||||||||||||
|Crop-wise Tokens<br>Reserved Tokens (Intra- and Inter-Crop)<br>Discarde|Crop-wise Tokens<br>Reserved Tokens (Intra- and Inter-Crop)<br>Discarde|d T|oken|s wi|th|Hi|gh A|tte|nt|ion||
|Figure 7: Illustration of HoloV. We re-rank highlighted visual tokens for ho<br>Based on our fndings about the positional bias, We frst rearrange visual<br>Let the total number of image tokens be_ Nv_, which is evenly partitioned int<br>the model to maintain spatial granularity and gather statistics both locally<br>5|Figure 7: Illustration of HoloV. We re-rank highlighted visual tokens for ho<br>Based on our fndings about the positional bias, We frst rearrange visual<br>Let the total number of image tokens be_ Nv_, which is evenly partitioned int<br>the model to maintain spatial granularity and gather statistics both locally<br>5|lis<br>    to<br>     o <br>           an|tic<br>    ken<br>_ C_ c<br>           d g|con<br>    s in<br>rop<br>            lob|te<br>     to<br>s.  <br>            all|xt<br>      l<br>T<br>            y.|ret<br>      oca<br>his<br> Gi|en<br>      l<br> en<br>ve|ti<br>       cr<br> a<br>n|o<br>       o<br> bl<br> t|n.<br>       ps<br> es<br> he|


normalized embeddings **Z** _[c]_ _v_ _[∈]_ [R] _[M]_ _[×][d]_ [ in] _[ c]_ [-th crop, we first compute intra-crop similarity matrix] **[ S]** _[c]_ [ as]

**S** _[c]_ = ( **1** _−_ **I** _M_ ) _⊙_ **Z** _[c]_ _v_ **[Z]** _[c]_ _v_ _⊤,_ (1)


where _⊙_ denotes Hadamard product, and **I** _M_ is the identity matrix masking self-similarities. Then,
we capture intra-crop diversity by the variance of semantic distribution, the formula is as follows



1
_Vi_ _[c]_ [=]
_M_ _−_ 1



�� **S** _[c]_ _i,j_ _[−]_ _[µ]_ _i_ _[c]_ �2 _,_ (2)



where a high value of _Vi_ _[c]_ [indicates that] _[ i]_ [-th token has diverse connections with others, the visual]
semantics expressed by the informative token is essential within the crop. To obtain holistic attention,
we establish a balanced scoring mechanism combining contextual diversity and attention saliency.
Specifically, we merge variance _V_ _[c]_ and [CLS] attention _A_ _[c]_ in the crop using adaptive scaling:


_H_ _[c]_ = _γcV_ _[c]_ + _A_ _[c]_ _,_ where _γc_ = E[ _∥A_ _[c]_ _∥_ ] _/_ E[ _∥V_ _[c]_ _∥_ ] _._ (3)


**Adaptive holistic token allocation.** To preserve overall scene semantics and spatial diversity, we
compute a crop-level priority score by averaging token scores within each crop. The total quota for
selected image tokens _T_ _[′]_ is dynamically allocated to crops according to their normalized crop-level
importance. The allocation to each crop is discrete and capped, ensuring spatial coverage while
preventing over-concentration on specific regions. We resolve rounding and overflow through an
iterative reallocation procedure, so that crops with excess quota donate surplus tokens to those with
remaining capacity, according to their crop-level scores.


We compute crop importance weights via



_C_




�( [1]

_M_

_c_ _[′]_ =1



_wc_ = ( [1]

_M_



_M_

- _Ht_ _[c]_ [)] _[τ]_ _[/]_

_t_ =1



_M_



_M_

- _Ht_ _[c][′]_ [)] _[τ]_ _[,]_ (4)

_t_ =1



where _τ_ controls the sharpness of allocation. Thus, initial quota _qc_ = _⌊wcN_ [ˆ] _v⌋_, where _N_ [ˆ] _v_ denotes the
number of retained tokens. When the allocated tokens overflow or fall short, we redistribute residual
tokens. For overflow, the quota is changed by _qc_ = min( _qc_ + ∆ _c, M_ ) _,_ ∆ _c_ _∝_ _wc ·_ ( _M_ _−_ _qc_ ), while
for fall short, we allocate the remaining quota to the crop with the highest weight. In this way, HoloV
adaptively adjusts its compression degree according to the informativeness of different crops.


**Top-** _k_ **visual token selection.** Within each crop, select visual tokens by maximizing:


argmaxΩ _c⊂{_ 1 _,...,M_ _}_           - _H_ _[c]_ _,_ subject to _|_ Ω _c|_ = _qc,_ (5)


which ensures both crop-wise local saliency and global relevance. We retain top- _k_ visual tokens in
each crop, where _k_ is determined by the quota _qc_ in the allocation. By performing token pruning
before the LLM decoder, we dynamically adjust the number of visual tokens as input to the language
model based on the actual computational budget, thus accelerating the MLLM inference.


**4.1.1** **Fast Visual Context Refetching**


Motivated by the attention sinks [94], and information loss during visual token pruning, we further
propose visual context refetching to fast supplement the visual holistic context. Specifically, we treat
pruned tokens as supplementary evidence, re-injecting them into the MLLM through Feed Forward
Network (FFN) as “key-value memory” at the middle trigger layer. This _refetch_ mechanism occurs
when the model exhibits high uncertainty during inference, achieving effective and efficient visual
information replenishment. Limited by space, the details can be found in Appendix D.


**4.2** **Theoretical Analysis**


To further justify the trustworthiness of our proposed HoloV, we provide a theoretical analysis of
it. Under Assumption 1, for any pruned token, there exists a retained token that is sufficiently close
in the embedding space, with bounded context variance. By leveraging the _Lipschitz continuity_ [8]
of the transformer layer, we can bound the semantic difference between the outputs on the original
and pruned token sets. The residual error introduced by the scoring threshold is also controlled.
Combining these components, we obtain the stated upper bound. More details are in Appendix C.


6


Table 1: Performance comparison of various methods across different benchmarks. Results are shown
for different pruning ratios, with accuracy and average performance highlighted. Best results in **blue** .
.
**Methods** **GQA** **MMB** **MMB** CN **MME** **POPE** **SQA** **VQA** V2 **VQA** Text **VizWiz** **Average**


**4.3** **Computational Complexity**


As language instructions are much shorter than visual tokens, we focus on the FLOPs contributed
by visual tokens. Let _n_ denote the number of visual tokens, _d_ the hidden size, and _m_ the FFN
intermediate size (with SwiGLU). For the prefill stage, the FLOPs per transformer layer can be
approximated as _an_ [2] _d_ + _bnd_ [2] + _cndm_, where _a_, _b_, and _c_ are constants. If the token count is reduced
by a ratio _R_ ( _n_ ˆ = (1 _−_ _R_ ) _n_ ), the FLOPs reduction ratio is:

_F_ = 1 _−_ _[a][n]_ [ˆ][2] _[d]_ [ +] _[ b][nd]_ [ˆ] [2][ +] _[ c][ndm]_ [ˆ] (6)

_an_ [2] _d_ + _bnd_ [2] + _cndm_ _[.]_


For large _n_, the quadratic term dominates, so _F_ _≈_ 1 _−_ (1 _−_ _R_ ) [2] = 2 _R −_ _R_ [2] . Thus, the reduction is
slightly better than linear in _R_ . In the decode stage (with KV cache), the complexity becomes linear
in _n_, and the FLOPs per layer are _bd_ [2] + ( _bd_ + _cdm_ ) _n_, so the reduction is nearly proportional to _R_ .
HoloV speeds up inference by pruning ahead of the LLM to avoid KV cache inefficiency.


**5** **Experiments**


**5.1** **Experimental Setup**


**Benchmarks.** We conducted experiments on several widely used visual understanding benchmarks.
For image understanding task, we performed experiments on ten widely used benchmarks, including
GQA [30], MMBench (MMB) and MMB-CN [51], MME [21], POPE [42], VizWiz [9], SQA
(ScienceQA) [52], VQAV2 (VQA V2) [23], VQAText (TextVQA) [65], and MM-Vet [90]. Video


7



![](images/paper.pdf-6-0.png)
80

75

70

65

60

55


90

80

70

60

50

40



![](images/paper.pdf-7-2.png)

![](images/paper.pdf-7-6.png)

0.25 0.50 0.75 0.95
pruning ratio



![](images/paper.pdf-7-1.png)



![](images/paper.pdf-7-3.png)



![](images/paper.pdf-7-0.png)

















0.25 0.50 0.75 0.95
pruning ratio



60


55


50


45


65


60


55


50



0.25 0.50 0.75 0.95
pruning ratio



0.25 0.50 0.75 0.95
pruning ratio



58
56
54
52
50
48


34
32
30
28
26
24
22
20



0.25 0.50 0.75 0.95
pruning ratio



![](images/paper.pdf-7-7.png)



![](images/paper.pdf-7-5.png)





![](images/paper.pdf-7-4.png)















0.25 0.50 0.75 0.95
pruning ratio



0.25 0.50 0.75 0.95
pruning ratio



70


65


60


55


50


60

55

50

45

40

35



0.25 0.50 0.75 0.95
pruning ratio



Figure 8: Comparison of different methods across multiple benchmarks under varying pruning ratios.


QA benchmarks include MSVD-QA and MSRVTT-QA [83]. All experiments on these benchmarks
follow the default settings. More details of the benchmarks are provided in Appendix A.1.


**Comparison methods.** We compare our approach with several representative methods for accelerating multi-modal language models (MLLMs) via token reduction, including ToMe [11], FastV [13],
SparseVLM [96], HiRED [4], LLaVA-PruMerge [64], PDrop [81], MustDrop [49], FasterVLM [91],
GlobalCom [2] [50], VisionZip [86], DART [79]. These baselines employ diverse strategies such as
token merging, attention-based pruning, adaptive allocation, and hierarchical retention to improve
efficiency by reducing redundant tokens. Each method offers a unique perspective on balancing
computational cost and model performance. More details of these baselines are provided in Appendix
A.2.


**5.2** **Main Results**


**General-purpose benchmarks** . We evaluate the performance of HoloV on general-purpose datasets,
_i.e._, GQA, MM-Vet, MME, MMBench, SQA, and VizWiz. As shown in Tab. 1, HoloV consistently outperforms competing approaches at different pruning ratios, _e.g._, HoloV removes up to
88.9% of visual tokens with only a 4.2% performance drop, and 77.8% with just 2% on average.



Further, we show more results under varying pruning ratios, as shown in
Fig. 8, the performance of FastV and SparseVLM drops dramatically under high pruning ratios, while HoloV maintains robust performance with
relatively minor losses at all pruning ratios on SQA and MMBench. On
MMBench _CN_ and MM-Vet, HoloV even achieves higher than baseline
(unpruned) scores at pruning ratios of 25%, 50%, and 75% (MM-Vet),
then the score slowly drops as the pruning ratio increases. For VizWiz
evaluation, the result in Fig. 9 indicates that HoloV can consistently
obtain performance improvements at different pruning ratios, even at
95%, which means HoloV effectively retains visual holistic semantics.



![](images/paper.pdf-7-8.png)

Figure 9: Performance of
different methods on VizWiz
under varying pruning ratios.



53

52

51

50

49

48







0.25 0.50 0.75 0.95
pruning ratio



**Hallucination benchmarks validation** . We conduct the hallucination under varying pruning ratios.
evaluations on POPE and MME benchmarks, with results on LLaVA1.5-7B presented in Tab. 1, where the proposed HoloV shows robust capabilities, and the performance
significantly exceeds the results of the compared SOTA methods, _e.g._, with a pruning rate of 88.9%,
HoloV achieves 80.3% accuracy compared to 76% for the second runner-up on POPE, and achieved
desirable performance on MME evaluation, compared to other comparative approaches.



**5.3** **HoloV with Higher Resolution**


For further comprehensive evaluation, we also evaluated HoloV for LLaVA-NeXT on different benchmarks mentioned above, with comparison to current SOTA approaches. LLaVA-NeXT introduces a
new image processing method, leading to dynamic lengths of visual embeddings for various image
inputs. Thus, during the evaluation, 320 visual tokens has been kept (from up to 2880 raw tokens).
As shown in Table 3, the evaluation results of all various benchmarks show that HoloV obtained the
highest score on almost every track, and has an average of 95. 6%, much higher than the current
SOTA of 93.3%.


8


Table 3: Performance comparison of various methods across different benchmarks. Results are shown
for different pruning ratios, with accuracy and average performance highlighted. Best results in **blue** .

|Methods|GQA MMB MMBCN MME POPE SQA VQAV2 VQAText VizWiz|Average|
|---|---|---|
|Upper Bound, 2880 Tokens|64.2<br>67.4<br>60.6<br>1851<br>86.5<br>70.1<br>81.8<br>64.9<br>57.6|100%|
|LLaVA-NeXT 7B<br>FastV (ECCV24)<br>LLaVA-PruMerge (ICCV25)<br>PDrop (CVPR25)<br>MustDrop (2024.11)<br>FasterVLM (ICCV25)<br>HiRED (AAAI25)<br>SparseVLM (ICML25)<br>GlobalCom2 (2025.3)<br>DART (EMNLP25)<br>HoloV (Ours)|_Retain 320 Tokens_ (_↓_**88**_._**9**%)<br>55.9<br>61.6<br>51.9<br>1661<br>71.7<br>62.8<br>71.9<br>55.7<br>53.1<br>88.0%<br>53.6<br>61.3<br>55.3<br>1534<br>60.8<br>66.4<br>69.7<br>50.6<br>54.0<br>85.6%<br>56.4<br>63.4<br>56.2<br>1663<br>77.6<br>67.5<br>73.5<br>54.4<br>54.1<br>90.9%<br>57.3<br>62.8<br>55.1<br>1641<br>82.1<br>68.0<br>73.7<br>**59.9**<br>54.0<br>92.2%<br>56.9<br>61.6<br>53.5<br>1701<br>83.6<br>66.5<br>74.0<br>56.5<br>52.6<br>91.1%<br>59.3<br>64.2<br>55.9<br>1690<br>83.3<br>66.7<br>75.7<br>58.8<br>54.2<br>93.3%<br>56.1<br>60.6<br>54.5<br>1533<br>82.4<br>66.1<br>71.5<br>58.4<br>52.0<br>89.7%<br>57.1<br>61.8<br>53.4<br>1698<br>83.8<br>67.4<br>76.7<br>57.2<br>54.6<br>92.2%<br>61.7<br>65.3<br>**58.2**<br>1710<br>**84.1**<br>68.4<br>79.1<br>58.7<br>**56.1**<br>93.9%<br>**61.7**<br>**65.3**<br>57.5<br>**1738**<br>83.9<br>**68.9**<br>**79.5**<br>58.7<br>55.3<br>**95.6%**|_Retain 320 Tokens_ (_↓_**88**_._**9**%)<br>55.9<br>61.6<br>51.9<br>1661<br>71.7<br>62.8<br>71.9<br>55.7<br>53.1<br>88.0%<br>53.6<br>61.3<br>55.3<br>1534<br>60.8<br>66.4<br>69.7<br>50.6<br>54.0<br>85.6%<br>56.4<br>63.4<br>56.2<br>1663<br>77.6<br>67.5<br>73.5<br>54.4<br>54.1<br>90.9%<br>57.3<br>62.8<br>55.1<br>1641<br>82.1<br>68.0<br>73.7<br>**59.9**<br>54.0<br>92.2%<br>56.9<br>61.6<br>53.5<br>1701<br>83.6<br>66.5<br>74.0<br>56.5<br>52.6<br>91.1%<br>59.3<br>64.2<br>55.9<br>1690<br>83.3<br>66.7<br>75.7<br>58.8<br>54.2<br>93.3%<br>56.1<br>60.6<br>54.5<br>1533<br>82.4<br>66.1<br>71.5<br>58.4<br>52.0<br>89.7%<br>57.1<br>61.8<br>53.4<br>1698<br>83.8<br>67.4<br>76.7<br>57.2<br>54.6<br>92.2%<br>61.7<br>65.3<br>**58.2**<br>1710<br>**84.1**<br>68.4<br>79.1<br>58.7<br>**56.1**<br>93.9%<br>**61.7**<br>**65.3**<br>57.5<br>**1738**<br>83.9<br>**68.9**<br>**79.5**<br>58.7<br>55.3<br>**95.6%**|



Table 4: Real inference comparison on POPE. Experiments adopt 66.7% and 90% pruning ratios.


|Methods|Time Prefill Latency Mem. Acc.|Time Prefill Latency Mem. Acc.|
|---|---|---|
|Upper Bound, 576 Tokens|49:41<br>0.5ms<br>0.334s<br>19.0G<br>100.%|49:41<br>0.5ms<br>0.334s<br>19.0G<br>100.%|
|LLaVA-1.5-7B<br>FastV (ECCV24)<br>MustDrop (2024.11)<br>FasterVLM (ICCV25)<br>HiRED (AAAI25)<br>SparseVLM (ICML25)<br>HoloV (Ours)|_Retain 192 Tokens_ (_↓_**66**_._**7**%)<br>_Retain 58 Tokens_ (_↓_**90**%)<br>35:34<br>0.5ms<br>0.239s<br>16.0G<br>75.4%<br>30:41<br>0.5ms<br>0.206s<br>15.6G<br>66.8%<br>32:30<br>0.5ms<br>0.273s<br>15.6G<br>96.2%<br>29:40<br>0.6ms<br>0.199s<br>14.5G<br>87.1%<br>**30:09**<br>0.5ms<br>**0.202s**<br>15.6G<br>100.%<br>25:08<br>0.5ms<br>**0.168s**<br>14.5G<br>92.5%<br>30:08<br>0.6ms<br>0.210s<br>15.7G<br>96.4%<br>**25:03**<br>0.6ms<br>0.168s<br>14.5G<br>92.7%<br>40:51<br>0.6ms<br>0.251s<br>15.8G<br>97.3%<br>31:28<br>0.6ms<br>0.212s<br>14.6G<br>92.3%<br>31:02<br>0.5ms<br>0.208s<br>**15.6G**<br>**99.7%**<br>27:36<br>0.5ms<br>0.176s<br>**14.5G**<br>**95.7%**|_Retain 192 Tokens_ (_↓_**66**_._**7**%)<br>_Retain 58 Tokens_ (_↓_**90**%)<br>35:34<br>0.5ms<br>0.239s<br>16.0G<br>75.4%<br>30:41<br>0.5ms<br>0.206s<br>15.6G<br>66.8%<br>32:30<br>0.5ms<br>0.273s<br>15.6G<br>96.2%<br>29:40<br>0.6ms<br>0.199s<br>14.5G<br>87.1%<br>**30:09**<br>0.5ms<br>**0.202s**<br>15.6G<br>100.%<br>25:08<br>0.5ms<br>**0.168s**<br>14.5G<br>92.5%<br>30:08<br>0.6ms<br>0.210s<br>15.7G<br>96.4%<br>**25:03**<br>0.6ms<br>0.168s<br>14.5G<br>92.7%<br>40:51<br>0.6ms<br>0.251s<br>15.8G<br>97.3%<br>31:28<br>0.6ms<br>0.212s<br>14.6G<br>92.3%<br>31:02<br>0.5ms<br>0.208s<br>**15.6G**<br>**99.7%**<br>27:36<br>0.5ms<br>0.176s<br>**14.5G**<br>**95.7%**|



Besides, on video understanding benchmarks, HoloV maintains close to the original performance, significantly outperforming
FasterVLM and FastV, as shown in Table 2.
This demonstrates the value of HoloV when
it comes to high-resolution visual input.


**5.4** **Efficiency Analysis**



Table 2: Video QA Evaluations of different methods with
50% of visual tokens retained. HoloV beats SOTA.


**MSVD-QA** **MSRVT-QA** **Avgerge**



![](images/paper.pdf-8-0.png)

To assess the efficiency of HoloV, we compare total inference time, prefill time, end-to-end latency,
GPU memory usage, and accuracy on LLaVA-1.5-7B. As shown in Tab. 4, under a 90% pruning ratio,
HoloV achieves a 42.7% reduction in inference time and a 42.8% decrease in latency, with only a 4.3%
drop in accuracy, similarly under 66.7% pruning ratio. Compared to FastV and SparseVLM, HoloV
uses less memory and runs faster. Although FasterVLM offers slightly quicker inference, HoloV
improves accuracy by 3.0%, demonstrating a better balance between efficiency and performance.


**5.5** **Ablation Analysis of Crop Numbers**



Table 5: Ablation of different crop numbers.



Partition granularity does not affect pruning effi- Table 5:
ciency: retained visual tokens are determined by prun- **Methods** **# = 4** **# = 8** **# = 12** **# = 16**

dynamically via intra-crop visual token informative
high-resolution images, dynamic crop number adjust
areas and more for low-detail regions. Specifically,
Table 5 shows results when total crops vary from 4 to
16, where the values represent percentages relative to
original performance. We observe no significant performance impact from varying crop numbers.



**Methods** **# = 4** **# = 8** **# = 12** **# = 16**



9



![](images/paper.pdf-8-1.png)
_Original image_ _Prune =50%_ _Prune =70%_ _Prune =87.5%_



_Original image_ _Prune =50%_ _Prune =70%_ _Prune =87.5%_



Figure 10: The case comparison between FastV and HoloV from the GQA. It presents original images
alongside their pruned versions at pruning rates of 50%, 70%, and 87.5%. The bounding boxes
highlight specific regions and objects across images, where HoloV well preserves the pivotal tokens.


**5.6** **Visualization Analysis**


Further, we visualize retained visual patches under different pruning rates. As shown in Fig. 10,
black areas indicate discarded tokens, while colored regions show key semantic areas aligned with
text. Compared to FastV, HoloV preserves more relevant visual cues even under high pruning (e.g.,
87.5%), effectively filtering out redundant visual tokens while keeping critical objects. This supports
better cross-modal alignment, allowing pivotal holistic tokens for visual overall understanding.


**5.7** **HoloV with Qwen Architecture**



Table 6: Comparative Experiments on Qwen2.5-VL-7B.



To verify the architectural generaliza- Table 6: Comparative Experiments on Qwen2.5-VL-7B.
tion of HoloV beyond LLaVA-based **Methods** **MMB** **MME** **POPE** **SQA** **VQA** Text **Avg.**

FastV at various reduction ratios, high
achieves average performance retention
rates of 94.6%, 92.7%, and 90.5% at 66.7%, 77.8%, and 88.9% token pruning rates respectively,
significantly higher than FastV’s 92.3%, 89.2%, and 84.3% performance. These results show that our
proposed holistic pruning strategy effectively generalizes across different MLLM architectures.



**Methods** **MMB** **MME** **POPE** **SQA** **VQA** Text **Avg.**





![](images/paper.pdf-9-20.png)

**6** **Conclusion**


We present HoloV, a holistic token pruning framework that addresses two critical limitations of
attention-based visual compression: 1) semantic fragmentation from over-pruning non-salient regions,
and 2) static importance estimation ignoring token interdependencies. The core innovation lies in
variance-modulated dynamic scoring and capacity-constrained allocation, which preserve holistic
context. Extensive experiments validate our method’s effectiveness in maintaining both perceptual
details and abstract spatial reasoning capabilities under aggressive token reduction.


**Acknowledgments and Disclosure of Funding**


This work was supported by the National Natural Science Foundation of China (Grant No.62506318);
Guangdong Provincial Department of Education Project (Grant No.2024KQNCX028); CAAI-Ant


10


Group Research Fund; Scientific Research Projects for the Higher-educational Institutions (Grant
No.2024312096), Education Bureau of Guangzhou Municipality; Guangzhou-HKUST(GZ) Joint
Funding Program (Grant No.2025A03J3957), Education Bureau of Guangzhou Municipality.


11


**References**


[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv_
_preprint arXiv:2303.08774_, 2023. 3


[2] Yael Adini, Dov Sagi, and Misha Tsodyks. Context-enabled learning in the human visual system. _Nature_,
415(6873):790–793, 2002. 2


[3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc,
Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for
few-shot learning. _Advances in Neural Information Processing Systems_, 35:23716–23736, 2022. 26


[4] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S Nikolopoulos, Hans Vandierendonck, Deepu John, and
Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language
models in resource-constrained environments. _arXiv preprint arXiv:2408.10945_, 2024. 2, 8, 20


[5] Vaswani Ashish. Attention is all you need. _Advances in neural information processing systems_, 30:I,
2017. 4


[6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and
Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint_
_arXiv:2308.12966_, 2023. 3


[7] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang,
Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo
Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_,
2025. 10


[8] Louis Béthune, Thibaut Boissin, Mathieu Serrurier, Franck Mamalet, Corentin Friedrich, and Alberto
Gonzalez Sanz. Pay attention to your loss: understanding misconceptions about lipschitz neural networks.
_Advances in Neural Information Processing Systems_, 35:20077–20091, 2022. 6


[9] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller,
Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual
questions. In _Proceedings of the 23nd annual ACM symposium on User interface software and technology_,
pages 333–342, 2010. 7, 19


[10] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy
Hoffman. Token merging: Your vit but faster. _arXiv preprint arXiv:2210.09461_, 2022. 3


[11] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy
Hoffman. Token merging: Your ViT but faster. In _International Conference on Learning Representations_,
2023. 3, 8, 20


[12] Davide Caffagni, Federico Cocchi, Luca Barsellotti, Nicholas Moratelli, Sara Sarto, Lorenzo Baraldi,
Marcella Cornia, and Rita Cucchiara. The revolution of multimodal large language models: A survey. In
_Findings of the Association for Computational Linguistics:_ _ACL 2024_, pages 13590–13618, 2024. 1


[13] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An
image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language
models. In _European Conference on Computer Vision_, pages 19–35, 2024. 1, 2, 3, 4, 8, 20, 21


[14] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin.
Sharegpt4v: Improving large multi-modal models with better captions. In _European_ _Conference_ _on_
_Computer Vision_, pages 370–387. Springer, 2024. 1


[15] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In _International_
_Conference on Learning Representations (ICLR)_, 2024. 3


[16] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. _Advances in Neural Information Processing Systems_, 35:16344–
16359, 2022. 3


[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, G Heigold, S Gelly, et al. An image is worth
16x16 words: Transformers for image recognition at scale. In _International Conference on Learning_
_Representations_, 2020. 2


12


[18] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv_
_preprint arXiv:2407.21783_, 2024. 3


[19] Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token pruning
for vision-language model acceleration. _arXiv preprint arXiv:2412.13180_, 2024. 1


[20] Zhanzhou Feng and Shiliang Zhang. Efficient vision transformer via token merger. _IEEE Transactions on_
_Image Processing_, 32:4156–4169, 2023. 2


[21] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu
Zheng, Ke Li, Xing Sun, et al. MME: A comprehensive evaluation benchmark for multimodal large
language models. _arXiv:2306.13394_, 2023. 5, 7, 19


[22] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are keyvalue memories. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language_
_Processing_, pages 5484–5495, 2021. 26


[23] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa
matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the_
_IEEE conference on computer vision and pattern recognition_, pages 6904–6913, 2017. 4, 7, 19


[24] Jiaxian Guo, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Boyang Li, Dacheng Tao, and Steven
Hoi. From images to textual prompts: Zero-shot visual question answering with frozen large language
models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages
10867–10877, 2023. 1


[25] Yuhang Han, Xuyang Liu, Pengxiang Ding, Donglin Wang, Honggang Chen, Qingsen Yan, and Siteng
Huang. Rethinking token reduction in mllms: Towards a unified paradigm for training-free acceleration.
_arXiv preprint arXiv:2411.17686_, 2024. 2, 3


[26] Yefei He, Feng Chen, Jing Liu, Wenqi Shao, Hong Zhou, Kaipeng Zhang, and Bohan Zhuang. Zipvl:
Efficient large vision-language models with dynamic token sparsification and kv cache compression.
_arXiv preprint arXiv:2410.08584_, 2024. 3


[27] Kai Huang, Hao Zou, Ye Xi, BoChen Wang, Zhen Xie, and Liang Yu. Ivtp: Instruction-guided visual
token pruning for large vision-language models. In _European Conference on Computer Vision_, pages
214–230. Springer, 2024. 3


[28] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models:
Principles, taxonomy, challenges, and open questions. _ACM_ _Transactions_ _on_ _Information_ _Systems_,
43(2):1–55, 2025. 3


[29] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming
Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multi-modal large language models via
over-trust penalty and retrospection-allocation. In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, pages 13418–13427, 2024. 3


[30] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and
compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and_
_pattern recognition_, pages 6700–6709, 2019. 7, 19


[31] Lei Jiang, Weizhe Huang, Tongxuan Liu, Yuting Zeng, Jing Li, Lechao Cheng, and Xiaohua Xu. Fopru:
Focal pruning for efficient large vision-language models. _arXiv preprint arXiv:2411.14164_, 2024. 3


[32] Yifan Jiang, Kexuan Sun, Zhivar Sourati, Kian Ahrabian, Kaixin Ma, Filip Ilievski, Jay Pujara, et al.
Marvel: Multidimensional abstraction and reasoning through visual evaluation and learning. _Advances in_
_Neural Information Processing Systems_, 37:46567–46592, 2024. 3


[33] Shibo Jie, Yehui Tang, Ning Ding, Zhi-Hong Deng, Kai Han, and Yunhe Wang. Memory-space visual
prompting for efficient vision-language fine-tuning. In _Forty-first International Conference on Machine_
_Learning_, 2024. 26


[34] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual
representation empowers large language models with image and video understanding. In _Proceedings of_
_the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 13700–13710, 2024. 1


13


[35] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language
models. _Advances in Neural Information Processing Systems_, 36:21487–21506, 2023. 1


[36] Jiayi Kuang, Ying Shen, Jingyou Xie, Haohao Luo, Zhe Xu, Ronghao Li, Yinghui Li, Xianfeng Cheng,
Xika Lin, and Yu Han. Natural language understanding and inference with mllm in visual question
answering: A survey. _ACM Computing Surveys_, 57(8):1–36, 2025. 1


[37] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang,
Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_,
2024. 3


[38] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li.
Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. _arXiv preprint_
_arXiv:2407.07895_, 2024. 1


[39] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training
for unified vision-language understanding and generation. In _International_ _Conference_ _on_ _Machine_
_Learning_, pages 12888–12900. PMLR, 2022. 26


[40] Yanwei Li, Chengyao Wang, and Jiaya Jia. LLaMA-VID: An image is worth 2 tokens in large language
models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2024.
3


[41] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng
Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. _arXiv_
_preprint arXiv:2403.18814_, 2024. 1, 3


[42] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object
hallucination in large vision-language models. _arXiv:2305.10355_, 2023. 5, 7, 19


[43] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united
visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_, 2023. 1, 20


[44] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united
visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_, 2023. 3


[45] Hanxiao Liu, Andy Brock, Karen Simonyan, and Quoc Le. Evolving normalization-activation layers.
_Advances in Neural Information Processing Systems_, 33:13539–13550, 2020. 26


[46] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction
tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages
26296–26306, 2024. 3


[47] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next:
Improved reasoning, ocr, and world knowledge, 2024. 3, 5, 20


[48] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in_
_neural information processing systems_, 36, 2024. 3, 20


[49] Ting Liu, Liangtao Shi, Richang Hong, Yue Hu, Quanjun Yin, and Linfeng Zhang. Multi-stage vision
token dropping: Towards efficient multimodal large language model. _arXiv preprint arXiv:2411.10803_,
2024. 1, 3, 8, 20


[50] Xuyang Liu, Ziming Wang, Yuhang Han, Yingyao Wang, Jiale Yuan, Jun Song, Bo Zheng, Linfeng
Zhang, Siteng Huang, and Honggang Chen. Compression with global guidance: Towards training-free
high-resolution mllms acceleration. _arXiv preprint arXiv:2501.05179_, 2025. 8, 20


[51] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi
Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In
_European Conference on Computer Vision_, pages 216–233. Springer, 2025. 5, 7, 19, 21


[52] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord,
Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science
question answering. _Advances in Neural Information Processing Systems_, 35:2507–2521, 2022. 7, 19


[53] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes:
Mixture-of-resolution adaptation for multimodal large language models. _arXiv preprint arXiv:2403.03003_,
2024. 3


14


[54] Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. Llm as
dataset analyst: Subpopulation structure discovery with large language model. In _European Conference_
_on Computer Vision_, pages 235–252. Springer, 2025. 3


[55] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed
video understanding via large vision and language models. In _Proceedings of the 62nd Annual Meeting of_
_the Association for Computational Linguistics (Volume 1:_ _Long Papers)_, pages 12585–12602, 2024. 1


[56] Junzhu Mao, Yang Shen, Jinyang Guo, Yazhou Yao, and Xiansheng Hua. Efficient token compression for
vision transformer with spatial information preserved. _arXiv preprint arXiv:2503.23455_, 2025. 1


[57] Junzhu Mao, Yang Shen, Jinyang Guo, Yazhou Yao, Xiansheng Hua, and Hengtao Shen. Prune and
merge: Efficient token compression for vision transformer with spatial information preserved. _IEEE_
_Transactions on Multimedia_, 2025. 1


[58] Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. Towards interpreting
visual information processing in vision-language models. _arXiv preprint arXiv:2410.07149_, 2024. 2


[59] Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, and Ludwig Schmidt. Improving
multimodal datasets with image captioning. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_,
36:22047–22069, 2023. 1


[60] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with
human feedback. _Advances in neural information processing systems_, 35:27730–27744, 2022. 3


[61] Marius V Peelen, Li Fei-Fei, and Sabine Kastner. Neural mechanisms of rapid natural scene categorization
in human visual cortex. _Nature_, 460(7251):94–97, 2009. 2


[62] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large
language model for long video understanding. In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, pages 14313–14323, 2024. 1


[63] Michael Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner:
Adaptive space-time tokenization for videos. _Advances_ _in_ _neural_ _information_ _processing_ _systems_,
34:12786–12797, 2021. 3


[64] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token
reduction for efficient large multimodal models. _arXiv preprint arXiv:2403.15388_, 2024. 2, 3, 8, 20


[65] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach.
Towards VQA models that can read. In _Proceedings of the IEEE Conference on Computer Vision and_
_Pattern Recognition_, pages 8317–8326, 2019. 5, 7, 19


[66] Dingjie Song, Wenjun Wang, Shunian Chen, Xidong Wang, Michael Guan, and Benyou Wang. Less
is more: A simple yet effective token reduction method for efficient multi-modal llms. _arXiv preprint_
_arXiv:2409.10994_, 2024. 3


[67] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan
Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable
multimodal models. _arXiv preprint arXiv:2312.11805_, 2023. 3


[68] Simon Thorpe, Denis Fize, and Catherine Marlot. Speed of processing in the human visual system. _nature_,
381(6582):520–522, 1996. 2


[69] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut?
exploring the visual shortcomings of multimodal llms. In _Proceedings of the IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition_, pages 9568–9578, 2024. 3


[70] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation
language models. _arXiv preprint arXiv:2302.13971_, 2023. 3


[71] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and
fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023. 3


[72] Joel A Tropp. Greed is good: Algorithmic results for sparse approximation. _IEEE_ _Transactions_ _on_
_Information theory_, 50(10):2231–2242, 2004. 26


15


[73] Dezhan Tu, Danylo Vashchilenko, Yuzhe Lu, and Panpan Xu. Vl-cache: Sparsity and modality-aware kv
cache compression for vision-language model inference acceleration. _arXiv preprint arXiv:2410.23317_,
2024. 3


[74] Shakti N Wadekar, Abhishek Chaurasia, Aman Chadha, and Eugenio Culurciello. The evolution of
multimodal model architectures. _arXiv preprint arXiv:2405.17927_, 2024. 4


[75] Ao Wang, Fengyuan Sun, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding. [cls] token tells
everything needed for training-free efficient mllms. _arXiv preprint arXiv:2412.05819_, 2024. 2, 3


[76] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any
resolution. _arXiv preprint arXiv:2409.12191_, 2024. 1


[77] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun
Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding.
In _European Conference on Computer Vision_, pages 396–416. Springer, 2024. 1


[78] Zichen Wen, Yifeng Gao, Weijia Li, Conghui He, and Linfeng Zhang. Token pruning in multimodal large
language models: Are we solving the right problem? _arXiv preprint arXiv:2502.11501_, 2025. 2


[79] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and
Linfeng Zhang. Stop looking for important tokens in multimodal language models: Duplication matters
more. _arXiv preprint arXiv:2502.11494_, 2025. 2, 8, 20


[80] Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and S Yu Philip. Multimodal large language
models: A survey. In _2023 IEEE International Conference on Big Data (BigData)_, pages 2247–2256.
IEEE, 2023. 1


[81] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He,
Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid
visual redundancy reduction. _arXiv preprint arXiv:2410.17247_, 2024. 3, 8, 20


[82] Bingxin Xu, Yuzhang Shang, Yunhao Ge, Qian Lou, and Yan Yan. freepruner: A training-free approach
for large multimodal model acceleration. _arXiv preprint arXiv:2411.15446_, 2024. 3


[83] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video
question answering via gradually refined attention over appearance and motion. In _Proceedings of the_
_ACM international conference on Multimedia_, pages 1645–1653, 2017. 8, 19, 20


[84] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu,
Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution
images. _arXiv preprint arXiv:2403.11703_, 2024. 3


[85] Yibo Yan, Guangwei Xu, Xin Zou, Shuliang Liu, James Kwok, and Xuming Hu. Docpruner: A storageefficient framework for multi-vector visual document retrieval via adaptive patch-level embedding pruning.
_arXiv preprint arXiv:2509.23883_, 2025. 1


[86] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip:
Longer is better but not necessary in vision language models. _arXiv preprint arXiv:2412.04467_, 2024. 2,
8, 20


[87] Te Yang, Jian Jia, Xiangyu Zhu, Weisong Zhao, Bo Wang, Yanhua Cheng, Yan Li, Shengyuan Liu,
Quan Chen, Peng Jiang, et al. Enhancing instruction-following capability of visual-language models by
reducing image redundancy. _arXiv preprint arXiv:2411.15453_, 2024. 3


[88] Linli Yao, Lei Li, Shuhuai Ren, Lean Wang, Yuanxin Liu, Xu Sun, and Lu Hou. DeCo: Decoupling token
compression from semantic abstraction in multimodal large language models. _arXiv:2405.20985_, 2024. 3


[89] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token
pruning for multi-modal large language models. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, volume 39, pages 22128–22136, 2025. 2, 3


[90] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and
Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In _Forty-first_
_International Conference on Machine Learning_, 2024. 5, 7, 19


[91] Qizhe Zhang, Aosong Cheng, Ming Lu, Zhiyong Zhuo, Minqi Wang, Jiajun Cao, Shaobo Guo, Qi She,
and Shanghang Zhang. [cls] attention is all you need for training-free visual token pruning: Make vlm
inference faster. _arXiv preprint arXiv:2412.01818_, 2024. 2, 3, 4, 8, 20


16


[92] Renshan Zhang, Yibo Lyu, Rui Shao, Gongwei Chen, Weili Guan, and Liqiang Nie. Token-level
correlation-guided compression for efficient multimodal document understanding. _arXiv_ _preprint_
_arXiv:2407.14439_, 2024. 2


[93] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher
Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models.
_arXiv preprint arXiv:2205.01068_, 2022. 3


[94] Xiaofeng Zhang, Chen Shen, Xiaosong Yuan, Shaotian Yan, Liang Xie, Wenxiao Wang, Chaochen Gu,
Hao Tang, and Jieping Ye. From redundancy to relevance: Enhancing explainability in multimodal large
language models. _arXiv e-prints_, pages arXiv–2406, 2024. 6


[95] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin.
Beyond llava-hd: Diving into high-resolution large multimodal models. _arXiv preprint arXiv:2406.08487_,
2024. 1


[96] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy,
Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient
vision-language model inference. _arXiv preprint arXiv:2410.04417_, 2024. 1, 2, 3, 8, 20


[97] Henry Hengyuan Zhao, Pan Zhou, Difei Gao, Zechen Bai, and Mike Zheng Shou. Lova3: Learning to
visual question answering, asking and assessment. _Advances in Neural Information Processing Systems_,
37:115146–115175, 2024. 1


[98] Kening Zheng, Junkai Chen, Yibo Yan, Xin Zou, and Xuming Hu. Reefknot: A comprehensive benchmark
for relation hallucination evaluation, analysis and mitigation in multimodal large language models. _arXiv_
_preprint arXiv:2408.09429_, 2024. 2


[99] Yuke Zhu, Chi Xie, Shuang Liang, Bo Zheng, and Sheng Guo. Focusllava: A coarse-to-fine approach for
efficient and effective visual token compression. _arXiv preprint arXiv:2411.14228_, 2024. 3


[100] Xin Zou, Chang Tang, Xiao Zheng, Zhenglai Li, Xiao He, Shan An, and Xinwang Liu. Dpnet: Dynamic
poly-attention network for trustworthy multi-modal classification. In _Proceedings_ _of_ _the_ _31st_ _ACM_
_international conference on multimedia_, pages 3550–3559, 2023. 26


[101] Xin Zou, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Kening Zheng, Sirui Huang, Junkai Chen, Peijie Jiang,
Jia Liu, Chang Tang, and Xuming Hu. Look twice before you answer: Memory-space visual retracing for
hallucination mitigation in multimodal large language models. _Forty-second International Conference on_
_Machine Learning (ICML)_, 2025. 2


17


**Contents of Technical Appendices**


**A** **Detailed Experiment Settings** **19**


A.1 Benchmarks and Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19


A.2 Backbones and Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20


A.3 Reproducibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21


**B** **More Sparsification Visualization** **21**


B.1 MMBench Finegrained Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21


**C** **Theoretical Analysis of HoloV** **25**


**D** **Fast Visual Context Refetching** **26**


D.1 Preliminary: Reformulation of FFN . . . . . . . . . . . . . . . . . . . . . . . . . 26


D.2 FFN with Visual Context Refetching . . . . . . . . . . . . . . . . . . . . . . . . . 26


D.3 Further Efficiency Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26


**E** **Impact Statement** **27**


18


_∞_ **Technical Appendices and Supplements**


In this appendix, we first provide the details of the experimental setup, including information about the
datasets, model architectures, and comparison methods. Then, we offer a more detailed computational
complexity and theoretical analysis, along with more visualizations and insights.


**A** **Detailed Experiment Settings**


**A.1** **Benchmarks and Metrics**


We conducted experiments on several widely used visual understanding benchmarks. For image
understanding task, we performed experiments on ten widely used benchmarks, including GQA [30],
MMBench (MMB) and MMB-CN [51], MME [21], POPE [42], VizWiz [9], SQA (ScienceQA) [52],
VQAV2 (VQA V2) [23], VQAText (TextVQA) [65], and MMVet [90].


**GQA** [30] The GQA benchmark is composed of three main components: scene graphs, questions,
and images. The image section encompasses not only the images themselves but also their spatial
features and the attributes of all objects within the images. The questions in GQA are specifically
crafted to assess the model’s ability to comprehend visual scenes and engage in reasoning about
different aspects of the images.


**MMBench** [51]. MMBench provides a comprehensive evaluation of a model’s performance across
multiple dimensions. It is structured into three levels of ability dimensions. The first level (L-1)
focuses on two core abilities: perception and reasoning. Building on this foundation, the second level
(L-2) includes six sub-abilities, further elaborating the model’s capabilities. At the third level (L-3),
the evaluation becomes more granular, encompassing 20 specific ability dimensions, thus ensuring a
detailed and multi-faceted analysis of the model’s performance.


**MME** [21]. The MME benchmark is another holistic evaluation framework, designed to thoroughly
assess various facets of a model’s performance. It includes 14 distinct subtasks, each targeting specific
perceptual and cognitive abilities of the model. By employing carefully crafted instruction-answer
pairs and maintaining concise instruction designs, the benchmark minimizes issues such as data
leakage and unfair evaluation, ensuring a fair and reliable performance assessment.


**POPE** [42]. POPE focuses on evaluating the degree of Object Hallucination in models. It reformulates
hallucination evaluation by prompting the model with specific binary questions regarding the presence
of objects in images. Key metrics such as Accuracy, Recall, Precision, and F1 Score are utilized to
measure the hallucination level across three different sampling strategies, providing a robust and
precise evaluation of the model’s object detection and hallucination behavior.


**ScienceQA** [52]. ScienceQA spans many domains, including natural sciences, language sciences,
and social sciences. Questions are categorized within each domain according to topics, categories,
and skills, which results in 26 topics, 127 categories, and 379 skills. This hierarchical categorization
facilitates a thorough and diverse range of scientific questions, enabling an in-depth evaluation of the
model’s multimodal understanding, multi-step reasoning abilities, and interpretability.


**VQA-V2** [23]. VQA-V2 is designed to evaluate a model’s visual perception capabilities through
open-ended questions. It consists of 265,016 images representing a wide variety of real-world scenes
and objects, providing rich visual contexts for the associated questions. Each question is accompanied
by 10 ground truth answers provided by human annotators, enabling a comprehensive evaluation of
the model’s ability to answer questions accurately and effectively.


**TextVQA** [65]. TextVQA focuses on the integration of text within images, evaluating the model’s
ability to comprehend and reason about both the visual and textual information present. The benchmark includes a series of visual question-answering tasks where the model must not only interpret
the visual content but also read and understand the embedded text in order to respond correctly.


**MMVet** [90]. MMVet is designed to assess a model’s ability to solve complex tasks by leveraging
various core vision-language capabilities. It defines six core vision-language capabilities and examines
16 distinct integrations of these capabilities. This allows for a nuanced evaluation of how well models
integrate and utilize multiple vision-language abilities to solve tasks.


**MSVD-QA** [83]. The MSVD-QA benchmark is derived from the Microsoft Research Video Description (MSVD) dataset and consists of 1970 video clips paired with approximately 50.5K question

19


answer pairs. The questions span a wide range of topics and aspects related to the video content,
making it suitable for video question-answering and video captioning tasks. The questions fall into
five categories: what, who, how, when, and where, providing a comprehensive set of queries for
model evaluation.


**MSRVTT-QA** [83]. MSRVTT-QA includes 10,000 video clips and 243,000 question-answer pairs.
One of its primary challenges lies in understanding and reasoning about video content, which involves
both visual and temporal aspects. To answer questions accurately, models must effectively integrate
and process these components. Similar to MSVD-QA, the tasks in MSRVTT-QA are categorized into
five question types: what, who, how, when, and where, allowing for detailed performance evaluation
across multiple dimensions.


**A.2** **Backbones and Baselines**


**Models** . We evaluate HoloV using various open-source MLLMs. For image understanding tasks,
experiments are conducted on the LLaVA family, including LLaVA-1.5 [2] [48] and LLaVA-NeXT [3] [47],
with the latter used to validate performance on high-resolution images. For video understanding tasks,
we use Video-LLaVA [43] as the baseline model. Following the settings reported in their paper.


We analyze multiple representative methods for accelerating MLLM inference through visual token
pruning. These methods share the goal of improving efficiency by reducing redundant visual tokens.


**ToMe** [11] merges similar tokens in visual transformer layers through lightweight matching techniques, achieving acceleration without requiring additional training.


**LLaVA-PruMerge** [64] combines pruning and merging strategies by dynamically removing less
important tokens using sparse CLS-visual attention and clustering retained tokens based on key
similarity.


**FastV** [13] focuses on early-stage token pruning by leveraging attention maps, effectively reducing
computational overhead in the initial layers.


**HiRED** [4] allocates token budgets across image partitions based on CLS token attention, followed
by the selection of the most informative tokens within each partition, ensuring spatially aware token
reduction.


**PDrop** [81] adopts a progressive token-dropping strategy across model stages, forming a pyramid-like
token structure that balances efficiency and performance.


**FasterVLM** [91] evaluates token importance via CLS attention in the encoder and performs pruning
before interaction with the language model, streamlining the overall process.


**MustDrop** [49] integrates multiple strategies, including spatial merging, text-guided pruning, and
output-aware cache policies, to reduce tokens across various stages.


**GlobalCom** [2] [50] introduces a hierarchical approach by coordinating thumbnail tokens to allocate
retention ratios for high-resolution crops while preserving local details.


**SparseVLM** [96] ranks token importance using cross-modal attention and introduces adaptive
sparsity ratios, complemented by a novel token recycling mechanism.


**VisionZip** [86] evaluates token importance via attention in the encoder and clustering retained tokens
based on key similarity.


**DART** [79] introduces a duplication-aware token reduction method that selects a small subset of pivot
tokens, calculates cosine similarity between pivot tokens and remaining tokens, retains those with the
lowest duplication to pivots, achieving significant acceleration while maintaining performance and
good compatibility with efficient attention operators. These methods collectively highlight diverse
approaches to token reduction, ranging from attention-based pruning to adaptive merging, offering
complementary solutions for accelerating MLLMs.


[2https://huggingface.co/liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
[3https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)


20


Table 7: Fine-grained comparison MMBench [51] between FastV and HoloV at high pruning ratios.



HoloV _↓_ **90** % FastV _↓_ **75** % HoloV _↓_ **75** %

(58 Tokens) (144 Tokens) (144 Tokens)



(144 Tokens)



Vanilla
**Category (dev)**
(576 Tokens)



FastV _↓_ **90** %



FastV _↓_ **90** % HoloV _↓_ **90** %

(58 Tokens) (58 Tokens)



Action Recognition 90.7 85.2 85.3 87.0 89.7
Attribute Comparison 50.0 50.0 53.9 52.3 48.7
Attribute Recognition 79.7 68.9 71.7 77.0 79.7
Celebrity Recognition 79.8 76.8 74.7 78.8 78.8
Function Reasoning 75.9 72.2 83.9 75.9 83.9
Future Prediction 45.0 30.0 58.3 40.0 58.3
Identity Reasoning 93.3 86.7 97.5 95.6 97.7
Image Emotion 78.0 58.0 68.7 78.0 76.0
Image Quality 35.8 22.6 38.8 28.3 40.1
Image Scene 96.2 90.4 91.5 96.2 97.1
Image Style 77.4 73.6 71.7 77.4 77.4
Image Topic 83.3 80.6 92.9 83.3 83.3
Nature Relation 41.7 39.6 49.4 37.5 37.5
Object Localization 39.5 35.8 23.3 37.0 38.3
OCR 59.0 59.0 81.8 59.0 84.4
Physical Property Reasoning 50.7 60.3 49.3 53.3 58.0
Physical Relation 33.3 41.7 32.7 41.7 41.7
Social Relation 88.4 53.5 75.8 72.1 75.7
Spatial Relationship 17.8 17.8 18.5 17.8 18.5
Structured Image-Text Understanding 26.9 30.8 21.8 28.2 21.9


**A.3** **Reproducibility**


**Implementaion** **Details** . All of our experiments are conducted on Nvidia A800-80G GPU. The
implementation was carried out in Python 3.10, utilizing PyTorch 2.1.2, and CUDA 11.8. All baseline
settings follow the original paper. We set _numcrop_ = [1024 _/N_ ], where _N_ denotes the number of
retained visual tokens, thus the smaller the quota, the more crops there will be for visual holistic
context retention.


**B** **More Sparsification Visualization**


We conduct a detailed visualization of retained visual patches across varying pruning rates to
illustrate the effectiveness of HoloV. As depicted in Fig. 11, 12, 13, the black regions represent
discarded visual tokens, whereas the colored areas highlight key semantic zones that align with
textual descriptions, demonstrating how HoloV strategically preserves informative content. Compared
to FastV, a representative attention-based pruning method, HoloV exhibits superior capability in
retaining relevant visual cues even at extremely high pruning ratios, such as 87.5%. This is achieved
through its holistic pruning strategy, which prioritizes spatial-semantic diversity over isolated attention
scores. By dynamically allocating pruning budgets across different image crops, HoloV effectively
filters out redundant tokens while safeguarding critical objects and their contextual relationships.
For instance, in complex scenes with multiple interacting elements, HoloV ensures that tokens
corresponding to both focal objects and their surrounding environmental cues are preserved, whereas
FastV tends to over-concentrate on high-attention regions, leading to loss of contextual coherence.
This enhanced preservation of visual holistic understanding facilitates more accurate cross-modal
alignment between visual features and language tokens, enabling MLLMs to maintain robust semantic
reasoning capabilities even under aggressive token reduction. The visualization not only validates
the superiority of HoloV’s design philosophy but also provides empirical evidence of its ability to
balance efficiency and semantic integrity in visual token pruning.


**B.1** **MMBench Finegrained Results**


As shown in Table 7, in the MMBench [51] fine-grained comparison between FastV [13] and HoloV
at 90% and 75% pruning ratios, significant performance improvements are evident with HoloV in
several categories. Specifically, HoloV shows enhanced outcomes in Action Recognition, Attribute
Recognition, Future Prediction, Identity Reasoning, Image Emotion, Image Quality, and Image Scene.
These results underline HoloV’s ability to retain crucial visual information for complex understanding
and response capabilities within dynamic environments.


21


_Original image_ _Prune =25%_ _Prune =50%_ _Prune =70%_ _Prune =87.5%_


Figure 11: The case comparison between FastV and HoloV from the GQA. It presents original images
alongside their pruned versions at pruning rates of 25%, 50%, 70%, and 87.5%. The bounding boxes
highlight specific regions and objects across images, where HoloV well preserves the pivotal tokens.


22



![](images/paper.pdf-21-0.png)

![](images/paper.pdf-21-1.png)

![](images/paper.pdf-21-2.png)

![](images/paper.pdf-21-3.png)

![](images/paper.pdf-21-4.png)

![](images/paper.pdf-21-5.png)

![](images/paper.pdf-21-6.png)

![](images/paper.pdf-21-7.png)

![](images/paper.pdf-21-8.png)

![](images/paper.pdf-21-9.png)

![](images/paper.pdf-21-10.png)

![](images/paper.pdf-21-11.png)

![](images/paper.pdf-21-12.png)

![](images/paper.pdf-21-13.png)

![](images/paper.pdf-21-14.png)

![](images/paper.pdf-21-15.png)

![](images/paper.pdf-21-16.png)

![](images/paper.pdf-21-17.png)

![](images/paper.pdf-21-18.png)

![](images/paper.pdf-21-19.png)

![](images/paper.pdf-21-20.png)

![](images/paper.pdf-21-21.png)

![](images/paper.pdf-21-22.png)

![](images/paper.pdf-21-23.png)

![](images/paper.pdf-21-24.png)

![](images/paper.pdf-21-25.png)

![](images/paper.pdf-21-26.png)

![](images/paper.pdf-21-27.png)

![](images/paper.pdf-21-28.png)

![](images/paper.pdf-21-29.png)
_Original image_ _Prune =25%_ _Prune =50%_ _Prune =70%_ _Prune =87.5%_


Figure 12: The case comparison between FastV and HoloV from the GQA. It presents original images
alongside their pruned versions at pruning rates of 25%, 50%, 70%, and 87.5%. The bounding boxes
highlight specific regions and objects across images, where HoloV well preserves the pivotal tokens.


23



![](images/paper.pdf-22-0.png)

![](images/paper.pdf-22-1.png)

![](images/paper.pdf-22-2.png)

![](images/paper.pdf-22-3.png)

![](images/paper.pdf-22-4.png)

![](images/paper.pdf-22-5.png)

![](images/paper.pdf-22-6.png)

![](images/paper.pdf-22-7.png)

![](images/paper.pdf-22-8.png)

![](images/paper.pdf-22-9.png)

![](images/paper.pdf-22-10.png)

![](images/paper.pdf-22-11.png)

![](images/paper.pdf-22-12.png)

![](images/paper.pdf-22-13.png)

![](images/paper.pdf-22-14.png)

![](images/paper.pdf-22-15.png)

![](images/paper.pdf-22-16.png)

![](images/paper.pdf-22-17.png)

![](images/paper.pdf-22-18.png)

![](images/paper.pdf-22-19.png)

![](images/paper.pdf-22-20.png)

![](images/paper.pdf-22-21.png)

![](images/paper.pdf-22-22.png)

![](images/paper.pdf-22-23.png)

![](images/paper.pdf-22-24.png)

![](images/paper.pdf-22-25.png)

![](images/paper.pdf-22-26.png)

![](images/paper.pdf-22-27.png)

![](images/paper.pdf-22-28.png)

![](images/paper.pdf-22-29.png)
_Original image_ _Prune =25%_ _Prune =50%_ _Prune =70%_ _Prune =87.5%_


Figure 13: The case comparison between FastV and HoloV from the GQA. It presents original images
alongside their pruned versions at pruning rates of 25%, 50%, 70%, and 87.5%. The bounding boxes
highlight specific regions and objects across images, where HoloV well preserves the pivotal tokens.


24



![](images/paper.pdf-23-0.png)

![](images/paper.pdf-23-1.png)

![](images/paper.pdf-23-2.png)

![](images/paper.pdf-23-3.png)

![](images/paper.pdf-23-4.png)

![](images/paper.pdf-23-5.png)

![](images/paper.pdf-23-6.png)

![](images/paper.pdf-23-7.png)

![](images/paper.pdf-23-8.png)

![](images/paper.pdf-23-9.png)

![](images/paper.pdf-23-10.png)

![](images/paper.pdf-23-11.png)

![](images/paper.pdf-23-12.png)

![](images/paper.pdf-23-13.png)

![](images/paper.pdf-23-14.png)

![](images/paper.pdf-23-15.png)

![](images/paper.pdf-23-16.png)

![](images/paper.pdf-23-17.png)

![](images/paper.pdf-23-18.png)

![](images/paper.pdf-23-19.png)

![](images/paper.pdf-23-20.png)

![](images/paper.pdf-23-21.png)

![](images/paper.pdf-23-22.png)

![](images/paper.pdf-23-23.png)

![](images/paper.pdf-23-24.png)

![](images/paper.pdf-23-25.png)

![](images/paper.pdf-23-26.png)

![](images/paper.pdf-23-27.png)

![](images/paper.pdf-23-28.png)

![](images/paper.pdf-23-29.png)
**C** **Theoretical Analysis of HoloV**


To further justify the trustworthiness of our proposed HoloV, we provide a theoretical analysis of it.


**Assumption 1 (Contextual Stability)** _Let Xv_ _be the original visual tokens set, and Rv_ _⊆Xv_ _the_
_retained visual tokens subset, We assume the following:_


_**(C1)**_ _._ _For any pruned visual token xj_ _∈Xv \ Rv, there exists xi_ _∈Rv_ _with:_


_d_ ( _xi, xj_ ) _≥_ _ϵ_ _and_ Var( _d_ ( _xi, N_ ( _xj_ ))) _≤_ _δ_ _,_


_where d means distance function like cosine similarity, N_ ( _xj_ ) _denotes xj’s local context neighbors._


_**(C2)**_ _._ _For H_ ( _xi_ ) = _γV_ ( _xi_ ) + _A_ ( _xi_ ) _satisfies H_ ( _xi_ ) _≥_ _γ_ _for all retained tokens xi_ _∈Rv_


**Lemma C.1 (Token Coverage Guarantee)** _Under_ _**(A1)**_ _, for any pruned token xj, there exists xi_ _∈_
_R such that:_ _√_
_∥xi −_ _xj∥≤_ �2(1 _−_ _ϵ_ ) _∥xj∥_ + _δ_


**Proof C.1** _From the cosine similarity bound, there have x_ _[⊤]_ _i_ _[x][j]_ _[≥]_ _[ϵ][∥][x][i][∥∥][x][j][∥][.]_ _[Using the variance]_
_constraint:_
E[( _x_ _[⊤]_ _i_ _[x][k]_ _[−]_ _[µ]_ [)][2][]] _[ ≤]_ _[δ,]_ _∀xk_ _∈N_ ( _xj_ )

_where µ_ = E[ _x_ _[⊤]_ _i_ _[x][k]_ []] _[.]_ _[Combining via the triangle inequality:]_



_∥xi −_ _xj∥_ [2] = _∥xi∥_ [2] + _∥xj∥_ [2] _−_ 2 _x_ _[⊤]_ _i_ _[x][j]_



_√_
_≤_ 2 _B_ [2] _−_ 2 _ϵB_ [2] +



_√_
= 2(1 _−_ _ϵ_ ) _B_ [2] +



_δ_



_δ_



The lemma shows that pruned tokens can be approximated by retained tokens in Euclidean space.


**Theorem C.1 (Semantic Preservation)** _Let f_ _be a transformer layer with Lipschitz constant L._ _For_
_input embeddings Xv_ _and pruned set Rv_ _satisfying_ _**(C1)-(C2)**_ _:_



_∥f_ ( _Xv_ ) _−_ _f_ ( _Rv_ ) _∥≤_ _L_ - ~~�~~ 2(1 _−_ _ϵ_ ) _B_ + _√_




 _δ_ + _η_ ( _B, γ_ )



_where η_ ( _B, γ_ ) = _O_ - _B_ [2] _/γ_ - _is the residual error from the scoring threshold._


**Proof C.2** _Decompose_ _the_ _error_ _into_ _three_ _components:_ _1)_ _**Geometric**_ _**distortion**_ _:_ _Bounded_ _by_
_√_
_Lemma C.1 2)_ _**Context variance**_ _:_ _Controlled by_ _δ 3)_ _**Scoring residual**_ _:_


_For any x ∈Xv \ Rv_ _with S_ ( _x_ ) _< γ:_


_V_ _[c]_ + _A_ _[c]_ _< γ_ _⇒V_ ( _x_ ) _< γ −A_ ( _x_ )


_Using Cauchy-Schwarz inequality:_



_η_ _≤_ [1]

_γ_




- _∥WV x∥_ [2] _≤_ _[CB]_ [2]

_γ_

_x/∈Rv_



_γ_







_Combining terms via the triangle inequality completes the proof._


This theorem guarantees that, even after pruning, the semantic difference between the outputs of the
transformer for the original.


**Corollary 1 (Dynamic Allocation Optimality)** _The token allocation in Section 4 achieves:_







_kp_




_Spt_

_t_ =1






   _s.t._ _kp_ = _Ntarget_

_p_



max
_{kp}_



_P_

- log


_p_ =1







_with approximation ratio_ 1 _−_ 1 _/e when using greedy selection._


25


**Proof C.3** _The allocation problem is equivalent to maximizing a monotone submodular function._
_Greedy algorithms provide_ (1 _−_ 1 _/e_ ) _-approximation guarantees [72] for such problems._


This corollary shows that your token allocation strategy is not only efficient but also theoretically
near-optimal.


This theoretical framework demonstrates that HoloV: 1) Preserves semantic relationships through
bounded geometric distortion. 2) Context variance is controlled via stability-aware pruning. 3) Token
allocation is provably near-optimal, balancing efficiency and effectiveness.


**D** **Fast Visual Context Refetching**


**D.1** **Preliminary:** **Reformulation of FFN**


Vanilla FFN comprises two fully connected layers with non-linear activation in between. We suppose
_**x**_ _∈_ R _[d]_ as an input token of the FFN, and FFN function can be formulated as

FFN( _**x**_ ) = _ϕ_ ( _**xW**_ 1) _**W**_ _[⊤]_ 2 _[,]_ (7)

where _ϕ_ is activation function like ReLU or SiLU [45], and _**W**_ 1 _,_ _**W**_ 2 _∈_ R _[d][×][D]_ are the weight
matrices, in usual _D_ = 4 _d_ . Peculiarly, _**W**_ 1 and _**W**_ 2 can be rewritten as


_**W**_ 1 = ( _**k**_ 1 _,_ _**k**_ 2 _, . . .,_ _**k**_ _D_ ) _,_ _**W**_ 2 = ( _**v**_ 1 _,_ _**v**_ 2 _, . . .,_ _**v**_ _D_ ) _,_ (8)


where _**k**_ _i,_ _**v**_ _i_ _∈_ R _[d]_ denote entries of key and value, respectively. As a result, the FFN can be
reformulated as
FFN( _**x**_ ) =            - _ϕ_ ( _⟨_ _**x**_ _,_ _**k**_ _i⟩_ ) _·_ _**v**_ _i_ _._ (9)


Thus, the FFN function can be construed as using input _**x**_ as a query to measure similarity with keys,
find matching values, and gather values by similarity, which works like a key-value memory storing
the factual knowledge as found in previous studies [22, 33].


**D.2** **FFN with Visual Context Refetching**


We propose visual context refetching (VCR), _i.e._, reinjecting pruned visual information into the
middle layer of the text decoder during elevated uncertainty during reasoning. This strategy treats
pruned visual tokens as anchors to recalibrate off-target predictions and reduces uncertainties in
_object, attribute, relationship_ tokens. The reason we call this pattern of reinjecting visual evidence
VCR is that the model finds and refreshes key visual memories based on the hidden states in this
process. In particular, inspired by the fact that FFN executes analogous retrieval from its key-value
memory, we consider VCR to serve as a simplified and efficient information re-retrieval process.
Given a hidden token _**x**_ _∈_ R _[d]_ and dimension-aligned vision tokens _**z**_ _v_, FFN with visual context
refetching at _l_ -th layer can be written as follows


FFN [(] _[l]_ [)] ( _**x**_ _∝_ _**z**_ _v_ ) = _α_ ∆ + (1 _−_ _α_ ) FFN [(] _[l]_ [)] ( _**x**_ ) _,_ (10)

where _**z**_ _v_ = ( _**z**_ _v,_ 1 _, . . .,_ _**z**_ _v,Nv_ ) _∈_ R _[d][×][N][v]_, _x ∝_ _**z**_ _v_ denotes execute VCR ∆ from _**x**_ to visual features
_**z**_ _v_, and _α ∈_ [0 _,_ 1] denotes injection ratio of visual memory through the FFN layer which proportional
to image complexity. Specifically, instead of performing retrieval via cross-attention layers as in
previous approaches [39, 3, 100], we consider a simple retrieval process for VCR as,

∆( _**z**_ _v_ _|_ _**x**_ ) =           - _Nv_ (11)

_i_ =1 _[ϕ]_ [(] _[⟨]_ _**[x]**_ _[,]_ _**[ z]**_ _[v,i][⟩]_ [)] _[ ·]_ _**[ z]**_ _[v,i][.]_

From the perspective of FFN, VCR works by treating _**x**_ as a query, and _⟨_ _**z**_ _v,i_ : _**z**_ _v,i⟩_ as new keyvalue entries (visual evidence) to supplement vision-related information in the hidden states. In this
information re-retrieval process, MemVCR does not introduce any parameters that need to be trained.
Notably, since the size of key-value memory _D_ in FFN typically far exceeds the number of visual
tokens _Nv_ (for instance, _D_ = 11008 in LLaMA-7B and _Nv_ = 256 for ViT-L/14, _Nv_ _≪_ _D_ ), the
computation of MemVCR is negligible. Thus, VCR operation is more efficient than the cross-attention
mechanism with quadratic complexity.


**D.3** **Further Efficiency Analysis**


26


![](images/paper.pdf-26-0.png)







As shown in Fig. 14, we conduct efficiency evaluation on LLaVANeXT 7B at 95% pruning ratio, where we also introduce baseline
(unpruned Vanilla) and FastV (95% pruned) for comparison. We
evaluate these approaches using QA pairs from GQA, and the output
length has been set to 1. During evaluation, an A800 80GB GPU has
been used, and the average FLOPs, VMemory usage and throughput
has been calculated, shown in Fig. 14. HoloV reduces over 90% of
FLOPs requirement, 37% lower than FastV, and its VMemory usage
is at the lowest level, while keeping throughput at 5.2 per second,
2.16x and 1.13x faster than baseline and FastV, respectively.


**E** **Impact Statement**



50


40


30


20


10













0

|41 Vanilla FastV HoloV|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||~~17~~<br>~~17~~<br>|~~17~~<br>~~17~~<br>|~~17~~<br>~~17~~<br>|~~17~~<br>~~17~~<br>|
|||15<br>|15<br>|15<br>|15<br>|
|||15<br>||||
||~~6.~~|~~1~~<br>3.8|||2.4<br>4.6<br>5.2|
||~~6.~~|||||

FLOPs(T) Memory(G) Throughput



Figure 14: Inference efficiency comparison between
FastV and HoloV.



This paper presents HoloV, a visual token pruning framework for
MLLMs, and discusses its potential societal impacts. On the positive side, HoloV enhances the
accessibility of multimodal technologies by reducing computational overhead, making advanced applications like medical image analysis and autonomous driving more feasible in resource-constrained
environments such as edge devices or underserved regions. Its efficiency also contributes to energy
sustainability by lowering the energy consumption of MLLM inference, aligning with global efforts
to mitigate the environmental impact of AI. Additionally, by preserving holistic visual context instead
of relying solely on attention-based "highlighted tokens," HoloV may reduce biases in model outputs,
improving fairness in diverse scenarios like visual reasoning involving underrepresented communities.
The framework’s plug-and-play design further accelerates its integration into real-world systems,
driving innovations in education, accessibility tools, and emergency response to enhance societal
resilience.


However, the work also carries potential negative implications. The reduced computational barriers
enabled by HoloV could facilitate misuse, such as the creation of deepfakes or misinformation,
particularly in regions with limited regulatory oversight. While aiming to mitigate attention-based
biases, the framework’s crop-wise token allocation might inadvertently reinforce other biases if
training data lacks diversity, potentially disadvantaging underrepresented groups. Moreover, the
focus on inference efficiency might lead developers to prioritize speed over model interpretability,
raising concerns about accountability in "black-box" deployments for high-stakes tasks like healthcare
diagnostics. Lastly, over-reliance on post-hoc pruning could deter investments in more equitable
training data or architectural improvements, potentially accumulating technical debt and masking
foundational issues in MLLM development.


**Limitations and Future Work** . HoloV demonstrates robust performance in preserving holistic
visual context but faces two key limitations: its dependence on fixed spatial crop partitioning may
hinder fine-grained semantic capture in complex scenes, and minor accuracy declines persist even at
high pruning ratios (e.g., 4.2% drop when pruning 88.9% visual tokens). To address these, future
work could prioritize adaptive crop, sparse attention, multi-modality extensions ( _e.g._, 3D data), and
integration with hallucination mitigation, while optimizing for edge computing energy efficiency.


27


**NeurIPS Paper Checklist**


1. **Claims**


Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?


Answer: [Yes]


Justification: The claims are clearly stated in the abstract and the introduction.


Guidelines:


       - The answer NA means that the abstract and introduction do not include the claims
made in the paper.

       - The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.

       - The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.

       - It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.


2. **Limitations**


Question: Does the paper discuss the limitations of the work performed by the authors?


Answer: [Yes]


Justification: The discussion on the limitations of our work is stated in the paragraph E.


Guidelines:


       - The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.

       - The authors are encouraged to create a separate "Limitations" section in their paper.

       - The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.

       - The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.

       - The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.

       - The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.

       - If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.

       - While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.


3. **Theory assumptions and proofs**


Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?


Answer: [NA]


28


Justification: Our work is motivated by an interesting experimental phenomenon and proposes methods based on this observation, which improves the baseline by a large margin.
There are no assumptions and no following proofs.


Guidelines:


    - The answer NA means that the paper does not include theoretical results.

    - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

    - All assumptions should be clearly stated or referenced in the statement of any theorems.

    - The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.

    - Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.

    - Theorems and Lemmas that the proof relies upon should be properly referenced.


4. **Experimental result reproducibility**


Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?


Answer: [Yes]

Justification: The paper includes the implementation details in the experiment section and
the appendix.


Guidelines:


    - The answer NA means that the paper does not include experiments.

    - If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.

    - If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.

    - Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.

    - While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.


5. **Open access to data and code**


29


Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We provide the dataset URL and code URL as full submission.

Guidelines:


    - The answer NA means that paper does not include experiments requiring code.

    - Please see the NeurIPS code and data submission guidelines [(https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy)
[public/guides/CodeSubmissionPolicy) for more details.](https://nips.cc/public/guides/CodeSubmissionPolicy)

    - While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).

    - The instructions should contain the exact command and environment needed to run to
reproduce the results. [See the NeurIPS code and data submission guidelines (https:](https://nips.cc/public/guides/CodeSubmissionPolicy)
[//nips.cc/public/guides/CodeSubmissionPolicy) for more details.](https://nips.cc/public/guides/CodeSubmissionPolicy)

    - The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.

    - The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.

    - At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).

    - Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.

6. **Experimental setting/details**


Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: We specific experiment settings in Section 5 and Appendix A.

Guidelines:


    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.

    - The full details can be provided either with the code, in appendix, or as supplemental
material.

7. **Experiment statistical significance**


Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [No]

Justification: We don’t need to conduct such an evaluation.

Guidelines:


    - The answer NA means that the paper does not include experiments.

    - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

    - The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

    - The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)


30


    - The assumptions made should be given (e.g., Normally distributed errors).

    - It should be clear whether the error bar is the standard deviation or the standard error
of the mean.

    - It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.

    - For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).

    - If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.

8. **Experiments compute resources**


Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justification: We specific experiment settings in Section 5.4.

Guidelines:


    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.

    - The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).

9. **Code of ethics**


Question: Does the research conducted in the paper conform, in every respect, with the
[NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?](https://neurips.cc/public/EthicsGuidelines)

Answer: [Yes]

Justification: We conducted the research in the paper conform, in every respect, with the
NeurIPS Code of Ethics.

Guidelines:


    - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

    - If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

10. **Broader impacts**


Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [Yes]

Justification: The discussion on both potential positive societal impacts and negative societal
impacts is stated in Appendix E.

Guidelines:


    - The answer NA means that there is no societal impact of the work performed.

    - If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.

    - Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.


31


    - The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.

    - The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.

    - If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).


11. **Safeguards**


Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?


Answer: [NA]


Justification: The paper poses no such risks.


Guidelines:


    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.

    - Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.

    - We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.


12. **Licenses for existing assets**


Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?


Answer: [NA]


Justification: The paper does not use existing assets.


Guidelines:


    - The answer NA means that the paper does not use existing assets.

    - The authors should cite the original paper that produced the code package or dataset.

    - The authors should state which version of the asset is used and, if possible, include a
URL.

    - The name of the license (e.g., CC-BY 4.0) should be included for each asset.

    - For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.

    - If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.

    - For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.


32


    - If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.


13. **New assets**


Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?


Answer: [NA]


Justification: The paper does not release new assets.


Guidelines:


    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.

    - The paper should discuss whether and how consent was obtained from people whose
asset is used.

    - At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.


14. **Crowdsourcing and research with human subjects**


Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?


Answer: [NA]


Justification: The paper does not involve crowdsourcing experiments or research with human
subjects, so no related details are included.


Guidelines:


    - The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.

    - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be
included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.


15. **Institutional** **review** **board** **(IRB)** **approvals** **or** **equivalent** **for** **research** **with** **human**
**subjects**


Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?


Answer: [NA]


Justification: The research described in the paper does not involve study participants or
human subjects, thus questions regarding potential risks, disclosure, or IRB approvals are
not applicable.


Guidelines:


    - The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.

    - Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.


33


    - For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.

16. **Declaration of LLM usage**


Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The paper does not mention the usage of LLMs as a significant or original
component of the core methods.

Guidelines:


    - The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.

    - [Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for](https://neurips.cc/Conferences/2025/LLM)
what should or should not be described.


34


