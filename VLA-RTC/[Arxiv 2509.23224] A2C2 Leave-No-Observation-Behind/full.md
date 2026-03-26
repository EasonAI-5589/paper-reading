# LEAVE NO OBSERVATION BEHIND: REAL-TIME COR-RECTION FOR VLA ACTION CHUNKS

Kohei Sendai Maxime Alvarez Tatsuya Matsushima The University of Tokyo {kohei.sendai, maxime.alvarez, matsushima}@weblab.t.u-tokyo.ac.jp

Yutaka Matsuo Yusuke Iwasawa

The University of Tokyo {matsuo, iwasawa} $@$ weblab.t.u-tokyo.ac.jp

# ABSTRACT

To improve efficiency and temporal coherence, Vision-Language-Action (VLA) models often predict action chunks; however, this action chunking harms reactivity under inference delay and long horizons. We introduce Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA’s action chunk. The module combines the latest observation, the predicted action from VLA (base action), a positional feature that encodes the index of the base action within the chunk, and some features from the base policy, then outputs a per-step correction. This preserves the base model’s competence while restoring closed-loop responsiveness. The approach requires no retraining of the base policy and is orthogonal to asynchronous execution schemes such as Real Time Chunking (RTC). On the dynamic KINETIX task suite (12 tasks) and LIBERO SPATIAL, our method yields consistent success rate improvements across increasing delays and execution horizons $( + 2 3 \%$ point and $+ 7 \%$ point respectively, compared to RTC), and also improves robustness for long horizons even with zero injected delay. Since the correction head is small and fast, there is minimal overhead compared to the inference of large VLA models. These results indicate that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control.

# 1 INTRODUCTION

Recent advances in large vision–language–action (VLA) models have significantly expanded the capability of robots to generalize across tasks and environments (Black et al., 2024; Gemini Robotics Team et al., 2025; NVIDIA et al., 2025; TRI LBM Team et al., 2025). However, large model size requires high computational cost to output the actions for each step, which leads to high inference latency (Kawaharazuka et al., 2025; Black et al., 2025). Especially in dynamic control, such delays become critical. A robot relying on long action sequences predicted from outdated observations can drift, overlook cues, or fail in tasks demanding rapid reactions, such as catching moving objects or stabilizing unstable systems.

The trend of scaling up neural network policies using foundation models brings representational benefits (Sartor & Thompson, 2025), but also incurs a latency problem. For instance, large VLA models such as $\pi _ { 0 }$ (Black et al., 2024) or OpenVLA (Kim et al., 2024) have billions of parameters and often require hundreds of milliseconds to generate a single action chunk. These action chunks are predicted solely from the previous observation and then executed in an open-loop manner, without incorporating new sensory input during their execution. In addition, latency not only delays execution but also prevents the policy from incorporating the latest observations, thereby weakening its ability to produce reactive behaviors. This is particularly problematic in tasks where the environment changes rapidly during inference. For instance, following a moving object on a cluttered table or grasping a utensil while other objects are being placed, the robot should adjust its action sequence to new sensory inputs. In these scenarios, actions computed from outdated observations accumulate errors over time, which lowers success rates and, in some cases, leads to task failure. This is the central challenge we address in this work.

![](images/8d5a7917ccb501d07be482a58b25c9f6de3ff17ee001299f50de04c206737eb8.jpg)  
Figure 1: Illustration of asynchronous action chunk execution and its problem. $H$ denotes the horizon length of an action chunk. $e$ is the execution horizon, and $d$ is the inference delay caused by policy inference. (Policy generates a horizon $H$ length action chunk. Inference of the policy takes $d$ steps. While the next chunk is inferred, we execute $e$ steps from the current chunk. Each executed action is based on an observation at least $d$ steps old, and in the worst case, the action may correspond to an observation that is $d + e$ steps old.

Conventional approaches attempt to mitigate the latency of large models through action chunking (Zhao et al., 2023; Black et al., 2024). By predicting long sequences of actions at once, these methods reduce the frequency of expensive inference calls. However, the chunking strategies can impact performance; robots may experience waiting time during inference, and inconsistencies can arise between successive chunks (Liu et al., 2025). To address this, SmolVLA (Shukor et al., 2025) introduces synchronous execution of the policy, and Real Time Chunking (RTC) (Black et al., 2025) ensures smoother continuity between chunks under asynchrony for diffusion-based action generation. However, these methods still assume that the model predicts fixed-length horizons, which means reactivity to new sensory input remains limited.

Another line of work adopts hierarchical architectures inspired by dual-system reasoning (Kahneman, 2011). Large models serve as a high-level planner (System 2), while smaller policies act as fast executors (System 1). Examples include Hi Robot (Shi et al., 2025), which combines a VLM at the high level with a VLA at the low level, and GR00T-N1 (NVIDIA et al., 2025), which uses a compact policy to refine continuous action chunks. However, since the low-level executor has to wait for predictions from the high-level model, the latency still persists. Consequently, while chunking and hierarchical approaches alleviate some issues, they do not fundamentally solve the challenge of maintaining responsiveness to new observations under the inference delays inherent to VLAs with a large number of parameters.

To mitigate this problem, in this paper, we propose Asynchronous Action Chunk Correction (A2C2), which is a lightweight correction head that can be executed at every timestep to complement the outputs of large VLA models. Unlike conventional approaches such as action chunking and asynchronous inference, our method introduces a lower-level correction layer that directly integrates the most recent observation referring to the action chunks that high-level model outputs. This correction head does not compete with base (high-level) policies like diffusion- or VLA-based chunk generators; instead, it enhances them by injecting real-time feedback to maintain responsiveness under inference delays and long horizons. Through this design, the proposed framework achieves robustness against dynamic environmental changes and external disturbances, thereby mitigating the critical latency bottleneck in deploying large-scale VLA models for real-time robotic control.

In our experiments on the Kinetix tasks, we measure a $3 5 \%$ point increase in success rate over naive execution and $2 3 \%$ point increase over RTC in the presence of delay. For long execution horizons, we measure a $1 2 \%$ point success rate increase over naive execution and $7 \%$ point increase over RTC.

In summary, the contributions of this work are as follows:

• We first formulated delays in policy inference with VLAs that generate action chunks. • A lightweight add-on action correction policy (A2C2) is introduced to improve reactivity, which can be applied to any VLA model independent of the underlying architecture. • The method showed substantial improvements in success rates on dynamic tasks and robot manipulation benchmarks with varied inference delays.

# 2 PROBLEM FORMULATION

We consider an action chunk execution with an imitation learning (IL) policy. As illustrated in Figure 1, an action chunk $A _ { t } = \left\{ { a _ { t } , \ldots , a _ { t + H - 1 } } \right\}$ is from $\mathrm { I L }$ policy $\pi$ based on the observation $o _ { t }$ and a language instruction $l$ . $H$ is the horizon length, the training sequence length of the $\mathrm { I L }$ model $\pi$ . We assume it uses $e$ steps of the action chunk, and define it as the execution horizon. Policy predicts the action chunk every $e$ steps as follows:

$$
A _ { t } = \left\{ a _ { t } , \ldots , a _ { t + H - 1 } \right\} = \pi ( o _ { t } , l ) .
$$

Also, there is an inference latency. We define the delay $d$ as the number of control steps between receiving an observation $o _ { t }$ and obtaining the corresponding action chunk $A _ { t }$ . Formally, it is computed as

$$
d = \left\lfloor { \frac { \delta } { \Delta t } } \right\rfloor ,
$$

where $\delta$ represents the combined inference and communication time, and $\Delta t$ denotes the duration of a single control step.

To control delayed, chunked action execution, the agent executes one action per step till a new chunk arrives asynchronously. Additionally, we assume that the policy server can handle only one inference at a time. If the execution horizon $e$ is shorter than the delay $d$ , there will be no action during the model inference, which leads to waiting time. On the other hand, if the execution horizon $e$ is longer than $H - d$ , there is no action remaining during the inference time. Therefore, the execution horizon $e$ needs to be longer than the delay $d$ , and $e$ must be shorter than $H - d$ $\begin{array} { r } { d \leq e \leq H - d ) } \end{array}$ .

In this setting, the agent needs to use the actions that are always based on past observations. Each executed action corresponds to an observation at least $d$ steps old. And in the worst case, the agent may need to execute an action that is generated from the $d + e$ steps past observations.

# 3 METHOD

# 3.1 OVERVIEW

We extend the action chunk–based policy $\pi$ by Asynchronous Action Chunking Correction (A2C2), introducing a lightweight correction head $\pi _ { a 2 c 2 }$ that refines each action within a predicted chunk using the most recent observation, features of the base policy, and a temporal position feature. This framework enables step-wise online correction without retraining the base policy and is complementary to methods such as RTC (Black et al., 2025).

At time t, Observation $o _ { t }$ is sent to the policy server. Then, the base policy $\pi$ generates the action chunk $A _ { t } = \{ a _ { t } ^ { \mathrm { b a s e } } , \dots , a _ { t + H - 1 } ^ { \mathrm { b a s e } } \}$ within inference delay $d$ as

$$
A _ { t } = \{ a _ { t } ^ { \mathrm { b a s e } } , \ldots , a _ { t + H - 1 } ^ { \mathrm { b a s e } } \} = \pi ( o _ { t } , l ) .
$$

Subsequently, at time $t + k$ $\left( d \leq k \leq d + e \right)$ , time feature $\tau _ { k }$ , and base action $a _ { t + k }$ , latest observation $o _ { t + k }$ , base policy latest representation $z _ { t }$ and language instruction $l$ are added to the correction head $\pi _ { a 2 c 2 }$ . The positional feature $\tau _ { k }$ is represented by a sinusoidal embedding that provides periodic structure over the chunk length $( \sin ( 2 \pi { \frac { k } { H } } ) , \cos ( 2 \pi { \frac { k } { H } } ) )$ . The correction head integrates this information and predicts the residual action $\Delta a _ { t + k }$ as

$$
\Delta a _ { t + k } = \pi _ { a 2 c 2 } ( o _ { t + k } , a _ { t + k } ^ { \mathrm { b a s e } } , \tau _ { k } , z _ { t + k } , l ) .
$$

![](images/a2c93a6b6a8dbd08c0844987e3fb33be2b9a5420872a447eb5d46694db28a25f.jpg)  
Figure 2: The Base policy $\pi$ outputs an action chunk $A _ { t } = \left\{ { a _ { t } , \ldots , a _ { t + H - 1 } } \right\}$ from the current observation $o _ { t }$ . For each step within the chunk, a lightweight correction head refines the corresponding base action $a _ { t + k }$ using the latest observation $O _ { t + k }$ and a time feature $\tau _ { k }$ indicating the relative position within the chunk. The refined actions $a _ { t + k } ^ { e x e c }$ mitigate performance degradation under inference delays and long horizons .

The residual action $\Delta a _ { t + k }$ is added to the base action $a _ { t + k }$ and output the execution action $a _ { t } ^ { \mathrm { { e x e c } } }$ as

$$
a _ { t + k } ^ { \mathrm { e x e c } } = a _ { t + k } ^ { \mathrm { b a s e } } + \Delta a _ { t + k } .
$$

Base policy $\pi$ infers an action chunk every $e$ steps with $d$ delay. On the other hand, we assume that the model size of the correction head $\pi _ { a 2 c 2 }$ is small enough to run every step, which means the inference time of the head is smaller than the duration of a single control step $\Delta t$ . Refer to Figure 2 for the overview.

Our method differs from existing approaches for asynchronous inference in the following aspects:

• Time-aware correction: The correction head explicitly conditions on the position within the action chunking VLA using a temporal feature.   
• Chunk-level smoothness: By specifying which element of the chunk is being corrected, the method produces smoother corrections across horizons.   
• Data compatibility: Training uses the same demonstration datasets as the base VLA policy, which does not require reinforcement learning fine-tuning.   
• Real-time feedback: New observations are always incorporated, improving robustness under inference delay in dynamic tasks.

# 3.2 MODEL TRAINING PROCEDURE

First, we train the base policy $\pi$ with the dataset

$$
{ \cal D } _ { b a s e } = \{ \{ \{ o _ { t } , a _ { t } \} _ { t = 0 \ldots T _ { n } } ^ { n } , l ^ { n } \} _ { n = 1 \ldots N } \} ,
$$

where $N$ denotes the number of episodes in the dataset. Afterward, we add the output action chunk $\hat { A } _ { t }$ of the inference from base policy $\pi$ for each step in the dataset $D _ { b a s e }$ as

$$
\hat { A } _ { t } = \{ \hat { a } _ { t } , . . . , \hat { a } _ { t + H - 1 } \} = \pi ( o _ { t } , l ) .
$$

With these inference results, we created a new dataset for correction head training $D _ { c o r }$ as

$$
D _ { c o r } = \{ \{ \{ o _ { t } , a _ { t } , \hat { a } _ { t - k } ^ { k } , \tau _ { k } \} _ { t = 0 \ldots T _ { 0 } , k = 0 \leq k \leq m i n ( t , H - 1 ) } ^ { n } , l ^ { n } \} _ { n = 1 \ldots N } \} .
$$

$\hat { a } _ { t - k } ^ { k }$ is the $\mathbf { k }$ -th action in the action chunk inferred by the base policy from the observation at time $t - k$ . Then, the Correction head $\pi _ { a 2 c 2 }$ is trained to predict the residual action, i.e., the difference between the target action and the base policy output. The target action is the action in the dataset that was originally collected from expert demonstrations. Formally, given the target action $a _ { \mathrm { t a r g e t } }$ and the base policy output $a _ { \mathrm { b a s e } }$ , the residual target is defined as

$$
\Delta a _ { \mathrm { r e s i d u a l } } = a - \hat { a } .
$$

$\hat { a }$ is a base action inferred by the base policy. There are some possible combinations of the base action with different time features $\tau$ . The predicted residual action is denoted by $\Delta a _ { \mathrm { r e s i d u a l } }$ . The loss function is the mean squared error (MSE):

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \Delta a _ { \mathrm { r e s i d u a l } } ^ { ( i ) } - \left( a ^ { ( i ) } - \hat { a } ^ { ( i ) } \right) \right. _ { 2 } ^ { 2 } .
$$

Where $N$ denotes the batch size, i.e., the number of training samples in a mini-batch.

# 4 EXPERIMENTAL SETUP

# 4.1 BENCHMARK AND DATASETS

We use the two simulation environments, Kinetix and LIBERO Spatial, for the experiments. Kinetix is first used for evaluating the performance under highly dynamic manipulation and locomotion tasks. Secondly, we used the LIBERO Spatial benchmark to evaluate the performance as a standard benchmark of robot manipulation. Especially, because Shukor et al. (2025) reports that long-horizon significantly degrades performance in LIBERO Spatial, making the task a natural choice for evaluating robustness under long horizons.

# 4.1.1 KINETIX

We used the Kinetix, which provides demonstrations across 12 highly dynamic (see Appendix A.1 ). It includes environments ranging from locomotion and grasping to game-like settings. Importantly for our setting, Kinetix contains highly dynamic environments where delayed or inconsistent action generation quickly leads to failure. This makes it a natural testbed for studying the limitations of action chunking and for benchmarking inference-time algorithms such as RTC, which aim to preserve responsiveness and continuity under latency.

Unlike quasi-static benchmarks, Kinetix environments employ torque- and force-based actuation, making asynchronous inference crucial. Kinetix consists of 12 tasks without language input. 1 million steps data was collected by using expert model. Following RTC experiments, we first train expert policies using RPO (Rahman & Xue, 2022) and a binary success reward. For each environment, 1-million transition dataset is generated with the expert policy.

# 4.1.2 LIBERO

LIBERO is a benchmark suite designed to study lifelong robot learning with a focus on knowledge transfer across tasks (Liu et al., 2023). They offer several task suites and datasets. In this work, we specifically use the LIBERO Spatial dataset, which emphasizes spatial reasoning in manipulation tasks as a widely used benchmark for robot manipulation.

For benchmarking 3D robot manipulation, we used LIBERO spatial benchmark, which provides 432 episodes and 52,970 frames across 10 tasks. The dataset consists of multimodal input, including top and wrist RGB images $( 2 5 6 \times 2 5 6 )$ ), an 8-dimensional state, and language instructions.

# 4.2 MODEL TRAINING

In Kinetix, we used a flow-matching policy as the base model, following prior work on RTC Black et al. (2025). The Correction head network is a 3-layer multilayer perceptron (MLP). The input layer receives the concatenation of the state vector (2722-dim), the base action (6-dim), and the

![](images/afe58ccd4c8fde9892c30b7a451cc9b14b16d0fb9da42d447bf41930fe4789ac.jpg)  
Figure 3: Correction head architecture in the Kinetix environment. The MLP takes as input the current state, the base action, and a positional embedding indicating the index within the action chunk. It outputs a residual action that is added to the base action, yielding the refined action.

![](images/9921238921874492fbd5cc718d66ab81b5d87fd13e2df094564369fa6fb63fe9.jpg)  
Figure 4: Architecture of the proposed Correction head in the LIBERO environment. The base policy is a SmolVLA that produces an action chunk from image, language instruction, and state inputs. A transformer encoder processes the latent representation from smolVLM $z _ { t }$ , state $s _ { t + k }$ , base action to produce based on th $a _ { t + k }$ , image feature, nt representation nt representation e action chunk, time feature . A lightweight MLP predic the transformer encoder an $\tau _ { k }$ , and langa residual ase action $l$ $e _ { t + k }$ $\Delta a _ { t + k } ^ { r e s i d u a l }$ $a _ { t + k }$ $s _ { t + k }$ and time feature $\tau _ { k }$ . Then, added to the selected base action to obtain the refined action executed in the environment.

2-dimensional sinusoidal positional feature. We did not use language instructions or latent representations from base policies, as the model was trained and evaluated separately for each task. Hidden layers have 512 units each with ReLU activation (Nair & Hinton, 2010) and layer normalization (Ba et al., 2016). The output layer produces a 6-dimensional residual vector, which is added elementwise to the base action. The total parameter count is 0.31M. Figure 3 shows the implementation detail for the Kinetix experiment.

For LIBERO spatial, we adopted SmolVLA (Shukor et al., 2025) (450M parameters) as the base, since it provides competitive performance among VLA models. The correction head consists of a transformer encoder and a lightweight MLP. Visual observations (top and wrist cameras) are encoded into 512-dimensional tokens using a ResNet-18 (He et al., 2016) pretrained on ImageNet (Deng et al., 2009). Language instructions are embedded by the smolVLM encoder provided in the base fipolicy. The base action, latent features of the base policy, and the sinusoidal time embedding are also

![](images/75650e0fe21f28db3fa5842764e11abf205b248311d775f6093d37fdd8a15f2b.jpg)

![](images/f2e385506290d688730aeab1dbd09347b3a440285838f69d4215c371e8d8d1fc.jpg)  
(b) Average Success rate as a function of execution horizon $e$ with delay fixed at $d \ : = \ : 1$ . A2C2 (red) remains robust across horizons, while baselines degrade as horizon length increases.

(a) Average Success rate as a function of inference delay $d$ with execution horizon fixed at $e =$ $\operatorname* { m a x } ( d , 1 )$ . A2C2 (red) consistently outperforms both naive and real-time baselines, maintaining higher success rates even under large delays.

Figure 5: Overall performance comparison in Kinetix tasks. Each data point averages over 2048 rollouts. Residual correction improves robustness under both increasing inference delay and longer execution horizons.

projected into 512-dim tokens. All tokens are concatenated and processed by a 6-layer transformer encoder. The pooled embedding, along with the base action and state vector, is passed through a 3-layer MLP (hidden size 512) to predict the residual action. The number of total parameters is 32M. Figure 4 shows the implementation detail for the LIBERO experiment. We also release the source code for both Kinetix and LIBERO experiments. See Appendix A.3 for the details.

# 5 RESULTS

# 5.1 KINETIX

We evaluate the proposed action chunk correction framework in the Kinetix benchmark under varying inference delays $d$ and execution horizons $e$ . Figure 5 reports success rates aggregated across all 12 tasks. There are two baseline comparisons. First is Naive async. This strategy does not pay attention to the previous action chunk at all when generating a new one, naively switching chunks as soon as the new one is ready. Second is RTC. As expected, both the naive async and RTC baselines degrade significantly as either the delay $d$ increases or the horizon $H$ becomes longer. In particular, when $d \geq 3$ , the na¨ıve baseline suffers a sharp drop in success rate due to compounding errors from executing outdated action chunks. RTC inference partially mitigates this issue by overlapping prediction and execution, but performance still declines as the execution horizon increases.

In contrast, the action chunk correction maintains consistently higher success rates across all settings. Because it refines each action using the most recent observation, the action chunk correction can counteract both the temporal misalignment introduced by inference delay and the drift that accumulates within long action horizons. For example, at delay $d = 4$ , our proposed method achieves nearly $3 5 \%$ higher success than the na¨ıve baseline, and remains above $85 \%$ even for horizons $H = 7$ . This demonstrates that real-time correction of action chunks maintains performance both with inference delays and with long-horizon execution.

# 5.2 LIBERO SPATIAL

Figure 6 and Table 1 summarize the evaluation on the LIBERO Spatial benchmark. We tested the Na¨ıve async and A2C2 on this setting. Across 10 manipulation tasks with multimodal inputs, the correction head consistently improved success rates over the na¨ıve baseline under both long horizons and injected delays. For example, with execution horizon $H = 4 0$ and delay $d = 1 0$ , the na¨ıve baseline achieved only $67 \%$ success, whereas the A2C2 reached $84 \%$ . Even when no delay was present, Action chunk correction provided notable gains at long horizons $\mathit { \Delta } H \ : = \ : 5 0$ , $d = 0$ ), raising success from $7 2 . 2 \%$ to $8 1 . 6 \%$ . These results demonstrate that residual refinement by correction head mitigates the degradation caused by outdated action chunks and restores closed-loop

(a) Success Rate vs Inference Delay $d$ (Execution Horizon: $\scriptstyle \mathrm { e = } 4 0$ ). A2C2 remains robust under inference delays.

![](images/ea68b0370e9c802562f44ff4314146629812c48355b99c1992e476c4733566d8.jpg)  
(b) Success Rate vs Execution Horizon $e$ (Inference Delay: $\mathrm { d } { = } 0$ ). A2C2 consistently improves performance across horizons.

![](images/7daed863ff3e368eae61b0ae47d3c244c7153b57ce3459f0365fd2c9e060d0cd.jpg)

Figure 6: Results of LIBERO Spatial: Comparison of Success Rate under different conditions. (a) Effect of inference delay with fixed execution horizon. (b) Effect of execution horizon with no inference delay. Each data point is evaluated on 10 tasks, each with 10 rollouts, resulting in a total of 100 rollouts.

Table 1: LIBERO Spatial: success rate $( \% )$ . 50 rollouts per task. Action chunk correction mitigates performance degradation under delay and long horizons.   

<table><tr><td>Method</td><td>Execution horizon e</td><td>Delay d</td><td>Success Rate (%)</td></tr><tr><td>Naive</td><td>10</td><td>0</td><td>81.8</td></tr><tr><td>A2C2 (Ours)</td><td>10</td><td>0</td><td>89.2</td></tr><tr><td>Naive</td><td>40</td><td>10</td><td>64.4</td></tr><tr><td>A2C2(Ours)</td><td>40</td><td>10</td><td>84.2</td></tr><tr><td>Naive</td><td>50</td><td>0</td><td>72.2</td></tr><tr><td>A2C2(Ours)</td><td>50</td><td>0</td><td>81.6</td></tr></table>

responsiveness, enabling large VLA models to maintain high success rates that require fine-grained spatial reasoning.

# 6 RELATED WORK

Imitation learning and VLAs: Imitation learning (IL) trains agents from demonstrations provided by humans or expert policies, and has been a representative approach in learning robotic control (Osa et al., 2018). Recent advances have introduced generative sequence models to improve consistency and scalability. Diffusion Policy (Chi et al., 2023) utilizes diffusion models for action generation, enabling it to handle the multimodality of data distribution in imitation learning. In parallel, the Action Chunking Transformer (ACT) (Zhao et al., 2023) proposes a transformer-based policy that outputs action chunks rather than single-step actions, producing coherent behaviors while enabling faster inference. In addition, flow-based approaches, such as Flow Policy (Zhang et al., 2024), generate actions by learning continuous transport maps instead of iterative denoising.

Building on these foundations, a new class of vision–language–action (VLA) foundation models has emerged (Kawaharazuka et al., 2024), including $\pi _ { 0 }$ Black et al. (2024), openVLA Kim et al. (2024), GR00T NVIDIA et al. (2025), and SmolVLA Shukor et al. (2025). These models adopt chunk-based prediction as the de facto standard for inference, similar to ACT (Zhao et al., 2023). Vision–Language–Action (VLA) models achieve broad task generalization by aligning multimodal inputs, but their architectures are considerably larger than diffusion- or transformer-based imitation policies. For instance, $\pi _ { 0 }$ has about 3B parameters and openVLA around 7B, which makes inference latency significant even on modern GPU-accelerated hardware. While these models demonstrate the promise of scaling and multimodal grounding, their computational footprint exacerbates the latency problem in real-time control.

Asynchronous chunk execution: As model sizes increase, inference latency becomes a significant bottleneck, motivating asynchronous policy frameworks. In particular, the SmolVLA (Shukor et al., 2025) proposed a server–client architecture for mitigating inference delays. In this setup, the server receives observations and performs inference with a delay of $d$ control steps (including communication latency), then transmits an action chunk of horizon $H$ to the client. Then, the client executes these actions sequentially. However, because the $d$ delayed actions are not yet available at execution time, the client continues executing actions from the previous chunk until the new chunk arrives. This design ensures continuity but introduces the risk of inconsistency between consecutive chunks. For example, the earlier chunk may predict avoiding an obstacle by moving left, while the newly received chunk may instead suggest moving right. Such mismatches across chunks can cause jerky motion and noticeable performance degradation, especially in dynamic environments.

To fix the chunk mismatches, Real Time Chunking (RTC) (Black et al., 2025) is proposed. It is an inference-time algorithm that enables smooth asynchronous execution for action-chunking policies by posing chunk switching as an inpainting problem. Specifically, it generates the next action chunk while executing the current one, “freezing” actions guaranteed to execute and “inpainting” the rest.

Reducing inference latency: One natural way to enhance a model’s real-time performance is to reduce its inference time. Streaming Diffusion Policy (Høeg et al., 2024) or Streaming Flow Policy (Jiang et al., 2025) presents a new training procedure that enables faster inference. More generally, optimizations such as model compression (Lin et al., 2024) or memory optimization (Kwon et al., 2023) of models can also improve the inference speeds. However, as long as model scale and communication overhead prevent action generation from being faster than the control step, the challenges highlighted in this work remain unresolved.

# 7 CONCLUSION

In this paper, we propose Asynchronous Action Chunk Correction (A2C2), which introduces a lightweight action correction head by augmenting a large base policy, such as VLAs. A2C2 addresses the challenge of preserving reactivity under inference delays and long execution horizons of action chunking policies. The correction head is trained on the same dataset as the base policy and, in principle, can be added to any off-the-shelf VLAs. Our experiments in both the Kinetix simulation suite and the LIBERO Spatial benchmark demonstrated that Asynchronous Action Chunk Correction (A2C2) consistently maintained high success rates, even in settings where na¨ıve or RTC degraded significantly.

While our approach adds minimal overhead compared to full model inference, further work is needed to validate its scalability to richer language instructions, out-of-distribution settings, and more dynamic tasks beyond those in LIBERO Spatial. Addressing these challenges would broaden the applicability of action chunk correction and strengthen its role as a general mechanism for enhancing reactivity in large policy architectures.

Recently, Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated improved generality through parameter scaling, as established by neural scaling laws (Kaplan et al., 2020). Since recent VLA policies are built upon these models, it is reasonable to expect that future VLAs will continue to grow in size to support deployment across diverse environments and tasks. Our work can be viewed as a step toward enabling such scaled VLAs to operate in real time without sacrificing responsiveness by introducing a lightweight correction mechanism that mitigates the effects of inference latency.

Moreover, inference of models with billions of parameters already exceeds the computational capacity of on-board processors on most robotic platforms. In practice, this motivates client–server architectures where the VLA runs on a remote server and the robot queries it over a network. In this setting, by explicitly treating communication delay as part of the inference latency in our formulation, our framework naturally extends to client–server architectures where large VLAs are executed remotely. Thus, our framework provides a pathway to leverage the generalization benefits of largescale VLAs while still maintaining reactivity in real-world deployments, enabling the design of next-generation VLA systems that combine scalability with responsiveness.

# ETHICS STATEMENT

This work does not involve human subjects, personally identifiable information, or sensitive data.   
All experiments were conducted on publicly available datasets.

# REPRODUCIBILITY STATEMENT

We provide implementation details and dataset preprocessing in the Appendix, and full hyperparameter settings in the appendix. We released our source code for the experiments below:

• Kinetix: https://github.com/k1000dai/a2c2-kinetix • LIBERO: https://github.com/k1000dai/a2c2-libero

REFERENCES

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. $\pi _ { 0 }$ : A visionlanguage-action flow model for general robot control, 2024. URL https://arxiv.org/ abs/2410.24164.

Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-time execution of action chunking flow policies, 2025. URL https://arxiv.org/abs/2506.07339.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Sigmund H. Høeg, Yilun Du, and Olav Egeland. Streaming diffusion policy: Fast policy synthesis with variable noise diffusion models, 2024. URL https://arxiv.org/abs/2406. 04806.

Sunshine Jiang, Xiaolin Fang, Nicholas Roy, Tomas Lozano-P ´ erez, Leslie Pack Kaelbling, and Sid- ´ dharth Ancha. Streaming flow policy: Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories, 2025. URL https://arxiv.org/abs/2505. 21851.

Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, New York, 2011. ISBN 9780374275631.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.08361.

Kento Kawaharazuka, Tatsuya Matsushima, Andrew Gambardella, Jiaxian Guo, Chris Paxton, and Andy Zeng. Real-world robot applications of foundation models: A review. Advanced Robotics, 38(18):1232–1254, 2024.

Kento Kawaharazuka, Jihoon Oh, Jun Yamada, Ingmar Posner, and Yuke Zhu. Vision-languageaction models for robotics: A review towards real-world applications. IEEE Access, 13:162467– 162504, 2025. doi: 10.1109/ACCESS.2025.3609980.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model, 2024. URL https://arxiv.org/ abs/2406.09246.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023. URL https://arxiv.org/abs/2309.06180.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration, 2024. URL https://arxiv.org/abs/2306.00978.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. URL https://arxiv. org/abs/2306.03310.

Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, and Chelsea Finn. Bidirectional decoding: Improving action chunking via guided test-time sampling, 2025. URL https://arxiv.org/abs/2408.17355.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Michael Matthews, Michael Beukman, Chris Lu, and Jakob Foerster. Kinetix: Investigating the training of general agents through open-ended physics-based control tasks, 2025. URL https: //arxiv.org/abs/2410.23208.

Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML-10), pp. 807–814, 2010.

NVIDIA, Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, ˜ Linxi ”Jim” Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.

Takayuki Osa, Joni Pajarinen, Gerhard Neumann, J Andrew Bagnell, Pieter Abbeel, Jan Peters, et al. An algorithmic perspective on imitation learning. Foundations and Trends® in Robotics, 7(1-2): 1–179, 2018.

Md Masudur Rahman and Yexiang Xue. Robust policy optimization in deep reinforcement learning, 2022. URL https://arxiv.org/abs/2212.07536.

Sebastian Sartor and Neil Thompson. Neural scaling laws in robotics, 2025. URL https:// arxiv.org/abs/2405.14005.

Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, and Chelsea Finn. Hi robot: Open-ended instruction following with hierarchical vision-language-action models, 2025. URL https://arxiv.org/abs/2502. 19417.

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. Smolvla: A vision-language-action model for affordable and efficient robotics, 2025. URL https://arxiv.org/abs/2506.01844.

TRI LBM Team, Jose Barreiros, Andrew Beaulieu, Aditya Bhat, Rick Cory, Eric Cousineau, Hongkai Dai, Ching-Hsin Fang, Kunimatsu Hashimoto, Muhammad Zubair Irshad, Masha Itkina, Naveen Kuppuswamy, Kuan-Hui Lee, Katherine Liu, Dale McConachie, Ian McMahon, Haruki Nishimura, Calder Phillips-Grafflin, Charles Richter, Paarth Shah, Krishnan Srinivasan, Blake Wulfe, Chen Xu, Mengchao Zhang, Alex Alspach, Maya Angeles, Kushal Arora, Vitor Campagnolo Guizilini, Alejandro Castro, Dian Chen, Ting-Sheng Chu, Sam Creasey, Sean Curtis, Richard Denitto, Emma Dixon, Eric Dusel, Matthew Ferreira, Aimee Goncalves, Grant Gould, Damrong Guoy, Swati Gupta, Xuchen Han, Kyle Hatch, Brendan Hathaway, Allison Henry, Hillel Hochsztein, Phoebe Horgan, Shun Iwase, Donovon Jackson, Siddharth Karamcheti, Sedrick Keh, Joseph Masterjohn, Jean Mercat, Patrick Miller, Paul Mitiguy, Tony Nguyen, Jeremy Nimmer, Yuki Noguchi, Reko Ong, Aykut Onol, Owen Pfannenstiehl, Richard Poyner, Leticia Priebe Mendes Rocha, Gordon Richardson, Christopher Rodriguez, Derick Seale, Michael Sherman, Mariah Smith-Jones, David Tago, Pavel Tokmakov, Matthew Tran, Basile Van Hoorick, Igor Vasiljevic, Sergey Zakharov, Mark Zolotas, Rares Ambrus, Kerri Fetzer-Borelli, Benjamin Burchfiel, Hadas Kress-Gazit, Siyuan Feng, Stacie Ford, and Russ Tedrake. A careful examination of large behavior models for multitask dexterous manipulation. 2025. URL https://arxiv.org/abs/2507.05331.

Qinglun Zhang, Zhen Liu, Haoqiang Fan, Guanghui Liu, Bing Zeng, and Shuaicheng Liu. Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation, 2024. URL https://arxiv.org/abs/2412.04987.

Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL https://arxiv.org/abs/2304.13705.

# A APPENDIX

A.1 KINETIX SIMULATION DETAIL

# A.1.1 ENVIRONMENT

We reused the 12 tasks from the Kinetix benchmark (Matthews et al., 2025) used in the RTC paper Black et al. (2025). A sample visualization of each of the environments is shown in Figure 7. The Kinetix environment has an observation space with 2722 dimensions which do not include any images. Instead, it encodes information about polygons, circles, joints, thrusters, gravity, and the states of motors and thrusters described below. For entities not used in a given task, their corresponding entries are zero-padded. The action space has 6 dimensions. The first four correspond to motor controls, and the last two correspond to thruster controls. For unused actuators, their entries are set to zero via padding.

![](images/039532fc864510cb90dd2313db5ac875d52902a91a0de042eccd18c87bed226b.jpg)  
Figure 7: Visualization of the 12 tasks from the Kinetix simulation environment. Each subfigure corresponds to one task used in our experiments.

# A.1.2 DATASET GENERATION AND TRAINING DETAIL

An imitation learning dataset was required to test the flow policy and our correction head. In the Kinetix simulation, we follow the RTC implementation. First, we trained the expert policy with RPO (Rahman & Xue, 2022) on 8 seeds per task for 64 million environment steps each. For each task, we load the best-performing checkpoint for each seed and discard some seeds if they did not reach a certain success threshold. Then, we used the expert model to generate 1 million environment steps for each task. After that, we train the flow policy with the generated dataset. We saved the checkpoints for each, but used the last checkpoint for the evaluation.

Table 2: Training hyperparameters for the Kinetix flow policy.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>3 × 10−4</td></tr><tr><td>Gradient norm clip</td><td>10.0</td></tr><tr><td>Weight decay</td><td>1 × 10-2</td></tr><tr><td>Warmup steps</td><td>1000</td></tr><tr><td>Batch size</td><td>512</td></tr><tr><td>Number of epochs</td><td>32</td></tr></table>

Table 3: Training hyperparameters for the Kinetix Correction head.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>512</td></tr><tr><td>Number of epochs</td><td>16</td></tr><tr><td>learning rate</td><td>1 × 10−4</td></tr><tr><td>weight decay</td><td>1 × 10−3</td></tr><tr><td>Gradient norm clip</td><td>5.0</td></tr><tr><td>Warmup steps</td><td>500</td></tr></table>

The correction head is then trained with the flow policy. The correction policy requires the base action from the base policy, so at every step, we infer the action chunk from the base policy and use it and the dataset to train the correction head. During the base flow policy training, we used a constant learning rate and added some warmup state. See Table 2 for more details on the settings. For the Correction Head training, we used the parameters shown in Table 3. In both the flow policy and A2C2 training, the AdamW optimizer (Loshchilov & Hutter, 2017) was used.

# A.1.3 EVALUATION DETAILS

In the evaluation, we rolled out 2048 per task and computed the success rate for different delays and execution horizon lengths. In the Kinetix simulation, we tested all combinations of delay and execution horizons compatible with the chosen action chunk size. All results are in Table 4.

# A.2 LIBERO SIMULATION DETAIL

# A.2.1 ENVIRONMENT

LIBERO Spatial consists of 10 tasks. We evaluated all tasks, and the corresponding language instructions are listed below. The language instructions are:

1. pick up the black bowl between the plate and the ramekin and place it on the plate   
2. pick up the black bowl next to the ramekin and place it on the plate   
3. pick up the black bowl from the table center and place it on the plate   
4. pick up the black bowl on the cookie box and place it on the plate   
5. pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate   
6. pick up the black bowl on the ramekin and place it on the plate   
7. pick up the black bowl next to the cookie box and place it on the plate   
8. pick up the black bowl on the stove and place it on the plate   
9. pick up the black bowl next to the plate and place it on the plate   
10. pick up the black bowl on the wooden cabinet and place it on the plate

Table 4: Kinetix: success rate (percent) under different execution horizons (e) and inference delays $( d )$ . 10 tasks and 10 rollouts per task. Residual correction consistently improves over the na¨ıve baseline. The first, second, and third row of each cell denote the success rate of Na¨ıve, RTC (Black et al., 2025), and A2C2(Ours), respectively.   

<table><tr><td rowspan="2">Delay (d)</td><td colspan="8">Execution Horizon (e)</td></tr><tr><td></td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td rowspan="3">0</td><td>90.8</td><td>90.4</td><td>89.8</td><td>88.9</td><td>88.1</td><td>87.4</td><td>86.6</td><td>86.0</td></tr><tr><td>90.9</td><td>90.0</td><td>89.2</td><td>88.6</td><td>87.5</td><td>86.6</td><td>86.3</td><td>86.0</td></tr><tr><td>90.3</td><td>89.5</td><td>89.6</td><td>88.9</td><td>88.8</td><td>88.9</td><td>88.2</td><td>87.8</td></tr><tr><td rowspan="3">1</td><td>78.1</td><td>83.7</td><td>82.6</td><td>80.9</td><td>78.5</td><td>77.4</td><td>75.7</td><td rowspan="3"></td></tr><tr><td>88.2</td><td>87.5</td><td>86.3</td><td>84.8</td><td>83.1</td><td>81.1</td><td>80.2</td></tr><tr><td>89.5</td><td>88.8</td><td>88.4</td><td>88.3</td><td>87.9</td><td>87.3</td><td>87.8</td></tr><tr><td rowspan="3">2</td><td rowspan="3"></td><td>75.7</td><td>72.7</td><td>70.1</td><td>67.3</td><td>66.4</td><td rowspan="3"></td><td rowspan="3"></td></tr><tr><td>84.1</td><td>81.4</td><td>79.3</td><td>76.4</td><td>74.8</td></tr><tr><td>87.4</td><td>87.5</td><td>87.5</td><td>87.1</td><td>86.6</td></tr><tr><td rowspan="3">3</td><td rowspan="3"></td><td rowspan="3"></td><td>59.3</td><td>59.5</td><td>56.5</td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td></tr><tr><td>74.8</td><td>71.0</td><td>67.5</td></tr><tr><td>87.4</td><td>87.2</td><td>86.0</td></tr><tr><td rowspan="2">4</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td>51.2 62.9</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td>86.5</td></tr></table>

# A.2.2 DATASET AND TRAINING DETAIL

We used the LIBERO Dataset with the LeRobot dataset format available on Huggingface and we used the LeRobot framework to read the dataset. LeRobot also has a well-organized training pipeline and makes it easy to create and try new architectures.

First, we trained SmolVLA as a base policy. There is an option for training the policy from scratch or fine-tuning the pretrained model. In our setting, we chose the training from scratch because SmolVLA is pretrained mainly with S0-101, which is a different embodiment from the Franka arm used in the LIBERO benchmark.

In the Kinetix simulation, the base policy predicts the action chunk every time in the correction head training. However, it is too time-consuming with a large VLA model. Then, we added the inference result of SmolVLA on the dataset for training the correction head. The new dataset has all the LIBERO Spatial data, the action chunk result, and the VLM latent representation from the SmolVLA policy for each step.

After that, we trained the correction head with the dataset we created. For SmolVLA training, we trained a model from scratch with a cosine learning scheduler, which is the default setting for SmolVLA training. The parameter for SmolVLA training is in Table 5

For Correction head training, we use a constant learning rate of 1e-5. High learning rates, such as 1e-4, do not work well for the Correction head training. See Table 6

In both SmolVLA and Correction head training, the AdamW optimizer was used (Loshchilov & Hutter, 2017).

# A.2.3 EVALUATION DETAIL

For the evaluation, we tested various combinations of delay steps and horizon steps first. We tested 10 rollouts per task, and LIBERO Spatial has 10 tasks. Then, to evaluate more precisely, we select 3 pairs of delay and horizon, (0,10), (10,40), (0,50), and rollouts 50 per task. All results for LIBERO Spatial are shown in Table 7.

Table 5: Training hyperparameters for LIBERO with SmolVLA.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>1 × 10−4</td></tr><tr><td>Scheduler</td><td>Cosine</td></tr><tr><td>Warmup steps</td><td>1000 30000</td></tr><tr><td>Decay steps Minimum learning rate</td><td>2.5 × 10−6</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Training steps</td><td>100000</td></tr><tr><td>Optimizer €</td><td>1 × 10−8</td></tr><tr><td></td><td></td></tr><tr><td>Optimizer weight decay Gradient norm clip</td><td>1 × 10−10 10</td></tr></table>

Table 6: Training hyperparameters for LIBERO Correction head.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>1 × 10−5 (constant)</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Training steps</td><td>200000</td></tr><tr><td>Optimizer weight decay</td><td>1 × 10−5</td></tr><tr><td>Model dimension</td><td>512</td></tr><tr><td>Number of heads</td><td>8</td></tr><tr><td>Number of encoder layers</td><td>6</td></tr></table>

# A.3 SOURCE CODE FOR EXPERIMENTS

To facilitate reproducibility, we have released the source code for our experiments:

• Kinetix: https://github.com/k1000dai/a2c2-kinetix • LIBERO: https://github.com/k1000dai/a2c2-libero

# A.4 COMPUTATIONAL RESOURCES

We trained both models on NVIDIA RTX A6000 and H200 GPUs. Training in Kinetix required about 20 minutes per task on A6000, while LIBERO residual training (200k steps) took about 4 hours on H200.

# A.5 INFERENCE TIME COMPARISON

We benchmarked the average inference time per step for SmolVLA (450M parameters) and our Correction head (32M parameters) over 100 trials each. All measurements were performed on an NVIDIA RTX 5080 laptop GPU (16GB VRAM).

The results confirm that the correction head is significantly faster, with an average step time of 0.0047s compared to SmolVLA’s 0.101s. This $\sim 2 0 \times$ speed difference highlights that the proposed correction head can be integrated into high-frequency control loops without introducing prohibitive overhead, while still preserving the benefits of large foundation models at the chunk level.

# A.6 THE USE OF LARGE LANGUAGE MODELS

We used Large Language Models to polish our writing.

Table 7: LIBERO Spatial: success rate under different execution horizons and inference delays. 10 tasks and 10 rollouts per task. Residual correction consistently improves over the na¨ıve baseline.   

<table><tr><td>Execution Horizon</td><td>Inference Delay d</td><td>Naive</td><td>A2C2 (Ours)</td></tr><tr><td>40</td><td>10</td><td>0.67</td><td>0.84</td></tr><tr><td>40</td><td>5</td><td>0.66</td><td>0.86</td></tr><tr><td>40</td><td>3</td><td>0.65</td><td>0.86</td></tr><tr><td>40</td><td>1</td><td>0.74</td><td>0.83</td></tr><tr><td>10</td><td>10</td><td>0.75</td><td>0.88</td></tr><tr><td>10</td><td>5</td><td>0.82</td><td>0.92</td></tr><tr><td>10</td><td>3</td><td>0.81</td><td>0.89</td></tr><tr><td>10</td><td>1</td><td>0.83</td><td>0.92</td></tr><tr><td>50</td><td>0</td><td>0.71</td><td>0.84</td></tr><tr><td>40</td><td>0</td><td>0.79</td><td>0.89</td></tr><tr><td>30</td><td>0</td><td>0.79</td><td>0.89</td></tr><tr><td>10</td><td>0</td><td>0.85</td><td>0.87</td></tr><tr><td>5</td><td>0</td><td>0.83</td><td>0.85</td></tr><tr><td>1</td><td>0</td><td>0.77</td><td>0.84</td></tr></table>

Table 8: Average inference time per step (seconds). 100 trials.   

<table><tr><td>Model</td><td>Avg. Inference Time</td></tr><tr><td>SmolVLA (base policy)</td><td>101 msec</td></tr><tr><td>Correction head (Ours)</td><td>4.7 msec</td></tr></table>