# WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL

Zhennan Jiang2,3,4,\* Shangqing Zhou2,3,\* Yutong Jiang3 Zefang Huang4 Mingjie Wei4 Yuhui Chen2,3 Tianxing Zhou4 Zhen Guo5 Hao Lin5 Quanlu Zhang5 Yu Wang1 Haoran Li2,3, † Chao $\mathbf { V } \mathbf { u } ^ { 1 , }$ † Dongbin Zhao2,3,4

∗Equal contribution †Corresponding authors

1Tsinghua university 2Institute of Automation, Chinese Academy of Sciences 3University of Chinese Academy of Sciences 4Zhongguancun Academy 5Infinigence AI https://huggingface.co/Collections/RLinf/wovr $\pmb { \mathcal { Q } }$ https://github.com/RLinf/RLinf

Abstract: Reinforcement learning (RL) promises to unlock capabilities beyond imitation learning for Vision–Language–Action (VLA) models, but its requirement for massive real-world interaction prevents direct deployment on physical robots. Recent work attempts to use learned world models as simulators for policy optimization, yet closed-loop imagined rollouts inevitably suffer from hallucination and long-horizon error accumulation. Such errors do not merely degrade visual fidelity—they corrupt the optimization signal, encouraging policies to exploit model inaccuracies rather than genuine task progress. We propose WoVR, a reliable world-model-based reinforcement learning framework for post-training VLA policies. Instead of assuming a faithful world model, WoVR explicitly regulates how RL interacts with imperfect imagined dynamics. It improves rollout stability through a controllable action-conditioned video world model, reshapes imagined interaction to reduce effective error depth via Keyframe-Initialized Rollouts, and maintains policy–simulator alignment through World Model-Policy co-evolution. Extensive experiments on LIBERO benchmarks and real-world robotic manipulation demonstrate that WoVR enables stable long-horizon imagined rollouts and effective policy optimization, improving average LIBERO success from $3 9 . 9 5 \%$ to $6 9 . 2 \%$ $+ 2 9 . 3$ points) and real-robot success from $6 1 . 7 \%$ to $9 1 . 7 \%$ $\cdot + 3 0 . 0$ points). These results show that learned world models can serve as practical simulators for reinforcement learning when hallucination is explicitly controlled.

# 1 Introduction

Vision–Language–Action (VLA) models [1, 2, 3, 4] have been increasingly adopted for robotic manipulation, where actions are generated end-to-end by conditioning on language instructions and visual observations. Most existing VLA systems are trained via imitation learning. While effective in many downstream tasks, this paradigm fundamentally limits the performance ceiling of VLA policies, as it is tightly constrained by the quality and coverage of demonstration data.

Recent studies have demonstrated that reinforcement learning for VLA [5, 6, 7, 8, 9, 10] can substantially improve policy performance and reduce reliance on imitation data [11]. However, most approaches rely on standard on-policy optimization algorithms such as Proximal Policy Optimization (PPO) [12] or Group Relative Policy Optimization (GRPO) [13, 14], which require large-scale environment parallelism to achieve stable and efficient training. This requirement is impractical for real-world robotic reinforcement learning, where physical robot interaction is expensive [15], slow, and often requires substantial human supervision [16]. Although simulation-based alternatives have been explored [7, 17], accurately aligning simulators with real-world dynamics remains highly challenging, particularly for contact-rich manipulation tasks. These constraints motivate replacing real-environment interaction with a learned world model that serves as a simulator for policy optimization.

![](images/61a06b85f4c425f38571339f906d041144de0ef8158596dbaabd4093c174d1a7.jpg)  
Figure 1: Hallucination in Closed-Loop World Model Rollouts. The world model imagines a successful grasp (green frames), but real-world execution fails (red frames).To address this critical mismatch, we propose three hallucination-aware mechanisms.

Recent advances in large-scale generative video models [18, 19, 20] have made this direction increasingly feasible. Several works directly treat pretrained video generators as simulators and perform reinforcement learning entirely in imagination [21, 22, 23]. However, learned world models are not faithful simulators. In this work, we define hallucination as a systematic mismatch between imagined and real outcomes in closed-loop interaction. The world model may produce visually plausible rollouts while predicting physically incorrect state transitions or even spurious success signals under the policy’s actions(Fig. 1).

Hallucination is not merely a generation artifact — it fundamentally undermines reinforcement learning. In closed-loop autoregressive rollouts, prediction errors compound with horizon length due to:

• Autoregressive feedback: the model conditions on its own generated frames, amplifying small early errors; • Distribution shift: as the policy evolves, its action distribution drifts away from the data used to train the world model, increasing out-of-distribution prediction failures.

If hallucinated trajectories are directly used for policy optimization, reinforcement learning is incentivized to exploit systematic model errors rather than true task progress. This leads to a critical question:

If world models inevitably hallucinate, how can reinforcement learning remain reliable under imperfect imagined dynamics?

We argue that using world models for RL is not primarily a modeling problem, but a reliability problem. To make world-model-based reinforcement learning viable, one must control hallucination at three interconnected levels: controllable simulator design, reliable interaction protocol, and policy–model alignment. To this end, we propose WoVR, a World-model-based framework for posttraining Vision–Language–Action policies with Reinforcement Learning, built upon RLinf [24]. Rather than assuming the learned world model to be a faithful simulator, WoVR explicitly regulates how reinforcement learning interacts with imperfect imagined dynamics. We first strengthen the simulator itself by constructing a rollout-stable, action-controllable video world model with stabilized autoregressive context modeling, reducing long-horizon drift and structural collapse. However, improving the simulator alone is insufficient, as prediction errors inevitably accumulate over extended rollouts. We therefore reshape imagined interaction through Keyframe-Initialized Rollouts (KIR), which shorten the effective prediction depth by initializing trajectories near task-critical states, limiting the compounding of hallucination during learning. Finally, as policy optimization shifts the action distribution and induces distribution mismatch between the policy and the world model, we introduce $P A C E$ , a policy-aligned co-evolution strategy that restores alignment by iteratively refining the world model under the evolving policy distribution, without requiring continuous online supervision. Together, these components form a unified hallucination-aware reinforcement learning framework that enables reliable policy optimization in imagination.

In summary, our contributions are as follows.

• We identify hallucination under closed-loop imagined interaction as a fundamental reliability challenge in world-model-based reinforcement learning for VLA, showing that autoregressive error accumulation and policy-induced distribution shift can systematically corrupt optimization signals.   
• We propose WoVR, a hallucination-aware reinforcement learning framework that jointly regulates controllable simulator design, reliable imagined interaction, and a policy-aligned co-evolution strategy, enabling stable on-policy optimization entirely in imagination.   
• Extensive experiments demonstrate that WoVR achieves state-of-the-art world-model quality across perceptual and temporal metrics while maintaining high rollout efficiency (23 FPS). More importantly, WoVR delivers strong downstream task performance, improving average LIBERO success from $3 9 . 9 5 \%$ to $6 9 . 2 \%$ $+ 2 9 . 3$ points) and real-robot success from $6 1 . 7 \%$ to $9 1 . 7 \%$ $+ 3 0 . 0$ points).

# 2 Related Work

# 2.1 Online RL Fine-tuning for VLA Models

On-policy reinforcement learning [5, 12] has been increasingly adopted to fine-tune VLA models beyond imitation learning [5, 6]. However, directly transferring online on-policy fine-tuning to real robots remains impractical, as such methods require large-scale parallel rollouts, repeated environment resets, and tightly coupled policy–environment interaction, which are difficult to support under real-world hardware. To mitigate this, some off-policy approaches [8, 25] introduce offline data reuse or human intervention, but often suffer from limited scalability and performance degradation during online updates. An alternative direction builds large-scale real-robot infrastructures, yet existing systems [26, 27] still cannot practically support fully on-policy algorithms at scale. These limitations suggest that the challenge of online RL for VLA is systemic rather than algorithmic, motivating world-model-based approaches that decouple policy optimization from real-world interaction.

# 2.2 World Models

Recent progress in large-scale general-purpose world models [28, 29, 30, 31, 32, 33, 34] has demonstrated strong long-horizon generation and spatial memory under large viewpoint changes. However, these models rely on complex Self-Forcing/DMD training pipelines [35, 36, 37, 38, 39, 40], require massive pretraining data, cannot be trained from scratch, and are primarily designed for navigationstyle tasks with mouse–keyboard control. In contrast, embodied manipulation exhibits fixed viewpoints, locally constrained dynamics, and fine-grained object interactions, leading to fundamentally different modeling objectives, data distributions, and inference requirements.

To address embodied settings, prior works adapt pretrained video models into action-conditioned world models using projected end-effector position [41, 42], AdaLN-based frame-wise action injection [43, 22, 44, 45], cross-attention [46, 47], and MoE-based conditioning [48]. Despite improved action responsiveness, these approaches commonly suffer from slow inference, severe error accumulation in long-horizon autoregressive generation, and unstable modeling of fine-grained physical interactions, limiting their scalability for reinforcement learning.

# 2.3 World Models as Simulators

Many works have validated the correlation between VLA performance in real environments and in learned world models [43, 46, 42, 49] , demonstrating the potential of world models for outof-distribution generalization [43], and exploring the use of WM-generated synthetic data to train VLAs [46, 42]. However, these approaches do not treat world models as true simulators.

World-Env [21] and WMPO [22] take an important step toward treating learned world models as simulators, aiming to avoid costly interaction with real environments during reinforcement learning. Despite these advances, both approaches largely treat the world model as a drop-in replacement for a standard simulator, mechanically coupling on-policy reinforcement learning with imagined rollouts. They do not explicitly address the fundamental challenge of reinforcement learning under hallucinated dynamics, where closed-loop prediction errors accumulate and incentivize policies to exploit model inaccuracies. As a result, these methods lack dedicated mechanisms to regulate rollout horizons, suppress post-success hallucinations, or align policy optimization with the reliability regime of the world model.

# 3 Preliminary

# 3.1 Reinforcement Learning for VLA Fine-tuning

Recent work has increasingly adopted reinforcement learning to post-train VLA policies beyond imitation learning. In this setting, policy optimization is commonly formulated as a Markov Decision Process (MDP), defined by the tuple $\mathcal { M } = ( \mathcal { O } , \mathcal { A } , P , R , \gamma )$ . At each time step $t$ , given a visual observation $o _ { t } \in \mathcal { O }$ and a language instruction $l _ { t }$ , the VLA policy outputs a chunked action $a _ { t } \sim \pi _ { \theta } ( \cdot \mid o _ { t } , l _ { t } ) \in \mathcal { A }$ , which induces a state transition $o _ { t + 1 } \sim P ( \cdot \mid o _ { t } , a _ { t } )$ and yields a scalar reward $r _ { t } = R ( o _ { t } , a _ { t } )$ . The goal of reinforcement learning is to maximize the expected discounted return:

$$
J ( \pi _ { \theta } ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ \sum _ { t = 1 } ^ { T } \gamma ^ { t - 1 } r _ { t } \right] ,
$$

where $\gamma \in ( 0 , 1 ]$ is the discount factor.

Following standard on-policy reinforcement learning, the policy is optimized via a policy gradient objective:

$$
\nabla _ { \theta } J ( \pi _ { \theta } ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ \sum _ { t = 1 } ^ { T } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } \mid o _ { t } , l _ { t } ) A ( o _ { t } , a _ { t } ) \right] ,
$$

where $A ( o _ { t } , a _ { t } )$ denotes the advantage function. In general, the advantage function is defined as

$$
A ( o _ { t } , a _ { t } ) = Q ( o _ { t } , a _ { t } ) - V ( o _ { t } ) ,
$$

where $Q ( o _ { t } , a _ { t } )$ is the action-value function representing the expected return after taking action $a _ { t }$ at observation $o _ { t }$ , and $V ( o _ { t } )$ is the state-value function estimating the expected return under the current policy.

# 3.2 Problem Formulation

While on-policy reinforcement learning provides a principled framework for fine-tuning VLA policies, applying it directly in real-world robotic environments is often impractical due to high interaction cost and limited environment parallelism. To address this limitation, our goal is to replace real-environment interaction with a learned world model that can serve as a simulator for policy optimization.

We reformulate the original MDP as a World Model-MDP (WM-MDP), defined as

$$
\mathcal { M } _ { \mathrm { W M } } = ( \mathcal { O } , \mathcal { A } , \hat { P } _ { \phi } , \hat { R } _ { \psi } , \gamma ) ,
$$

![](images/7968f0473a5b7c7121a5d0cd2cf8fa686171725803b1b7b61877a1f236681334.jpg)  
Figure 2: Overview of WoVR. WoVR builds a reliability-driven reinforcement learning framework entirely around the learned world model. It first strengthens the world model as a controllable simulator, ensuring rollout-stable and action-responsive generation. On top of this simulator, it designs a reliable interaction protocol via Keyframe-Initialized Rollouts (KIR) and masked GRPO to reduce effective error depth and prevent optimization on hallucinated success. Finally, it maintains policy–model alignment through PACE, which co-evolves the world model with the evolving policy to mitigate distribution shift and preserve simulator reliability.

where the transition dynamics $\hat { P } _ { \phi } ( o _ { t + 1 } \mid o _ { t } , a _ { t } )$ are approximated by a learned world model parameterized by $\phi$ , and $\hat { R } _ { \psi } ( o _ { t } , a _ { t } )$ denotes a reward function parameterized by $\psi$ .

In $\mathcal { M } _ { \mathrm { W M } }$ , trajectories are generated by interacting with the world model rather than the physical environment. Specifically, given an observation $o _ { t }$ and an action $a _ { t }$ sampled from the policy, the next observation and reward are obtained as

$$
\tilde { o } _ { t + 1 } \sim \hat { P } _ { \phi } ( \cdot \mid o _ { t } , a _ { t } ) , \quad \tilde { r } _ { t } = \hat { R } _ { \psi } \big ( \tilde { o } _ { t + 1 } \big ) ,
$$

which allows the policy to perform closed-loop rollouts entirely in imagination. Under this formulation, the policy optimization objective remains unchanged in form:

$$
J _ { \mathrm { W M } } ( \pi _ { \theta } ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } , \hat { P } _ { \phi } } \left[ \sum _ { t = 1 } ^ { T } \gamma ^ { t - 1 } \hat { r } _ { t } \right] .
$$

# 4 Methods

We propose WoVR, a reliability-driven world-model-based reinforcement learning framework for post-training Vision–Language–Action (VLA) policies without requiring parallel real-world interaction. As illustrated in Fig. 2, WoVR treats the learned world model as a generative simulator and builds the entire reinforcement learning pipeline around controlling hallucination in closed-loop imagination.

Specifically, WoVR regulates reliability at three interconnected levels. (1) Simulator-level control: we construct an action-controllable, rollout-stable video world model with dual-channel action injection and first-frame anchoring to suppress long-horizon drift. (2) Interaction-level reshaping: we redesign imagined interaction through Keyframe-Initialized Rollouts (KIR) and masked GRPO to reduce effective error depth and prevent optimization on hallucinated success. (3) Alignment-level regulation: we introduce PACE, a policy–model co-evolution strategy that mitigates distribution shift by periodically aligning the world model with the evolving policy.

![](images/d962ce5e0af5bb1c0788cd8fd05b2f58c4aeae3a301f2d98206a58585e3d032b.jpg)  
Figure 3: Architecture of the proposed action-conditioned world model. The world model is built upon a video diffusion backbone and conditioned on actions via a dual-channel action injection design, enabling frame-level controllability and stable chunk-by-chunk autoregressive generation for long-horizon imagined rollouts.

Together, these components enable scalable and reliable on-policy optimization entirely in imagination.

# 4.1 Stabilized Action-Conditioned World Model

WoVR relies on a learned video world model as a generative simulator for closed-loop imagined interaction. However, long-horizon rollouts are prone to hallucination, where global scene structure gradually drifts and the background collapses as the rollout length increases. We therefore design the world model to be both action-controllable and rollout-stable, so that the simulated dynamics remain consistent under iterative, policy-driven generation.

Backbone and action conditioning. Our world model is built upon the Wan2.2-TI2V-5B video diffusion backbone [20]. Unlike conventional image-to-video generation, embodied simulation requires explicit action conditioning to ensure that predicted state transitions respond causally to the policy. To this end, we reformulate Wan2.2-TI2V into an action-conditioned generator via a dualchannel action injection design (Fig. 3), which preserves the original DiT structure while enabling frame-level controllability. Concretely, in each DiT block, action embeddings influence generation through two complementary pathways. First, action embeddings are fused with the diffusion timestep embeddings and applied via AdaLN-Zero-style modulation, directly shaping the denoising dynamics at the feature level. Second, we retain the original cross-attention operator but replace textual embeddings with action embeddings, allowing actions to condition the network globally across layers. Together, these two pathways provide both local modulation and global context for action-conditioned video generation.

First-frame anchoring for rollout stability. Even with strong action conditioning, chunk-bychunk autoregressive generation can still accumulate errors, leading to spatial drift and gradual background collapse. To suppress such long-horizon degradation, we adopt a first-frame–anchored inference context. At each autoregressive step, the model conditions on $\left[ o _ { 0 } , o _ { t ^ { \prime } - c ^ { \prime } : t ^ { \prime } } \right]$ , which concatenates the episode’s initial reference frame with the most recent memory frames from the previous chunk. This persistent reference constrains global appearance and scene layout, because many selfattention heads naturally attend to the first frame during denoising (Fig. 4), consistent with prior findings [50, 51, 31].

![](images/fea056911b3bdef4d6a7cd3e981d5ca8fb456431d5fef51d9e473071f288be7a.jpg)  
Figure 4: Visualization of self attention probability map. During the denoising process, many attention heads focus on the first frame of the sequence.

With dual-channel action conditioning and first-frame anchoring, we obtain a rollout-stable world model that can be used as a generative simulator for closed-loop interaction. Starting from the firstframe–anchored context $\left[ o _ { 0 } , o _ { t ^ { \prime } - c ^ { \prime } : t ^ { \prime } } \right]$ , we first encode it into latent representations $[ z _ { 0 } , z _ { t - c : t } ]$ using the Wan encoder. We then sam tents $z _ { t + 1 : t + H } ^ { \mathrm { n o i s e } } \sim { \mathcal { N } } ( 0 , \mathbf { I } )$ for the next chunk and feed the concatenated latents [z0, zt−c:t, znoiset+1:t+H ] The model predicts future latents $\hat { z } _ { t + 1 : t + H }$ , which are decoded into frames $\widehat { O } _ { t ^ { \prime } + 1 : t ^ { \prime } + H ^ { \prime } }$ and appended to the context. By iterating this chunk-by-chunk procedure, the world model generates long-horizon imagined rollouts under the policy’s actions while maintaining global scene consistency, and these rollouts are subsequently used as the simulated environment for reinforcement learning.

We train the world model with the Rectified Flow objective [52]. Let $x _ { 1 } = z _ { t + 1 : t + H }$ denote the target future latents and $x _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ be noise of the same shape. Given a sampled time $t \in [ 0 , 1 ]$ , we obtain the intermediate latent $x _ { t }$ and train the model $u ( \cdot ; \phi )$ to predict the velocity $v _ { t }$ :

$$
\mathcal { L } = \mathbb { E } _ { x _ { 0 } , x _ { 1 } , c , t } \left[ \left\| u ( x _ { t } , c , t ; \phi ) - v _ { t } \right\| ^ { 2 } \right] ,
$$

where the condition $c$ includes both the first-frame–anchored context and actions. To reduce the train–inference gap in closed-loop rollouts, we additionally apply noisy context by injecting diffusion noise into the non-reference context latents $z _ { t - c : t }$ during training. This discourages brittle visual copying from context and improves robustness when the model consumes self-generated frames over long horizons [53, 46].

When serving as a simulator for reinforcement learning, the world model must also provide a reward signal. In real-world robotic manipulation, designing dense rewards is often impractical, and training typically relies on sparse success annotations. Therefore, we introduce a learned reward classifier that produces a binary success signal based on the predicted next observation. Concretely, given the generated observation $\tilde { o } _ { t + 1 }$ from the world model, the reward model $R \psi$ predicts the probability of task success, and the sparse reward is defined as

$$
r _ { t + 1 } = \mathbb { I } \left( R _ { \psi } ( \tilde { o } _ { t + 1 } ) \geq 0 . 5 \right) ,
$$

where $\mathbb { I } ( \cdot )$ denotes the indicator function. Following HiL-SERL [16], the reward classifier is implemented as a lightweight network and trained using binary cross-entropy (BCE) loss on labeled success states.

# 4.2 Hallucination-Aware Policy Optimization in Imagination

WoVR optimizes the VLA policy by interacting with the learned world model, which serves as a generative simulator for closed-loop imagined rollouts. The key difficulty is that, in long-horizon rollouts starting from the initial state, world-model errors accumulate early and can eventually produce visually plausible but physically incorrect transitions and even spurious success signals. If reinforcement learning naively trusts such rollouts, the policy is encouraged to optimize toward hallucinated outcomes rather than real task progress.

![](images/a5716fc4ffdfb1d12df145fcc98cf67ab9cf4abecdfca1d7b5dab1347ca6103c.jpg)  
Figure 5: Illustration of the effect of Keyframe-Initialized Rollouts (KIR). Starting from the initial state, long-horizon rollouts accumulate prediction errors in early stages, leading to hallucinated success that contradicts the ground-truth failure. In contrast, Keyframe-Initialized Rollouts initialize rollouts near critical states, enabling physically consistent predictions that correctly model failure, which in turn facilitates more efficient and stable policy learning.

To reduce the effective error depth of imagined interaction, we introduce Keyframe-Initialized Rollouts (KIR). As illustrated in Figure 5, instead of always initializing rollouts from the episode start $o _ { 0 }$ , we initialize a portion of rollouts from keyframes $o _ { k }$ that lie near task-critical intermediate states, especially failure states encountered by the current policy. The motivation is that many decisive contacts and corrections happen locally around these states, whereas starting from $o _ { 0 }$ forces the world model to predict a long prefix before reaching them, during which compounding errors can already derail the rollout.

We adopt Group Relative Policy Optimization (GRPO) to update the policy using imagined rollouts. For each update, we sample a group of imagined trajectories and compute a group-relative advantage, then optimize a clipped GRPO objective. Because hallucinations often dominate after success has been reached in imagination, we mask post-success steps and normalize each trajectory by its valid length.

Formally, given a group of imagined trajectories $\{ \tau ^ { ( i ) } \} _ { i = 1 } ^ { G }$ sampled in the world model,

$$
\tau ^ { ( i ) } = \{ ( o _ { t } ^ { ( i ) } , a _ { t } ^ { ( i ) } , \hat { r } _ { t } ^ { ( i ) } ) \} _ { t = 1 } ^ { T } , \qquad o _ { t + 1 } ^ { ( i ) } \sim \hat { P } _ { \phi } ( \cdot \mid o _ { t } ^ { ( i ) } , a _ { t } ^ { ( i ) } ) ,
$$

we compute the return $R ( \tau ^ { ( i ) } )$ and group-relative advantage $\hat { A } ^ { ( i ) }$ ,

$$
R ( \tau ^ { ( i ) } ) = \sum _ { t = 1 } ^ { | \tau ^ { ( i ) } | } \gamma ^ { t - 1 } \hat { r } _ { t } ^ { ( i ) } , \qquad \hat { A } ^ { ( i ) } = R ( \tau ^ { ( i ) } ) - \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R ( \tau ^ { ( j ) } ) .
$$

Let $T _ { i } ^ { \mathrm { v a l i d } }$ be the number of valid timesteps up to (and including) the first success, and define $\begin{array} { r } { \rho _ { t } ^ { ( i ) } ( \theta ) = \frac { \pi _ { \theta } ( a _ { t } ^ { ( i ) } | o _ { t } ^ { ( i ) } , l ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { t } ^ { ( i ) } | o _ { t } ^ { ( i ) } , l ) } } \end{array}$ . The masked, trajectory-length–normalized GRPO objective is

$$
J _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { T _ { i } ^ { \mathrm { v a l i d } } } \sum _ { t = 1 } ^ { T _ { i } ^ { \mathrm { v a l i d } } } \operatorname* { m i n } \Bigl ( \rho _ { t } ^ { ( i ) } ( \theta ) \hat { A } ^ { ( i ) } , \mathrm { c l i p } \Bigl ( \rho _ { t } ^ { ( i ) } ( \theta ) , 1 - \epsilon , 1 + \epsilon \Bigr ) \hat { A } ^ { ( i ) } \Bigr ) \right] .
$$

This objective also complements KIR: keyframe-initialized rollouts tend to reach task resolution with fewer valid steps, and trajectory-length normalization increases their per-timestep contribution, so gradients are dominated by short, task-critical segments rather than long, drift-prone continuations.

# 4.3 PACE: Policy–Aligned Co-Evolution

While policy optimization proceeds entirely within the learned world model, the policy’s action distribution continuously evolves and drifts away from the data used to train the initial world model. This inherent distribution shift leads to accumulating mismatch between the simulator and the improving policy, ultimately degrading the reliability of imagined rollouts.

To address this issue, we introduce PACE, a World Model–Policy co-evolution strategy. Instead of treating the world model as a fixed, static simulator throughout policy optimization, PACE allows the world model and VLA policy to evolve together throughout training.

Concretely, we realize this co-evolution through low-frequency, policy-driven refinement: we first train an initial world model, denoted as $\mathrm { W M _ { B a s e } }$ , using trajectories collected from the base VLA policy. After the first stage of policy optimization within $\mathrm { W M _ { B a s e } }$ , we collect a limited set of additional rollouts under the evolved policy and use them to further refine the world model. The refined model is referred to as $\mathrm { W M _ { E v o } }$ .

Importantly, this refinement is performed only once (or at very low frequency), distinguishing $P A C E$ from classical model-based reinforcement learning methods, which continuously update the dynamics model at high frequency during policy optimization.

This low-frequency refinement provides two key advantages. First, unlike real-world online RL, it does not require continuous human supervision or environment resets during policy training, significantly reducing operational overhead. Second, by aligning the world model with the evolving policy distribution, $P A C E$ mitigates compounding model errors and maintains simulator reliability without sacrificing training stability.

System Implementation. We build WoVR on top of RLinf [24] to support efficient distributed imagined rollouts and training. Concretely, we replace RLinf’s environment back-end with our learned world model, enabling scalable closed-loop rollouts without a ground-truth simulator. GPU allocation details are provided in Appendix A.

# 5 Experiments

We conduct extensive experiments to evaluate the effectiveness of WoVR as a world-model-based reinforcement learning framework for post-training VLA policies. Our experimental design aims to systematically answer the following three questions:

• Q1: Is the proposed world model stable, controllable, and efficient enough to serve as a simulator for closed-loop reinforcement learning?   
• Q2: Can WoVR effectively improve VLA task performance compared to existing worldmodel-based reinforcement learning methods?   
• Q3: Do the policies optimized with WoVR reliably transfer to real-world robotic manipulation tasks?

To answer these questions, we evaluate both the quality of the learned world model and the downstream policy performance. For world model evaluation, we focus on long-horizon, actionconditioned video generation under closed-loop, chunk-by-chunk autoregressive inference. We adopt standard perceptual and distributional metrics, including LPIPS [54], FID [55], FVD [56] and FloLPIPS [57]. Specifically, LPIPS (Learned Perceptual Image Patch Similarity) measures frame-level perceptual similarity using deep feature representations; FID (Frechet Inception Dis- ´ tance) evaluates distributional similarity between generated and real frames via feature statistics; FVD (Frechet Video Distance) extends this comparison to the temporal domain to assess video- ´ level realism and motion consistency; and FloLPIPS measures motion-aligned perceptual similarity along estimated optical flow trajectories, emphasizing temporal coherence under action-conditioned dynamics. Since the world model is intended to be used as a simulator for closed-loop on-policy reinforcement learning, we also report inference throughput (frames per second) to quantify generation efficiency.

For policy evaluation, we use task success rate (SR) as the primary metric, reflecting the sparsereward setting commonly encountered in real-world robotic manipulation. All success rates are computed over multiple independent rollouts with fixed initial conditions.

We compare WoVR against several representative baselines spanning both world model quality and policy optimization. For world model quality, we include EVAC [42], which conditions generation on absolute end-effector actions, as well as Cosmos-Predict2 [58] and OpenSora [19], the latter serving as the world-model backbone adopted in WMPO [22]. All compared models are evaluated under the same chunk-wise autoregressive generation protocol to ensure a fair comparison.

For policy optimization, we consider the following baselines:

• OpenVLA-OFT-base [1]: a base VLA policy trained purely with imitation learning; • GRPO (Online) [14]: trained with real-environment interaction under the same rollout budget; • WMPO [22]: performs reinforcement learning using an OpenSora-based world model.

All world-model-based methods are trained until convergence within their respective simulators, while GRPO is reported under the same rollout budget to ensure a fair comparison. All experiments are conducted using eight NVIDIA H100 GPUs.

# 5.1 Q1: Is the World Model Stable, Controllable, and Efficient?

We first investigate whether the proposed world model is sufficiently stable, controllable, and efficient to serve as a simulator for closed-loop reinforcement learning. In particular, we focus on long-horizon, action-conditioned video generation under chunk-by-chunk autoregressive inference, where modeling errors may accumulate and severely affect downstream policy optimization.

Experimental Setup. We conduct all world model evaluations in the LIBERO environment [59]. A total of 3,000 VLA rollout trajectories, each with a length of 512 frames, are collected to train the world models. In addition, 200 held-out trajectories of the same length are used exclusively for evaluation. We compare WoVR against three representative action-conditioned world models: EVAC, Cosmos-Predict2, and OpenSora as adopted in WMPO. Among them, EVAC conditions video generation on absolute end-effector actions, while Cosmos-Predict2, OpenSora, and WoVR all use residual action representations.

During evaluation, all baseline models follow the same chunk-wise autoregressive generation protocol to ensure a fair comparison. Specifically, each model predicts future video segments by conditioning on a 4-frame visual context together with an 8-step action chunk, and autoregressively generates the subsequent 8 frames. For the first chunk, where only a single initial image is available, the initial frame is replicated to fill the context window in order to align the inference procedure across methods. We quantitatively evaluate the generated rollouts by comparing predicted videos with ground-truth trajectories using standard video generation metrics, including LPIPS, FID, FVD and FloLPIPS.

Table 1: World model quality, motion consistency, and efficiency comparison. Rollout denotes the rollout horizon length.   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Rollout</td><td colspan="5">Metrics</td></tr><tr><td>FPS ↑</td><td>LPIPS [54]↓</td><td>FID [55]↓</td><td>FVD [56]↓</td><td>FloLPIPS [57]↓</td></tr><tr><td rowspan="3">EVAC [42]</td><td>512</td><td rowspan="3">2.7</td><td>0.146</td><td>46.528</td><td>345.818</td><td>0.205</td></tr><tr><td>256</td><td>0.130</td><td>49.153</td><td>354.983</td><td>0.192</td></tr><tr><td>128</td><td>0.106</td><td>44.337</td><td>423.132</td><td>0.166</td></tr><tr><td rowspan="3">Cosmos-Predict2 [58]</td><td>512</td><td rowspan="3">3.50</td><td>0.315</td><td>165.862</td><td>275.737</td><td>0.265</td></tr><tr><td>256</td><td>0.226</td><td>106.324</td><td>203.853</td><td>0.306</td></tr><tr><td>128</td><td>0.164</td><td>77.555</td><td>304.456</td><td>0.281</td></tr><tr><td rowspan="3">OpenSora [22]</td><td>512</td><td rowspan="3">7.00</td><td>0.105</td><td>38.478</td><td>89.391</td><td>0.156</td></tr><tr><td>256</td><td>0.082</td><td>33.577</td><td>94.998</td><td>0.122</td></tr><tr><td>128</td><td>0.069</td><td>33.413</td><td>111.643</td><td>0.113</td></tr><tr><td rowspan="3">WoVR (Ours)</td><td>512</td><td></td><td>0.091</td><td>34.252</td><td>68.011</td><td>0.154</td></tr><tr><td>256</td><td rowspan="2">23.0</td><td>0.063</td><td>24.378</td><td>50.041</td><td>0.102</td></tr><tr><td>128</td><td>0.047</td><td>18.553</td><td>39.047</td><td>0.079</td></tr></table>

Quantitative Results. Table 1 summarizes the quantitative comparison across different rollout horizons. As shown in the table, WoVR consistently outperforms EVAC, Cosmos-Predict2, and OpenSora across all evaluation metrics. In particular, WoVR achieves the lowest LPIPS, FID, FVD, and FloLPIPS scores at all tested rollout lengths, indicating higher visual fidelity, stronger temporal consistency, and more accurate action-conditioned dynamics. These improvements become more pronounced as the rollout horizon increases, suggesting that WoVR is more robust to error accumulation in long-horizon autoregressive generation.

Despite adopting a larger backbone (Wan, ${ \sim } 5 \mathrm { B }$ ) than OpenSora $( \sim 1 . 3 \mathrm { B } )$ , WoVR achieves higher inference throughput by requiring only five diffusion steps and leveraging a 3D VAE for spatiotemporal latent encoding, whereas OpenSora typically relies on more sampling steps and a 2D VAE.

# 5.2 Q2: Can WoVR Effectively Improve VLA Task Performance?

We next evaluate whether the proposed world model can effectively support reinforcement learning and lead to improved task performance of VLA policies. Beyond world model fidelity, this experiment directly assesses the practical value of WoVR as a simulator for policy optimization.

Experimental Setup. We conduct policy optimization experiments on multiple LIBERO task suites, including the Spatial, Object, Goal, and Long suites [59]. As the base policy, following SimpleVLA-RL [17], we initialize from OpenVLA-OFT and perform one-trajectory supervised fine-tuning.

Each LIBERO suite contains 10 tasks. For each suite, we allocate a total real-environment rollout budget of 2,500 trajectories. We first collect 1,500 trajectories (150 per task) using the base VLA policy to train the initial world model $\mathrm { W M _ { B a s e } }$ . After the first stage of policy optimization in imagination, we collect an additional 1,000 trajectories under the updated policy and use them to refine the world model $\mathrm { W M _ { E v o } }$ , aligning the simulator with the evolving policy distribution. To balance alignment quality and computational efficiency, we perform only a single co-evolution step in practice, resulting in one refinement from $\mathrm { W M _ { B a s e } }$ to $\mathrm { W M _ { E v o } }$ rather than multiple iterative alternations.

To ensure a fair comparison, all methods are allocated the same real-environment rollout budget of 2,500 trajectories per suite. For world-model-based methods, including WMPO and WoVR, these trajectories are used exclusively for world model training and refinement, while policy optimization is conducted entirely within the learned world model via imagined rollouts, without further interaction with the ground-truth simulator. In contrast, GRPO directly interacts with the ground-truth simulator and consumes the same 2,500-trajectory budget for on-policy policy optimization.

Table 2: Task success rates $( \% )$ across LIBERO task suites. The base policy is OpenVLAOFT trained with one-trajectory supervised fine-tuning (one-trajectory SFT). All methods use 2,500 trajectories collected from the ground-truth simulator. GRPO (Online) consumes them for on-policy interaction, whereas WMPO and $\mathbf { W o V R }$ use them only for world model training and perform policy optimization via imagined rollouts. Improvements are shown in parentheses relative to the base policy.   

<table><tr><td></td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg ↑</td></tr><tr><td>OpenVLA-OFT-base [1]</td><td>61.5</td><td>36.3</td><td>48.2</td><td>13.7</td><td>39.9</td></tr><tr><td>GRPO (online) [14]</td><td>66.6</td><td>45.1</td><td>52.1</td><td>14.5</td><td>44.6</td></tr><tr><td>WMPO [22]</td><td>67.8</td><td>48.0</td><td>54.6</td><td>13.7</td><td>46.2</td></tr><tr><td>WoVR (Ours)</td><td>81.5 (+20.0)</td><td>82.0 (+45.7)</td><td>77.5 (+29.3)</td><td>35.8 (+22.1)</td><td>69.2 (+29.3)</td></tr></table>

Table 2 summarizes success rates across LIBERO suites under this shared simulator-trajectory budget: GRPO uses the 2,500 trajectories for on-policy optimization, whereas WMPO and WoVR use them only to train the world model and optimize the policy via imagined rollouts.

Quantitative Results. Table 2 reports task success rates across LIBERO suites under a shared simulator-trajectory budget. The base policy achieves moderate performance, reflecting the limitations of imitation learning under sparse rewards and limited demonstrations. While GRPO improves over the base policy, its gains come at a high interaction cost. In practice, each policy update requires close to a thousand additional simulator trajectories, making the optimization process sample-inefficient under realistic interaction constraints. This highlights the fundamental limitation of purely online reinforcement learning in data-scarce robotic settings.

WMPO further improves performance on short- and medium-horizon suites (Spatial, Object, and Goal), demonstrating that world-model-based optimization can provide benefits beyond online interaction. However, WMPO does not achieve performance gains on the LIBERO-Long suite, which consists of longer-horizon tasks. In these tasks, rollout instability in later stages of autoregressive generation degrades policy optimization, resulting in no improvement over the base policy.

In contrast, WoVR consistently achieves the highest success rates across all evaluated suites. Notably, WoVR improves performance by $+ 2 0 . 0 \%$ on Spatial, $+ 4 5 . 7 \%$ on Object, $+ 2 9 . 3 \%$ on Goal, and $+ 2 2 . 1 \%$ on the long-horizon LIBERO-Long suite compared to the base policy. On average, WoVR achieves a success rate of $6 9 . 2 \%$ , substantially outperforming both GRPO $( 4 4 . 6 \% )$ and WMPO $( 4 6 . 2 \% )$ .

These results indicate that the improved stability and controllability of the proposed world model directly translate into more effective policy optimization. In particular, the strong performance on long-horizon tasks highlights that suppressing error accumulation in imagined rollouts is critical for reliable reinforcement learning with learned simulators.

# 5.3 Q3: Do Policies Optimized with WoVR Reliably Transfer to the Real World?

Finally, we evaluate whether policies optimized with WoVR reliably transfer to real-world robotic manipulation tasks.

Experimental Setup Our experiments are conducted on a Franka Emika Panda robot. We consider two contact-rich manipulation tasks: (i) Pick Banana, which requires picking a banana and placing it onto a plate, and (ii) Pick Bread, which requires picking a bread item and placing it onto a designated bread marker. For each task, we collect 10 teleoperated demonstrations to pre-train the base VLA policy, and additionally collect 150 rollouts from the base policy to train the world model. After training, we deploy the resulting policies on the physical robot and evaluate success rates over 30 independent trials per task.

![](images/d5f8c696d64d952c958bde4cd370679b1e85466c04657e4bfd63a7de7f994738.jpg)  
Figure 6: Real-world setup on a Franka Panda for Pick Banana and Pick Bread.

Table 3: Real-world success rates ( $\%$ , 30 trials per task) on a Franka Panda robot. Improvements are shown in parentheses relative to the base policy.   

<table><tr><td>Method</td><td>Pick Banana</td><td>Pick Bread</td><td>Avg</td></tr><tr><td>OpenVLA-OFT-base</td><td>46.7 (14/30)</td><td>76.7 (23/30)</td><td>61.7</td></tr><tr><td>WoVR (Ours)</td><td>93.3 (28/30) (+46.6)</td><td>90.0 (27/30) (+13.3)</td><td>91.7 (+30.0)</td></tr></table>

Quantitative Results. Table 3 reports the real-world success rates. On Pick Banana, the base policy achieves a success rate of $4 6 . 6 7 \%$ (14/30), while WoVR improves it to $9 3 . 3 \%$ (28/30). On Pick Bread, WoVR increases the success rate from $7 6 . 6 7 \%$ (23/30) to $9 0 . 0 \%$ (27/30). These results demonstrate that $\mathbf { W o V R }$ delivers consistent real-world gains over imitation learning without requiring additional online interaction during policy optimization, indicating strong sim-to-real transfer of the optimized behaviors.

# 6 Ablation Study

# 6.1 Ablation on World Model Mechanisms

We first conduct ablation studies on the core design choices of the proposed world model, aiming to understand how different context modeling mechanisms affect long-horizon video generation stability. Specifically, we investigate the following factors: (i) the number of memory frames used as visual context, (ii) the use of a fixed reference frame, and (iii) the effect of adding noise to context frames during training.

Experimental Variants. We compare the full WoVR model against three ablated variants:

• WoVR w/o ref, which removes the fixed reference frame from the context window;   
• WoVR w. mem $_ { | = 1 }$ , which uses only a single-frame context;   
• WoVR w/o noisy context, which disables noise injection on context frames during training.

All variants are trained and evaluated on the LIBERO-Spatial suite only. We train the world model using 1,500 VLA rollout trajectories and evaluate on a held-out set of 24 trajectories.

Quantitative Results. Table 4 reports the quantitative results measured by LPIPS, FID, and FVD under different rollout horizons. Compared to using a single-frame context, employing a multi-frame context with a fixed reference anchor significantly improves performance across all metrics.

Table 4: Ablation study on world model mechanisms (LIBERO-Spatial). Rollout denotes the rollout horizon length.   

<table><tr><td></td><td></td><td colspan="4">Metrics</td></tr><tr><td>Method</td><td>Rollout</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td><td>FloLPIPS↓</td></tr><tr><td rowspan="3">WoVR (Ours)</td><td>512</td><td>0.091</td><td>36.687</td><td>73.493</td><td>0.154</td></tr><tr><td>256</td><td>0.069</td><td>27.238</td><td>63.948</td><td>0.110</td></tr><tr><td>128</td><td>0.051</td><td>20.780</td><td>49.017</td><td>0.081</td></tr><tr><td rowspan="3">WoVR w/o ref</td><td>512</td><td>0.133</td><td>73.942</td><td>123.502</td><td>0.168</td></tr><tr><td>256</td><td>0.089</td><td>49.406</td><td>86.000</td><td>0.116</td></tr><tr><td>128</td><td>0.064</td><td>35.559</td><td>86.146</td><td>0.090</td></tr><tr><td rowspan="3">WoVR w. mem=1</td><td>512</td><td>0.120</td><td>64.501</td><td>86.042</td><td>0.165</td></tr><tr><td>256</td><td>0.086</td><td>46.790</td><td>81.742</td><td>0.117</td></tr><tr><td>128</td><td>0.065</td><td>36.047</td><td>79.605</td><td>0.095</td></tr><tr><td rowspan="3">WoVR w/o noisy context</td><td>512</td><td>0.099</td><td>44.712</td><td>77.284</td><td>0.160</td></tr><tr><td>256</td><td>0.074</td><td>31.691</td><td>61.660</td><td>0.115</td></tr><tr><td>128</td><td>0.054</td><td>23.444</td><td>58.836</td><td>0.085</td></tr></table>

To better understand the failure modes behind these quantitative trends, we provide qualitative comparisons in Fig. 7. As shown in the figure, models without a fixed reference frame or noisy context exhibit noticeable spatial drift and object disappearance over long-horizon rollouts, whereas the full WoVR model remains visually stable and consistent with the ground truth.

Removing the reference frame leads to a clear degradation in performance, especially under longer rollout horizons. This result suggests that anchoring the context with a fixed reference frame effectively suppresses error accumulation in the autoregressive feedback loop, which is critical for maintaining stability in long-horizon video generation.

Furthermore, disabling noise injection on context frames also results in noticeable performance drops. While the degradation is moderate for short rollouts, the gap becomes more pronounced as the rollout length increases. This observation indicates that adding mild noise to context frames improves robustness in long-horizon generation by reducing over-reliance on precise conditioning inputs, thereby alleviating the train–inference gap.

Overall, these results demonstrate that the proposed context modeling strategy—combining a fixed reference frame, a multi-frame memory window, and noisy context augmentation—plays a crucial role in stabilizing long-horizon video generation. Together, these mechanisms enable WoVR to maintain high fidelity and temporal consistency under closed-loop autoregressive inference, providing a more reliable simulator for downstream reinforcement learning.

# 6.2 Ablation on Policy Optimization Mechanisms

We next ablate key components in the policy optimization pipeline of WoVR, aiming to understand how different design choices affect downstream VLA task performance. In particular, we focus on mechanisms that facilitate stable policy learning and effective utilization of the learned world model.

Experimental Setup. All experiments are conducted on the LIBERO-Spatial suite, with the same training protocol, data budget, and evaluation procedure as described in Sec. 5.2. Specifically, the base VLA policy is pre-trained following the same demonstration setup as in Q2, and policy optimization is performed using world-model-based reinforcement learning. Task performance is measured by the average success rate over the LIBERO-Spatial tasks.

![](images/b8f0811df06a5de93d42a9823c5640576d94712712c5d4d7d535d877c3f907ae.jpg)  
Figure 7: Qualitative ablation results on LIBERO-Spatial. Ablated variants exhibit error accumulation and visual drift under long-horizon rollouts, while the full WoVR model remains stable and consistent with the ground truth.

We compare the full WoVR framework against the following ablated variants:

• WoVR w/o KIR, which removes keyframe-based initialization and starts policy optimization from randomly sampled initial states in the world model; • WoVR w/o PACE, which disables the co-evolution of the world model with the updated policy and keeps the world model fixed during policy optimization.

Quantitative Results. Table 5 reports the success rates on the LIBERO-Spatial suite. The full WoVR framework achieves the highest performance, with an average success rate of 0.815. Removing keyframe-based initialization leads to a noticeable drop in performance, reducing the success rate to 0.782. This result indicates that KIR plays an important role in stabilizing early-stage policy learning by providing meaningful initial states.

Disabling the co-evolution of the world model further degrades performance to 0.71. This suggests that continuously refining the world model with updated policy rollouts is critical for maintaining simulator accuracy and preventing compounding model errors during policy optimization.

# 7 Conclusion

In this work, we revisited world-model-based reinforcement learning for VLA policies through the lens of reliability. Rather than assuming a learned world model to be a faithful simulator, we identified hallucination under closed-loop imagined interaction as the central obstacle: autoregressive error accumulation and policy-induced distribution shift can systematically corrupt optimization signals, causing reinforcement learning to exploit model inaccuracies instead of genuine task progress.

Table 5: Ablation on policy optimization mechanisms on LIBERO-Spatial. Success rate is averaged over all tasks in the suite.   

<table><tr><td>Method</td><td>Success Rate ↑</td></tr><tr><td>WoVR (Ours)</td><td>0.815</td></tr><tr><td>WoVR w/o KIR</td><td>0.782</td></tr><tr><td>WoVR w/o PACE</td><td>0.710</td></tr></table>

To make RL in imagination viable under imperfect dynamics, we introduced WoVR, a hallucinationaware framework that controls hallucination at three interconnected levels. First, we strengthen the simulator itself by building a rollout-stable, action-controllable video world model, improving longhorizon consistency under policy-driven generation. Second, because residual prediction errors are unavoidable, we reshape the interaction protocol with Keyframe-Initialized Rollouts (KIR) to reduce the effective error depth and concentrate learning on task-critical segments where dynamics must be correct. Third, to prevent the evolving policy from drifting out of the simulator’s training distribution, we maintain policy–simulator alignment via PACE, a policy-aligned co-evolution strategy that mitigates distribution mismatch without requiring continuous online supervision.

Extensive experiments on LIBERO and real-world manipulation tasks demonstrate that WoVR enables stable long-horizon imagined rollouts and effective on-policy optimization, yielding substantial gains over imitation learning and reliable transfer to physical robots. Overall, our results suggest that learned world models can serve as practical simulators for reinforcement learning when hallucination is explicitly regulated by design, interaction, and alignment. Nevertheless, WoVR reduces but does not fully eliminate hallucination, particularly in extremely long-horizon or highly contactsensitive settings, and it still relies on learned reward modeling and limited real-data refinement, leaving broader reliability guarantees as an open direction for future work.

# References

[1] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.   
[2] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. $\pi _ { 0 }$ : A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.   
[3] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, et al. $\pi _ { 0 . 5 }$ : a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.   
[4] H. Li, Y. Chen, W. Cui, W. Liu, K. Liu, M. Zhou, Z. Zhang, and D. Zhao. Survey of visionlanguage-action models for embodied manipulation. arXiv preprint arXiv:2508.15201, 2025.   
[5] H. Zang, M. Wei, S. Xu, Y. Wu, Z. Guo, Y. Wang, H. Lin, L. Shi, Y. Xie, Z. Xu, Z. Liu, K. Chen, W. Tang, Q. Zhang, W. Zhang, C. Yu, and Y. Wang. Rlinf-vla: A unified and efficient framework for vla+rl training, 2025.   
[6] J. Liu, F. Gao, B. Wei, X. Chen, Q. Liao, Y. Wu, C. Yu, and Y. Wang. What can rl bring to vla generalization? an empirical study. arXiv preprint arXiv:2505.19789, 2026.   
[7] K. Chen, Z. Liu, T. Zhang, Z. Guo, S. Xu, H. Lin, H. Zang, X. Li, Q. Zhang, Z. Yu, G. Fan, T. Huang, Y. Wang, and C. Yu. $\pi _ { \tt R L }$ : Online rl fine-tuning for flow-based vision-language-action models, 2026.   
[8] Y. Chen, S. Tian, S. Liu, Y. Zhou, H. Li, and D. Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. In Proceedings of Robotics: Science and Systems, RSS 2025, Los Angeles, CA, USA, Jun 21-25, 2025, 2025. doi:10.15607/RSS.2025.XXI.019.   
[9] G. Lu, W. Guo, C. Zhang, Y. Zhou, H. Jiang, Z. Gao, Y. Tang, and Z. Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025.   
[10] Y. Li, X. Ma, J. Xu, Y. Cui, Z. Cui, Z. Han, L. Huang, T. Kong, Y. Liu, H. Niu, W. Peng, J. Qiao, Z. Ren, H. Shi, Z. Su, J. Tian, Y. Xiao, S. Zhang, L. Zheng, H. Li, and Y. Wu. Gr-rl: Going dexterous and precise for long-horizon robotic manipulation, 2025.   
[11] K. Lei, H. Li, D. Yu, Z. Wei, L. Guo, Z. Jiang, Z. Wang, S. Liang, and H. Xu. Rl-100: Performant robotic manipulation with real-world reinforcement learning, 2025.   
[12] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
[13] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[14] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[15] W. Cui, C. Zhao, S. Wei, J. Zhang, H. Geng, Y. Chen, and H. Wang. Gapartmanip: a large-scale dataset for generalizable and actionable part manipulation with material-agnostic articulated objects. In IEEE International Conference on Robotics and Automation. IEEE, 2025.   
[16] J. Luo, C. Xu, J. Wu, and S. Levine. Precise and dexterous robotic manipulation via humanin-the-loop reinforcement learning. Science Robotics, 10(105):eads5033, 2025.   
[17] H. Li, Y. Zuo, J. Yu, Y. Zhang, Z. Yang, K. Zhang, X. Zhu, Y. Zhang, T. Chen, G. Cui, et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025.   
[18] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.   
[19] X. Peng, Z. Zheng, C. Shen, T. Young, X. Guo, B. Wang, H. Xu, H. Liu, M. Jiang, W. Li, et al. Open-sora 2.0: Training a commercial-level video generation model in $2 0 0 k$ . arXiv preprint arXiv:2503.09642, 2025.   
[20] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
[21] J. Xiao, Y. Yang, X. Chang, R. Chen, F. Xiong, M. Xu, W.-S. Zheng, and Q. Zhang. Worldenv: Leveraging world model as a virtual environment for vla post-training. arXiv preprint arXiv:2509.24948, 2025.   
[22] F. Zhu, Z. Yan, Z. Hong, Q. Shou, X. Ma, and S. Guo. Wmpo: World model-based policy optimization for vision-language-action models. arXiv preprint arXiv:2511.09515, 2025.   
[23] H. Li, P. Ding, R. Suo, Y. Wang, Z. Ge, D. Zang, K. Yu, M. Sun, H. Zhang, D. Wang, and W. Su. Vla-rft: Vision-language-action reinforcement fine-tuning with verified rewards in world simulators. arXiv preprint arXiv:2510.00406, 2025.   
[24] C. Yu, Y. Wang, Z. Guo, H. Lin, S. Xu, H. Zang, Q. Zhang, Y. Wu, C. Zhu, J. Hu, et al. Rlinf: Flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation. arXiv preprint arXiv:2509.15965, 2025.   
[25] X. Yuan, T. Mu, S. Tao, Y. Fang, M. Zhang, and H. Su. Policy decorator: Model-agnostic online refinement for large policy model, 2024.   
[26] M. Pan, S. Feng, Q. Zhang, X. Li, J. Song, C. Qu, Y. Wang, C. Li, Z. Xiong, Z. Chen, Y. Liu, and J. Luo. Sop: A scalable online post-training system for vision-language-action models. arXiv preprint arXiv:2601.03044, 2026.   
[27] H. Zang, S. Yu, H. Lin, T. Zhou, Z. Huang, Z. Guo, X. Xu, J. Zhou, Y. Sheng, S. Zhang, F. Gao, W. Tang, Y. Yue, Q. Zhang, X. Chen, C. Yu, and Y. Wang. Rlinf-user: A unified and extensible system for real-world online policy learning in embodied ai, 2026.   
[28] X. He, C. Peng, Z. Liu, B. Wang, Y. Zhang, Q. Cui, F. Kang, B. Jiang, M. An, Y. Ren, B. Xu, H.-X. Guo, K. Gong, S. Wu, W. Li, X. Song, Y. Liu, Y. Li, and Y. Zhou. Matrixgame 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.   
[29] Y. Zhang, C. Peng, B. Wang, P. Wang, Q. Zhu, F. Kang, B. Jiang, Z. Gao, E. Li, Y. Liu, and Y. Zhou. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.   
[30] J. Li, J. Tang, Z. Xu, L. Wu, Y. Zhou, S. Shao, T. Yu, Z. Cao, and Q. Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.   
[31] J. Tang, J. Liu, J. Li, L. Wu, H. Yang, P. Zhao, S. Gong, X. Yuan, S. Shao, and Q. Lu. Hunyuan-gamecraft-2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025.   
[32] X. Mao, S. Lin, Z. Li, C. Li, W. Peng, T. He, J. Pang, M. Chi, Y. Qiao, and K. Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025.   
[33] X. Mao, Z. Li, C. Li, X. Xu, K. Ying, T. He, J. Pang, Y. Qiao, and K. Zhang. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.   
[34] R. Team, Z. Gao, Q. Wang, Y. Zeng, J. Zhu, K. L. Cheng, Y. Li, H. Wang, Y. Xu, S. Ma, Y. Chen, J. Liu, Y. Cheng, Y. Yao, J. Zhu, Y. Meng, K. Zheng, Q. Bai, J. Chen, Z. Shen, Y. Yu, X. Zhu, Y. Shen, and H. Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.   
[35] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.   
[36] Y. Lu, Y. Zeng, H. Li, H. Ouyang, Q. Wang, K. L. Cheng, J. Zhu, H. Cao, Z. Zhang, X. Zhu, Y. Shen, and M. Zhang. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025.   
[37] H. Zhu, M. Zhao, G. He, H. Su, C. Li, and J. Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.   
[38] T. Yin, Q. Zhang, R. Zhang, W. T. Freeman, F. Durand, E. Shechtman, and X. Huang. From slow bidirectional to fast autoregressive video diffusion models. arXiv preprint arXiv:2412.07772, 2025.   
[39] T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and W. T. Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024.   
[40] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park. One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828, 2024.   
[41] Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, J. Cai, S. Liu, J. Luo, L. Chen, S. Yan, M. Yao, and G. Ren. Genie envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025.   
[42] Y. Jiang, S. Chen, S. Huang, L. Chen, P. Zhou, Y. Liao, X. He, C. Liu, H. Li, M. Yao, and G. Ren. Enerverse-ac: Envisioning embodied environments with action condition. arXiv preprint arXiv:2505.09723, 2025.   
[43] J. Quevedo, A. K. Sharma, Y. Sun, V. Suryavanshi, P. Liang, and S. Yang. Worldgym: World model as an environment for policy evaluation. arXiv preprint arXiv:2506.00613, 2025.   
[44] A. Bagchi, Z. Bao, H. Bharadhwaj, Y.-X. Wang, P. Tokmakov, and M. Hebert. Walk through paintings: Egocentric world models from internet priors. arXiv preprint arXiv:2601.15284, 2026.   
[45] F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong. Irasim: A fine-grained world model for robot manipulation. arXiv preprint arXiv:2406.14540, 2025.   
[46] Y. Guo, L. X. Shi, J. Chen, and C. Finn. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125, 2025.   
[47] Y. Li, Y. Zhu, J. Wen, C. Shen, and Y. Xu. Worldeval: World model as real-world robot policies evaluator. arXiv preprint arXiv:2505.19017, 2025.   
[48] Y. Zhu, J. Feng, W. Zheng, Y. Gao, X. Tao, P. Wan, J. Zhou, and J. Lu. Astra: General interactive world model with autoregressive denoising. arXiv preprint arXiv:2512.08931, 2026.   
[49] Z. Jiang, K. Liu, Y. Qin, S. Tian, Y. Zheng, M. Zhou, C. Yu, H. Li, and D. Zhao. World4rl: Diffusion world models for policy refinement with reinforcement learning for robotic manipulation. arXiv preprint arXiv:2509.19080, 2025.   
[50] J. Shin, Z. Li, R. Zhang, J.-Y. Zhu, J. Park, E. Shechtman, and X. Huang. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.   
[51] S. Yang, W. Huang, R. Chu, Y. Xiao, Y. Zhao, X. Wang, M. Li, E. Xie, Y. Chen, Y. Lu, S. Han, and Y. Chen. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.   
[52] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, ¨ F. Boesel, D. Podell, T. Dockhorn, Z. English, K. Lacey, A. Goodwin, Y. Marek, and R. Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.   
[53] B. Chen, D. M. Monso, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. arXiv preprint arXiv:2407.01392, 2024.   
[54] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.   
[55] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
[56] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.   
[57] D. Danier, F. Zhang, and D. Bull. Flolpips: A bespoke video quality metric for frame interpoation. arXiv preprint arXiv:2207.08119, 2022.   
[58] J. Pennington, P. Joshi, and A. Bhide. Develop custom physical ai foundation models with nvidia cosmos predict-2. https://developer.nvidia.com/blog/ develop-custom-physical-ai-foundation-models-with-nvidia-cosmos-predict-2/, June 11 2025. NVIDIA Developer Blog.   
[59] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

# A GPU Allocation Strategy

The reinforcement-learning pipeline can be decomposed into three components: Generation, Simulator, and Training. In WoVR, the Simulator is instantiated by the learned world model, which generates the next observation given the current observation and action. In the rollout phase, Generation performs policy inference to produce an (optionally chunked) action from the current observation, while the Simulator executes the action and returns the next observation; this closed-loop interaction repeats until a batch of trajectories is collected. In the optimization phase, Training updates the VLA policy using the collected trajectories, after which the system alternates back to rollout for the next iteration.

Following the system abstraction in RLinf-VLA, WoVR adopts a collocated (shared) GPU allocation strategy, where the three RL pipeline components co-exist on the same set of GPUs, with the Simulator implemented as the world-model rollout module. Unlike physical simulators that require dedicated device-side state, WoVR’s simulator is a neural network; thus, offload/onload can be naturally realized by swapping only the model parameters between GPU and host memory, without migrating any external simulator state. In its original form, collocated execution relied on frequent $\mathrm { G P U } \{  \mathrm { C P U }$ offload/onload to keep only one component resident on GPUs at a time; however, in embodied settings the simulator and generator must interact iteratively, making per-interaction offload/onload prohibitively expensive. Therefore, we use the modified collocated strategy: offload/onload for Generation and Simulator happens only at the beginning and end of the rollout phase, avoiding repeated transfers during closed-loop imagined interaction, as illustrated in Fig. 8.

![](images/68e75b960e9d1fdc4f0d148645fdbbba65952f68e413d32c24a9581a2e9c7389.jpg)  
Figure 8: Collocated GPU Allocation Strategy