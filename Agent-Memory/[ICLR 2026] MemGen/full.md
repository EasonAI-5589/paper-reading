# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents

Guibin Zhang $^ \dagger$ , Muxin $\bar { \mathbf { F } } \mathbf { u } ^ { \dagger }$ , Shuicheng Yan

National University of Singapore †Equal Contribution

Agent memory shapes how Large Language Model (LLM)-powered agents, akin to the human brain, progressively refine themselves through environment interactions. Existing paradigms remain constrained: parametric memory forcibly adjusts model parameters, and retrieval-based memory externalizes experience into structured databases, yet neither captures the fluid interweaving of reasoning and memory that underlies human cognition. To address this gap, we propose MemGen, a dynamic generative memory framework that equips agents with a human-esque cognitive faculty. It consists of a memory trigger, which monitors the agent’s reasoning state to decide explicit memory invocation, and a memory weaver, which takes the agent’s current state as stimulus to construct a latent token sequence as machine-native memory to enrich its reasoning. In this way, MemGen enables agents to recall and augment latent memory throughout reasoning, producing a tightly interwoven cycle of memory and cognition. Extensive experiments across eight benchmarks show that MemGen surpasses leading external memory systems such as ExpeL and AWM by up to $3 8 . 2 2 \%$ , exceeds GRPO by up to $1 3 . 4 4 \%$ , and exhibits strong cross-domain generalization ability. More importantly, we find that without explicit supervision, MemGen spontaneously evolves distinct human-like memory faculties, including planning memory, procedural memory, and working memory, suggesting an emergent trajectory toward more naturalistic forms of machine cognition.

z Date: October 14, 2025 § Github: https://github.com/KANABOON1/MemGen

# 1 Introduction

The ascent of Large Language Model (LLM)-powered agents marks a paradigm shift across diverse domains (Luo et al., 2025b; Yang et al., 2024b; Qian et al., 2025; Singh et al., 2025; Pantiukhin et al., 2025; Ren et al., 2025). Pivotal to this success is the concept of agent memory (Zhang et al., 2024b; Wu et al., 2025b), which enables LLM agents to learn progressively from environmental interactions (Zhang et al., 2025a; Qiu et al., 2025b). Crucially, this conception of agent memory extends beyond that of conversational agents (i.e., personalized memory (Wu et al., 2025b)), whose primary role is to sustain coherence across long-horizon, multi-turn dialogues (Chhikara et al., 2025; Xu et al., 2025a; Packer et al., 2024; Zhong et al., 2023). Rather, the scope of this paper is primarily on enabling agents to internalize experience, simulate human-like cognitive iteration, and progressively enhance problem-solving competence (Gao et al., 2025).

The memory serving as this self-evolving engine typically manifests in two dominant paradigms. The first is (I) parametric memory, which internalizes experiences by directly updating agents’ parameters (Yao et al., 2024; Zeng et al., 2023; Chen et al., 2024b, 2025). While this approach can yield substantial performance gains, its reliance on parameter modification inevitably entails catastrophic forgetting, i.e., the erosion of general knowledge (Dou et al., 2024). Conversely, the second paradigm is (II) retrieval-based memory, which externalizes past experiences into a structured database, such as (i) raw trajectories (Luo et al., 2025a; Zhang et al., 2025a; Zhao et al., 2024), (ii) high-level experiences (Zhao et al., 2024; Fang et al., 2025; Wang et al., 2024c), and (iii) condensed skills like reusable APIs (Zheng et al., 2025) or MCP boxes (Qiu et al., 2025b,a). Although this non-invasive approach circumvents catastrophic forgetting, its efficacy is fundamentally tethered to context engineering. It adheres to a rigid execution pipeline, providing retrieved context to the agent without achieving the fluid, seamless integration characteristic of truly internalized memory (Su et al., 2025b).

![](images/542a39c60163cb9d76c5ee6989c29a616e07df80b7d799151f7299ad4cb7697d.jpg)  
Figure 1 The comparison among parametric memory, retrieval-based memory and MemGen. We drew inspiration from the layout presented in Figure 1 of Li et al. (2025a).

Given these deficiencies, latent memory offers a compelling alternative, leveraging latent states as a machinenative, high-density medium for memory. Existing approaches either use the (i) key-value (KV) cache to maintain dynamic memory set (Gim et al., 2024; Jin et al., 2024; Hongkang Yang et al., 2024), yet which is primarily confined to addressing long-context issues, or (ii) latent token embeddings to store agent experiences (Wang et al., 2024b, 2025a), which still rely on invasive LLM parameter updates. LatentSeek (Li et al., 2025a) and SoftCoT (Xu et al., 2025b,c) similarly belong to this category, utilizing latent embeddings to steer agent generation. Nevertheless, all these methods diverge from human cognition in two critical dimensions: they lack the seamless interleaving of reasoning and memory, a process where thought and memory dynamically reshape one another, and remain largely retrieval-based, fetching memories by embedding similarity (Wang et al., 2024b) rather than generatively reconstructing them into novel, coherent insights. This leads to our pivotal research question:

How can we architect agent memory as a dynamic cognitive faculty, capable of fluid, reconstructive processes that interweave seamlessly with reasoning?

To address this challenge, we introduce MemGen, a dynamic and generative memory framework designed to endow any LLM agent with a more human-esque cognitive faculty. At its core, MemGen continuously monitors an agent’s cognitive state, enabling it to dynamically invoke a generative process that synthesizes a bespoke latent memory at any critical juncture during its reasoning process. Practically, MemGen comprises two synergistic components: a reinforcement learning (RL)-trained $\clubsuit$ memory trigger, which acts as a metacognitive monitor to discern the opportune moments for explicit memory invocation; and a $\spadesuit$ memory weaver, which takes the agent’s current state as a stimulus to draw upon relevant implicit parametric memory (potentially augmented with externally retrieved information) and then reconstructs this synthesis into a succinct, machine-native latent memory. With the reasoning core fixed, MemGen inherently mitigates catastrophic forgetting when exposed to new data, and, moving beyond the static and extractive paradigm of prior memory systems, equips agents with a fluid, generative faculty deeply integrated with reasoning cores.

Experimental Observation. Extensive experiments across nine benchmarks and four baseline categories demonstrate that MemGen delivers $\bullet$ substantial performance gains, with improvements of up to $3 1 . 7 \%$ on ALFWorld (Shridhar et al., 2021) and $2 7 . 1 \%$ on KodCode (Xu et al., 2025d) with $_ { \mathsf { Q w e n 3 - 8 B } }$ , surpassing parametric memory (REINFORCE $^ { + + }$ , $+ 5 . 8 \%$ ) and the GRPO method $\left( + 5 . 3 2 \% \right)$ ; $\pmb { \varrho }$ strong cross-domain generalization, where training in the math domain not only avoids degradation elsewhere but also boosts performance in science reasoning $( + 6 . 0 6 \% )$ and code generation $( + 5 . 1 \% )$ ; and $\otimes$ continual learning ability, maintaining stable performance in previously trained domains even after fine-tuning on three additional ones.

Analysis & Interpretation. Beyond quantitative evaluation, we sought to interpret the learned behavior of MemGen. Through post-hoc interventions examining the impact of removing specific latent memory on different agent failure modes, we found that MemGen implicitly evolves a human-like memory hierarchy without any external guidance, including $\bullet$ planning memory, where certain latent tokens specifically support high-level task planning, $\pmb { \varrho }$ procedural memory, where some latent memory tokens facilitate the agent’s recall of task-specific procedural skills, such as tool usage and answer formatting, and $\otimes$ working memory, where certain tokens help the agent maintain coherence and understanding over long contexts within a single task session. These specializations strongly reveal that MemGen endows the agent with precise, functionally distinct memory.

# 2 Related Work

LLM & Agent Memory. As outlined in Section 1, existing memory mechanisms designed to evolve the problem-solving capacity of LLM agents can be broadly categorized into three classes: (I) parametric memory, which either integrates past experiences directly into agent parameters through finetuning, as in FireAct (Chen et al., 2023), AgentLumos (Yin et al., 2024), and others (Zhang et al., 2024a; Fu et al., 2025), or maintains them in external parameter modules (Tack et al., 2024; Wang et al., 2024a); (II) retrieval-based memory, which abstracts prior experiences into transferable knowledge (Zhang et al., 2025a; Zhao et al., 2024), or distills them into reusable tools and skills (Zheng et al., 2025; Wang et al., 2025b; Qiu et al., 2025b,a); and (III) latent memory, which leverages implicit representations to encode and retrieve experience (Wang et al., 2024b, 2025a; Hu et al., 2025b; Liu et al., 2024; Sun et al., 2025). Our MemGen falls within the latent memory paradigm, yet distinguishes itself from prior approaches through its more human-esque interweaving of reasoning and memory, as well as its generative, rather than purely retrieval-based, nature.

Latent Computation. Our method is also closely related to latent computation, wherein latent states are employed to intervene in or reshape the LLM’s reasoning process (Zhu et al., 2025). Prominent paradigms include: (I) architecturally enabling native latent reasoning, exemplified by Coconut (Hao et al., 2024), CODI (Shen et al., 2025), LatentR3 (Zhang et al., 2025b) and CoLaR (Tan et al., 2025), which render the LLM’s inference process inherently latent and machine-native; and (II) employing latent computation to steer LLM generation, as in LaRS (Xu et al., 2023), LatentSeek (Li et al., 2025a), SoftCoT (Xu et al., 2025c,b), and Coprocessor (Liu et al., 2024), which leverage latent representations to modulate the quality of generated outputs. These aforementioned works have greatly inspired the latent memory design in this paper: Latent memory can likewise be viewed as an instantiation of the latter, supplementing essential memory context to enhance the problem-solving capacity of agents (Wang et al., 2024b, 2025a).

LLM Decoding $\&$ RL. Two additional topics that relate to our work are LLM decoding and reinforcement learning (RL). From the decoding perspective, MemGen dynamically generates and inserts latent tokens, which shares similarity with speculative decoding where a drafter model receives the current decoding context and produces subsequent drafted tokens (Cai et al., 2024; Fu et al., 2024; Li et al., 2025b; Goel et al., 2025). . However, these methods primarily aim to accelerate LLM inference, whereas MemGen focuses on leveraging latent states as effective carriers of memory. From the RL perspective, MemGen employs rule-based RL to train the memory trigger, which is closely related to reinforcement learning with variable reward (RLVR), including GRPO from DeepSeek-R1 (DeepSeek-AI et al., 2025) and its various derivatives (Qian et al., 2025; Wu et al., 2025a; Wei et al., 2025; Fu et al., 2025). While there exist efforts combining RL with agent memory, to our knowledge, most do not address self-improving memory; for example, MemAgent (Yu et al., 2025) and MEM1 (Zhou et al., 2025) focus on handling long-context inputs rather than evolving memory mechanisms.

# 3 Preliminary

Notation. We formalize the agent’s interaction within an environment $\varepsilon$ . An agent, powered by an LLM parameterized by $\theta$ , is denoted as $\pi \theta$ . For a given task $\mathbf { x }$ , the agent’s interaction unfolds as a high-level trajectory, denoted as follows $\tau = ( s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , \dots , s _ { T } )$ , where $s _ { t }$ represents the state of the environment and $a _ { t }$ is the high-level action taken by the agent. More internally, each action $a _ { t }$ is essentially a sequence of tokens, $a _ { t } = ( \mathbf { z } _ { t , 1 } , \mathbf { z } _ { t , 2 } , \ldots , \mathbf { z } _ { t , L _ { t } } )$ , generated autoregressively by the LLM. The generation of the $j$ -th token is conditioned on the current state $s _ { t }$ and all previously generated tokens within that action:

$$
\mathbf { z } _ { t , j } \sim \pi _ { \boldsymbol { \theta } } \big ( \cdot \mid s _ { t } , \mathbf { z } _ { t , < j } \big ) .
$$

![](images/3c8220ce763ce1b3b463c3e2b717cabe2749d92772d197497242f92c7b24ca29.jpg)  
Figure 2 The overview of our proposed MemGen.

After an entire action sequence $\mathbf { a } _ { t }$ is generated, it is executed in the environment, which transitions the state from $s _ { t }$ to $s _ { t + 1 }$ . The success of the trajectory $\tau$ is evaluated by a reward function $R ( \tau )$ .

Problem Formalization Given a history of past experiences $\mathcal { H } = \{ ( x _ { i } , \tau _ { i } ) \} _ { i = 1 } ^ { N }$ , the objective is to leverage this history to maximize the agent’s performance on new tasks. The policy $\pi \theta$ and a memory system $\mathcal { M }$ are thus jointly optimized to maximize the expected reward over a task distribution $\mathcal { D }$ :

$$
\operatorname* { m a x } _ { \theta , \mathcal { M } } \mathbb { E } _ { { x } \sim \mathcal { D } , \tau \sim \pi _ { \theta , \mathcal { M } } } \left[ R ( \tau ) \right] ,
$$

during which $\mathcal { M }$ is to produce a memory representation, $m$ , which conditions the agent’s policy. The action at any timestep $t$ is thus sampled as $a _ { t } \sim \pi _ { \theta } ( \cdot \mid s _ { t } , m _ { t } )$ , where $m _ { t }$ is the inserted memory at that step. Crucially, the nature and timing of memory generation, which we denote as the function $f _ { \mathcal { M } }$ , vary across different paradigms. We express the generation of the memory $m _ { t }$ as:

$$
\begin{array} { r } { m _ { t } = f _ { \mathcal { M } } ( s _ { t } , \mathcal { H } , m _ { < t } ) , } \end{array}
$$

which accommodates diverse memory invocation granularities. For task-level memory (e.g., Expel (Zhao et al., 2024) and G-Memory (Zhang et al., 2025a)), $f _ { \mathcal { M } }$ is invoked only at $t = 0$ , and $m _ { t } = m _ { 0 }$ for all subsequent steps. For step-level memory (e.g., AgentKB (Tang et al., 2025)), $f _ { \mathcal { M } }$ is invoked at every step $t$ to update the memory. In parametric memory, the influence of $\mathcal { H }$ is compiled into $\theta$ , rendering memory generation implicit in the model parameters. Our work, which introduces dynamic latent memory, focuses on designing a more fine-grained $f _ { \mathcal { M } }$ that decides for itself the optimal moments to regenerate $m _ { t }$ at the token level during the agent’s reasoning process.

# 4 Methodology

# 4.1 MemGen: Interleaving Memory and Reasoning

Just as a person is the sum of their past experiences (McAdams, 2001), memory critically shapes an agent’s actions (Xiong et al., 2025). Existing agent memory systems, however, often lack the flexibility of human cognition. In the human brain, reasoning and memory form a seamless continuum: active reasoning in the frontoparietal control network and memory retrieval in the hippocampus and prefrontal cortices operate interweavingly, generating a “continuous flow of thoughts” (Su et al., 2025a). By contrast, many agent memory paradigms retrieve information once at task onset and append it coarsely to the query. MemGen is designed precisely to bridge this gap.

As shown in Figure 2, the reasoning process in an agent equipped with MemGen unfolds autoregressively, driven by a frozen core LLM, the reasoner $\pi \theta$ . For a given state $s _ { t }$ , $\pi \theta$ generates the action $\boldsymbol a _ { t } = ( \mathbf z _ { t , 1 } , \dots , \mathbf z _ { t , L _ { t } } )$ MemGen continuously monitors the token-by-token generation process and performs on-demand memory insertion. At each token-generation step $j$ , a memory trigger $\tau _ { \mathrm { t r i g g e r } }$ monitors the reasoner’s internal cognitive state to determine if a moment of reflection, i.e., a memory invocation, is necessary. Specifically, as the reasoner generates the token sequence $\mathbf { z } _ { t , < j }$ , it produces a corresponding sequence of hidden state vectors, $\mathbf { H } _ { t , < j } = ( \mathbf { h } _ { t , 1 } , \dots , \mathbf { h } _ { t , j - 1 } )$ , where each $\mathbf { h } _ { t , k } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ . The trigger takes the current hidden states $\mathbf { H } _ { t , < j }$ as a representation of the reasoner’s current metacognitive state and computes an invocation probability:

$$
p _ { j } = \sigma \left( \mathcal T _ { \mathrm { t r i g g e r } } ( \mathbf h _ { t , 1 } , \dots , \mathbf h _ { t , j - 1 } ) \right) ,
$$

from which a binary decision, $d _ { j } \sim \mathrm { B e r n o u l l i } ( p _ { j } ) \in \{ \mathrm { I N V O K E } , \mathrm { S K I P } \}$ , is sampled. If the decision is to [SKIP], $\pi \theta$ proceeds with its standard autoregressive generation, i.e., $\mathbf { z } _ { t , j } \sim \pi _ { \boldsymbol \theta } \big ( \cdot \mid s _ { t } , \mathbf { z } _ { t , < j } \big )$ . However, if the decision is to INVOKE, the reasoning process is momentarily paused. This summons the second core component of our framework: the memory weaver $\mathcal { W } _ { \mathrm { w e a v e r } }$ , which takes the same cognitive state $\mathbf { H } _ { t , < j }$ as a stimulus to perform a generative act of recollection. It synthesizes a bespoke, machine-native latent memory, formalized as $\mathbf { M } _ { t } \in \mathbb { R } ^ { K \times d _ { \mathrm { m o d e l } } }$ with fixed length $K$ :

$$
\mathbf { M } _ { t } : = [ \mathbf { m } _ { t , 1 } , \mathbf { m } _ { t , 2 } , \cdots , \mathbf { m } _ { t , K } ] = { \mathcal { W } } _ { \mathrm { w e a v e r } } ( \mathbf { H } _ { t , < j } ) ,
$$

where the memory is generated not merely from the parametric knowledge encoded within $\mathcal { W } _ { \mathrm { w e a v e r } }$ but may also incorporate cues retrieved from external memory databases (detailed implementation is elaborated in Section 4.3). Crucially, $\mathbf { M } _ { t }$ is not a verbatim restatement of prior content but a selective reconstruction, filtered and integrated through $\mathcal { W } _ { \mathrm { w e a v e r } }$ , akin to the hippocampus consolidating fragments of recollection into human memory (Spens and Burgess, 2024). Once formed, the latent memory is woven seamlessly into the reasoner’s ongoing dynamics: its hidden states are prepended to $\mathbf { H } _ { t , < j }$ , upon which the reasoner resumes generation conditioned on this enriched context,

$$
\mathbf { z } _ { t , j } \sim \pi _ { \boldsymbol \theta } \big ( \cdot \mid s _ { t } , \mathbf { z } _ { t , < j } , \mathbf { M } _ { t } \big ) .
$$

This iterative cycle of generation, monitoring, invocation, weaving, and reintegration elevates reasoning from a linear unfolding to a recursive dialogue with memory, all without altering the frozen reasoner $\pi \theta$ , and thereby preserving its general capabilities. In the following sections, we detail the implementations of the memory trigger (▷ Section 4.2) and the memory weaver (▷ Section 4.3).

# 4.2 Memory Trigger: Learning to Invocate Memory

In this section, we describe the concrete instantiation and training recipe of the memory trigger $\tau _ { \mathrm { t r i g g e r } }$ Recall from Section 4.1 that $\tau _ { \mathrm { t r i g e r } }$ serves as a metacognitive monitor, observing the evolving reasoning state of the frozen reasoner $\pi \theta$ and producing a binary decision $d _ { j } \in \{ \mathrm { I N V O K E , S K I P } \} ( 0  \mathbf { S K I P } , 1  \mathrm { I N V O K E } )$ ) that determines whether the memory weaver should be invoked at token $j$ .

Instantiation. We instantiate $\tau _ { \mathrm { t r i g g e r } }$ as a lightweight LoRA adapter attached to the reasoner $\pi _ { \theta }$ . At the decoding step $j$ of the timestep $t$ , it receives the sequence of all hidden states, $\mathbf { H } _ { t , < j } \in \mathbb { R } ^ { ( j - 1 ) \times d _ { \mathrm { m o d e l } } }$ conditioned on this context, $\tau _ { \mathrm { t r i g g e r } }$ outputs the action probability $P ( d _ { j } = \mathrm { I N V O K E } ) \in [ 0 , 1 ]$ . To avoid excessive computational overhead, we adopt a sentence-granularity activation strategy, inspired by recent studies on LLM interpretability (Anthropic, 2025; Chen et al., 2024a), which find that interventions between sentences can more effectively guide LLMs’ reasoning path. Specifically, we define a delimiter token set $\mathcal { D }$ (e.g., commas, periods) and let the trigger act only when the current token falls in $\mathcal { D }$ . The invocation decision is computed as:

$$
\begin{array} { r } { d _ { j } = \mathrm { B e r n o u l l i } ( p _ { j } ) , p _ { j } = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } z _ { j } \notin \mathcal { D } , } \\ { \mathcal { T } _ { \mathrm { t r i g g e r } } ( \mathbf { H } _ { t , < j } ) } & { \mathrm { i f ~ } z _ { j } \in \mathcal { D } , } \end{array} \right. } \end{array}
$$

which ensures that $\tau _ { \mathrm { t r i g g e r } }$ is invoked only at semantically significant boundaries, preserving decoding efficiency.   
We validate that MemGen does not incur excessive inference delay in Section D.3.3.

Training Recipe. The memory trigger is trained via reinforcement learning, motivated by the need to balance two competing desiderata: ensuring that critical latent memories are invoked to improve task performance, while avoiding unnecessary or spurious invocations that could disrupt reasoning or incur computational overhead. Given a batch of seen tasks $\mathcal { H } = \{ ( x _ { i } , \tau _ { i } ) \} _ { i = 1 } ^ { N }$ , the frozen reasoner $\pi _ { \theta }$ generates candidate trajectories while the memory weaver $\mathcal { W } _ { \mathrm { w e a v e r } }$ remains fixed. At each activated step, the trigger selects an action $\tilde { d } _ { j } \in \{ \mathrm { I N V O K E } , \mathrm { S K I P } \}$ and receives a reward $r ( \tau _ { i } )$ reflecting the quality of the resulting trajectory with respect to the task objective. To encourage sparse yet strategically critical memory invocation, we introduce a reward-adaptive penalty, which discourages unnecessary activations while preserving essential ones, into the objective:

$$
\operatorname* { m a x } _ { \phi } \mathbb { E } _ { \tau _ { i } \sim \pi _ { \theta } , \tilde { \mathbf { d } } \sim \mathcal { T } _ { \mathrm { t r i g g e r } } ^ { \phi } } \Big [ R ( \tau _ { i } ) - \lambda \sum _ { i , j } \operatorname* { m a x } ( 0 , \tilde { d } _ { i , j } - \bar { p } ) \Big ] ,
$$

where $p$ is computed as the mean activation probability across high-reward trajectories, i.e., those with reward exceeding the batch median:

$$
\bar { p } = \frac { 1 } { \left| \mathcal { H } _ { \mathrm { h i g h } } \right| } \sum _ { i \in \mathcal { H } _ { \mathrm { h i g h } } } \frac { 1 } { \left| \tau _ { i } \right| } \sum _ { j } \tilde { d } _ { i , j } , \ \mathcal { H } _ { \mathrm { h i g h } } = \{ i : R ( \tau _ { i } ) \ge \mathrm { m e d i a n } _ { k } ( R ( \tau _ { k } ) ) \} ,
$$

where ensures that $\tau _ { \mathrm { t r i g g e r } }$ learns to invoke memory selectively at key decision points, maximizing task reward while maintaining computational efficiency.

# 4.3 Memory Weaver: Synthesizing and Inserting Latent Memory

In this section, we elaborate on the weaver $\mathcal { W } _ { \mathrm { w e a v e r } }$ , the memory carrier within the MemGen framework. When the agent assimilates new experiences, this information is exclusively internalized into the parameters of $\mathcal { W } _ { \mathrm { w e a v e r } }$ , leaving the core reasoner $\pi _ { \theta }$ entirely unmodified. At junctures where the reasoner requires experiential support, a context-dependent hook activates the weaver to synthesize and externalize pertinent knowledge as a usable memory. To be more specific, recall from Equation (5) that after the $\tau _ { \mathrm { t r i g g e r } }$ signals the need for memory at step $j$ , $\mathcal { W } _ { \mathrm { w e a v e r } }$ accepts $\mathbf { H } _ { t , < j }$ (as the hook) and generates a latent token sequence $\mathbf { M } _ { t }$ (as the memory) for $\pi \theta$ .

Instantiation. We instantiate $\mathcal { W } _ { \mathrm { w e a v e r } }$ using anthoer LoRA adapter attached to $\pi \theta$ . Formally, given the incoming hook $\mathbf { H } _ { t , < j } \in \mathbb { R } ^ { ( j - 1 ) \times d _ { \mathrm { m o d e l } } }$ , the weaver outputs a latent memory matrix: $\mathbf { M } _ { t } = \mathcal { W } _ { \mathrm { w e a v e r } } ^ { \theta ^ { \prime } } ( \mathbf { H } _ { t , < j } ) \in$ $\mathbb { R } ^ { K \times d _ { \mathrm { m o d e l } } }$ , where $K$ denotes the fixed length of the latent memory sequence and $\theta ^ { \prime }$ are the trainable LoRA parameters. The synthesized $\mathbf { M } _ { t }$ is then prepended to the current hidden states of $\pi \theta$ to guide subsequent token generation, as described in Equation (6).

Training Recipe. The training of $\mathcal { W } _ { \mathrm { w e a v e r } }$ proceeds over a batch of past trajectories $\mathcal { H } = \{ ( x _ { i } , \tau _ { i } ) \} _ { i = 1 } ^ { N }$ . Distinct from conventional agent tuning, which directly integrates experiential data into the parameters of $\pi \theta$ (Chen et al., 2025; Yin et al., 2024), MemGen internalizes experiential knowledge solely into $\mathcal { W } _ { \mathrm { w e a v e r } }$ which ensures that $\pi \theta$ ’s general capabilities remain intact.

Crucially, this separation makes MemGen agnostic to optimization strategies and compatible with diverse LLM backbones. Whether employing supervised fine-tuning (SFT) or RL-based objectives such as GRPO or DAPO, the weaver can be updated under a unified goal: optimizing the generation process of latent memory so as to maximize downstream reward. Formally, let $\Pi _ { \theta } ^ { \mathcal { W } _ { \theta ^ { \prime } } , \mathcal { I } } \left( \cdot \mid x \right)$ denote the process of rolling out a trajectory for a task $x$ by $\pi \theta$ in conjunction with weaver $\mathcal { W } _ { \theta ^ { \prime } }$ and trigger $\tau$ . Given a reward functional $R$ , the objective updates only $\theta ^ { \prime }$ by maximizing the expected reward:

$$
\operatorname* { m a x } _ { \theta _ { \mathrm { l o r a } } } \ { \mathbb { E } } _ { ( x _ { i } , \tau _ { i } ) \sim \mathcal { H } } \ { \mathbb { E } } _ { \tau \sim \Pi _ { \theta } ^ { \mathcal { W } _ { \theta ^ { \prime } } , \tau } ( \cdot | x _ { i } ) } \big [ R ( x _ { i } , \tau ) \big ] ,
$$

where the gradients from $R$ are propagated solely to $\theta ^ { \prime }$ , thereby equipping the weaver to supply precisely the memories that improve end-to-end performance without altering $\pi \theta$ . Equation (10) enables $\mathcal { W } _ { \mathrm { w e a v e r } }$ to absorb diverse experiential signals and externalize them as dynamic, context-sensitive latent memories, independent of the architectural or training paradigm of the base reasoner.

Integration with Retrieval-based Memory. Although the memory generation above primarily draws on the weaver’s parametric knowledge, it can be combined with external memory sources. When triggered, any retrieval-based system (e.g., MemoryBank, ExpeL) can provide textual memory, which is merged with the hook $\mathbf { H } _ { t , < j }$ and fed into $\mathcal { W }$ to produce latent memory. This allows $\mathcal { W }$ to integrate internal knowledge and external information, supplying the reasoner with richer memory support. Implementation details and results are placed in Appendix E.

# 5 Experiments

In this section, we conduct extensive experiments to answer the following research questions:

• (RQ1) Can MemGen surpass both parametric and retrieval-based memory? • (RQ2) Is the memory learnt by MemGen generalizable across task domains? And why? • (RQ3) Can MemGen facilitate continual learning and mitigate catastrophic forgetting? • (RQ4) Does MemGen implicitly evolve human-like memory hierachy?

# 5.1 Experimental Setup

Evaluation and Benchmarks. Our evaluation covers nine datasets from five domains, including $\bullet$ web search: TriviaQA (Joshi et al., 2017) and PopQA (Mallen et al., 2023); ❷ embodied action: ALFWorld (Shridhar et al., 2021); $\pmb { \Theta }$ math reasoning: AQuA (Ling et al., 2017), GSM8K (Cobbe et al., 2021), and MATH (Hendrycks et al., 2021); ❹ scientific reasoning: GPQA (Rein et al., 2023); and $\pmb { \ 6 }$ coding: KodCode (Xu et al., 2025d) and BigCodeBench (Jain et al., 2024).

Baselines. We compare MemGen against twelve baselines, categorized into four groups: (I) Prompt-based methods: Vanilla model, CoT (Wei et al., 2023); (II) Parametric memory, where experiential knowledge directly modifies model parameters via: SFT, GRPO (DeepSeek-AI et al., 2025), REINFORCE (Williams, 1992), REINFORCE $^ { + + }$ (Hu et al., 2025a), Agent-FLAN (Chen et al., 2024b); (III) Retrieval-based memory, where processing tasks sequentially and storing the experiences in an external database, represented by MemoryBank (Zhong et al., 2023), ExpeL (Zhao et al., 2024), Agent Workflow Memory (AWM) (Wang et al., 2024c); and (IV) Latent computation, where leveraging latent tokens as carriers of experiential knowledge, including SoftCoT (Xu et al., 2025c) and Co-processor (Liu et al., 2024).

Implementation Details. We select LLM backbones of varying sizes, including Qwen-2.5-1.5B (Yang et al., 2024a), HuggingFace’s SmolLM3-3B (HuggingFace, 2025), and Qwen3-8B (Yang et al., 2025). The length of each latent memory sequence $K$ is set among $\{ 2 , 4 , 8 \}$ . MemGen does not rely on a specific optimization algorithm, so we implement two variants: MemGen SFT and MemGen GRPO, in which the weaver is updated using SFT and GRPO signals. Details on these variants are provided in Appendix B. More training setup and parameter configurations are listed in Appendix C.

# 5.2 Main Restuls

[For RQ1] MemGen provides high-performing memory across domains. As shown in Tables 1 and 3, existing baselines exhibit clear limitations in cross-domain adaptivity. Retrieval-based memories (e.g., ExpeL, MemoryBank, AWM) occasionally surpass parazmetric tuning in embodied action; for instance, AWM reaches $3 6 . 1 8 \%$ on ALFWorld with SmolLM3-3B, exceeding SFT by $3 . 1 5 \%$ . Yet their effectiveness deteriorates on reasoning-intensive tasks: ExpeL achieves only $8 . 1 2 \%$ on GPQA+Qwen2.5-1.5B, and even underperforms the vanilla model by $6 . 9 \%$ on TriviaQA, underscoring its heavy reliance on backbone capacity. Parametric finetuning methods display the opposite tendency: they excel in structured domains such as code generation, where REINFORCE $^ { + + }$ reaches $6 3 . 3 3 \%$ on KodCode with $\mathtt { Q w e n 2 . 5 - 1 . 5 B }$ , but remain weak in knowledge-intensive reasoning, with GPQA below $1 4 \%$ . In contrast, MemGen consistently advances performance across all domains. For example, on ALFWorld+SmolLM3-3B, MemGen SFT and MemGen GRPO attain $5 0 . 6 0 \%$ and $6 3 . 6 0 \%$ , improving over vanilla by $3 1 . 6 4 \%$ and $4 4 . 6 4 \%$ , respectively. Similar gains appear with the larger $_ { \mathrm { Q w e n 3 - 8 B } }$ , where MemGen GRPO achieves $+ 2 7 . 0 6 \%$ on KodCode and $+ 2 8 . 1 7 \%$ on PopQA, surpassing GRPO by up to $3 . 4 \%$ . Overall, the dynamic memory insertion of MemGen delivers substantial improvements across diverse task domains.

[For RQ2] MemGen Exhibits Strong Cross-Domain Generalization. To evaluate whether the memory learned by MemGen can transfer across tasks, we train MemGen on one dataset and test it on several others. We include two out-of-domain datasets, ScienceWorld (Wang et al., 2022) and FEVER (Thorne et al., 2018), to further probe this. As shown in Figures 3, 9 and 10, baselines such as SFT and MemoryBank achieve gains within their training domains (e.g., on ALFWorld, SFT $+ 1 4 . 1 \%$ and MemoryBank $+ 5 . 4 \%$ compared with vanilla), yet fail to generalize, with performance dropping sharply on FEVER by $1 6 . 2 \%$ . In contrast, MemGen not only attains substantial in-domain improvements $2 4 . 5 5 \%  5 8 . 1 6 \%$ on KodCode, Figure 10), but also exhibits effective transfer: when trained on KodCode, performance on MATH rises from $3 6 . 6 \%  5 4 . 2 \%$ . Having empirically validated MemGen’s generalizability, we take a further step to investigate the mechanism that underlies this capability.

Table 1 Results on SmolLM3-3B and Qwen3-8B. All values represent the performance metric for each task (e.g., accuracy $\%$ ). We highlight the best and second best results.   

<table><tr><td>Backbone</td><td>Method</td><td>ALFWorld</td><td>TrivialQA</td><td>PopQA</td><td>KodCode</td><td>BigCodeBench</td><td>GPQA</td><td>GSM8K</td><td>MATH</td></tr><tr><td rowspan="10">SmolLM3-3B</td><td>Vanilla CoT</td><td>18.96</td><td>10.47</td><td>8.23</td><td>37.05</td><td>35.96</td><td>9.35</td><td>47.63</td><td>16.22</td></tr><tr><td></td><td>17.60</td><td>12.88</td><td>9.95</td><td>38.45</td><td>39.42</td><td>20.70</td><td>58.91</td><td>56.33</td></tr><tr><td>SFT</td><td>32.36</td><td>55.25</td><td>37.22</td><td>59.25</td><td>40.79</td><td>19.70</td><td>63.48</td><td>45.65</td></tr><tr><td>GRPO</td><td>55.35</td><td>65.88</td><td>45.16</td><td>68.48</td><td>72.44</td><td>22.73</td><td>80.03</td><td>61.23</td></tr><tr><td>REINFORCE</td><td>53.13</td><td>63.20</td><td>46.81</td><td>65.53</td><td>67.14</td><td>23.44</td><td>82.03</td><td>58.75</td></tr><tr><td>REINFORCE++</td><td>53.95</td><td>63.20</td><td>44.10</td><td>65.90</td><td>68.80</td><td>22.73</td><td>81.50</td><td>59.89</td></tr><tr><td>Agent-FLAN</td><td>34.00</td><td>56.70</td><td>39.50</td><td>56.80</td><td>37.20</td><td>17.80</td><td>59.60</td><td>36.84</td></tr><tr><td>ExpeL</td><td>36.18</td><td>46.20</td><td>28.16</td><td>51.14</td><td>40.22</td><td>15.15</td><td>56.23</td><td>38.11</td></tr><tr><td>MemoryBank</td><td>32.80</td><td>43.30</td><td>25.81</td><td>44.50</td><td>31.80</td><td>10.20</td><td>58.30</td><td>43.53</td></tr><tr><td>AWM</td><td>40.50</td><td>49.80</td><td>29.60</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SoftCoT</td><td>35.03</td><td>50.38</td><td>34.90</td><td>59.20</td><td>39.10</td><td></td><td>17.22 56.34</td><td>44.62</td></tr><tr><td>Co-processor</td><td>38.36</td><td>53.28</td><td>38.96</td><td>56.25</td><td>45.40</td><td>20.10</td><td>57.60</td><td>38.81</td></tr><tr><td>MemGen SFT</td><td>50.60</td><td>68.13</td><td>42.34</td><td>62.65</td><td>42.99</td><td>26.75</td><td>70.42</td><td>57.44</td></tr><tr><td>MemGen GRPO Vanilla</td><td>63.60</td><td>79.30</td><td>58.60</td><td>72.85</td><td>74.24</td><td>25.20</td><td>83.47</td><td>63.65</td></tr><tr><td>CoT SFT</td><td>58.93</td><td>52.18 53.80</td><td>34.13 33.20</td><td>49.10 51.25</td><td>33.33 35.59</td><td>38.18 35.15</td><td>89.48 87.67</td><td>79.82 78.24</td></tr><tr><td rowspan="10">Qwen3-8B</td><td></td><td>57.10</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>83.59</td><td>74.55</td><td>51.12</td><td>64.75</td><td>41.33</td><td>40.33</td><td>90.76</td><td>81.35</td></tr><tr><td>GRPO</td><td>85.60</td><td>76.15</td><td>58.90</td><td>73.35</td><td>70.24</td><td>39.54</td><td>92.30</td><td>83.54</td></tr><tr><td>REINFORCE</td><td>82.10</td><td>75.22</td><td>57.96</td><td>72.11</td><td>70.20</td><td>37.12</td><td>91.25</td><td>83.27</td></tr><tr><td>REINFORCE++</td><td>84.80</td><td>75.90</td><td>58.30</td><td>72.90</td><td>71.88</td><td>37.68</td><td>91.90</td><td>85.24</td></tr><tr><td>Agent-FLAN</td><td>80.32</td><td>70.32</td><td>50.08</td><td>62.99</td><td>43.40</td><td>39.50</td><td>87.60</td><td>80.05</td></tr><tr><td>ExpeL</td><td>78.97</td><td>65.54</td><td>40.33</td><td>57.20</td><td>34.23</td><td>35.15</td><td>86.20</td><td>77.40</td></tr><tr><td>MemoryBank</td><td>70.41</td><td>60.56</td><td>41.60</td><td>56.39</td><td>40.61</td><td>35.66</td><td>90.35</td><td>80.35</td></tr><tr><td>AWM</td><td>80.33</td><td>69.30</td><td>43.69</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SoftCoT</td><td>75.60</td><td>59.42</td><td>39.42</td><td>63.28</td><td>38.27</td><td>39.60</td><td>86.30</td><td>76.23</td></tr><tr><td>Co-processor</td><td>73.28</td><td>61.42</td><td>45.55</td><td>64.90</td><td>42.19</td><td>39.15</td><td>76.23</td><td>79.20</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MemGen SFT MemGen GRPO</td><td>85.82 90.60</td><td>77.22 80.65</td><td>54.65 62.30</td><td>66.15 76.16</td><td>40.35 75.56</td><td>43.23 40.24</td><td>91.25 93.20</td><td>83.30 88.24</td></tr></table>

![](images/f19cfb7088de428f0384592ce1a482aae2e53222cf1c2476474ffcddc80be6b5.jpg)  
Figure 3 The generalization study of MemGen. We train MemGen SFT on one dataset (ALFWorld or TriviaQA) and evaluate it on four datasets (TriviaQA, ALFWorld, ScienceWorld, and FEVER).

[For RQ2] The Memory Trigger Intelligently Determines When to Activate Memory Insertion, Mitigating Domain Conflict. After training MemGen on GSM8K, we evaluate 150 samples each from GSM8K, KodCode, and GPQA, visualizing the frequency with which the memory trigger invoked the memory weaver at each relative position in the model output. We observe that the invocation frequency varies across domains and correlates directly with performance in Figure 9: GSM8K exhibits the largest improvement $( + 1 9 . 6 4 \% )$ and maximal invocations, GPQA achieves moderate gains $( + 6 . 0 6 \% )$ with medium invocations, and KodCode shows the smallest improvement $( + 3 . 1 \% )$ with the fewest invocations. This indicates that MemGen autonomously assesses, based on task-specific context, when memory insertion will be beneficial, invoking the weaver less frequently in unfamiliar domains.

![](images/2ae402513ea6e5dc9d9615d0b4600eee291227f674045ce9a353250f7536aac8.jpg)  
Figure 5 (Left) $\mathrm { t - S N E }$ visualization of latent memories generated by MemGen $+ Q \mathrm { w e n } 3 - 8 \mathrm { B }$ across datasets; (Middle and Right) Latent memory visualization within the TriviaQA and GSM8K datasets, clustered using $K$ -means. The text at each cluster center represents the common pattern shared by many memory sequences in the cluster, such as Cluster 0 in GSM8K, where many sequences end with “_check”.

[For RQ3] MemGen Effectively Mitigates Catastrophic Forgetting. In Table 4, we sequentially train on four datasets and evaluate on all benchmarks after each stage, where MemGen exhibits stronger knowledge retention ability compared to baseline methods. For example, unlike SFT which primarily improves performance on the most recent task $( 5 4 . 1 0 \%$ on KodCode but only $2 . 5 3 \%$ on GPQA), MemGen demonstrates more balanced cross-task generalization, attaining $3 8 . 4 3 \%$ on AQuA and $2 1 . 7 2 \%$ on GPQA after GSM8K training. Finally, it mitigates forgetting on earlier tasks, preserving $4 0 . 3 4 \%$ on AQuA following KodCode training compared to $2 7 . 1 4 \%$ for ExpeL and $2 8 . 6 1 \%$ for SFT, indicating a more stable continual learning ability. More analysis is placed in Section D.1.

![](images/93a3d11f77243e55d38ef9e564d389b15a8622d1d370960a21861e7d8b0626f1.jpg)  
Figure 4 Memory invocation frequency across benchmarks at inference (trained on MemGen SFT+Qwen3-8B $^ +$ GSM8K).

# 5.3 Framework Analysis

Having established the expressive capabilities of MemGen, we further investigate its underlying mechanisms: what do the learned latent memories look like? Do they have specialized functions?

[For RQ4] The Latent Memory Is Machine-Native and Human-Unreadable. We first visualized the latent memory sequences learned by MemGen across different datasets using t-SNE in Figures 5 and 11. As shown in Figure 5 (Left), sequences from distinct domains form separate distributions, with related domains clustering closely (e.g., KodCode and BigCodeBench, GSM8K and MATH). Examining latent memories within the same dataset, we observed pronounced clustering patterns (as shown in Figure 5 (Middle and Right)). To explore potential commonalities within these clusters, we forcibly decoded the latent tokens. Although the decoded sequences are not human-readable, they exhibit intriguing regularities: many tokens within a cluster share structural conventions. For example, Cluster 0 in TriviaQA frequently follows the pattern “ [...]SOC”, whereas Cluster 3 in GSM8K often adopts the format “ [...]_pick”. A large corpus of latent memory tokens is provided in Appendix F. Despite these sequences being machine-native and human-unreadable, we further investigate whether their underlying semantics can be interpreted.

![](images/f51207d6a3bc45baf38e256e5cca24bc6d67ac4df26c8b1ef021d3744c456927.jpg)  
Figure 6 (Left) Parameter sensitivity analysis on the latent memory length $K$ ; (Right) Effects of selectively removing latent memory clusters on different agent failure modes on the TriviaQA dataset.

[For RQ4] MemGen Implicitly Learns a Human-like Memory Hierarchy. To uncover the functional roles of different latent memory clusters, we conducted a post-hoc intervention study. Following the taxonomy from (Song et al., 2025), we study eight distinct types of agent failure, including planning errors, tool response/parsing failures, answer formatting mistakes, etc, providing a structured framework to assess how memory influences performance. During evaluation, we selectively removed latent tokens close to a specific cluster while keeping others intact, measuring the resulting changes in these failure modes. Details on (1) the visualization process, (2) failure mode annotation, and (3) token filtration are in Appendix G. As shown in Figure 6 (Right), distinct memory clusters exhibit varying influence on failure modes and can be mapped to different memory functions:

• Planning Memory supports high-level task planning and strategic reasoning. Removal of Cluster 2 substantially increases planning and compositional reasoning failures, indicating that this cluster is crucial for guiding the LLM agent’s decision-making and sequencing of reasoning steps.   
• Procedural Memory captures task-specific operational knowledge, such as tool usage and formatting ability. Cluster 3 corresponds to this role, as its removal leads to a marked increase in tool response errors, parsing failures, and answer formatting mistakes.   
• Working Memory manages the retention and effective use of prior context to maintain reasoning consistency. Clusters 1 and 4 contribute to this function: for instance, removing Cluster 1’s memory tokens results in more frequent task misunderstandings and think-act inconsistency.

Nevertheless, these memory clusters are not entirely independent: for example, removing Cluster 1 also negatively affects planning ability, indicating that these memory faculties interact and jointly enable the LLM to leverage past experience effectively. This analysis reveals that MemGen spontaneously organizes latent memory into a structured, human-like hierarchy.

Ablation Study $\&$ Sensitivity Analysis. We conduct a sensitivity analysis on the length of the latent memory sequence $K$ , as shown in Figure 6 (Left). It can be observed that as the latent token length increases from $2  3 2$ , MemGen ’s performance correspondingly improves, likely reflecting the expanded memory capacity. We then perform an ablation study on the memory trigger module in Table 5, demonstrating the necessity of a dedicatedly trained trigger for effective memory invocation. Furthermore, we analyze different training paradigms of the memory weaver in Table 6. Additional results and discussions are in Section D.3.

Efficiency Analysis. To confirm that the memory insertion process of MemGen does not introduce significant inference overhead, we show in Section D.3.3 that, while achieving up to $5 7 . 6 6 \%$ performance improvement over vanilla LLMs, the per-query inference delay remains consistently below the baseline, ranging from $2 4 \%$ to $9 4 \%$ of the vanilla LLM latency. This clearly demonstrates that MemGen delivers substantial performance gains without compromising efficiency.

# 6 Conclusion

In this work, we introduced MemGen, a dynamic and generative memory framework designed for LLM Agents. By interleaving reasoning with memory synthesis through a reinforcement-learned memory trigger and a generative memory weaver, MemGen transcends the limitations of parametric and retrieval-based paradigms. Extensive experiments showcase substantial performance gains, robust cross-domain generalization, strong continual learning ability, and MemGen’s explicitly modeled memory hierarchy (i.e., planning, procedural, and working memory). These results suggest a promising path toward self-evolving LLM agents capable of fluid and reconstructive intelligence.

# References

Anthropic. On the Biology of a Large Language Model. https://transformer-circuits.pub/2025/attribution-graphs/ biology.html, 2025. [Accessed 24-08-2025].   
Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024. https://arxiv.org/abs/2401.10774.   
Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. Fireact: Toward language agent fine-tuning, 2023. https://arxiv.org/abs/2310.05915.   
Guoxuan Chen, Han Shi, Jiawei Li, Yihang Gao, Xiaozhe Ren, Yimeng Chen, Xin Jiang, Zhenguo Li, Weiyang Liu, and Chao Huang. Sepllm: Accelerate large language models by compressing one segment into one separator. arXiv preprint arXiv:2412.12094, 2024a.   
Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei Zhang, Jiangning Liu, Dahua Lin, Kai Chen, and Feng Zhao. Agent-flan: Designing data and methods of effective agent tuning for large language models, 2024b. https: //arxiv.org/abs/2403.12881.   
Zhixun Chen, Ming Li, Yuxuan Huang, Yali Du, Meng Fang, and Tianyi Zhou. Atlas: Agent tuning via learning critical steps, 2025. https://arxiv.org/abs/2503.02197.   
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory, 2025. https://arxiv.org/abs/2504.19413.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li,

Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. https://arxiv.org/abs/2501.12948.

Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. Loramoe: Alleviate world knowledge forgetting in large language models via moe-style plugin, 2024. https://arxiv.org/abs/2312.09979.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. arXiv preprint arXiv:2508.06433, 2025.

Dayuan Fu, Keqing He, Yejie Wang, Wentao Hong, Zhuoma Gongque, Weihao Zeng, Wei Wang, Jingang Wang, Xunliang Cai, and Weiran Xu. Agentrefine: Enhancing agent generalization through refinement tuning, 2025. https://arxiv.org/abs/2501.01702.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding, 2024. https://arxiv.org/abs/2402.02057.

Huanang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Qihan Ren, Cheng Qian, Zhenhailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, and Mengdi Wang. A survey of self-evolving agents: On path to artificial super intelligence, 2025. https://arxiv.org/abs/2507.21046.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. Prompt cache: Modular attention reuse for low-latency inference. Proceedings of Machine Learning and Systems, 6:325–338, 2024.

Raghavv Goel, Sudhanshu Agrawal, Mukul Gagrani, Junyoung Park, Yifan Zao, He Zhang, Tian Liu, Yiping Yang, Xin Yuan, Jiuyan Lu, Chris Lott, and Mingu Lee. Vocabtrim: Vocabulary pruning for efficient speculative decoding in llms, 2025. https://arxiv.org/abs/2506.22694.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2024. https://arxiv.org/abs/2412.06769.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. https://arxiv.org/abs/2103. 03874.

Hongkang Yang Hongkang Yang, Zehao Lin Zehao Lin, Wenjin Wang Wenjin Wang, Hao Wu Hao Wu, Zhiyu Li Zhiyu Li, Bo Tang Bo Tang, Wenqiang Wei Wenqiang Wei, Jinbo Wang Jinbo Wang, Zeyun Tang Zeyun Tang, Shichao Song Shichao Song, Chenyang Xi Chenyang Xi, Yu Yu Yu Yu, Kai Chen Kai Chen, Feiyu Xiong Feiyu Xiong, Linpeng Tang Linpeng Tang, and Weinan E Weinan E. Memory $^ 3$ : Language modeling with explicit memory. Journal of Machine Learning, 3(3):300–346, January 2024. ISSN 2790-203X. doi: 10.4208/jml.240708. http://dx.doi.org/10.4208/jml.240708.

Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models, 2025a. https://arxiv.org/abs/2501.03262.

Zhiyuan Hu, Yibo Wang, Hanze Dong, Yuhui Xu, Amrita Saha, Caiming Xiong, Bryan Hooi, and Junnan Li. Beyond’aha!’: Toward systematic meta-abilities alignment in large reasoning models. arXiv preprint arXiv:2505.10554, 2025b.

HuggingFace. SmolLM3: smol, multilingual, long-context reasoner — huggingface.co. https://huggingface.co/blog/ smollm3, 2025. [Accessed 23-09-2025].

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. https://arxiv.org/abs/2403.07974.

Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin Jin. Ragcache: Efficient knowledge caching for retrieval-augmented generation. arXiv preprint arXiv:2404.12457, 2024.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601– 1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. https://aclanthology.org/P17-1147/.

Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, and Zilong Zheng. Seek in the dark: Reasoning via test-time instance-level policy gradient in latent space, 2025a. https://arxiv.org/abs/2505.13308.   
Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty, 2025b. https://arxiv.org/abs/2401.15077.   
Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. arXiv preprint arXiv:1705.04146, 2017.   
Luyang Liu, Jonas Pfeiffer, Jiaxing Wu, Jun Xie, and Arthur Szlam. Deliberation in latent space via differentiable cache augmentation. arXiv preprint arXiv:2412.17747, 2024.   
Hanjun Luo, Shenyu Dai, Chiming Ni, Xinfeng Li, Guibin Zhang, Kun Wang, Tongliang Liu, and Hanan Salam. Agentauditor: Human-level safety and security evaluation for llm agents, 2025a. https://arxiv.org/abs/2506.00641.   
Michael Luo, Naman Jain, Jaskirat Singh, Sijun Tan, Ameen Patel, Qingyang Wu, Alpay Ariyak, Colin Cai, Shang Zhu Tarun Venkat, Ben Athiwaratkun, Manan Roongta, Ce Zhang, Li Erran Li, Raluca Ada Popa, Koushik Sen, and Ion Stoica. Deepswe: Training a state-of-the-art coding agent from scratch by scaling rl. https://pretty-radio-b75.notion.site/ DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabb 2025b. Notion Blog.   
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9 (Nov):2579–2605, 2008.   
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories, 2023. https: //arxiv.org/abs/2212.10511.   
Dan P McAdams. The psychology of life stories. Review of general psychology, 5(2):100–122, 2001.   
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Memgpt: Towards llms as operating systems, 2024. https://arxiv.org/abs/2310.08560.   
Dmitrii Pantiukhin, Boris Shapkin, Ivan Kuznetsov, Antonia Anna Jost, and Nikolay Koldunov. Accelerating earth science discovery via multi-agent llm systems, 2025. https://arxiv.org/abs/2503.05854.   
Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs, 2025. https://arxiv.org/abs/2504.13958.   
Jiahao Qiu, Xinzhe Juan, Yimin Wang, Ling Yang, Xuan Qi, Tongcheng Zhang, Jiacheng Guo, Yifu Lu, Zixin Yao, Hongru Wang, Shilong Liu, Xun Jiang, Liu Leqi, and Mengdi Wang. Agentdistill: Training-free agent distillation with generalizable mcp boxes, 2025a. https://arxiv.org/abs/2506.14728.   
Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, Xing Zhou, Dongrui Liu, Ling Yang, Yue Wu, Kaixuan Huang, Shilong Liu, Hongru Wang, and Mengdi Wang. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution, 2025b. https://arxiv.org/abs/2505.20286.   
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. https://arxiv.org/abs/2311. 12022.   
Shuo Ren, Pu Jian, Zhenjiang Ren, Chunlin Leng, Can Xie, and Jiajun Zhang. Towards scientific intelligence: A survey of llm-based scientific agents, 2025. https://arxiv.org/abs/2503.24047.   
Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation, 2025. https://arxiv.org/abs/2502.21074.   
Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning, 2021. https://arxiv.org/abs/2010. 03768.   
Joykirat Singh, Raghav Magazine, Yash Pandya, and Akshay Nambi. Agentic reasoning and tool integration for llms via reinforcement learning, 2025. https://arxiv.org/abs/2505.01441.   
Peiyang Song, Pengrui Han, and Noah Goodman. A survey on large language model reasoning failures. In 2nd AI for Math Workshop @ ICML 2025, 2025. https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } }$ hsgMn4KBFG.   
Eleanor Spens and Neil Burgess. A generative model of memory construction and consolidation. Nature human behaviour, 8(3):526–543, 2024.   
Haowen Su, Xian Li, Savannah Born, Christopher J Honey, Janice Chen, and Hongmi Lee. Neural dynamics of spontaneous memory recall and future thinking in the continuous flow of thoughts. Nature communications, 16(1): 6433, 2025a.   
Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. Parametric retrieval augmented generation, 2025b. https://arxiv.org/abs/2501.15915.   
Yuchang Sun, Yanxi Chen, Yaliang Li, and Bolin Ding. Enhancing latent computation in transformers with latent tokens, 2025. https://arxiv.org/abs/2505.12629.   
Jihoon Tack, Jaehyung Kim, Eric Mitchell, Jinwoo Shin, Yee Whye Teh, and Jonathan Richard Schwarz. Online adaptation of language models with a memory of amortized contexts, 2024. https://arxiv.org/abs/2403.04317.   
Wenhui Tan, Jiaze Li, Jianzhong Ju, Zhenbo Luo, Jian Luan, and Ruihua Song. Think silently, think fast: Dynamic latent compression of llm reasoning chains, 2025. https://arxiv.org/abs/2505.16552.   
Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, Hao Cheng, Chi Wang, and Wangchunshu Zhou. Agent kb: Leveraging cross-domain experience for agentic problem solving, 2025. https://arxiv.org/abs/2507.06229.   
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale dataset for fact extraction and verification. arXiv preprint arXiv:1803.05355, 2018.   
Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. Wise: Rethinking the knowledge memory for lifelong model editing of large language models, 2024a. https://arxiv.org/abs/2405.14768.   
Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader?, 2022. https://arxiv.org/abs/2203.07540.   
Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, et al. Memoryllm: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624, 2024b.   
Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley, Dan Gutfreund, Rogerio Feris, and Zexue He. M+: Extending memoryllm with scalable long-term memory, 2025a. https://arxiv.org/abs/ 2502.00592.   
Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. Mobileagent-e: Self-evolving mobile assistant for complex tasks, 2025b. https://arxiv.org/abs/2501.11733.   
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory, 2024c. https: //arxiv.org/abs/2409.07429.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. https://arxiv.org/abs/2201.11903.   
Yifan Wei, Xiaoyan Yu, Yixuan Weng, Tengfei Pan, Angsheng Li, and Li Du. Autotir: Autonomous tools integrated reasoning via reinforcement learning, 2025. https://arxiv.org/abs/2507.21836.   
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.   
Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, and Klara Nahrstedt. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use, 2025a. https://arxiv.org/abs/2505.19255.   
Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey on memory mechanisms in the era of llms, 2025b. https: //arxiv.org/abs/2504.15965.   
Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Jiliang Tang, Himabindu Lakkaraju, and Zhen Xiang. How memory management impacts llm agents: An empirical study of experience-following behavior, 2025. https: //arxiv.org/abs/2505.16067.   
Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. A-mem: Agentic memory for llm agents, 2025a. https://arxiv.org/abs/2502.12110.   
Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot: Soft chain-of-thought for efficient reasoning with llms, 2025b. https://arxiv.org/abs/2502.12134.   
Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot++: Test-time scaling with soft chain-of-thought reasoning, 2025c. https://arxiv.org/abs/2505.11484.   
Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, and Radha Poovendran. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding, 2025d. https://arxiv.org/abs/2503.02951.   
Zifan Xu, Haozhu Wang, Dmitriy Bespalov, Xuan Wang, Peter Stone, and Yanjun Qi. Latent skill discovery for chain-of-thought reasoning. arXiv preprint arXiv:2312.04684, 2023.   
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024a.   
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, and Bo Zheng et al. Qwen3 technical report, 2025. https://arxiv.org/abs/2505.09388.   
John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Sweagent: Agent-computer interfaces enable automated software engineering, 2024b. https://arxiv.org/abs/2405.15793.   
Weiran Yao, Shelby Heinecke, Juan Carlos Niebles, Zhiwei Liu, Yihao Feng, Le Xue, Rithesh Murthy, Zeyuan Chen, Jianguo Zhang, Devansh Arpit, Ran Xu, Phil Mui, Huan Wang, Caiming Xiong, and Silvio Savarese. Retroformer: Retrospective large language agents with policy gradient optimization, 2024. https://arxiv.org/abs/2308.02151.   
Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. Agent lumos: Unified and modular training for open-source language agents, 2024. https://arxiv.org/abs/2311. 05657.   
Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, and Hao Zhou. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent, 2025. https://arxiv.org/abs/2507.02259.   
Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. Agenttuning: Enabling generalized agent abilities for llms, 2023. https://arxiv.org/abs/2310.12823.   
Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, and Shuicheng Yan. G-memory: Tracing hierarchical memory for multi-agent systems, 2025a. https://arxiv.org/abs/2506.07398.   
Jianguo Zhang, Tian Lan, Rithesh Murthy, Zhiwei Liu, Weiran Yao, Ming Zhu, Juntao Tan, Thai Hoang, Zuxin Liu, Liangwei Yang, Yihao Feng, Shirley Kokane, Tulika Awalgaonkar, Juan Carlos Niebles, Silvio Savarese, Shelby Heinecke, Huan Wang, and Caiming Xiong. Agentohana: Design unified data and training pipeline for effective agent learning, 2024a. https://arxiv.org/abs/2402.15506.   
Yang Zhang, Wenxin Xu, Xiaoyan Zhao, Wenjie Wang, Fuli Feng, Xiangnan He, and Tat-Seng Chua. Reinforced latent reasoning for llm-based recommendation, 2025b. https://arxiv.org/abs/2505.19092.   
Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents, 2024b. https://arxiv.org/abs/2404.13501.   
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners, 2024. https://arxiv.org/abs/2308.10144.   
Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills, 2025. https://arxiv.org/abs/2504.07079.   
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory, 2023. https://arxiv.org/abs/2305.10250.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents, 2025. https://arxiv.org/abs/2506.15841. Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, et al. A survey on latent reasoning. arXiv preprint arXiv:2507.06203, 2025.

# Appendix

# A Additional Related Works

LLM Decoding & RL. Two additional topics that relate to our work are LLM decoding and reinforcement learning (RL). From the decoding perspective, MemGen dynamically generates and inserts latent tokens, which shares similarity with speculative decoding where a drafter model receives the current decoding context and produces subsequent drafted tokens (Cai et al., 2024; Fu et al., 2024; Li et al., 2025b; Goel et al., 2025). . However, these methods primarily aim to accelerate LLM inference, whereas MemGen focuses on leveraging latent states as effective carriers of memory. From the RL perspective, MemGen employs rule-based RL to train the memory trigger, which is closely related to reinforcement learning with variable reward (RLVR), including GRPO from DeepSeek-R1 (DeepSeek-AI et al., 2025) and its various derivatives (Qian et al., 2025; Wu et al., 2025a; Wei et al., 2025; Fu et al., 2025). While there exist efforts combining RL with agent memory, to our knowledge, most do not address self-improving memory; for example, MemAgent (Yu et al., 2025) and MEM1 (Zhou et al., 2025) focus on handling long-context inputs rather than evolving memory mechanisms.

# B Optimization Algorithm on Memory Weaver

In this section, we provide a detailed exposition of the optimization algorithms for training the memory weaver, $\mathcal { W } _ { \mathrm { w e a v e r } }$ , as mentioned in Section 4.3. The core principle, as established in Equation (10), is to update only the weaver’s parameters, denoted as $\theta ^ { \prime }$ , while keeping the reasoner $\pi _ { \theta }$ frozen. This modularity allows MemGen to be compatible with various optimization paradigms. We detail the specific implementations for Supervised Fine-Tuning (SFT) and a reinforcement learning approach, Group Relative Policy Optimization (GRPO).

# B.1 Combining MemGen with SFT

The objective of Supervised Fine-Tuning is to train the memory weaver to generate latent memories that guide the frozen reasoner $\pi \theta$ to replicate the behavior observed in a dataset of high-quality demonstration trajectories. We leverage the provided history of past experiences, $\mathcal { H } = \{ ( x _ { i } , \tau _ { i } ^ { * } ) \} _ { i = 1 } ^ { N }$ , where each $\tau _ { i } ^ { * }$ is treated as an expert demonstration.

Each expert trajectory $\tau _ { i } ^ { * }$ consists of a sequence of states and actions, $\tau _ { i } ^ { * } = ( s _ { i , 0 } , a _ { i , 0 } ^ { * } , s _ { i , 1 } , a _ { i , 1 } ^ { * } , \ldots )$ . Each expert action $a _ { i , t } ^ { * }$ is a sequence of tokens, $a _ { i , t } ^ { * } = ( \mathbf { z } _ { i , t , 1 } ^ { * } , \mathbf { z } _ { i , t , 2 } ^ { * } , \ldots , \mathbf { z } _ { i , t , L _ { t } } ^ { * } )$ . The goal is to maximize the conditional log-likelihood of generating these expert token sequences.

During the training of the weaver, both the reasoner $\pi \theta$ and the memory trigger $\tau$ are held fixed. At each token generation step $j$ where the trigger activates, the weaver $\mathcal { W } _ { \theta ^ { \prime } }$ takes the reasoner’s hidden states $\mathbf { H } _ { i , t , < j }$ as input and produces a latent memory $\mathbf { M } _ { i , t }$ . The reasoner then generates the next token conditioned on this memory. The SFT objective is to adjust the weaver’s parameters $\theta ^ { \prime }$ to maximize the probability of the ground-truth token $\mathbf { z } _ { i , t , j } ^ { * }$ .

Formally, the optimization problem is to minimize the negative log-likelihood of the expert trajectories, averaged over the dataset $\mathcal { H }$ . The loss function for the weaver’s parameters $\theta ^ { \prime }$ is defined as:

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ^ { \prime } ) = - \mathbb { E } _ { ( x _ { i } , \tau _ { i } ^ { * } ) \sim \mathcal { H } } \left[ \sum _ { \ell = 0 } ^ { T _ { i } - 1 } \sum _ { j = 1 } ^ { L _ { t } } \log \pi _ { \theta } ( \mathbf { z } _ { i , t , j } ^ { * } \mid s _ { i , t } , \mathbf { z } _ { i , t , < j } ^ { * } , \mathbf { M } _ { i , t , j } ) \right] ,
$$

where the latent memory $\mathbf { M } _ { i , t , j }$ is synthesized by the weaver at that specific step:

$$
\begin{array} { r } { \mathbf { M } _ { i , t , j } = \mathcal { W } _ { \theta ^ { \prime } } ( \mathbf { H } _ { i , t , < j } ) . } \end{array}
$$

Note that the generation of $\mathbf { M } _ { i , t , j }$ only occurs if the fixed trigger $\tau$ determines an invocation is needed at step $j$ . In steps where no memory is invoked, the conditioning term $\mathbf { M } _ { i , t , j }$ is omitted. The gradients are computed

exclusively with respect to the weaver’s parameters $\theta ^ { \prime }$ and used to update them via gradient descent:

$$
\theta ^ { \prime }  \theta ^ { \prime } - \eta \nabla _ { \theta ^ { \prime } } \mathcal { L } _ { \mathrm { S F T } } ( \theta ^ { \prime } ) ,
$$

where $\eta$ is the learning rate. Through this process, the memory weaver learns to synthesize latent memories that effectively steer the frozen reasoner’s generative process to align with the training data’s behavior, thereby internalizing the knowledge from the demonstration data without corrupting the general capabilities of the core LLM.

# B.2 Combing MemGen wirh GRPO

The memory weaver can also be trained using a reinforcement learning objective. We specifically adapt the GRPO algorithm. The training process begins by sampling a batch of tasks from the experience history $\mathcal { H }$ . For each task $x _ { i }$ , we use the policy $\Pi _ { \theta } ^ { \mathcal { W } _ { \theta ^ { \prime } } , \mathcal { I } }$ (composed of the frozen reasoner $\pi \theta$ and the current memory weaver $\mathcal { W } _ { \theta ^ { \prime } }$ ) to generate a group of $K$ distinct trajectories, denoted as $\mathcal { G } _ { i } = \left\{ \tau _ { i , 1 } , \tau _ { i , 2 } , \ldots , \tau _ { i , K } \right\}$ . Each trajectory is generated by the agent’s interaction with the environment and results in a final reward $R ( \tau _ { i , k } )$ , evaluated by the reward function. Going forward, GRPO computes a group-relative baseline by averaging the rewards of all trajectories within the group $\beta _ { i }$ :

$$
\bar { R } ( \mathcal { G } _ { i } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R ( \tau _ { i , k } ) .
$$

The advantage for a specific trajectory $\tau _ { i , k }$ is then its reward relative to this baseline:

$$
A ( \tau _ { i , k } ) = R ( \tau _ { i , k } ) - \bar { R } ( \mathcal { G } _ { i } ) .
$$

This formulation allows the weaver to learn by differentiating between better and worse outcomes within a set of its own generations, promoting policies that produce trajectories with above-average rewards. The final objective function, maximized with respect to the weaver’s parameters $\theta ^ { \prime }$ , is:

$$
\begin{array} { r } { \nu ( \theta ^ { \prime } ) = \mathbb { E } _ { \boldsymbol { x } _ { i } \sim \mathcal { H } , \mathcal { G } _ { i } \sim \Pi _ { \theta } ^ { W _ { \theta ^ { \prime } } , T } } \left[ \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } A ( \tau _ { i , k } ) \log \Pi _ { \theta } ^ { \mathcal { W } _ { \theta ^ { \prime } } , T } ( \tau _ { i , k } \mid \boldsymbol { x } _ { i } ) - \beta \mathrm { K L } \left( \Pi _ { \theta } ^ { \mathcal { W } _ { \theta ^ { \prime } } , T } ( \cdot \mid \boldsymbol { x } _ { i } ) \parallel \Pi _ { \mathrm { r e f } } ( \cdot \mid \boldsymbol { x } _ { i } ) \right) \right] , } \end{array}
$$

where $\scriptstyle \mathrm { { 1 l _ { r e f } } }$ is a fixed reference policy and $\beta$ is a coefficient controlling the strength of the KL regularization. The gradients are computed only for the weaver’s parameters $\theta ^ { \prime }$ , thus refining its ability to synthesize impactful latent memories while preserving the integrity of the core reasoner.

# C Experimental Details

# C.1 Training Dataset Setup

Training Datasets. We employ the official training splits of all evaluated datasets, with the sole exception of PopQA. These datasets provide the supervisory signals necessary to train both the memory weaver and the memory trigger. PopQA does not include a dedicated training set; therefore, for this benchmark, we rely on the model trained on TriviaQA, selected for its conceptual proximity, and conduct evaluation directly on PopQA to assess cross-dataset generalization.

Training the Memory Weaver. We begin by training the memory weaver independently, in the absence of a memory trigger. To expose the weaver to varied invocation contexts, we explore two insertion strategies for latent memory tokens: inserting at every punctuation boundary, or sampling a subset of these boundaries at random. This staged setup allows the weaver to learn how to synthesize latent memory from local reasoning states before any explicit decision mechanism is introduced.

Training the Memory Trigger. Once the memory weaver is trained and fixed, we subsequently train the memory trigger. The trigger learns to detect when memory should be invoked, conditioned on the agent’s evolving cognitive state. During this phase, the trigger operates atop the frozen weaver, ensuring that its decision policy is aligned with a stable memory synthesis process.

# C.2 Parameter Configurations

Table 2 Hyperparameters used in the training of MemGen.   

<table><tr><td>Settings</td><td>Hyperparameters</td></tr><tr><td>Training (SFT)</td><td>train batch size = 4 learning_rate = 1e-5 epochs = 2 warmup_ratio = 0.1 optim = adamw torch schedular = cosine rollout batch size = 8</td></tr><tr><td>Training (GRPO)</td><td>train batch size = 8 epochs = 2 beta = 0.0 num_iterations = 1 learning_rate = 1e-5 warmup_ratio = 0.1 optim = adamw torch schedular = cosine</td></tr><tr><td>LoRA</td><td>r = 16 lora_alpha = 32 target_modules = [q_proj, v_proj] lora _dropout = 0.1 bias = none task_type = CAUSAL_LM</td></tr><tr><td>Optimization</td><td>adam_offload flash attn deepspeed_enable _sleep</td></tr></table>

# D Extra Results

# D.1 Continual Learning Result

The results in Table 4 indicate three main findings. First, MemGen exhibits stronger knowledge retention and forward transfer than SFT and ExpeL. For example, when trained on GPQA, MemGen reaches $4 7 . 9 6 \%$ on GSM8K and $2 8 . 8 0 \%$ on KodCode, surpassing SFT at $4 5 . 7 4 \%$ and $1 8 . 5 0 \%$ by margins of $+ 2 . 2 2 \%$ and $+ 1 0 . 3 0 \%$ , respectively. Similarly, when trained on KodCode, MemGen maintains $4 0 . 3 4 \%$ on AQuA and $2 0 . 0 9 \%$ on GPQA, whereas SFT yields $2 8 . 6 1 \%$ and $2 . 5 3 \%$ . Second, MemGen demonstrates more balanced cross-task generalization. Unlike SFT, which primarily improves performance on the most recent task (e.g., $5 4 . 1 0 \%$ on KodCode but only $2 . 5 3 \%$ on GPQA), MemGen achieves competitive results across tasks. After training on GSM8K, it attains $3 8 . 4 3 \%$ on AQuA and $2 1 . 7 2 \%$ on GPQA, both higher than SFT and ExpeL, suggesting that latent memory captures task-invariant reasoning. Third, MemGen effectively mitigates forgetting on earlier tasks. After sequential training on KodCode, it preserves $4 0 . 3 4 \%$ on AQuA compared to $2 7 . 1 4 \%$ of ExpeL and $2 8 . 6 1 \%$ of SFT, showing stronger robustness to catastrophic forgetting. These findings suggest that MemGen provides a more stable and transferable mechanism for continual learning.

# D.2 Trigger Frequency Visualization

More visualizations of the trigger frequency are displayed in Figures 7 and 8, where we paired Qwen2.5-1.5B or SmolLM3-3B with the GSM8K dataset and tested on subsets of GSM8K, KodCode, and GPQA (each

Table 3 Results on Qwen2.5-1.5B. All values represent the performance metric for each task (e.g., accuracy $\%$ ). We highlight the best and second best results.   

<table><tr><td>Vanilla</td><td>Method</td><td>ALFWorld</td><td>TrivialQA 32.10</td><td>PopQA</td><td>KodCode</td><td>BigCodeBench</td><td>GPQA</td><td>GSM8K</td><td>MATH</td></tr><tr><td colspan="16"></td></tr><tr><td rowspan="6">L 2</td><td>CoT</td><td>22.54 18.30</td><td>28.67</td><td>16.08 18.39</td><td>24.55 32.32</td><td>40.35 38.59</td><td>11.62 15.67</td><td>39.51 56.79</td><td>36.63 45.22</td></tr><tr><td>SFT</td><td>36.57</td><td>63.84</td><td>39.20</td><td>55.83</td><td>37.72</td><td>11.11</td><td>54.83</td><td>38.84</td></tr><tr><td>GRPO</td><td>43.55</td><td>68.21</td><td>43.15</td><td>62.11</td><td>70.34</td><td>15.65</td><td>68.10</td><td>47.42</td></tr><tr><td>REINFORCE</td><td>43.25</td><td>66.50</td><td>41.87</td><td>60.20</td><td>67.80</td><td>12.50</td><td>67.40</td><td>46.89</td></tr><tr><td>REINFORCE++</td><td>43.66</td><td>66.90</td><td>44.69</td><td>63.33</td><td>69.50</td><td>13.80</td><td>69.04</td><td>47.33</td></tr><tr><td>Agent-FLAN</td><td>35.80</td><td>64.28</td><td>38.90</td><td>56.21</td><td>43.83</td><td>9.35</td><td>53.02</td><td>29.82</td></tr><tr><td rowspan="6">AWM</td><td>ExpeL</td><td>28.96</td><td>25.20</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MemoryBank</td><td>27.89</td><td>38.14</td><td>20.20 22.78</td><td>31.15 37.93</td><td>39.78 35.87</td><td>8.12</td><td>45.12</td><td>38.12</td></tr><tr><td></td><td>30.42</td><td>55.69</td><td>32.54</td><td>-</td><td></td><td>13.87</td><td>47.88</td><td>30.47</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SoftCoT Co-processor</td><td>33.07 35.66</td><td>62.22 64.78</td><td>38.78 40.12</td><td>55.13 56.65</td><td>36.10 38.10</td><td>9.31 12.12</td><td>54.50 57.12</td><td>38.55</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>37.40</td></tr><tr><td></td><td>MemGen SFT</td><td>40.30</td><td>65.02</td><td>41.28</td><td>58.16</td><td>42.47</td><td>18.28</td><td>58.15</td><td>47.12</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>72.81</td><td>18.18</td><td>73.39</td><td>53.36</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>MemGen GRPO</td><td>54.27</td><td>73.42</td><td>49.28</td><td>65.43</td><td></td><td></td><td></td><td></td></tr></table>

subset having the same number of samples). We then tallied the frequency of memory trigger INVOKE judgments at each relative percentile position in the LLM output.

![](images/3169f885e5fec8f783562b988378fb8ac5df4cce4c66d5aa166bf25de85bdaca.jpg)  
Figure 7 Memory invocation frequency across benchmarks at inference (trained on MemGen SFT $^ +$ Qwen2.5-1.5B +GSM8K).

![](images/c0ec64ab4f6c5516d7928d4a4b96e9c794a9211cc8779691d5d4db9e195df8ba.jpg)  
Figure 8 Memory invocation frequency across benchmarks at inference (trained on MemGen SFT $^ +$ SmolLM3-3B +GSM8K).

![](images/22ef291c4ffd8d3bcf11d4b87285eb007cbd456916a864f424a073949cc6effe.jpg)  
Figure 9 The generalization study of MemGen. We train MemGen SFT on GSM8K and evaluate it on all four datasets.

![](images/fe4723fb9e3d5c660d5fd80d76980f12daaeae99a38205f859a7d1b2fb0ad7a7.jpg)  
Figure 10 The generalization study of MemGen. We train MemGen SFT on KodCode and evaluate it on all four datasets.   
Table 4 Continual learning results of Qwen2.5-1.5B-Instruct across four reasoning and programming datasets (AQuA, GPQA, GSM8K, KodCode). The model is sequentially trained on each dataset $\mathrm { \Delta A Q u A }  \mathrm { G P Q A }  \mathrm { G S M 8 K }$ $ \mathrm { K o d C o d e } )$ , and after each training stage, evaluation is conducted on all four benchmarks.

<table><tr><td rowspan="2">Trained On</td><td>Method</td><td>AQuA</td><td>GPQA</td><td>GSM8K</td><td>KodCode</td></tr><tr><td>Vanilla</td><td>41.34</td><td>11.62</td><td>39.51</td><td>24.55</td></tr><tr><td rowspan="3">AQuA</td><td>SFT</td><td>42.52</td><td>16.67</td><td>42.10</td><td>18.20</td></tr><tr><td>ExpeL</td><td>41.73</td><td>12.67</td><td>40.16</td><td>16.30</td></tr><tr><td>MemGen SFT</td><td>43.31</td><td>19.70</td><td>39.80</td><td>19.55</td></tr><tr><td rowspan="3">GPQA</td><td>SFT</td><td>38.55</td><td>17.17</td><td>45.74</td><td>18.50</td></tr><tr><td>ExpeL</td><td>37.24</td><td>14.35</td><td>42.67</td><td>15.20</td></tr><tr><td>MemGen SFT</td><td>39.85</td><td>20.72</td><td>47.96</td><td>28.80</td></tr><tr><td rowspan="3">GSM8K</td><td>SFT</td><td>33.46</td><td>13.13</td><td>52.31</td><td>19.45</td></tr><tr><td>ExpeL</td><td>34.89</td><td>12.42</td><td>48.78</td><td>13.65</td></tr><tr><td>MemGen SFT</td><td>38.43</td><td>21.72</td><td>55.67</td><td>19.75</td></tr><tr><td rowspan="3">KodCode</td><td>SFT</td><td>28.61</td><td>2.53</td><td>24.14</td><td>54.10</td></tr><tr><td>ExpeL</td><td>27.14</td><td>6.23</td><td>31.44</td><td>48.35</td></tr><tr><td>MemGen SFT</td><td>40.34</td><td>20.09</td><td>53.72</td><td>52.95</td></tr></table>

# D.3 Framework Analysis

# D.3.1 Ablation Study

To assess the effectiveness of the proposed memory trigger, we conduct an ablation study comparing different memory invocation strategies, as summarized in Table 5. Specifically, Random denotes a naïve baseline where latent memory tokens are inserted at arbitrary token positions with a fixed probability $p$ . All delimiters activated represents a sentence-level strategy that invokes the memory weaver at every delimiter position without any selection mechanism. Finally, MemGen’s dedicated Trigger corresponds to our standard approach, where the memory weaver is activated by the trained trigger $\tau _ { \mathrm { t r i g g e r } }$ . The results reveal several key observations. First, sentence-level intervention already improves performance compared to random invocation. For instance, activating the weaver at all delimiters yields $1 7 . 3 4 \%$ , $5 6 . 2 0 \%$ , and $6 4 . 1 5 \%$ on GPQA, Kodcode, and TriviaQA, respectively, consistently outperforming all random baselines (e.g., $p = 0 . 5$ achieves only $1 6 . 6 6 \%$ , $5 2 . 9 5 \%$ , and $5 7 . 2 8 \%$ ). This highlights the importance of aligning memory injection with semantic boundaries rather than distributing it across the sequence. More importantly, our trained trigger achieves the best overall performance, reaching $1 8 . 2 8 \%$ , $5 8 . 1 6 \%$ , and $6 5 . 0 2 \%$ on the three benchmarks. This demonstrates that selective activation, i.e., deciding when and where to weave in memory, provides the most effective support for reasoning, as it balances memory utility and interference more precisely than coarse-grained alternatives.

Table 5 Ablation study of different memory invocation strategies. Random denotes a naïve baseline where latent memory tokens are inserted at arbitrary token positions with a fixed probability $p$ . All delimiters activated represents a sentence-level strategy that invokes the memory weaver at every delimiter position without any selection mechanism. Finally, MemGen’s dedicated Trigger corresponds to our standard approach, where the memory weaver is activated by the trained trigger $\tau _ { \mathrm { t r i g g e r } }$ .   

<table><tr><td>Memory Invocation Strategy</td><td>GPQA</td><td>Kodcode</td><td>TriviaQA</td></tr><tr><td>Random (p = 0.2)</td><td>15.66</td><td>54.55</td><td>63.55</td></tr><tr><td>Random (p = 0.5)</td><td>16.66</td><td>52.95</td><td>57.28</td></tr><tr><td>Random (p = 0.8)</td><td>12.63</td><td>53.60</td><td>62.22</td></tr><tr><td>All delimiters activated</td><td>17.34</td><td>56.20</td><td>64.15</td></tr><tr><td>MemGen&#x27;s dedicated Trigger</td><td>18.28</td><td>58.16</td><td>65.02</td></tr></table>

# D.3.2 Analysis of Memory Weaver

We further investigate how the parameterization of the memory weaver influences downstream performance. As shown in Table 6, increasing the number of trainable parameters by adopting a full-parameter SFT setup surely enhances the weaver’s memory and learning capacity, leading to higher task performance across benchmarks. Nevertheless, the LoRA-based instantiation already achieves competitive results while retaining high parameter efficiency, demonstrating that even lightweight adaptations can endow the weaver with sufficient capacity to generate effective latent memories.

Table 6 Ablation study of the latent weaver. We instantiate MemGen SFT’s memory weaver with two strategies: LoRA and full-parameter SFT.   

<table><tr><td>Base LLM: Qwen2.5-1.5B-Instruct</td><td>GPQA</td><td>Kodcode</td><td>TriviaQA</td></tr><tr><td>LoRA (r = 16, α = 32)</td><td>18.28</td><td>58.16</td><td>65.02</td></tr><tr><td>Full SFT</td><td>21.21</td><td>60.00</td><td>67.10</td></tr></table>

# D.3.3 Efficiency Analysis

Table 7 reports average inference time and task performance across three benchmarks. Both SFT and MemGen drastically reduce per-task inference time compared with vanilla models due to fewer tokens required to reach correct answers. For instance, on KodCode with Qwen2.5-1.5B, MemGen SFT completes tasks in 2.94 s, a

$7 5 . 4 \%$ reduction from 11.96 s for vanilla, while improving accuracy by $3 3 . 6 1 \%$ . On ALFWorld+Qwen3-8B, MemGen SFT adds only $1 . 6 \%$ more time compared with SFT (20.08 s vs 19.76 s) but increases accuracy from $8 3 . 5 9 \%$ to $8 5 . 8 2 \%$ . These results confirm that MemGen efficiently enhances reasoning performance without incurring significant inference delays.

Table 7 Average per-task inference time (seconds) and task performance (accuracy $\%$ ) across three benchmarks. Performance improvement $( \% )$ of MemGen SFT over vanilla LLM is also reported.   

<table><tr><td rowspan="2">Model &amp; Method</td><td colspan="2">KodCode</td><td colspan="2">ALFWorld</td><td colspan="2">TriviaQA</td></tr><tr><td>Time (s)</td><td>Acc (%)</td><td>Time (s)</td><td>Acc (%)</td><td>Time (s)</td><td>Acc (%)</td></tr><tr><td colspan="7">Qwen2.5-1.5B</td></tr><tr><td>Vanilla</td><td>11.96</td><td>24.55</td><td>21.17</td><td>22.54</td><td>2.18</td><td>32.10</td></tr><tr><td>SFT</td><td>2.01</td><td>55.83</td><td>10.79</td><td>36.57</td><td>1.98</td><td>63.84</td></tr><tr><td>MemGen SFT</td><td>2.94</td><td>58.16</td><td>12.94</td><td>40.30</td><td>2.05</td><td>65.02</td></tr><tr><td>Improvement over Vanilla</td><td>-</td><td>+33.61</td><td>-</td><td>+17.76</td><td>-</td><td>+32.92</td></tr><tr><td colspan="7">SmolLM-3B</td></tr><tr><td>Vanilla</td><td>13.12</td><td>37.05</td><td>34.82</td><td>18.96</td><td>4.26</td><td>10.47</td></tr><tr><td>SFT</td><td>3.04</td><td>59.25</td><td>12.88</td><td>32.36</td><td>3.05</td><td>55.25</td></tr><tr><td>MemGen SFT</td><td>3.48</td><td>62.65</td><td>14.69</td><td>50.60</td><td>3.16</td><td>68.13</td></tr><tr><td>Improvement over Vanilla</td><td>-</td><td>+25.60</td><td>-</td><td>+31.64</td><td>-</td><td>+57.66</td></tr><tr><td colspan="7">Qwen3-8B</td></tr><tr><td>Vanilla</td><td>16.99</td><td>49.10</td><td>55.42</td><td>58.93</td><td>8.70</td><td>52.18</td></tr><tr><td>SFT</td><td>7.24</td><td>64.75</td><td>19.76</td><td>83.59</td><td>5.98</td><td>74.55</td></tr><tr><td>MemGen SFT</td><td>7.56</td><td>66.15</td><td>20.08</td><td>85.82</td><td>6.25</td><td>77.22</td></tr><tr><td>Improvement over Vanilla</td><td>-</td><td>+17.05</td><td>-</td><td>+26.89</td><td>-</td><td>+25.04</td></tr></table>

# E Integration with Retrieval-based Memory

# E.1 Formalizing the Integration Process

While the primary mechanism of MemGen leverages the parametric knowledge encapsulated within the memory weaver $\mathcal { W } _ { \mathrm { w e a v e r } }$ , the framework is designed to be extensible, allowing for seamless integration with external, retrieval-based memory systems. This hybrid approach enables the weaver to synthesize latent memories that are informed by both its internalized experiences and a vast corpus of external information, thereby providing a richer and more comprehensive context to the reasoner $\pi \theta$ . Let $\mathcal { M } _ { \mathrm { e x t } }$ denote an external memory database, and let $\mathcal { R } ( \cdot )$ be a retrieval function that, given a natural language query, returns a set of relevant textual memory snippets. When the memory trigger $\tau _ { \mathrm { t r i g e r } }$ determines an invocation is necessary at the token-generation step $j$ (i.e., $d _ { j } = \mathrm { I N V O K E } )$ , the natural language text generated thus far serves as the query for the external memory system. This query, denoted as $q _ { t , j }$ , is produced by decoding the sequence of tokens generated up to that point:

$$
q _ { t , j } = \operatorname { D e c o d e } ( \mathbf { z } _ { t , < j } ) .
$$

The retrieval process is then formalized as:

$$
\mathcal { C } _ { t } = \mathcal { R } ( q _ { t , j } ; \mathcal { M } _ { \mathrm { e x t } } ) ,
$$

where $\mathcal { C } _ { t } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { P } \}$ is a set of $P$ retrieved textual snippets. These snippets are subsequently encoded into a sequence of embeddings, $\mathbf { E } _ { t } \in \mathbb { R } ^ { L _ { c } \times d _ { \mathrm { m o d e l } } }$ , where $L _ { c }$ is the total length of the encoded text. This allows the weaver to process the retrieved information in its native latent space.

This retrieved information is subsequently merged with the reasoner’s internal cognitive state $\mathbf { H } _ { t , < j }$ . The combined context is then fed into the memory weaver $\mathcal { W } _ { \mathrm { w e a v e r } }$ to produce the final latent memory. The invocation of the weaver, as described in Equation (5), is thus modified to:

$$
\mathbf { M } _ { t } = \mathcal { W } _ { \mathrm { w e a v e r } } \big ( \big [ \mathbf { H } _ { t , < j } ; \mathbf { E } _ { t } \big ] \big ) ,
$$

where $[ \cdot ; \cdot ]$ denotes the concatenation of the hidden state sequences. This integrated process allows $\mathcal { W } _ { \mathrm { w e a v e r } }$ to reconstruct both internal parametric knowledge and externally retrieved information into a compact, potent latent memory $\mathbf { M } _ { t }$ for the reasoner.

# E.2 Experimental Results

As shown in Table 8, even when MemGen’s own parametric memory is disabled (which means that merely the retrieved textual snippets are fed into $\mathcal { W } _ { \mathrm { w e a v e r } }$ ), MemGen significantly enhances the retrieval baseline, boosting performance on ALFWorld from $3 6 . 1 8 \%$ to $4 5 . 6 0 \%$ and on PopQA from $2 8 . 1 6 \%$ to $3 9 . 5 0 \%$ . This demonstrates that MemGen serves as a powerful synthesizer, not merely appending but actively reconstructing externally retrieved information into a more potent, task-relevant latent memory, thereby underscoring the necessity of a generative memory mechanism. Most notably, when the weaver’s internal parametric memory is enabled alongside external retrieval, performance is dramatically amplified, achieving scores of $7 5 . 9 0 \%$ on ALFWorld and $7 6 . 4 0 \%$ on TriviaQA. This highlights the powerful synergy achieved when the memory weaver is empowered to dynamically integrate its own internalized experiences with external textual knowledge.

Table 8 Performance results when integrating MemGen with a retrieval-based memory baseline, ExpeL. The backbone LLM is set as SmolLM3-3B.   

<table><tr><td>Method</td><td>ALFWorld</td><td>TriviaQA</td><td>PopQA</td></tr><tr><td>Vanilla LLM</td><td>18.96</td><td>10.47</td><td>8.23</td></tr><tr><td>ExpeL</td><td>36.18</td><td>46.20</td><td>28.16</td></tr><tr><td>MemGen + ExpeL (w/o parametric memory)</td><td>45.60</td><td>53.20</td><td>39.50</td></tr><tr><td>MemGen + ExpeL (w/ parametric memory)</td><td>75.90</td><td>76.40</td><td>60.23</td></tr></table>

# F Latent Memory Token Demonstration

# Cases from TriviaQA

# Case 1

# system prompt

Answer the given question. You must conduct reasoning inside <think> and $< / \mathrm { t h i n k } >$ first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and $<$ </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside $<$ answer $>$ and $< ,$ /answer>, without detailed illustrations. For example, $<$ <answer $>$ Beijing $<$ /answer>.

# user prompt

Which Lloyd Webber musical premiered in the US on 10th December 1993?

assistant output   
[UPPORT...’,eniable certif] $<$ think> I need to find out which Lloyd Webber musical premiered in the US on 10th December 1993. </think>   
[essengeryyyyMMddELCOME certif]<search $>$ Lloyd Webber musical premiered in the US on 10th December 1993 $<$ </search>

# user prompt

<observation $>$ Doc 1(Title: "Andrew Lloyd Webber") opened at the Adelphi Theatre in London on 12 July 1993, and ran for 1,529 performances ... </observation>

# assistant output

[essengeryyyyMMddELCOME certif]<think> The search results indicate that "Sunset Boulevard" is the Lloyd Webber musical that premiered in the US on 10th December 1993. $< .$ /think>   
[UPPORT...’,eniable certif]<answe $>$ Sunset Boulevard $<$ </answer>

# Case 2

# system prompt

Answer the given question. You must conduct reasoning inside <think> and $< / \mathrm { t h i n k } >$ first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information $>$ and $<$ </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside $<$ answer $>$ and </answer>, without detailed illustrations. For example, <answer> Beijing $< /$ answer>.

# user prompt

Who had a 70s No 1 hit with Kiss You All Over?

assistant output

[UPPORT...’,eniable certif]<think $>$ I need to search who had a 70s No 1 hit with the song "Kiss You All Over." </think> [essengeryyyyMMddELCOME certif]<search> who had a 70s No 1 hit with Kiss You All Over? </search>

# user prompt

<observation $>$ Doc 1(Title: "Kiss You All Over") Exile’s only big hit in the pop rock market (they would later have great success on the country music charts)... </observation>

assistant output

[essengerUPPORT’options certif]<think $>$ The song "Kiss You All Over" was a 70s No 1 hit with the band Exile. </think> [UPPORT...’,eniable certif]<answe $>$ Exile $<$ </answer>

# Case 3

# system prompt

Answer the given question. You must conduct reasoning inside <think> and $< / \mathrm { t h i n k } >$ first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query $<$ </search> and it will return the top searched results between <information $>$ and $< .$ /information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside $<$ answer $>$ and $<$ </answer>, without detailed illustrations. For example, <answer $>$ Beijing $< /$ /answer>.

# user prompt

What was the name of Michael Jackson’s autobiography written in 1988?

assistant output   
[UPPORTUPPORTeniable certif] $<$ think> I need to search for the name of Michael Jackson’s autobiography written in 1988. </think>   
[UPPORT...’,eniable certif]<search $>$ Michael Jackson autobiography 1988 $<$ </search>

user prompt

<observation $>$ Doc 1(Title: "Michael Jackson") of 4.4 million people. In 1988, Jackson released his only autobiography, ""Moonwalk "", which took four years to complete and sold 200,000 copies... </observation>

assistant output

[cmsUPPORT’options Certif] $<$ <think> The name of Michael Jackson’s autobiography written in 1988 is "Moonwalk". </think> [UPPORTUPPORTeniable certif] $<$ <answer $>$ Moonwalk $<$ </answer>

# Case 4

# system prompt

Answer the given question. You must conduct reasoning inside <think> and $< / \mathrm { t h i n k } >$ first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information $>$ and $<$ </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside $<$ answer> and $< ,$ /answer>, without detailed illustrations. For example, <answer $>$ Beijing $<$ /answer>.

# user prompt

Which volcano in Tanzania is the highest mountain in Africa?

assistant output

[UPPORT...’,eniable certif]<think> I need to search for the highest mountain in Africa, which is located in Tanzania. </think> [UPPORTUPPORTeniable certif] $<$ search $>$ highest mountain in Africa in Tanzania </search>

# user prompt

<observation> Doc 1(Title: "Mount Kilimanjaro") Mount Kilimanjaro Mount Kilimanjaro or just Kilimanjaro (), with its three volcanic cones, ""Kibo"", ""Mawenzi"", and ""Shira"", is a dormant volcano in Tanzania... </observation $>$

assistant output

[essengeryyyyMMddeniable certif] $<$ <think> The highest mountain in Africa located in Tanzania is Mount Kilimanjaro. </think> [UPPORT...’,eniable certif]<answe $>$ Mount Kilimanjaro $< /$ answer>

# Case 2

MODEL INPUT:

Write an efficient and correct Python function to solve the following problem.Question: Write a python function to find the first maximum and the second maximum number in an array. If the array has less than two unique elements, return ‘None‘ for the missing values.

# MODEL OUTPUT:

[keyword-kindërgetAs-slide]def find_first_max_second_max(arr): " I "

[.keywordetical] Returns the first and second maximum numbers in the array. If the array has less than two unique elements,[.keyword_pick] returns None for the missing values.   
[.keyword_pick] """ if $\mathtt { l e n } ( \mathtt { a r r } ) < 2$ :   
[.keyword_pick] return None,[.keyword_pick] None

[.keywordetical] unique_elements $=$ set(arr) [.keyword_pick] if len(unique_elements) $< 2$ : [.keywordetical] return None,[.keyword_pick] None [.keywordetical] first_max $=$ max(unique_elements) second_max $=$ max(unique_elements, default $=$ None)

return first_max, second_max $<$ |im_end|>

# Case 3

# MODEL INPUT:

Write an efficient and correct Python function to solve the following problem.Question: Given a list of integers, write a function that returns the length of the longest contiguous subarray where all elements have the same value. The function should also return the value of the elements in this subarray. Consider the output to be a tuple where the first element is the length of the longest subarray and the second element is the value of the elements in the subarray.

For example, consider the following list as input, numbers $=$ [1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1]

Expected Output: (4, 2)

MODEL OUTPUT:

[kindgetAs?-kindërgetAs-slide]def longest_subarray_with_same_value(numbers):

[.keyword_pick] Returns the length of the longest contiguous subarray where all elements [.keyword_pick] The function also returns the value of the elements in this subarray. [LTR_pick] """   
[LTRetical] if not numbers:   
[.keyword_pick] return (0, None)

max_length $= 1$ [.keyword_pick] current_length $= 1$ max_value $=$ numbers[0] current_value $=$ numbers[0]

[.keyword_pick] for i in range(1,[LTR_pick] len(numbers)):   
[LTR_pick] if numbers[i] $= =$ current_value: current_length $+ = 1$ else: if current_length $>$ max_length:   
[LTR_pick] max_length $=$ current_length max_value $=$ current_value current_length $= 1$ current_value $=$ numbers[i] if current_length $>$ max_length: max_length $=$ current_length max_value $=$ current_value

return (max_length, max_value) $<$ <|im_end|> [ even. s. even even ... even] Billy’s first 3 customers buy $3 * 1 = < < 3 * 1 = 3 > > 3$ DVDs. [-gallery.]His next 2 customers buy $2 * 2 = < < 2 * 2 { = } 4 > > 4$ DVDs. [-routing.]Billy sells $3 + 4 = < < 3 + 4 = 7 > > 7$ DVDs on Tuesday.\boxed{7} $<$ |im_end|>

# Case 2

MODEL INPUT:   
Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}.   
Question: Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?

# MODEL OUTPUT:

[..... a even even even... even ]Henry traveled $6 0 { - } 1 5 { = } { < } { < } 6 0 { - } 1 5 { = } 4 5 { > } { > } 4 5$ miles before his second stop.   
[-android.]He traveled 20 miles before his first stop,[ecies.] so he traveled $4 5 { - } 2 0 { = } { < } { < } 4 5 { - } 2 0 { = } 2 5 { > } { > } 2 5$ miles between his first and second stops.\boxed{25} $<$ |im_end|>   
[ even.]The final answer is $\mathtt { 2 5 < }$ <|im_end|>

# Case 3

MODEL INPUT:

Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}.Question: A basket contains 25 oranges among which 1 is bad, $20 \%$ are unripe, 2 are sour and the rest are good. How many oranges are good?

# MODEL OUTPUT:

[. even.. pre. even even even... even ]The number of unripe oranges is $2 0 / 1 0 0 * 2 5 = < < 2 0 / 1 0 0 * 2 5 = 5 > > 5$ [-animate.]The number of sour oranges is 2   
[-login.]The number of bad oranges is 1   
[-login.]The number of good oranges is $2 5 \textrm { - 5 - 2 - } 1 = < < 2 5 \ – 5 \ – 2 \mathrm { - } 1 = 1 7 > > 1 7 \setminus$ boxed{17} $<$ <|im_end|>

# Cases from GSM8K Cluster

# Cluster 1

Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}.Question: Josh decides to try flipping a house. He buys a house for $\$ 80,000$ and then puts in $\$ 50,000$ in repairs. This increased the value of the house by $1 5 0 \%$ . How much profit did he make?

[..... pre. even even even... even ]The house was worth $\$ 80$

Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}.Question: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week? [..... out. even even even.. even even ]He runs $3 * 6 0 = < < 3 * 6 0 = 1 8 0 > > 1 8 0$ meters a day

Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}.Question: Kylar went to the store to buy glasses for his new apartment. One glass costs $\$ 5$ , but every second glass costs only $60 \%$ of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?   
[.... pre... even... even ]The first 16 glasses cost $1 6 * 5 = \ S < < 1 6 * 5 = 8 0 > > 8 0$

Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}.Question: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?

even even. s. even even even.. even even ]Charleston has $4 * 2 0 = < < 4 * 2 0 = 8 0 > > 8 0$ sheep.

Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}.Question: John drives for 3 hours at a speed of $6 0 ~ \mathrm { { m p h } }$ and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the next half-hour driving at a speed of 30 mph, before being able to drive the remaining time of the 4 hours going at $8 0 ~ \mathrm { { m p h } }$ . How far is he from home at the end of those 4 hours?

[...... even even even... even ]He drives $6 0 { * } 3 = < < 6 0 { * } 3 = 1 8 0 > > 1 8 0$ miles in the first 3 hours.

# Cluster 2

Janet’s ducks lay 16 eggs per day,[-animate.] and she eats 3 eggs for breakfast every morning

Wendi feeds her 20 chickens 15 cups of feed in the morning,[-animate.] 25 cups in the afternoon

She sold half of what was left at the orange house,[-animate.] she had $_ { 1 / 2 }$ left.

$\mathbf { x } = 1 6 2 / 1 8 = < < 1 6 2 / 1 8 = 9 > > 9$ [-animate.]Allen’s age is $1 1 \mathbf { x } = 1 1 \mathbf { * } 9 = < < 1 1 \mathbf { * } 9 = 9 9 > > 9 9$ years old now.

He runs 20 miles a day because $6 0 / 3 { = } { < } < 6 0 / 3 { = } 2 0 { > } > 2 0$ [-animate.]He runs 10 miles the other two days because $2 0 / 2 { = } { < } < 2 0 / 2 { = } 1 0 { > } { > } 1 0$

# Cluster 3

It takes $2 / 2 { = } { < } { < } 2 / 2 { = } 1 { > } { > } 1$ bolt of white fiber.   
[ecies.]So it takes $2 + 1 = < < 2 + 1 = 3 > > 3$ bolts in total. The total cost of the five cell phones is $\$ 150\mathrm { ~ x ~ 5 ~ } =\ S < < 150 * 5 = 7 5 0 > > 7 5 0$ .   
[ even.]The interest for each unit is $\$ 150\mathrm { ~ x ~ }2/100 = \ S < < 150 * 2 / 100 = 3 > > 3$ . The second level is $1 6 { * } 2 { = } { < } { < } 1 6 { * } 2 { = } 3 2 { > } { > } 3 2$ square footage.   
[ even.]The third level is $3 2 * 2 { = } { < } { < } { < } 3 2 * 2 { = } 6 4 { > } { > } 6 4$ square footage. $20 \%$ of 20 students is $2 0 \mathrm { ~ x ~ } 2 0 / 1 0 0 = < < 2 0 * 2 0 / 1 0 0 = 4 > > 4$ students.   
[ecies.]The remaining students are $2 0 \textrm { - } 4 = < < 2 0 { - } 4 { = } 1 6 { > } { > } 1 6$ students.

If Artie rounds the price of petunias to the nearest dollar,[-animateecies] he will sell them for $\$ 2$ each.

# Cluster 4

The groomer offers a $30 \%$ discount,[-loginecies] so the cost of the grooming is 100 - $^ { \cdot } 1 0 0 ^ { \circ } \ast 0$ .[ecies.] $\left| 3 0 \right. = \ S { < } { < } 1 0 0 { - } 3 0 { = } 7 0 { > } { > } 7 0$

f Dana skips at 3 miles per hour,[-login.] then she runs at $3 * 4 = < < 3 * 4 = 1 2 > > 1 2$ miles per hour.

There were $1 4 0 * 2 = < < 1 4 0 * 2 = 2 8 0 > > 2 8 0$ emeralds in the chest.   
[-login.]There were $1 7 5 + 1 4 0 + 2 8 0 = < < 1 7 5 + 1 4 0 + 2 8 0 = 5 9 5 > > 5 9 5$ gems in the chest.

There are $3 0 + 6 0 = < < 3 0 + 6 0 = 9 0 > > 9 0$ students at Dr.[-loginecies.] Wertz’s school.

Cecilia will feed her puppy 1 cup of dog food every day for 180 days,[-login.] so she will use 180 cups of dog food in the first 180 days.

![](images/6c71a22f0fe5ecd63ffc524191699f1a9eda967b4e4ea0ecd5313995395d0036.jpg)  
Figure 11 ( $U p )$ t-SNE visualization of latent memories generated by MemGen $^ +$ SmolLM3-3B across datasets; (Down) Latent memory visualization within the TriviaQA and GSM8K datasets, clustered using $K$ -means.

# G Memory Functional Study

# G.1 Visualization Process

This section details the methodology employed to visualize the geometric structure of the latent memory (as shown in Figures 5 and 11) and to identify semantically distinct clusters within it. The process involves two primary stages: obtaining a fixed-dimensional representation for each memory sequence and applying dimensionality reduction and clustering algorithms.

Sequence Representation. Let $\mathcal { D } _ { \mathrm { m e m } } = \{ \mathbf { M } _ { 1 } , \mathbf { M } _ { 2 } , \dots , \mathbf { M } _ { Q } \}$ be a corpus of $Q$ latent memory sequences collected from MemGen’s inference, where each sequence $\mathbf { M } _ { i } = ( \mathbf { m } _ { i , 1 } , \dots , \mathbf { m } _ { i , K } ) \in \mathbb { R } ^ { K \times d _ { \mathrm { { m o d e l } } } }$ consists of $K$ token embeddings of dimension $d _ { \mathrm { m o d e l } }$ . To obtain a single, holistic vector representation for each sequence, we compute its mean embedding $\bar { \bf m } _ { i } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ :

$$
\bar { \mathbf { m } } _ { i } = \frac { 1 } { K } \sum _ { l = 1 } ^ { K } \mathbf { m } _ { i , l } .
$$

This procedure yields a set of high-dimensional representations $\bar { \mathcal { M } } = \{ \bar { \mathbf { m } } _ { 1 } , \bar { \mathbf { m } } _ { 2 } , . . . , \bar { \mathbf { m } } _ { Q } \}$ , which serves as the input for the subsequent analysis.

For visualization, we employed t-SNE (Maaten and Hinton, 2008) to project the high-dimensional set $\bar { \mathcal { M } }$ into a two-dimensional space. Formally, this mapping can be expressed as $\mathbf { y } _ { i } \in \mathbb { R } ^ { 2 } = f _ { \mathrm { t - S N E } } ( \bar { \mathbf { m } } _ { i } )$ . These 2D data points are then utilized for the scatter plot.

To quantitatively identify distinct functional groups, we applied the K-means algorithm directly to the high-dimensional mean embeddings in $\bar { \mathcal { M } }$ . This procedure partitions the memory representations into a predefined number of $N$ discrete clusters, $\mathcal { C } = \{ C _ { 1 } , C _ { 2 } , \ldots , C _ { N } \}$ (we set $N = 4$ ).

# G.2 Failure Taxonomy Definitions

To systematically analyze the functional impact of different memory clusters, we adopt and refine the failure taxonomy proposed by Song et al. (2025). Below, we provide precise definitions of each failure category in the context of LLM-based agent behavior, which guide our annotation and evaluation process:

• Planning Failure. This category refers to errors in high-level task decomposition and strategic decisionmaking. The agent either formulates subgoals that do not align with the original objective, fails to plan intermediate steps necessary for task completion, or misorders the reasoning sequence, leading to suboptimal or incomplete solutions.

• Compositional Reasoning. This type of failure occurs when the agent struggles to integrate multiple pieces of information or reasoning steps into a coherent solution.

• Tool Parsing Error. This failure occurs when the agent’s generated output cannot be parsed into a valid tool call. Typical patterns include malformed function signatures, missing required arguments, or unrequired arguments in the invocation, which prevent the external tool from being executed as intended.

• Tool Response Error. This category refers to errors that arise after a tool has been successfully invoked. Either the tool itself returns incorrect or incomplete information, or the agent misuses or misinterprets the returned content.

• Answer Formatting Failure. This category includes errors in the final presentation or structuring of the output, despite the reasoning process being largely correct. Examples include incorrect output format (e.g., unable to include the answers with in \boxed), violation of task-specific response templates, or missing required components in the final answer.

Demand Misunderstanding. This failure indicates that the agent has misinterpreted the user’s intent or the task specification. It may solve a different problem from the one posed, ignore key constraints, or pursue irrelevant objectives due to misunderstanding the instruction semantics.

• Think-Act Inconsistency. This type refers to a mismatch between the agent’s internal reasoning and its subsequent action in ReAct mode. The agent may articulate a correct reasoning chain but execute a contradictory action or produce a final answer inconsistent with its prior deliberation.

• False Belief captures cases where the agent maintains and reasons with inaccurate assumptions about the external environment, user state, or task context.

# G.3 Annotating Failure Modes and Filtering Latent Memory

Annotation of Failure Modes. We manually annotated agent failures on the TriviaQA dataset based on the eight failure modes discussed above. Each trajectory generated by the agent was evaluated by human annotators who assigned only one failure label if the agent’s behavior deviated from a successful path. Trajectories exhibiting failures that did not fall into these predefined categories were excluded from this specific analysis.

Inference-time Filtering of Latent Memory Clusters. The core of our intervention study involved selectively removing the influence of a specific latent memory cluster during the agent’s reasoning process. Our methodology is designed to be consistent with the clustering process itself, which operates on sequence-level representations.

Let $\mathcal { C } = \{ C _ { 1 } , C _ { 2 } , \ldots , C _ { N } \}$ be the set of $N$ latent memory clusters. These clusters were derived by applying $K$ - means to a collection of historical memory representations, where each representation is the mean embedding of an entire latent memory sequence. For each cluster $C _ { i } \in \mathcal { C }$ , we compute its centroid $\mu _ { i } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ by averaging these historical sequence representations within the cluster.

During inference, when the memory weaver synthesizes a new latent memory sequence $\mathbf { M } _ { t } = ( \mathbf { m } _ { t , 1 } , \ldots , \mathbf { m } _ { t , K } ) \in$ $\mathbb { R } ^ { K \times d _ { \mathrm { m o d e l } } }$ , we first compute its single-vector representation, $\mathbf { m } _ { \mathrm { n e w } }$ , by averaging its token embeddings:

$$
\bar { \bf m } _ { \mathrm { n e w } } = \frac { 1 } { K } \sum _ { l = 1 } ^ { K } { \bf m } _ { t , l } .
$$

To determine the semantic affiliation of this new sequence, we compare its mean embedding $\mathbf { m } _ { \mathrm { n e w } }$ against a comprehensive reference set ${ \mathcal { E } } _ { \mathrm { c o m p } } = \mathbf { E } _ { \mathrm { v o c a b } } \cup \{ \mu _ { 1 } , \dots , \mu _ { N } \}$ , where $\mathbf { E } _ { \mathrm { v o c a b } } \in \mathbb { R } ^ { V \times d _ { \mathrm { m o d e l } } }$ is the LLM’s vocabulary embedding matrix. We then identify the set of top- $k$ nearest neighbors to $\bar { \bf m } _ { \mathrm { n e w } }$ based on cosine similarity, denoted as $S _ { k } ( \bar { \mathbf { m } } _ { \mathrm { n e w } } )$ . In our experiments, we set $k = 1 0$ .

To ablate the influence of a target cluster $C _ { j }$ , the entire latent memory sequence $\mathbf { M } _ { t }$ is filtered (i.e., discarded   
and not prepended to the reasoner’s context) if the centroid of that cluster, , is present within this top- $k$ $\mu _ { j }$   
set. Formally, $\mathbf { M } _ { t }$ is filtered if:

$$
\mu _ { j } \in S _ { k } \big ( \bar { \mathbf { m } } _ { \mathrm { n e w } } \big ) .
$$

This sequence-level filtering allows us to precisely ablate the contribution of a specific learned memory function and observe its impact on agent behavior.