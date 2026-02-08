[← 返回 README](../README.md)

# 4 Functions: Why Agents Need Memory?

> 💡 **批注**: Section 4 回答 "Why agents need memory?" — Factual / Experiential / Working memory 三种功能分类超越了传统的 long-term/short-term 二分法。这个分类更贴近实际应用需求。


The transition from large language models as general-purpose, stateless text processors to autonomous, goal-directed agents is not merely an incremental step but a fundamental paradigm shift. This shift exposes the critical limitation of statelessness. By definition, an agent must persist, adapt, and interact coherently over time. Achieving this relies not merely on a large context window but fundamentally on the capacity for memory. This section addresses the functions, or fundamental purpose, of agent memory, prioritizing the question of why it is essential over how it is implemented. We posit that agent memory is not a monolithic component but a set of distinct functional capabilities, each serving a unique objective in enabling persistent, intelligent behavior.

To provide a systematic analysis, this section organizes the why of memory around a functional taxonomy that maps directly to an agent’s core requirements. At the highest level, we distinguish between two temporal categories: long-term memory, which serves as the persistent, cross-session store for accumulated knowledge, and short-term memory, which functions as the transient, in-session workspace for active reasoning. This high-level temporal split is further resolved into three primary functional pillars, which form the structure of our analysis. An overview of this taxonomy is provided in Figure 6.

# Three Primary Memory Functions

1. Factual Memory (Section 4.1): The agent’s declarative knowledge base, established to ensure consistency, coherence, and adaptability by recalling explicit facts, user preferences, and environmental states. This system answers the question: “What does the agent know?”

2. Experiential Memory (Section 4.2): The agent’s procedural and strategic knowledge, accumulated to enable continual learning and self-evolution by abstracting from past trajectories, failures, and successes. This system answers: “How does the agent improve?”

3. Working Memory (Section 4.3): The agent’s capacity-limited, dynamically controlled scratchpad for active context management during a single task or session. This system answers: “What is the agent thinking about now?”

These three memory systems are not isolated but form a dynamic, interconnected architecture that defines the agent’s cognitive loop. The cycle begins with encoding, in which the outcomes of the agent’s interactions, such as newly acquired facts or the results of a failed plan, are consolidated into long-term memory through summarization, reflection, or abstraction. Processing subsequently occurs within working memory, which functions as the active workspace for immediate inference. To support this reasoning, the system relies on retrieval to populate the workspace with relevant context and skills drawn from the persistent stores of factual and experiential memory. This encoding-processing-retrieval sequence constitutes the central architectural

# (a) Long-term Memory

![](images/255bacf068c2331e7f94d192e011221907fe1d772e143354ba0b790b86bc3bbf.jpg)  
Figure 6 The functional taxonomy of agent memory. We organize memory capabilities based on their functions (purpose) into three primary pillars spanning two temporal domains: (1) Factual Memory serves as a persistent declarative knowledge base to ensure interaction consistency, coherence, and adaptability; (2) Experiential Memory encapsulates procedural knowledge to enable continual learning and self-evolution across episodes; and (3) Working Memory provides mechanisms for the active management of transient context.

pattern enabling agents to learn from the past simultaneously and reason in the present.

# 4.1 Factual Memory

> 💡 **批注**: Factual memory 类比认知科学的 declarative memory，但在 agent 系统中实现为 processing continuum 而非 rigid dichotomy。从 episodic traces 到 semantic fact bases 的渐进转化过程是关键。


Factual memory refers to the capacity of an agent to store and retrieve explicit, declarative facts about past events, user-specific information, and the state of the external environment. This information encompasses a wide range of content, including dialogue history, user preferences, and relevant properties of the external world. By allowing the agent to exploit historical information when interpreting current inputs, factual memory serves as the cornerstone for context awareness, personalized responses, and extended task planning.

To understand the structural composition of agent memory, we draw upon the cognitive science framework of declarative memory (Riedel and Blokland, 2015). In neuroscience, declarative memory denotes long-term storage for information that can be consciously accessed and is commonly analyzed in terms of two major components: episodic and semantic memory (Squire, 2004). Episodic memory stores personally experienced events associated with specific temporal and spatial contexts—the what, where, and when of an episode (Tulving, 1972, 2002). Its central characteristic is the capacity to mentally re-experience past events. Semantic memory retains general factual knowledge, concepts, and word meanings independent of the specific occasion on which they were acquired (Squire, 2004). While supported by a unitary declarative system in the human brain, these components represent distinct levels of abstraction.

In agent systems, this biological distinction is operationalized not as a rigid dichotomy but as a processing continuum. Systems typically initiate this process by logging concrete interaction histories as episodic traces, such as dialogue turns, user actions, and environment states (Zhong et al., 2024; Wang et al., 2024h; Chhikara et al., 2025). Subsequent processing stages apply summarization (Wang et al., 2025h; Chen et al., 2025d), reflection (Tan et al., 2025c; Park et al., 2023; Wang et al., 2025h), entity extraction (Gutierrez et al., 2024), and fact induction (Rasmussen et al., 2025). The resulting abstractions are stored in structures such as vector databases (Zhong et al., 2024), key-value stores, or knowledge graphs (Rasmussen et al., 2025; Sun et al., 2024), governed by procedures for deduplication and consistency checking. Through this sequence, raw event streams are gradually transformed into reusable semantic fact bases.

Functionally, this architecture ensures that the agent exhibits three fundamental properties during interaction: consistency, coherence, and adaptability.

• Consistency implies stable behavior and self-presentation over time. By maintaining a persistent internal state regarding user-specific facts and its own commitments, the agent avoids contradictions and arbitrary changes of stance.   
• Coherence is reflected in robust context awareness. The agent can recall and integrate relevant interaction history, refer to past user inputs, and preserve topical continuity, ensuring responses form a logically connected dialogue rather than isolated utterances.   
• Adaptability demonstrates the ability to personalize behavior based on stored user profiles and historical feedback. Consequently, response style and decision-making progressively align with the user’s specific needs and characteristics.

For exposition, we further organize factual memory according to the primary entity it refers to. This entitycentric taxonomy, together with representative methods and their technical design choices, is systematically summarized in subsection 4.1. This perspective highlights two central application domains:

# Two Types of Factual Memory

• User factual memory (Section 4.1.1) denotes facts that sustain the consistency of interactions between humans and agents, including identities, stable preferences, task constraints, and historical commitments.

• Environment factual memory (Section 4.1.2) denotes facts that sustain consistency with respect to the external world, such as document states, resource availability, and the capabilities of other agents.

Table 4 Taxonomy of factual memory methods. We categorize existing works based on the primary target entity: User Factual Memory focuses on sustaining interaction consistency, while Environment Factual Memory ensures consistency with the external world. Methods are compared across three technical dimensions: (1) Carrier (Section 3) identifies the storage medium, (2) Structure follows the taxonomy of token-level memory (Section 3.1), and (3) Optimization denotes the integration strategy, where $P E$ encompasses prompt engineering and inference-time techniques without parameter updates, distinct from gradient-based methods like $S F T '$ and $R L$ .   

![Table](images/abbf0461d9c6f3e19cf77547fc7f5527c1911825c49ec04d17648c3e3a1a316f.jpg)

Continued on next page

Table 4 Taxonomy of factual memory methods. (continued)   

![Table](images/c9e2c49149422a1148ee5fcbfc470f12cff6c14aa7effd2149b5d0c0a418ca35.jpg)

II. Environment factual Memory   

![Table](images/f9a866f3f7be3ba0df83079551ce6a61c8c0eb45508c01f8c77bedb65aae04c0.jpg)

# 4.1.1 User factual memory

User factual memory persists verifiable facts about a specific user across sessions and tasks, including identity, preferences, routines, historical commitments, and salient events.

Its primary function is to prevent characteristic failure modes of stateless interaction, such as coreference drift, repeated elicitation, and contradictory responses, thereby reducing interruptions to long-horizon goals (Tan et al., 2025c; Zhong et al., 2024). Engineering practice typically comprises selection and compression, structured organization, retrieval and reuse, and consistency governance, aiming to sustain long-range dialogic and behavioral coherence under bounded access cost.

Dialogue Coherence Dialogue coherence requires an agent to preserve conversational context, user-specific facts, and a stable persona over extended periods. This ensures that later turns remain sensitive to earlier disclosures and affective cues, rather than degrading into repeated clarifications or inconsistent replies. To achieve this, modern systems implement user factual memory through two complementary strategies: heuristic selection and semantic abstraction.

To navigate finite context windows efficiently, a primary strategy is to selectively retain and rank interaction histories. Rather than retaining all raw logs, systems (Xi and Wang, 2025; Zhong et al., 2024; Park et al., 2023; Lei et al., 2025) maintain structured stores of past interactions, ranking entries by metrics such as relevance, recency, importance, or distinctiveness. By filtering retrieval based on these scores, high-value items are preserved and periodically condensed into higher-level summaries, conditioning subsequent responses to maintain continuity without overwhelming the agent’s working memory.

Beyond mere selection, advanced frameworks emphasize the transformation and abstraction of raw dialogue fragments into higher-level semantic representations. Approaches such as Think in Memory (Liu et al., 2023a) and Reflective Memory Management (Tan et al., 2025c) convert raw interaction traces into thought representations or reflections via iterative update operations. This allows the agent to query a stable semantic memory, keeping later replies topically consistent and less repetitive. Similarly, COMEDY (Chen et al., 2025d) employs a single language model to generate, compress, and reuse memory while updating compact user profiles. These methods effectively stabilize persona and preference expression over long conversational histories by decoupling memory storage from the raw token surface form.

Goal Consistency Goal consistency requires an agent to maintain and refine an explicit task representation over time. This ensures that clarifying questions, information requests, and actions remain strictly aligned with the primary objective, minimizing intent drift.

To mitigate such drift, systems utilize factual memory to dynamically track and update the task state. Approaches like RecurrentGPT (Zhou et al., 2023b), Memolet (Yen and Zhao, 2024), and MemGuide (Du et al., 2025b) retain confirmed information while highlighting unresolved elements. By guiding retrieval based on task intent, these methods help agents satisfy missing constraints and maintain focus across sessions.

For complex, long-horizon tasks, memory forms are often structured to facilitate localized retrieval centered on the active goal (Wu et al., 2025h). For instance, A-Mem (Xu et al., 2025c) organizes memories as an interconnected graph of linked notes, while H-Mem (Limbacher and Legenstein, 2020) employs associative mechanisms to recall prerequisite facts when subsequent steps depend on prior observations.

In embodied scenarios, factual memory grounds agent behavior in user-specific habits and environmental context. Systems such as M3-Agent (Long et al., 2025) and MEMENTO (Kwon et al., 2025) persist data on household members, object locations, and routines, reusing this information to minimize redundant exploration and repeated instructions. Similarly, Encode-Store-Retrieve (Shen et al., 2024) processes egocentric visual streams into text-addressable entries, allowing agents to answer questions based on past visual experiences without requiring user repetition.

Summary Collectively, these mechanisms transform ephemeral interaction traces into a persistent cognitive substrate. By integrating retrieval-based ranking with generative abstraction, user factual memory upgrades the system from simple similarity matching to the active maintenance of explicit goals and constraints. This foundation yields a dual benefit: it fosters a sense of familiarity and trust through long-term behavioral coherence, while simultaneously enhancing operational efficiency by increasing task success rates, reducing redundancy, and lowering error recovery overhead.

# 4.1.2 Environment factual memory

Environment factual memory pertains to entities and states external to the user, encompassing long documents, codebases, tools, and interaction traces.

This memory paradigm addresses incomplete factual recall and unverifiable provenance, minimizes contradictions and redundancy in multi-agent collaboration, and stabilizes long-horizon tasks in heterogeneous environments. The central objective is to furnish an updatable, retrievable, and governable external fact layer, providing a stable reference across sessions and stages. Concretely, we categorize existing implementations along two complementary dimensions: knowledge persistence and multi-agent shared access.

Knowledge Persistence Knowledge memory refers to persistent representations of world knowledge and domain-specific knowledge that support long document analysis, factual question answering, multihop reasoning, and reliable retrieval of code and data resources.

In terms of knowledge organization, existing research focuses on structuring external data to enhance reasoning capabilities. For instance, HippoRAG (Gutierrez et al., 2024) utilizes knowledge graphs to facilitate evidence propagation, while MemTree (Rezazadeh et al., 2025c) employs a dynamic hierarchical structure to optimize aggregation and targeted access in growing corpora. Regarding storage form, LMLM (Zhao et al., 2025b) explicitly decouples factual knowledge from model weights by externalizing it into a database, thereby enabling direct knowledge edits and provenance verification without retraining. In narrative domains, CALYPSO (Zhu et al., 2023) distills lengthy game contexts into bite-sized prose, preserving critical story state accessibility.

In scenarios requiring continuous knowledge updates, parameter-centric approaches integrate persistence directly into the model architecture. Methods such as MEMORYLLM (Wang et al., 2024j), M+ (Wang et al., 2025n), and WISE (Wang et al., 2024e) incorporate trainable memory pools or side networks to absorb new information. Rather than relying solely on static external retrieval, these designs focus on the challenge of model editing, allowing agents to adapt to dynamic environments and correct obsolete facts while preserving the stability of the pre-trained backbone.

Shared Access Shared memory establishes a visible and manageable common factual foundation for multiagent collaboration, serving to align goals, carry intermediate artifacts, and eliminate redundant work. By maintaining a centralized repository of past queries and responses, frameworks such as Memory Sharing (Gao and Zhang, 2024b) enable agents to access and build on peers’ accumulated insights asynchronously. This mechanism ensures that individual agents directly benefit from collective knowledge, thereby suppressing contradictory conclusions and enhancing overall system efficiency.

For complex project coordination, systems such as MetaGPT (Hong et al., 2024) and GameGPT (Chen et al., 2023b) utilize shared message pools as central workspaces for publishing plans and partial results. Similarly, G-Memory (Zhang et al., 2025e) employs hierarchical memory graphs as a unified coordination medium. These architectures facilitate consistency maintenance around the current project state, which reduces communication overhead and enables the extraction of reusable workflows from historical collaborations.

In the domain of social simulation, platforms like Generative Agents (Park et al., 2023) and $\mathrm { S ^ { 3 } }$ (Gao et al., 2023a), alongside large-scale simulators such as OASIS (Yang et al., 2025) and AgentSociety (Piao et al., 2025), model the global environment and public interaction logs as a shared memory substrate. This substrate is incrementally updated and observed by the population, allowing information to diffuse naturally among agents and supporting coherent, history-aware social dynamics at scale.

Summary environment factual memory furnishes a continuously updatable, auditable, and reusable external fact layer. On the knowledge axis, it improves completeness, interpretability, and editability of factual recall through structured organization and long-term memory modules. On the collaboration axis, it maintains crossagent and cross-stage consistency through sharing and governance, thereby enabling robust decision-making and execution under long horizons, multiple actors, and multi-source information.

![](images/714d35abf490706e20697390a290ed69b48db6a9691e9ffb6122c7f75b50ee3a.jpg)  
Figure 7 Taxonomy of experiential memory paradigms. We classify approaches based on the abstraction level of stored knowledge: (1) Case-based Memory preserves raw trajectories and solutions as concrete exemplars; (2) Strategybased Memory abstracts experiences into high-level strategies, templates, or workflows; (3) Skill-based Memory distills procedural knowledge into executable functions and APIs; and (4) Hybrid Memory integrates multiple representations. Together, these systems mirror human procedural memory to enable continual learning and self-evolution. This figure draws inspiration from Gao et al. (2025).

# 4.2 Experiential Memory

> 💡 **批注**: Experiential memory 是 agent self-evolution 的基础。Case-based → Strategy-based → Skill-based 的抽象层次递进非常清晰：从原始轨迹到可迁移策略再到可执行技能。这直接对应了 Sutton 'Era of Experience' 的理念。


Experiential memory encapsulates the mechanism by which agents encode historical trajectories, distilled strategies, and interaction outcomes into durable, retrievable representations. Unlike working memory, which manages transient context, experiential memory focuses on the long-term accumulation and transfer of knowledge across distinct episodes.

Theoretically grounded in cognitive science, this paradigm parallels human nondeclarative memory, specifically the procedural and habit systems (Squire, 2004; Seger and Spiering, 2011). Biological systems rely on distributed neural circuits for implicit skill acquisition (Reber, 2013). In contrast, agentic experiential memory typically employs explicit data structures, such as vector databases or symbolic logs. This implementation difference grants agents a unique capability absent in biological counterparts: the ability to introspect, edit, and reason over their own procedural knowledge.

Crucially, experiential memory serves as a foundation for continual learning and self-evolution in the era of experience (Sutton, 2025; Gao et al., 2025). By maintaining a repository of structured experiences, agents achieve a non-parametric path to adaptation and avoid the prohibitive costs of frequent parametric updates. This mechanism effectively closes the learning loop by converting interaction feedback into reusable knowledge. Through this process, agents rectify past errors, abstract generalizable heuristics, and compile routine behaviors. Consequently, such adaptation minimizes redundant computations and refines decision-making over time (Zhao et al., 2024; Shinn et al., 2023b).

To systematically analyze existing literature, we classify experiential memory based on the abstraction level of the stored information. An overview of this abstraction-based taxonomy and representative paradigms is illustrated in Figure 7. Representative methods under this abstraction-based taxonomy, together with their storage carriers, representation forms, and optimization strategies, are summarized in Table 5.

# Three Types of Experiential Memory

• Case-based Memory (Section 4.2.1) stores minimally processed records of historical episodes, prioritizing high informational fidelity to support direct replay and imitation. By retaining the original alignment between situations and outcomes, it serves as a repository of concrete, verifiable evidence that functions as in-context exemplars for evidence-driven learning.

• Strategy-based Memory (Section 4.2.2) distills transferable reasoning patterns, workflows, and high-level insights from past trajectories to guide planning across diverse scenarios. Acting as a cognitive scaffold, it decouples decision-making logic from specific contexts, thereby enhancing cross-task generalization and constraining the search space for complex reasoning.

• Skill-based Memory (Section 4.2.3) encapsulates executable procedural capacities, ranging from atomic code snippets to standardized API protocols, that operationalize abstract strategies into verifiable actions. This category serves as the agent’s active execution substrate, enabling the modular expansion of capabilities and the efficient handling of tool-use environments.

Table 5 Taxonomy of experiential memory methods. We categorize existing works based on the abstraction level of stored knowledge: Case-based Memory preserves raw records for direct replay, Strategy-based Memory distills abstract heuristics for planning, and Skill-based Memory compiles executable capabilities for action. Methods are compared across three technical dimensions: (1) Carrier (Section 3) identifies the storage medium, (2) Form specifies the representation format of the experience, and (3) Optimization denotes the integration strategy, where $P E$ encompasses prompt engineering and inference-time techniques without parameter updates, distinct from gradient-based methods like $S F T$ and $R L$ .   

![Table](images/fde9fb80d823a477eebab986c56c60e6aff385b743a6892fe059e1303f152478.jpg)

Continued on next page

Table 5 Taxonomy of experiential memory methods. We categorize existing works based on the abstraction level of stored knowledge: Case-based Memory preserves raw records for direct replay, Strategy-based Memory distills abstract heuristics for planning, and Skill-based Memory compiles executable capabilities for action. (continued)   

![Table](images/0088bd60f7037110fa180a7d60357691522385111779081acaed2b69ce7a8717.jpg)

# 4.2.1 Case-based Memory

Case-based memory stores minimally processed records of historical events, prioritizing fidelity to ensure that episodes can be replayed or reused as in-context exemplars. Unlike strategy templates or skill modules, cases avoid extensive abstraction, thereby preserving the original alignment between situations and solutions.

Trajectories This category preserves interaction sequences to enable replay and evidence-driven learning. To optimize retrieval in text-based environments, Memento (Zhou et al., 2025a) employs soft Q-learning to dynamically refine the probability of selecting high-utility past trajectories. In multimodal settings, JARVIS1 (Wang et al., 2025q), EvoVLA (Liu et al., 2025i) and Auto-scaling Continuous Memory (Wu et al., 2025e) retain visual context, with the former storing survival experiences in Minecraft and the latter compressing GUI history into continuous embeddings. Furthermore, the early experience paradigm (Zhang et al., 2025k) constructs reward-free, agent-generated interaction traces and integrates them into model parameters via mid-training to enhance generalization.

Solutions This category treats memory as a repository of proven solutions. ExpeL (Zhao et al., 2024) autonomously gathers experience through trial-and-error, storing successful trajectories as exemplars while extracting textual insights to guide future actions. Synapse (Zheng et al., 2024a) similarly injects abstracted state-action episodes as contextual examples to align problem-solving patterns. In program synthesis, MapCoder (Islam et al., 2024) keeps relevant example code as a playbook-like case that multi-agent pipelines retrieve and adapt to improve reliability on complex tasks. In the financial domain, FinCon (Yu et al., 2024) maintains an episodic memory of past actions, PnL trajectories, and belief updates to facilitate robust cross-round decision-making.

Summary Case-based memory offers high informational fidelity and provides verifiable evidence for imitation. However, the reliance on raw data imposes challenges regarding retrieval efficiency and context window consumption. Distinguished from executable skills or abstract strategies, cases do not encompass orchestration logic or function interfaces. Instead, they serve as the factual substrate upon which higher-level reasoning operates.

# 4.2.2 Strategy-based Memory

Unlike case libraries that retain what happened, strategy-based memory extracts transferable knowledge of how to act, encompassing reusable reasoning patterns, task decompositions, insights, abstractions, and cross-situational workflows. It elevates experiences into editable, auditable, and composable high-level knowledge, thereby reducing dependence on lengthy trajectory replay and improving cross-task generalization and efficiency. We focus on non-code or weakly code-based templates and workflows in this section, while executable functions, APIs, MCP protocols, and code snippets are classified under Section 4.2.3. Based on the granularity and structural complexity of the retained knowledge, we categorize strategy-based memory into three distinct types: atomic Insights, sequential Workflows, and schematic Patterns.

Insights This category of approaches focuses on distilling discrete pieces of knowledge, such as granular decision rules and reflective heuristics, from past trajectories. $\mathrm { H ^ { 2 } R }$ (Ye et al., 2025b) explicitly decouples planning-level and execution-level memories, enabling high-level planning insights and low-level operational rules to be retrieved separately for fine-grained transfer in multi-task scenarios. R2D2 (Huang et al., 2025c) integrates remembering, reflecting, and dynamic decision-making for web navigation, deriving corrective insights from both failed and successful cases to inform subsequent episodes. For long-horizon web automation, BrowserAgent (Yu et al., 2025d) persists key conclusions as explicit memory to stabilize extended chains of reasoning and mitigate context drift.

Workflows Distinct from atomic, static insights, workflows encapsulate strategies as structured sequences of actions—executable routines abstracted from prior trajectories to guide multi-step execution at inference time. Agent Workflow Memory (AWM) (Wang et al., 2024m) induces reusable workflows on Mind2Web (Deng et al., 2023) and WebArena (Zhou et al., 2023a) and uses them as high-level scaffolds to guide subsequent generation, improving success rates and reducing steps without updating base model weights. This demonstrates that strategy templates can act as a top-level controller that complements case-level evidence. Agent KB (Tang et al., 2025d) establishes a unified knowledge base that treats workflows as transferable procedural knowledge. It employs hierarchical retrieval, accessing workflows first to structure the strategic approach and enabling problem-solving logic reuse across diverse agent architectures.

Patterns At a higher level of abstraction, reasoning patterns function as cognitive templates that encapsulate the structure of problem-solving, enabling agents to tackle complex reasoning tasks by instantiating these generalizable skeletons. Buffer of Thoughts (Yang et al., 2024b) maintains a meta-buffer of thought templates that are retrieved and instantiated to solve new problems. Similarly, ReasoningBank (Ouyang et al., 2025) abstracts both successes and failures into reusable reasoning units, facilitating test-time expansion and robust learning. RecMind’s self-inspiring planning algorithm (Wang et al., 2024h) generates intermediate self-guidance to structure subsequent planning and tool use. In the domain of dialogue agents, PRINCIPLES (Kim et al., 2025a) builds a synthetic strategy memory via offline self-play to guide strategy planning at inference, thereby eliminating the need for additional training. These advances indicate a paradigmatic shift from descriptive rules to portable reasoning structures.

Summary Strategy-based memory, which encompasses insights, workflows, and patterns, serves as a highlevel scaffold to guide generative reasoning. Unlike case-based memory that relies on retrieving specific, raw trajectories which may be noisy or context-dependent, this form of memory distills generalizable schemas to effectively constrain the search space and improve robustness on unseen tasks. However, a key distinction is that these strategies function as structural guidelines rather than executable actions; they direct the planning process but do not interact with the environment directly. This limitation necessitates skill-based memory, discussed in the following section, which stores callable capabilities and tools. Ultimately, robust agents typically synergize these components: strategies provide the abstract planning logic, while skills handle the grounded execution.

# 4.2.3 Skill-based Memory

Skill memory captures an agent’s procedural capacity and operationalizes abstract strategy into verifiable actions. It encodes what the agent can do, complements declarative knowledge of what the agent knows, and anchors the perception–reasoning–action loop by providing invocable, testable, and composable executables. Recent evidence shows that language models can learn when and how to call tools and scale reliably with large tool repertoires, establishing skill memory as the execution substrate of modern agents.

Skill memory spans a continuum from internal, fine-grained code to externalized, standardized interfaces. The unifying criteria are straightforward: skills must be callable by the agent, their outcomes must be verifiable to support learning, and they must compose with other skills to form larger routines.

Code Snippets Executable code stored as reusable snippets offers the fastest path from experience to capability. In open-ended tasks, agents distill successful sub-trajectories into interpretable programs and reuse them across environments. Voyager (Wang et al., 2024b) exemplifies this pattern with an ever-growing skill library; the Darwin Gödel Machine (Zhang et al., 2025i) goes further by safely rewriting its own code under empirical validation, yielding self-referential and progressively more capable skill sets.

Functions and Scripts Abstracting complex behaviors into modular functions or scripts enhances reusability and generalization. Recent advancements empower agents to autonomously create specialized tools for problemsolving (Qian et al., 2023; Yuan et al., 2024a), and to refine tool-use capabilities through demonstrations and environmental feedback across diverse domains such as mobile GUIs, web navigation, and software engineering (Fang et al., 2025d; Zheng et al., 2025a; Bouzenia et al., 2024). Furthermore, emergent mechanisms for procedural memory enable agents to distill execution trajectories into retrievable scripts, facilitating efficient generalization to novel scenarios (Liu et al., 2025b; Han et al., 2025a).

APIs APIs serve as the universal interface for encapsulated skills. While earlier work focused on fine-tuning models to correctly invoke tools (Schick et al., 2023; Patil et al., 2024), the exponential growth of API libraries has shifted the primary bottleneck to retrieval. Standard information retrieval methods often fail to capture the functional semantics of tools (Shi et al., 2025c). Consequently, recent approaches have moved towards learningbased retrieval and reranking strategies that account for tool documentation quality, hierarchical relationships, and collaborative usage patterns to bridge the gap between user intent and executable functions (Zheng et al., 2024b; Gao and Zhang, 2024c; Qu et al., 2024, 2025a).

MCPs To reduce protocol fragmentation in API-based ecosystems, the Model Context Protocol provides an open standard that unifies how agents discover and use tools and data, including code-execution patterns that load tools on demand and cut context overhead (Qiu et al., 2025c,b). Broad platform support indicates a convergence toward a common interface layer.

Beyond standard executables, research explores learnable memories of tool capabilities to handle uncertain neural tools, parametric integration that embeds tool symbols to unify retrieval and calling, and architectureas-skill perspectives where specialized agents are callable modules within a modular design space (Xiao et al., 2025b; Wang et al., 2025i; Zhao et al., 2025a). Collectively, these strands reframe skill memory as a learnable, evolving, and orchestrable capability layer.

Summary In conclusion, skill-based memory constitutes the active execution substrate of the agent, evolving from static code snippets and modular scripts to standardized APIs and learnable architectures. It bridges the gap between abstract planning and environmental interaction by operationalizing insights from case-based and strategy-based memories into verifiable procedures. As mechanisms for tool creation, retrieval, and interoperability (e.g., MCP) mature, skill memory moves beyond simple storage, enabling a continuous loop of capability synthesis, refinement, and execution that drives open-ended agent evolution.

# 4.2.4 Hybrid memory

Advanced agent architectures increasingly adopt a hybrid design that integrates multiple forms of experiential memory to balance grounded evidence with generalizable logic. By maintaining a spectrum of knowledge spanning raw episodes, distilled rules, and executable skills, these systems dynamically select the most appropriate memory format, ensuring both retrieval precision and broad generalization across contexts.

A prominent direction involves coupling case-based and strategy-based memories to facilitate complementary reasoning. For example, ExpeL (Zhao et al., 2024) synergizes concrete trajectories with abstract textual insights, allowing agents to recall specific solutions while applying general heuristics. Agent KB (Tang et al., 2025d) employs a hierarchical structure where high-level workflows guide planning and specific solution paths provide execution details. Similarly, R2D2 (Huang et al., 2025c) integrates a replay buffer of historical traces with a reflective mechanism that refines decision strategies from past errors, effectively bridging case retrieval and strategic abstraction. Complementing these, Dynamic Cheatsheet (Suzgun et al., 2025) prevents redundant computation by storing accumulated strategies and problem-solving insights for immediate reuse at inference time.

Furthermore, recent frameworks strive to unify the lifecycle of memory, incorporating Skill-based components or establishing comprehensive cognitive architectures (Sun et al., 2025a; Cai et al., 2025a). In scientific reasoning, ChemAgent (Tang et al., 2025c) constructs a self-updating library that pairs execution cases with decomposable skill modules, enabling the model to refine its chemical reasoning through accumulated experience. Taking a holistic approach, LARP (Yan et al., 2023) establishes a cognitive architecture for open-world games that harmonizes semantic memory for world knowledge, episodic memory for interaction cases, and procedural memory for learnable skills, ensuring consistent role-playing and robust decision-making. Finally, evolutionary systems like G-Memory (Zhang et al., 2025c) and Memp (Fang et al., 2025d) implement dynamic transitions, where repeated successful cases are gradually compiled into efficient skills, automating the shift from heavy retrieval to rapid execution. A recent effort, MemVerse (Liu et al., 2025e) combines both parametric memory and token-level prcedural memory.

# 4.3 Working Memory

> 💡 **批注**: Working memory 的定义从认知科学引入：capacity-limited, dynamically controlled mechanism。关键insight是当前 LLM 的 context window 只是 passive buffer，缺乏 active control — 这是工程上需要解决的核心问题。


In cognitive science, working memory is defined as a capacity-limited, dynamically controlled mechanism that supports higher-order cognition by selecting, maintaining, and transforming task-relevant information in the moment (Baddeley, 2012). Beyond mere temporary storage, it implies active control under resource constraints. This perspective is grounded in frameworks such as the multicomponent model and the embedded-processes account, both of which emphasize attentional focus, interference control, and bounded capacity (Cowan, 2014).

When transposed to LLMs, the standard context window functions primarily as a passive, read-only buffer. Although the model can consume the window’s contents during inference, it lacks explicit mechanisms to select, sustain, or transform the current workspace dynamically. Recent behavioral evidence suggests that current models do not exhibit human-like working memory characteristics, underscoring the necessity for explicitly engineered, operable working memory mechanisms (Huang et al., 2025a).

Throughout this section, we define working memory as the set of mechanisms for the active management and manipulation of context within a single episode (Zhang et al., 2025r). The objective is to transform the context window from a passive buffer into a controllable, updatable, and interference-resistant workspace. This transition offers immediate benefits: it increases the density of task-relevant information under fixed attention budgets, suppresses redundancy and noise, and enables the rewriting or compression of representations to preserve coherent chains of thought. We categorize these mechanisms based on the interaction dynamics.

Representative working memory approaches under this interaction-based taxonomy, together with their storage carriers, task domains, and optimization strategies, are systematically summarized in Table 6.

# Two Types of Working Memory

• Single-turn Working Memory (Section 4.3.1) focuses on input condensation and abstraction. In this setting, the system must process massive immediate inputs such as long documents or highdimensional multimodal streams within a single forward pass. The goal is to dynamically filter and rewrite evidence to construct a bounded computational scratchpad, thereby maximizing the effective information payload per token.

• Multi-turn Working Memory (Section 4.3.2) addresses temporal state maintenance. In sequential interactions, the challenge is to prevent historical accumulation from overwhelming the attention mechanism. This involves maintaining task states, goals, and constraints through a continuous loop of reading, executing, and updating, ensuring that intermediate artifacts are folded and consolidated across turns.

In summary, working memory for LLMs represents a paradigm shift towards active, within-episode context management. By aligning with the cognitive requirement of active manipulation, it suppresses interference and provides a practical solution to the engineering constraints of long-context inference.

# 4.3.1 Single-turn Working Memory

Single-turn working memory addresses the challenge of processing massive immediate inputs, including long documents (Chevalier et al., 2023) and high-dimensional multimodal streams (Wang et al., 2024g), within a single forward pass. Rather than passively consuming the entire context, the objective is to actively construct a writable workspace. This involves filtering and transforming raw information to increase density and operability under fixed attention and memory budgets (Jiang et al., 2023, 2024). We categorize these mechanisms into input condensation, which reduces physical token count, and observation abstraction, which transforms data into structured semantic representations.

Input Condensation Input condensation techniques aim to preprocess the context to minimize token usage while preserving essential information (Jiang et al., 2023). These methods generally fall into three paradigms: hard, soft, and hybrid condensation (Liao et al., 2025a).

Hard condensation discretely selects tokens based on importance metrics. Approaches like LLMLingua (Jiang et al., 2023) and LongLLMLingua (Jiang et al., 2024) estimate token perplexity to discard predictable or task-irrelevant content, while CompAct (Yoon et al., 2024) adopts an iterative strategy to retain segments that maximize information gain. Although efficient, hard selection risks severing syntactic or semantic dependencies. Soft condensation encodes variable-length contexts into dense latent vectors (memory slots). Methods such as Gist (Mu et al., 2023), In-Context Autoencoder (ICAE) (Ge et al., 2024), and AutoCompressors (Chevalier et al., 2023) train models to compress prompts into valid summary tokens or distinct memory embeddings. This achieves high compression ratios but requires additional training and may obscure fine-grained details. Hybrid approaches like HyCo2 (Liao et al., 2025a) attempt to reconcile these trade-offs by combining global semantic adapters (soft) with token-level retention probabilities (hard).

Observation Abstraction While condensation focuses on reduction, observation abstraction aims to transform raw observations into structured formats that facilitate reasoning. This mechanism maps dynamic, high-dimensional observation spaces into fixed-size memory states, preventing agents from being overwhelmed by raw data.

In complex interactive environments, abstraction converts verbose inputs into concise state descriptions. Synapse (Zheng et al., 2024a) rewrites unstructured HTML DOM trees into task-relevant state summaries to guide GUI automation. Similarly, in multimodal settings, processing every frame of a video stream is computationally prohibitive. Working memory mechanisms address this by extracting semantic structures: Context as Memory (Yu et al., 2025b) filters frames based on field-of-view overlap, VideoAgent (Wang et al.,

Table 6 Taxonomy of working memory methods. We categorize approaches into Single-turn and Multi-turn settings based on interaction dynamics. Methods are compared across three technical dimensions: (1) Carrier (Section 3) identifies the storage medium, (2) Task specifies the evaluation domain or application scenario, and (3) Optimization denotes the integration strategy, where PE encompasses prompt engineering and inference-time techniques without parameter updates, distinct from gradient-based methods like SFT and RL.   

![Table](images/82af4535ed593467054f314ec858ec690c8ffeb09621d3f8a3dac9802cef4ca8.jpg)

2024g) converts streams into temporal event descriptions, and MA-LMM (He et al., 2024) maintains a bank of visual features. These methods effectively rewrite high-dimensional, redundant streams into low-dimensional, semantically rich representations operable within a limited context window for efficient processing.

Summary Single-turn working memory functions as an active compression layer that maximizes the utility of the context window for immediate reasoning. By employing input condensation and observation abstraction, these mechanisms effectively increase the information density of the operational workspace, ensuring that critical evidence is retained despite capacity constraints. However, this optimization is strictly intra-turn; it addresses the breadth and complexity of static inputs rather than the temporal continuity of dynamic interactions.

# 4.3.2 Multi-turn Working Memory

Multi-turn working memory addresses a fundamentally different problem space than the single-turn setting. In long-horizon interactions, the primary bottleneck shifts from instantaneous context capacity to the continuous maintenance of task state and historical relevance. Even with extended context windows, the accumulation of history inevitably saturates attention budgets, increases latency, and induces goal drift (Lu et al., 2025b). To mitigate this, working memory in multi-turn settings functions as an externalized state carrier, organizing a continuous loop of reading, evaluation, and writing. The objective is to preserve critical state information accessible and consistent within a bounded resource budget. We categorize these mechanisms by their state management strategies: state consolidation, hierarchical folding, and cognitive planning.

State Consolidation In continuous interaction streams, state consolidation maps an ever-growing trajectory into a fixed-size state space through dynamic updates. Treating interaction as a streaming environment, MemAgent (Yu et al., 2025a), and MemSearcher (Yuan et al., 2025a) employ recurrent mechanisms to update fixed-budget memory and discard redundancy, answering queries from a compact, evolving state. ReSum (Wu et al., 2025f) extends this by periodically distilling history into reasoning states, utilizing reinforcement learning to optimize summary-conditioned behavior for indefinite exploration.

Beyond heuristic summarization, ACON (Kang et al., 2025c) frames state consolidation as an optimization problem, jointly compressing environment observations and interaction histories into a bounded condensation and iteratively refining compression guidelines from failure cases. IterResearch (Chen et al., 2025b) further adopts an MDP-inspired formulation with iterative workspace reconstruction, where an evolving report serves as persistent memory and periodic synthesis mitigates context suffocation and noise contamination in long-horizon research.

Regarding state representation, approaches vary to ensure constant-size footprints. MEM1 (Zhou et al., 2025b) maintains a shared internal state that merges new observations with prior memory. Distinct from explicit text, MemGen (Zhang et al., 2025d) injects latent memory tokens directly into the reasoning stream.

Hierarchical Folding For complex, long-horizon tasks, state maintenance requires structure beyond linear summarization. Hierarchical folding decomposes the task trajectory based on subgoals, maintaining fine-grained traces only while a subtask is active, and folding the completed sub-trajectory into a concise summary upon completion.

This decompose-then-consolidate strategy allows the working memory to expand and contract dynamically. HiAgent (Hu et al., 2025a) instantiates this by using subgoals as memory units, retaining only active action–observation pairs and writing back a summary after subgoal completion. Context-Folding (Sun et al., 2025b) and AgentFold (Ye et al., 2025a) extend this by making the folding operation a learnable policy, training agents to autonomously determine when to branch into sub-trajectories and how to abstract them into high-level states. DeepAgent (Li et al., 2025i) further applies this to tool-use reasoning, compressing interactions into structured episodic and working memories to support fine-grained credit assignment. By replacing finished sub-trajectories with stable high-level abstractions, these methods preserve essential context while keeping the active window small.

Cognitive Planning At the highest level of abstraction, working memory creates and maintains an externalized plan or world model. The state functions not merely as a summary of the past, but as a forward-looking structure that guides future actions.

PRIME (Tran et al., 2025) integrates retrieval directly into the planning loop, ensuring that memory updates actively support complex reasoning steps. In embodied and agentic environments, treating the language model as a high-level planner elevates the plan to the core of working memory. Approaches like SayPlan employ 3D scene graphs as queryable environmental memory to scale planning across large spaces (Rana et al., 2023). In GUI and household tasks, systems like Agent-S (Agashe et al., 2025) and KARMA (Wang et al., 2025r) stabilize long-horizon performance by anchoring reasoning to a hierarchical plan, using memory-augmented retrieval to bridge long-term knowledge with short-term execution.

By making plans and structured environment representations the readable and writable core of working memory, agents can maintain goal consistency and revise strategies robustly against perception failures (Song et al., 2023).

Summary Multi-turn working memory pivots on the construction of an operable state carrier rather than the retention of raw history. By integrating state consolidation to compress continuous streams, hierarchical folding to structure sub-trajectories, and cognitive planning to anchor future actions, these mechanisms effectively decouple reasoning performance from interaction length. This paradigm enables agents to maintain temporal coherence and goal alignment over indefinite horizons while adhering to strict computational and memory constraints.

