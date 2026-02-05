JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                        1




    Towards Efficient Multimodal Large Language
      Models: A Survey on Token Compression
                Linli Yao∗‡ , Long Xing∗ , Yang Shi∗ , Sida Li, Yuanxin Liu, Yuhao Dong, Yi–Fan Zhang,
                     Lei Li, Qingxiu Dong, Xiaoyi Dong, Qidong Huang, Haotian Wang, Feng Wu,
                                Yuanxing Zhang, Pengfei Wan, Zhouchen Lin† , Xu Sun†

      Abstract—Multimodal Large Language Models (MLLMs) have made significant strides in integrating vision-language perception,
      alignment, and reasoning. However, the increasing complexity of tasks such as high-resolution image processing and long video
      understanding has led to an exponential rise in visual context length within MLLMs. The resulting long-context token sequences
      impose substantial computational demands on large language models (LLMs), leading to quadratic complexity growth, heightened
      GPU resource consumption, and slower inference speeds. To address these challenges, token compression has emerged as a
      promising research direction that reduces the number of tokens processed within MLLMs while preserving essential cross-modal
      semantic information, thereby enhancing both training and inference efficiency. This survey provides a comprehensive review of token
      compression techniques for MLLMs, examining the current state of research and exploring future directions. We propose a taxonomy
      of token compression methods based on their application modules within the MLLM system, including the vision encoder, projector,
      LLM backbone, and hybrid approaches. We analyze the strengths and limitations of widely adopted algorithms, offering practitioners a
      structured framework for selecting appropriate token compression strategies. Finally, we discuss practical applications of token
      compression, identify key challenges in the field, and propose potential directions for future research and development. All related
      resources are available at https://github.com/yaolinli/MLLM-Token-Compression.

      Index Terms—Multimodal Large Language Model, Token Compression, Token Reduction, Efficient Multimodal Learning, Long-Context
      Modeling, Video Large Language Model, Vision and Language.

                                                                                 ✦

1    I NTRODUCTION


M       ULTIMODAL Large Language Models (MLLMs) [1]–
        [11] rapidly advanced the frontier of vision-language
joint perception, alignment, reasoning, and generation [12]–
                                                                                     and deployment. This tension between multimodal effec-
                                                                                     tiveness and computational efficiency has made compress-
                                                                                     ing multimodal token sequences an urgent research focus.
[17] [12]–[17]. By integrating the remarkable language                                   To build more efficient MLLMs, token compressing mul-
understanding capabilities of Large Language Models                                  timodal token sequences refers to methods that reduce the
(LLMs) [18]–[22] with comprehensive visual perception abil-                          number of tokens processed by MLLMs while preserving
ities from vision encoders [23], contemporary systems such                           critical cross-modal semantics. Conceptually, compression
as LLaVA [24], Qwen-VL [25] and GPT-4o [26] exhibit strong                           targets redundancy in spatial structure (e.g., repetitive back-
performance on diverse tasks spanning open-ended visual                              ground regions), temporal continuity (e.g., frame-to-frame
question answering, document understanding, and multi-                               similarities), and modality alignment (e.g., text-conditioned
step visual reasoning, among others.                                                 visual irrelevance), yielding shorter sequences with minimal
    However, these advanced cross-modal capabilities in-                             essential information degradation. Historically, token com-
cur substantial computational costs. High-resolution images                          pression originated in unimodal vision through patch drop-
and long videos can generate hundreds to thousands of vi-                            ping, token merging, and dynamic sparsification in Vision
sual tokens, while multi-turn dialogue and chain-of-thought                          Transformers [30]–[36]. These approaches have since been
reasoning further extend the historical context [27]–[29]. As                        extended to multimodal settings, where compression can
sequence lengths increase, the quadratic complexity of at-                           operate on visual streams, textual streams, or their fusion.
tention in Transformer-based MLLMs results in prohibitive                            As depicted in Figure 1, multimodal token compression
memory consumption and latency, limiting both scalability                            techniques [37]–[54] have evolved rapidly since 2022 and
                                                                                     experienced significant growth from 2024 onward. Recent
                                                                                     works [55]–[64] extend this research direction from spatial
•   L. Yao, Y. Shi, S. Li, Y. Liu, Q. Dong, Z. Lin, and X. Sun are with Peking       images to long-horizon video understanding with extreme
    University (Contact e-mail: linliyao@stu.pku.edu.cn).
•   L. Xing and F. Wu are with the University of Science and Technology of
                                                                                     compression ratios, where aggressive token compression
    China.                                                                           must be balanced against fine-grained localization, temporal
•   Y. Dong is with Nanyang Technological University.                                coherence, and temporal grounding performance.
•   Y.-F. Zhang is with the University of Chinese Academy of Sciences.                   Despite steady progress in token compression, practi-
•   L. Li is with the University of Hong Kong.
•   X. Dong is with Microsoft.                                                       tioners still face critical challenges in selecting or design-
•   Q. Huang is with Alibaba Cloud.                                                  ing token compression strategies for MLLMs. This survey
•   H. Wang is with the National University of Defense Technology.                   systematically examines the fundamental issues of token
•   Y. Zhang and P. Wan are with the Kling Team, Kuaishou Technology.
•   ∗ Equal contributions. † Corresponding authors. ‡ Project Leader.                compression from three perspectives.
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                                                        2

                                                            SpecVLM          HiPrune           STTM        VFlowOpt
                                                             METEOR          MMG-Vid           LLMC+       LaCo
            Code Open-source
                                                             TransPrune              GlimpsePrune        VisionThink                                  9-10
                                                                                                                                                7-8
                                                                                                                                4-6
                                                                                                                                                                        V " Drop
                                           LLaVA-Mini           FrameFusion                                           1-3
                                                                                                                                                      VCM               LangDC
            DyCoke      PruneVid              STORM            Video-XL-Pro
                                                                                                                                      HICom           PACT              TRIM
       VisionZip        MustDrop
                                                                                                        2025
                                           AdaFV                Skip-Vision
                                                                                                                                      HiCo                              HoloV
                                                                                                                                                      BTP
     Dynamic-VLM             QueCC                                                              11-12                ST !
                                                                                                                                      DART
         FastVLM       VisPruner                                                                                                                      HoliTom
                                                      TCA                                                            PVC              FastVID
                                                   TempMe                               9-10                         p-MoD                            Clapper
                                                                                                                                      TopV
                                   VTW                                                                                                                CDPruner
                                             SparseVLM                                                               Feather          DivPrune
                       MobileVLM V2                                           6-8
                                              TG-LLaVA                                                                                                SparseMM
                     LLaVA-PruMerge                                                                                            Video-XL
                                                                      1-5                                                                             FlexSelect
                               FastV                                                                                           LongVU
       SmartTrim                                                                                        SlowFast-LLaVA
                                                                                                                                                      Video-XL-2
                                     M3
                                                      2024                                              TokenPacker            FitPrune
        CrossGET                                                            MADTP                                                                     CoreMatching
                                                                                                        HiRED                  PyramidDrop
 BLIP2 (Q-Former)                          7-12                             LongVLM                                                                   Flash-VStream
                                                                                                        mPLUG-Owl3
                                     1-6                                    InternVL 1.5                                       mPLUG-DocOwl2
                                                  Honeybee                                                                                            LLaVA-Scissor
                     2023                                                                               LOOK-M
                                                                            PLLaVA                                             Learnable VTM
     2022                                         LLaMA-VID                                             LLaVolta                                      TimeChat-Online
                                                                            DeCo
                            ToMe                  TESTA                                                 VoCo-LLaMA

Fig. 1: A timeline of representative token compression methods for MLLMs is presented. The timeline is organized
primarily based on the earliest arXiv submission dates. Methods with publicly available code are highlighted. Due to
space constraints, only a subset of representative open-source approaches is included in the figure.


    First, where and how should token compression be applied ken compression techniques for MLLMs, with emphasis
within the MLLM architecture?                                    on efficient long-context sequence processing. A concurrent
    Different modules in MLLMs, including the vision en- survey [162] examines token compression across compre-
coder, projector, and large language model, introduce dis- hensive modalities including image, video, and audio. Our
tinct architectural characteristics, information bottlenecks, work is distinctly motivated by the goal of systematically
and computational trade-offs. The placement of compres- organizing existing token compression methods according
sion strongly influences the preservation of visual seman- to MLLM architectural components (where to compress) and
tics, the quality of cross-modal alignment, and downstream providing a practical roadmap of compression techniques
reasoning capability, yet lacks systematic analysis to guide (how to select).
architectural choices.                                                Our main contributions are summarized as follows:
    Second, which compression mechanism best suits specific (i) Taxonomy of token compression by MLLM architec-
deployment scenarios? The commonly-adopted design space              tural placement (§3). We introduce a systematic tax-
spans token merging versus pruning, text-guided versus               onomy that categorizes token compression methods by
purely visual compression, objectives for training versus            their application location within MLLMs—vision en-
inference acceleration, and plug-in modules versus end-to-           coder, projector, or large language model—clarifying
end retraining. Each paradigm offers distinct benefits and           how architectural placement interacts with compression
limitations that must be aligned with application-specific           objectives and how hybrid strategies can synergistically
constraints. We aim to clarify these trade-offs and provide          combine approaches across different modules.
decision guidelines for practitioners.                          (ii) Methodological analysis and design roadmap (§4).
    Third, what are the remaining open challenges and promis-        Complementing the architectural taxonomy, we analyze
ing future directions? As token compression represents an            the prevailing token compression mechanisms employed
active research field undergoing rapid development, it is            across these locations. We dissect critical design dimen-
essential to identify unresolved issues and emerging op-             sions, including text-guided versus vision-only compres-
portunities. We discuss key challenges including the lack            sion, token pruning versus merging, modular plug-ins
of theoretical foundations, adaptation to dynamic compres-           versus end-to-end retraining, and training-centric versus
sion requirements, efficiency-effectiveness trade-offs in fine-      inference-centric optimization. Based on this method-
grained tasks (e.g., chart understanding and OCR), and the           ological breakdown, we provide a selection roadmap to
need for more rigorous evaluation protocols. Based on these          guide researchers in choosing the optimal compression
perspectives, we aim to shed light on promising future               techniques tailored to specific tasks, accuracy targets, and
research directions                                                  latency constraints.
    This survey addresses these fundamental questions                 Grounded in the above analysis, we further summa-
through structured analysis. Compared to existing sur- rize open challenges in this field and aim to shed light
veys on efficient MLLMs [159], [160] and efficient vision on efficient next-generation MLLMs. We highlight pivotal
transformers [161], this work focuses specifically on to- future directions, such as task-aware adaptivity and refined
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                                                                                               3

                                                                                                                                       HiRED [65], TRIM [66], SAINT [67], VisPruner [68], HiPrune [69], VFlowOpt [70],
                                                                                                        Visual Token Dropping
                                                                                                                                       VScan [71], SmartTrim [72], MADTP [73], EgoPrune [74], TimeChat-Online [57]

                                                                                                                                       LaCo [75], LLaVA-STF [76], FastVLM [77], Clapper [78], TokenCorrCompressor [79],
                                                                        Inside-Encoder (§3.1.1)         Visual Token Merging           FiLA-Video [80], FiLA-Video [80], Chat-UniVi [81], CrossGET [82], TESTA [83],
                                                                                                                                       LLaVA-PruMerge [84], VisToG [85], FiCoCo [86], STTM [87], LookupViT [37]

                                                                                                                                       LLaVA-STF [76], LinVT [88], ADMIRE [89], VideoChat-Flash [90], LaCo [75],
                                                                                                       Multi-Scale Compression
                                             Vision Encoder (§3.1)                                                                     M3 [91], Cambrian-1 [7], MustDrop [92]

                                                                                                                                       VisionZip [93], LLaVA-PruMerge [84], FoPru [94], freePruner [95], HoloV [96],
                                                                                                      Purely-Vision Compression
                                                                                                                                       Fourier-VLM [97], LLaMA-VID [98], VideoLLaMA2 [99], LLaVA-PruMerge [84]
                                                                       Outside-Encoder (§3.1.2)
                                                                                                       Text-guided Compression         PAR [100], QG-VTC [101], LongVU [56], RecoverableCompression [102], VTC [103]

                                                                                                               Pooling                 MobileVLM V2 [104], DeCo [105], AVG-LLaVA [106], TC-LLaVA [107], PLLaVA [108]

                                                                     Transformation-Based (§3.2.1)           Pixel Shuffle             InternVL 1.5 [109], NVILA [110], InternVL 2.5 [111], Qwen2VL [112]
 Where to Compress Tokens in MLLMs (§3)




                                                                                                             Convolution               Honeybee [113], MobileVLM V2 [104], VideoLLaMA2 [114], LLaVA-STF [76]

                                                                                                               Q-Former                BLIP-2 [115], MiniGPT-4 [116], InstructBLIP [117]
                                                Projector (§3.2)
                                                                         Query-Based (§3.2.2)            Variants of Q-Former          Qwen-VL [118], Honeybee [113], MQT [119], TG-LLaVA [120], Cambrian-1 [7]

                                                                                                        Cross Attention-Based          CATP [121], TokenPacker [122], HiRes-LLaVA [123], mPLUG-DocOwl2 [124]

                                                                      Importance-Driven (§3.2.3)      Various Similarity Metrics       DynTok [125], LLaVA-Scissor [126], SeqCompression [127], DivPrune [128]

                                                                                                                                       FastV [129], PyramidDrop [130], VTW [131], SparseVLM [132], Feather [133],
                                                                                                           Importance-based
                                                                                                                                       ATP-LLaVA [134], AIM [135], ST3 [136] , G-Search [137], TopV [138], PACT [139]

                                                                                                                                       p-MoD [140], ATP-LLaVA [134], Dynamic-LLaVA [141], DyRate [142],
                                                                                                       Learnable Module-based
                                                                           Prefilling (§3.3.1)                                         GlimpsePrune [143]

                                                                                                         Token Merging-based           LLaVolta [144], FiCoCo [86], FrameFusion [145], HoliTom [146], VFlowOpt [70]
                                                  LLM (§3.3)
                                                                                                             Fusion-based              Flamingo [147], mPLUG-Owl3 [148], CrossLMM [149], VoCo-LLaMA [150]

                                                                                                     LOOK-M [151], MustDrop [92], DyCoke [152], SparseMM [153], InfiniPot-V [154],
                                                                           Decoding (§3.3.2)
                                                                                                     Dynamic-LLaVA [141], Video-XL-2 [58], LiveVLM [155], StreamMem [156]

                                                                           Collaborative
                                                                                                     CrossGET [82], LLaMA-VID [98], PAR [100]
                                                                         Compression (§3.4.1)
                                                Hybrid (§3.4)
                                                                           Progressive
                                                                                                     MustDrop [92], DyCoke [152], FiCoCo [86], VFlowOpt [70], VTC-CLS [157], METEOR [158]
                                                                         Compression (§3.4.2)

Fig. 2: A taxonomy of token compression methods for MLLMs, organized by the compression position in MLLMs(§3), with
leaf nodes illustrating representative works.


evaluation protocols, with the ultimate goal of making mul-                                                                visual features with the language model’s embedding space,
timodal intelligence both powerful and affordable at scale.                                                                and a powerful LLM that performs multimodal alignment,
                                                                                                                           reasoning and generation. This architectural design enables
                                                                                                                           end-to-end training and seamless integration of visual and
2                                         P RELIMINARIES                                                                   textual information processing. Throughout this survey, we
This section lays the foundation for token compression in                                                                  focus on token compression techniques designed for this
Multimodal Large Language Models (MLLMs). We begin                                                                         mainstream three-component architecture. Alternative ar-
with an overview of typical MLLM architectures (§2.1),                                                                     chitectural paradigms [169], [170] that deviate from this
followed by a formal definition of token compression tech-                                                                 design are beyond the scope of our discussion.
niques (§2.2).                                                                                                                 Formally, let X v = {I1 , I2 , . . . , Inv } with nv ≥ 1
                                                                                                                           denote the input image sequence or video frames, and
2.1                                       Multimodal Large Language Models                                                 X t = {x1 , x2 , . . . , xnt } represent the textual token sequence
                                                                                                                           comprising system prompts, user instructions, or dialogue
The rapid advancement of artificial intelligence has wit-
                                                                                                                           history. The MLLM architecture consists of three key com-
nessed a paradigm shift from unimodal models to sophis-
                                                                                                                           ponents:
ticated multimodal systems capable of understanding and
                                                                                                                               Vision Encoder. The vision encoder Ev transforms raw
reasoning across diverse data modalities. MLLMs represent
                                                                                                                           visual inputs into a sequence of dense visual token repre-
a significant milestone in this evolution, combining the
                                                                                                                           sentations:
remarkable language understanding capabilities of Large
Language Models (LLMs) [18], [163]–[166] with comprehen-                                                                                            Zv = Ev (X v ) ∈ Rnv ×dv ,              (1)
sive visual perception abilities to create systems that can                                                                where nv denotes the number of visual tokens and dv
process, understand, and generate responses based on both                                                                  represents the feature dimension of each visual token.
textual and visual information.                                                                                                Projector. To bridge the modality gap between visual
    Modern MLLMs typically adopt a three-component ar-                                                                     and textual representations, a projector P transforms visual
chitecture: A vision encoder (VE) (often based on SigLIP [167]                                                             features from dimension dv to the LLM’s embedding space:
or CLIP [168]) that processes visual inputs into high-
dimensional feature representations, a projector that aligns                                                                                           Hv = P(Zv ) ∈ Rnv ×dt ,                                          (2)
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                  4

where dt denotes the embedding dimension of the target              Compression Ratio is a widely-mentioned concept in
language model.                                                  token compression, defined as:
    Large Language Model. The LLM G processes the concate-                                          N
nated sequence of projected visual tokens and embedded                                  Rcomp =       ,                   (7)
                                                                                                    M
textual tokens:
                                                                 where higher values (e.g., 4× or 8×) indicate greater com-
                   Y = G [Hv ; Et (X t )] ,
                                         
                                                       (3)
                                                                 pression levels, more compact semantic representations, and
where Et (·) represents the embedding layer of the LLM, [·; ·]   consequently larger efficiency gains.
denotes concatenation along the sequence dimension, and              Since the number of visual tokens typically exceeds that
Y is the generated output sequence.                              of textual tokens by substantial margins (e.g., by 20× [93])
                                                                 in MLLMs, most existing token compression methods pri-
Computational Complexity. The aforementioned compo-              marily focus on reducing nv . To achieve more compact
nents in MLLMs primarily employ Transformer-based ar-            visual representations within MLLMs, two main types of
chitectures [171], renowned for their powerful representa-       redundancy can be exploited:
tion capabilities but also characterized by high computa-            (i) Intra-Visual Redundancy. Visual content inher-
tional costs for processing long input sequences. The com-       ently contains redundant information. In images, numer-
putational complexity predominantly stems from the self-         ous patches may represent background elements that are
attention mechanism and feed-forward networks (FFNs)             not crucial for understanding the primary subject matter.
within Transformer layers.                                       Similarly, in videos, consecutive frames often exhibit sub-
    Given a sequence of length n, a hidden dimension size        stantial similarity, resulting in temporal redundancy. This
d, and an intermediate dimension m in the FFN, the com-          redundancy can be leveraged to reduce the number of visual
putational cost per Transformer layer can be approximated        tokens requiring processing, thereby improving computa-
as                                                               tional efficiency while maintaining information quality.
           Layer FLOPs = 4nd2 + 2n2 d + 2ndm.           (4)          (ii) Cross-Modal Redundancy. In multimodal tasks, par-
                                                                 ticularly question-answering scenarios, textual input pro-
Thus, for an L-layer Transformer, the total cost is
                                                                 vides contextual guidance that can identify the most rele-
       Total FLOPs = L × 4nd2 + 2n2 d + 2ndm ,
                                            
                                                           (5)   vant visual tokens. For instance, when a question focuses
                                                                 on a specific object within an image, only visual tokens
where n = nt +nv is the overall sequence length (text tokens     corresponding to that object may be necessary for accurate
nt plus visual tokens nv ).                                      comprehension and response generation. By exploiting tex-
    As the sequence length n increases, the quadratic com-       tual information, it becomes possible to selectively retain
plexity term 2n2 d in the attention mechanism grows rapidly,     only those visual tokens that are pertinent to the specific
leading to prohibitive computational overhead. This compu-       task requirements.
tational bottleneck is particularly pronounced in scenarios
involving: (1) high-resolution images or long videos, where      3     W HERE TO C OMPRESS TOKENS IN MLLM S
nv typically dominates nt in MLLMs, and (2) multi-turn con-
                                                                 Based on the taxonomy illustrated in Figure 2, we systemati-
versations or complex reasoning tasks requiring extensive
                                                                 cally categorize existing token compression methods accord-
contextual history.
                                                                 ing to where compression is applied within the MLLM archi-
                                                                 tecture. Throughout the processing procedure from visual
2.2   Token Compression                                          input to textual output, token compression strategies can
                                                                 be progressively deployed at three architectural modules:
The quadratic computational complexity in MLLMs natu-
                                                                 (1) the Vision Encoder (§3.1), where compression reduces
rally motivates the development of token compression tech-
                                                                 computational overhead at the visual perception stage; (2)
niques (also known as token reduction), which aim to reduce
                                                                 the Projector (§3.2), which integrates token reduction during
the total context length in the MLLM while preserving
                                                                 the transformation from visual to linguistic representation
essential visual and textual semantics, thereby achieving
                                                                 space; and (3) the Large Language Model (§3.3), where
computational efficiency without remarkably compromising
                                                                 compression achieves holistic cross-modal efficiency opti-
model performance.
                                                                 mization.
    Formually, denote the total visual and textual token
number in the MLLM as N = nt + nv , token compression
aims to reduce the N to a smaller M to improve efficiency by     3.1   Token Compression in Vision Encoder
selecting or aggregating original tokens, where M < N . The      In MLLMs, visual data are inherently more redundant than
token compression process can be represented as a function       text [191]–[193], leading to a substantially larger number of
C that maps the original token sequence to a compressed          tokens on the vision side than on the language side. For
sequence:                                                        instance, a single high-resolution image can be divided into
                                                                 thousands of patch tokens [10], [112]. If these tokens are
                  Hcomp = C(H) ∈ RM ×dt ,                  (6)   simply concatenated with text tokens and processed as an
                                                                 “interleaved long sequence”, the subsequent pre-filling and
where H = [Hv ; Ht ] ∈ RN ×dt is the concatenated sequence       decoding stages of the LLM incur quadratic computational
of projected visual tokens and embedded textual tokens,          complexity with respect to the sequence length. Since the
and Hcomp is the compressed token sequence.                      vision encoder (VE) is the first module to encode visual
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                     5

TABLE 1: Summary of representative token compression works (venues up to Oct. 2025). Modality denotes the primary
application scenario. Compression Position indicates the application stage (Vision Encoder, Projector, or LLM). Text Query-
based marks dependency on text token guidance. Re-train vs. Plug-in distinguishes methods requiring additional training
from plug-and-play modules.
                                                                                                  Text
   #   Method                   Date          Venue    Modality     Compression Position                     Re-train/Plug-in
                                                                                               Query-based
   1   ToMe [34]               2022.01        ICLR      image            Vision Encoder            no        re-train,plug-in
   2   BLIP2 [172]             2023.01        ICML      image               Projector              yes            re-train
   3   MovieChat [173]         2023.07        CVPR       video           Vision Encoder            no             plug-in
   4   MobileVLM V2 [104]      2024.02        arXiv     image               Projector              no             re-train
   5   LLaVA-PruMerge [84]     2024.03        ICCV    image,video        Vision Encoder            no        re-train,plug-in
   6   FastV [129]             2024.03        ECCV    image,video             LLM                  yes            plug-in
   7   M3 [91]                 2024.05        ICLR    image,video        Vision Encoder            no             re-train
   8   DeCo [105]              2024.05        arXiv     image               Projector              no             re-train
   9   VoCo-LLaMA [150]        2024.06        CVPR    image,video             LLM                  no             re-train
  10   TokenPacker [122]       2024.07         IJCV     image               Projector              no             re-train
  11   HiRes-LLaVA [123]       2024.07        CVPR      image               Projector              no             re-train
  12   mPLUG-Owl3 [148]        2024.08        arXiv   image,video             LLM                  yes            re-train
  13   HiRED [65]              2024.08        AAAI      image            Vision Encoder            no        re-train,plug-in
  14   TempMe [174]            2024.09        ICLR       video           Vision Encoder            no             re-train
  15   Video-XL [55]           2024.09        CVPR       video                LLM                  no             re-train
  16   PyramidDrop [130]       2024.10        CVPR    image,video             LLM                  yes            plug-in
  17   SparseVLM [132]         2024.10        ICML    image,video             LLM                  yes            plug-in
  18   LongVU [56]             2024.10        ICML       video           Vision Encoder,           yes            re-train
                                                                            Projector
  19   TCA [175]               2024.10        ICCV      image            Vision Encoder            no             plug-in
  20   QueCC [176]             2024.11        ICLR      image               Projector              yes            re-train
  21   ATP-LLaVA [177]         2024.12        CVPR      image                 LLM                  yes            re-train
  22   VisPruner [68]          2024.12        ICCV    image,video        Vision Encoder            no             plug-in
  23   VisionZip [93]          2024.12        CVPR    image,video        Vision Encoder,           no        re-train,plug-in
                                                                            Projector
  24   Dynamic-VLM [41]        2024.12     ICCV       image,video        Vision Encoder            no             re-train
  25   PVC [178]               2024.12     CVPR       image,video   Vision Encoder,Projector       no             re-train
  26   PruneVid [179]          2024.12      ACL          video           Projector,LLM             yes            plug-in
  27   Feather [133]           2024.12     ICCV         image                 LLM                  yes            plug-in
  28   HiCo [180]              2025.01     arXiv         video        Vision Encoder,LLM           yes            re-train
  29   LLaVA-Mini [181]        2025.01     ICLR       image,video           Projector              yes            re-train
  30   FALCON [44]             2025.01     ICCV         image            Vision Encoder            no             re-train
  31   FCoT-VL [182]           2025.02     arXiv        image               Projector              no             re-train
  32   DART [183]              2025.02    EMNLP         image            Vision Encoder            no             plug-in
  33   DivPrune [128]          2025.03     CVPR       image,video           Projector              no             plug-in
  34   FastVID [184]           2025.03    NeurIPS        video              Projector              no             plug-in
  35   TopV [138]              2025.03     CVPR       image,video             LLM                  no             plug-in
  36   Skip-Vision [185]       2025.03     ICCV         image         Vision Encoder,LLM           no             plug-in
  37   TimeChat-Online [57]    2025.04   ACM MM          video              Projector              no        re-train,plug-in
  38   VCM [186]               2025.04     arXiv        image               Projector              yes            re-train
  39   HoliTom [146]           2025.05    NeurIPS        video           Projector,LLM             no             plug-in
  40   ToDRE [187]             2025.05     arXiv      image,video        Projector,LLM             no             plug-in
  41   BTP [188]               2025.05    NeurIPS       image                 LLM                  yes            plug-in
  42   DynTok [125]            2025.06     arXiv         video              Projector              no             plug-in
  43   LLaVA-Scissor [126]     2025.06     arXiv         video              Projector              no             plug-in
  44   SparseMM [153]          2025.06     ICCV         image                 LLM                  yes            plug-in
  45   Video-XL-2 [58]         2025.06     arXiv         video        Vision Encoder,LLM           yes            re-train
  46   FlexSelect [189]        2025.06    NeurIPS        video           Vision Encoder            yes       re-train,plug-in
  47   VisionThink [190]       2025.07    NeurIPS       image               Projector              yes            re-train
  48   STTM [87]               2025.07     ICCV          video           Vision Encoder            no             plug-in
  49   METEOR [158]            2025.07     ICCV         image            Vision Encoder,           yes            re-train
                                                                         Projector, LLM
  50   VFlowOpt [70]           2025.08     ICCV       image,video     Vision Encoder,LLM           yes       re-train,plug-in
  51   CATP [121]              2025.08     arXiv        image            Projector,LLM             yes            plug-in
  52   SpecVLM [61]            2025.08    EMNLP          video                LLM                  yes            plug-in
  53   LangDC [64]             2025.09    EMNLP          video              Projector              yes            re-train
  54   HoloV [96]              2025.10    NeurIPS     image,video        Vision Encoder            no             plug-in
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                                                                    6

inputs, reducing visual tokens at this initial stage yields dis-                                   Forward
                                                                                                                                                  vision token         text token

proportionately large efficiency gains throughout the entire                                                                                        Tokenizer
MLLM system. As shown in Figure 3, we first review and




                                                                                                                          · ··
                                                                                         ViT        Token             ViT
                                                                                        Layer                        Layer                                                     LLM
categorize vision-side token compression methods applied                 or
                                                                                          N
                                                                                                  Compression
                                                                                                                     N+1
                                                                                                                                     Token
                                                                                                                                   Compression
                                                                                                                                                    Projector
at the vision encoder module into two broad categories:
   • Inside Vision Encoder Compression (Inside-VE,
                                                                                    Inside-Encoder Compression                            Outside-Encoder Compression
     §3.1.1): Compression is applied within the ViT or video
                                                                    (a) Important Token Selection                            (a) Purely-Vision Compression
     encoder itself. Methods in this category either discard         •   Self-attention
                                                                                                                                                                        Multi-scale Visual
                                                                                                                                                                            Features

     redundant tokens or merge similar ones. Since differ-           •   [cls] token
                                                                         relevance,             Select Metrics                                              Vision
                                                                     •   KNN cluster,                                                                     Similarity
     ent layers capture multi-scale semantics—ranging from           •
                                                                     …
                                                                         Similarity graph
                                                                                                                                                                        low            high
     low-level textures to high-level concepts—multi-scale          (b) Low-Importance Token Reduction
                                                                                                                                                                       level           level
                                                                                                                             (b)Text-guided Compression
     compression schemes have been developed to coordi-                       Token Pruning               Token Merging                               Text-Vision
                                                                                                                                                       Similarity
     nate compression across layers.
   • Outside Vision Encoder Compression (Outside-VE,                             Drop                                                                                   low            high
                                                                                                           Aggregation                                                 level
     §3.1.2): Compression occurs after the vision encoder                                                                                                                              level


     produces its output tokens but before the projector
     maps these tokens into the language model space. This         Fig. 3: Illustration of token compression strategies applied
     design is plug-and-play and minimally invasive to the         at the vision encoder module in MLLMs.
     original architecture. Depending on whether textual
     signals are incorporated, methods can be grouped into
     purely vision-based approaches and text-guided ap-
                                                                   HiPrune [69] leverage the CLS token attention in the vi-
     proaches.
                                                                   sion transformer to assess the visual importance of image
                                                                   partitions. VFlowOpt [70] constructs an importance map
3.1.1   Inside-Encoder Compression                                 by integrating visual attention-derived context relevance
Inside-VE compression directly alters token flow within            with patch-level information entropy to determine which
the encoder, reducing self-attention complexity at an early        tokens to prune. The second category incorporates cross-
stage and shortening the propagation path of tokens. The           modal attention to evaluate token significance. MADTP [73]
design revolves around two questions: (1) how to handle            introduces a Token Importance Score (TIS) that integrates
“unimportant” tokens through pruning or merging; and               three attention mechanisms—class attention, self-attention,
(2) how to coordinate compression across multiple layers           and cross-modal alignment attention—and employs learn-
or encoders to leverage multi-scale visual features. Here          able thresholds with sparsemax activation to dynamically
we focus exclusively on methods applied in multimodal              determine pruning masks. SmartTrim [72] adopts a cross-
LLMs, and do not review token compression for pure vision          modal guidance approach by feeding the CLS token into
tasks [161].                                                       a lightweight policy network that learns importance scores
                                                                   based on cross-modal information.
Visual Token Dropping. Token dropping methods com-
pute importance scores for visual tokens within the vision             Heuristic-based scoring. These methods exploit task-
encoder and retain only the most salient ones, directly            specific priors to guide token selection. EgoPrune [74]
discarding the remainder. Implementation typically follows         leverages domain-specific heuristics from egocentric videos,
a “ranking + Top-K ” paradigm with defined thresholds. To          utilizing geometric stability and field-of-view dynamics to
identify important visual tokens within the encoder, existing      prioritize motion-relevant regions while pruning static back-
methods employ three principal scoring strategies:                 grounds. METEOR [158] adopts a layer-adaptive strategy
                                                                   based on the prior that shallow and deep layers encode
    Similarity-based scoring. These methods quantify token re-
                                                                   fundamentally different types of information. Specifically,
dundancy by measuring the similarity between each visual
                                                                   METEOR employs similarity to the average token as the
token and a global representation (e.g., CLS token or aggre-
                                                                   pruning criterion in shallow layers, where low-level redun-
gated feature vector). Tokens exhibiting high similarity are
                                                                   dancies dominate, and class attention scores in deep layers,
deemed redundant and removed. Representative works in-
                                                                   where semantic information is more concentrated.
clude TRIM [66] and SAINT [67], which employ global sim-
ilarity metrics with layer-adaptive thresholds. TRIM lever-        Visual Token Merging. Unlike pruning, which deletes to-
ages CLIP embeddings to measure the relevance between              kens outright, merging aggregates similar tokens into com-
textual queries and visual tokens, employing an adaptive           pact representations to preserve information while short-
Interquartile Range (IQR)-based thresholding mechanism to          ening sequences [34]. A fundamental principle underlying
select the most query-relevant tokens. SAINT advances this         merging operations is proximity-based redundancy: tokens
paradigm by leveraging token similarity within a graph-            that are close to each other spatially or temporally tend to
based formulation to dynamically optimize both pruning             exhibit high redundancy. Beyond proximity heuristics, more
rates and redundancy thresholds, offering greater flexibility      sophisticated methods leverage explicit similarity measure-
than fixed strategies.                                             ments or hybrid compression pipelines to achieve semantic
    Attention-based scoring. These approaches leverage at-         merging.
tention weights from the vision transformer to derive                 Proximity-based Merging. Spatial and temporal adjacency
token saliency. The first category restricts pruning deci-         provide natural bases for identifying redundant visual to-
sions to vision-only attention patterns. VisPruner [68] and        kens, as neighboring patches or consecutive frames typically
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                  7

share similar features. For spatial merging, structured ap-     level cascade aggregation, progressively extracting coarse,
proaches perform deterministic aggregation through down-        medium, and fine-grained token sets for unified multi-
sampling operations [77] or pixel-shuffle with channel          scale representation. LaCo [75] performs aggressive early-
merging [75], while learnable methods adopt adaptive con-       layer compression followed by pixel shuffle and MLP-based
volution kernels [76] or density-based clustering [81] to       detail recovery.
capture task-specific patterns beyond uniform averaging.            Multi-Encoder Compression. Combining vision encoders
    In video understanding, temporal proximity enables          with different architectures or training paradigms yields
cross-frame consolidation through two complementary             complementary representations. Cambrian-1 [7] demon-
strategies: joint temporal-spatial aggregation that merges      strates that integrating self-supervised models (e.g., DI-
similar frames and patches simultaneously [71], [83], and       NOv2 [194]) with language-supervised encoders (e.g.,
frame-level fusion that adaptively integrates consecutive       CLIP [195]) consistently improves performance on vision-
frames with learnable importance weighting [80], [87]. By       centric and OCR tasks, underscoring the value of diverse
exploiting the inductive bias that adjacent tokens exhibit      visual representations. METEOR [158] proposes a system-
high correlation across both spatial and temporal dimen-        atic multi-encoder framework that eliminates cross-encoder
sions, these proximity-based methods achieve efficient com-     redundancy to maximize complementarity while minimiz-
pression while preserving local coherence.                      ing computational overhead.
    Similarity-based Merging. While proximity heuristics pro-       Multi-Resolution Compression. Processing inputs at multi-
vide strong inductive bias, semantic redundancy often tran-     ple resolutions balances efficiency with visual detail preser-
scends geometric or temporal adjacency, focusing explicit       vation. High-resolution inputs capture fine-grained infor-
feature-space similarity. Global similarity methods compute     mation for vision-sensitive tasks, while low-resolution in-
token importance via patch-to-class correlation [79] or clus-   puts provide efficient global context. FastVLM [77] achieves
ter semantically similar patches into abstracted representa-    optimal token-resolution balance through a novel hybrid
tions [85], enabling merging of spatially distant yet seman-    vision encoder called FastViTHD. ADMIRE [89] employs
tically related regions.                                        dual-path Multi-Resolution Adaptation, comprising a low-
    Cross-modal merging methods leverage textual context        resolution backbone for global processing and a high-
to refine token merging decisions. This can be achieved         resolution bypass for detail injection, excelling at document
through bidirectional tokens that exchange language-aware       understanding and small object detection with minimal
signals between modalities [82] or through pipelines that       overhead.
combine semantic and spatial similarity [84]. By prioritizing       For video understanding, LinVT [88] and M3 [91] apply
semantic relationships over spatial proximity, these methods    multi-scale temporal pooling to capture both short-term
enable compression that adapts to content meaning rather        dynamics and long-term context across different timescales.
than token positions.                                           VideoChat-Flash [180] introduces Hierarchical Condensa-
    Hybrid Strategies. Combining multiple compression tech-     tion (HiCo), progressively refining video semantics from
niques can achieve better efficiency-quality trade-offs than    clip-level to segment-level through selective filtering and
individual methods alone. Sequential approaches [86] first      backfill.
apply attention-based pruning to remove coarse-grained
redundancy, then use weighted merging to recover infor-
                                                                3.1.2   Outside-Encoder Compression
mation from discarded tokens and integrate it into re-
tained ones. Learnable abstraction methods [37] employ a        Outside-encoder compression occurs after vision encoder
small set of trainable compressed tokens while maintaining      output but before the projector. At this stage, visual tokens
cross-attention with high-resolution lookup tokens for fine-    are encoded but not yet aligned with the language modal-
grained details, allowing flexible compression ratios with-     ity. This position offers stronger plug-and-play capability
out architectural changes. These hybrid strategies show that    than inside-encoder approaches, requiring no modification
pruning, merging, and learnable abstraction work synergis-      to encoder layers. Compression here reduces visual token
tically when properly combined.                                 count by measuring semantic relevance between vision-
Multi-Scale Visual Compression. Single-scale compres-           vision or vision-text representations. We categorize methods
sion methods operate at fixed granularity, struggling to        into purely-vision and text-guided compression.
obtain comprehensive visual details. Multi-scale approaches     Purely-Vision Compression.         Purely-vision methods
address this limitation by coordinating compression across      downsample or aggregate encoder outputs based solely
layers, encoders, or resolutions, enabling flexible exploita-   on vision-vision semantic relevance, independent of user
tion of hierarchical visual semantics.                          queries or prompts. A widely adopted paradigm is
    Multi-Layer Compression. While most MLLMs extract vi-       “selection-then-merge”. VisionZip [93] identifies reusable
sual features from the penultimate ViT layer, aggregating       tokens through importance estimation and representa-
multi-layer features complements high-level visual seman-       tiveness constraints. Fourier-VLM [97] suppresses high-
tics with low-level visual details. LLaVA-STF [76] extracts     frequency redundancy via low-pass filtering in the fre-
tokens from multiple ViT blocks, fusing them via channel        quency domain before mapping back to token space.
concatenation and convolutions to combine spatial and se-       LLaVA-STF [76] generates compact visual summaries
mantic information across layers. METEOR [158] applies          through cross-layer concatenation and Multi-Block Token
hierarchical pruning, using token-to-average similarity in      Fusion (MBTF).
shallow layers and CLS-to-token attention in deep layers for        Visual Attention Bias Problem. Early works such as
layer-adaptive compression. Chat-UniVi [81] employs three-      LLaVA-PruMerge [84], VTC-CLS [157], and FasterVLM [196]
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                           8

leverage the CLS token for patch attention and represen-
tation similarity-based sparsification. Similarly, FoPru [94]                              Tokenizer
and freePruner [95] calculate token contribution via self-
attention scores, selecting high-contribution tokens as piv-                                                                       text token
                                                                                                                    LLM
ots. However, recent works [71], [96], [133] reveal that                        Vision                                            vision token
                                                                                           Projector
attention-based selection exhibits bias toward salient re-                     Encoder
gions (e.g., foreground objects), neglecting global context.
HoloV [96] addresses this by incorporating global visual
context to balance foreground and background tokens from           (a) Transformation-based            (b) Query-based        (c) Importance-driven
a holistic perspective.                                                                                                          Importance
    Extreme Compression. For long videos, LLaMA-VID [98]
                                                                                                           Cross Attention
compresses each frame into a single Content Token, pro-
viding fixed-budget compression. Flash-VStream [197] em-
ploys K-means clustering of low-resolution features as Con-             Pooling、Convolution
                                                                               ，            …             Learnable Queries

text Synopsis Memory to retain global temporal informa-           Fig. 4: Illustration of token compression strategies applied
tion. VideoLLaMA 2 [99] integrates frame-level patches via        at the projector module in MLLMs.
Spatial-Temporal Convolution (STC) with separable con-
volution and local aggregation. LLaVA-PruMerge [84] per-
forms learnable token merging via nearest-neighbor clus-          employs a Stable Diffusion [200] decoder to reconstruct
tering, maintaining near-uncompressed performance under           images from completed tokens, using reconstruction error
10x compression.                                                  to recover missing visual information.
    These methods share a common principle: enhancing
per-token information density without text reliance, demon-
strating particular advantages in multi-image and multi-          3.2     Token Compression in Projector
turn scenarios.                                                   The projector module plays a pivotal role in bridging the
Text-Guided Compression. When textual prompts pro-                vision encoder and the language model in MLLMs. It acts
vide semantic priors, compression can focus on question-          as the interface that transforms raw visual embeddings
relevant regions or frames, realizing context-oriented effi-      into language-compatible representations, ensuring that the
ciency. PAR [100] parses queries into entities and actions and    information extracted by the vision backbone can be effec-
re-weights visual tokens accordingly. QG-VTC [101] com-           tively leveraged by the LLM. While projector architectures
putes question-to-vision similarity to guide token retention,     such as Q-Former [115] inherently perform token compres-
enabling from 4× to 8× compression with minimal per-              sion by distilling a large set of visual embeddings into a
formance loss. LongVU [56] integrates cross-modal queries         compact set of query tokens, subsequent research [104],
with frame or region candidates, first filtering at the segment   [105], [108], [113] has introduced additional design enhance-
level and then refining token-level selection.                    ments to the projector to enable more fine-grained and
    Text-guided compression methods demonstrate partic-           task-adaptive compression. Therefore, in this section, we
ular robustness at the Outside-VE position: visual token          focus on token compression within the projector, referring
semantics are fully encoded while cross-modal interaction         to methods that operate on the visual features produced by
has not yet begun, minimizing textual bias interference           the vision encoder before they are fed into the language
with low-level visual encoding. These methods often cas-          model. As Figure 4 shows, these approaches can be broadly
cade with purely-vision approaches: first applying text-          categorized into three main types: transformation-based
agnostic compression, then refining based on query rele-          (Sec. 3.2.1), query-based (Sec. 3.2.2), and importance-driven
vance, achieving both stronger generalization and higher          (Sec. 3.2.3).
effective compression rates.
                                                                  3.2.1     Transformation-Based Compression
Token Recovery Mechanisms.           Under aggressive com-
pression, dynamic recovery mechanisms enable closed-loop          Transformation-based token compression methods reduce
refinement for enhanced robustness. When MLLMs detect             the number of visual tokens by directly transforming the
semantic uncertainty by confidence or entropy, they can           spatial structure of visual feature maps. Instead of relying on
trigger resampling visual information, reinjecting tokens         learnable queries or complex attention mechanisms, these
to compensate for missing visual evidence. Recoverable-           approaches perform lightweight, deterministic transforma-
Compression [102] triggers targeted resampling based on           tions to achieve token reduction while preserving essential
confidence and conflict thresholds post-compression. Must-        visual information. In this section, we review representative
Drop [92] integrates recovery throughout a multi-stage            transformation-based techniques, including pooling-based,
pipeline via uncertainty gating, balancing aggressive reduc-      pixel shuffle-based, and convolution-based methods.
tion with stability. Beyond runtime recovery, ToCom [198]         Pooling-Based. Pooling is a widely used downsampling
addresses train-test compression mismatches as a plug-and-        operation in computer vision, which can directly and ef-
play layer. It bridges performance gaps across compression        fectively reduce the number of tokens while preserving the
ratios without retraining.                                        main semantic information.
    VTC [103] and Video-XL-Pro [199] optimize compression             Given an input feature map X ∈ RH×W ×C , a pool-
via visual reconstruction supervision. For example, VTC           ing window of size k × k , and an output feature map
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                      9
           ′     ′
Y ∈ RH ×W ×C , the average pooled feature at spatial                  Formally, a 2D convolution that maps an input feature
position (i, j) in channel c is defined as:                         map X ∈ RH×W ×Cin to an output feature map Y ∈
                                                                      ′  ′
                                                                    RH ×W ×Cout can be defined as:
                              1          X                                         kh X
                                                                               Cin X  kw
                 Yi,j,c =                          Xu,v,c ,   (8)       (o)
                                                                               X
                                                                                              (o)       (c)
                            |Ωi,j |                                   Yi,j =                 Wm,n,c · X i+m−1, j+n−1 + b(o) , (10)
                                      (u,v)∈Ωi,j
                                                                               c=1 m=1 n=1

where Ωi,j denotes the set of spatial locations within the          where W ∈ Rkh ×kw ×Cin ×Cout are the learnable convolu-
k × k neighborhood centered at (i, j).                              tional kernels, kh and kw denote the kernel height and
     Owing to its parameter-free nature and computational           width, and b(o) is the bias term for the o-th output channel.
efficiency, pooling has been widely employed in many token              In token compression methods, convolution is often
compression approaches [104]–[108]. MobileVLM V2 [104]              combined with other operations such as average pooling.
proposes the Lightweight Downsample Projector (LDP),                For example, the C-Abstractor proposed in Honeybee [113]
which performs a simple 2×2 average pooling to effectively          integrates convolution with average pooling to achieve
reduce the number of image tokens. DeCo [105] validates             improved local context modeling. Similarly, MobileVLM
the effectiveness of the adaptive average pooling through           V2 [104] employs an LDP that combines pointwise and
extensive experimental analysis, showing that it not only fa-       depthwise convolutions with average pooling.
cilitates stable and efficient convergence but also effectively
                                                                    3.2.2 Query-Based Compression
extracts visual features. Following this line of pooling-based
compression, AVG-LLaVA [106] proposes the Visual Granu-             Query-based token compression leverages a limited num-
larity Scaler, which constructs multi-granularity visual fea-       ber of learnable query embeddings to attend to dense
tures by stacking average pooling layers and employs the            visual features and distill them into a compact repre-
Visual Granularity Router to select the most appropriate            sentation for the subsequent processing. This paradigm
granularity.                                                        provides a flexible and parameter-efficient alternative to
                                                                    purely transformation-based methods, as the queries can
     For video-focused models, pooling also serves as a
                                                                    adaptively select task-relevant information while discarding
simple yet effective way to reduce the number of tokens.
                                                                    redundancy. In the following, we discuss the canonical Q-
TC-LLaVA [107] employs simple global average pooling to
                                                                    Former framework, explore its enhanced and simplified
reduce the number of tokens per frame, while PLLaVA [108]
                                                                    variants, and introduce other cross-attention–based token
applies adaptive average pooling across both spatial and
                                                                    compression approaches.
temporal dimensions.
                                                                    Q-Former. Q-Former, introduced in BLIP-2 [115], is a
Pixel Shuffle-Based. Pixel shuffle is a method that trades          lightweight Transformer designed for query-based token
token count for channel dimensionality, rearranging high-           compression and vision–language alignment.
resolution spatial tokens into fewer tokens with increased              Q-Former employs a small set of learnable query vectors
channel depth.                                                      that interact with frozen visual features via stacked self-
   Given an input feature map X ∈ RH×W ×C and a                     attention and cross-attention layers. In this mechanism,
downsampling factor r, this approach rearranges the spatial         the queries (Q) are trainable embeddings initialized as a
resolution into the channel dimension as:                           small set of tokens that aim to retrieve task-relevant in-
                                                                    formation; the keys and values (K/V) are the fixed output
               Y = PixelShuffle(X, r)                               features from the frozen vision encoder (e.g., patch em-
                                                              (9)
                 = reshape X, H/r, W/r, C · r2 ,                    beddings of the image). Through this process, the queries
                                              
                                                                    selectively aggregate task-relevant visual information into
                 H   W      2                                       a compact set of embeddings, which are then linearly pro-
where Y ∈ R r × r ×(Cr ) .
                                                                    jected into the language embedding space and fed to the
   This operation reduces the spatial token count by a factor       LLM as visual tokens. Building upon this principle, Q-
of r2 while increasing the channel dimension accordingly,           Former efficiently compresses hundreds of visual tokens
thus effectively trading token number for richer per-token          into only a few while preserving essential semantics, pro-
channel representation. An additional module, typically an          viding a parameter-efficient and highly adaptable bridge
MLP, is then applied to align the expanded channel dimen-           between vision encoders and LLMs. This design not only
sion with the embedding dimension required by the LLM.              enables effective multimodal understanding but has also
Such a token compression strategy has also been employed            been widely adopted and extended in subsequent works
in many well-known models, including InternVL 1.5 [109]             such as MiniGPT-4 [116] and InstructBLIP [117].
and NVLM [201].
                                                                    Variants of Q-Former. Some later works [113], [118]–[120],
Convolution-Based. Compared with parameter-free meth-               [181] have proposed simplified and enhanced variants of the
ods such as pooling or pixel shuffle, convolutions selectively      Q-Former architecture. For example, Qwen-VL [118] adopts
integrate local information through learnable weights rather        a single-layer cross-attention module, reducing architectural
than merely taking the mean or maximum, thus preserving             complexity while retaining the ability to aggregate visual in-
more task-relevant details. By stacking convolutional layers        formation and perform token compression. Honeybee [113]
or using variable kernel sizes, the model can also capture          further observes that the conventional Q-Former may lead
multi-scale abstract features, offering greater flexibility than    to the loss of fine-grained spatial information. To address
simple pooling.                                                     this issue, it introduces two locality-enhanced projectors:
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                     10

C-Abstractor and D-Abstractor. The C-Abstractor combines           kens. To address this limitation, they introduce cross-modal
ResNet blocks with average pooling to perform downsam-             attention mechanisms, enabling more precise identification
pling while preserving local structures, whereas the D-            and extraction of task-relevant information. AdaFV [204]
Abstractor leverages the idea of Deformable Attention [202],       proposes a self-adaptive cross-modality attention mixture
using reference points and sampling offsets to enhance lo-         mechanism that dynamically selects visual tokens based
cality while maintaining flexibility in the number of output       on visual saliency and text–image similarity. VCM [186]
tokens. MQT [119] proposes a variant of the Q-Former               introduces the concept of Vision Concept Modeling, which
architecture that allows a variable number of query tokens.        dynamically determines the number and spatial locations of
Specifically, given M query tokens, MQT randomly samples           required visual concepts according to a given instruction. It
the first m (m < M ) tokens during training, enabling the          employs a multi-head cross-attention layer as a key compo-
model to learn visual representations at varying granulari-        nent for semantic alignment in keyword selection, aligning
ties. On average, this strategy reduces the number of visual       visual features with training signals to guide subsequent
tokens by about half compared with the original Q-Former,          token retention or aggregation. Based on the number and
while maintaining effective information compression. TG-           relevance of the selected keywords, VCM further estimates
LLaVA [120] emphasizes the role of textual instructions in         the optimal number of tokens to retain.
guiding key visual feature extraction. It introduces learnable
latent embeddings to encode global text semantics and em-          3.2.3   Importance-Driven Compression
ploys a single-layer Q-Former to integrate textual and visual
information. The resulting mask is applied to the visual           Importance-driven token compression refers to methods
features, refining them under text guidance. Considering           that reduce visual token redundancy by estimating the
that using a standalone Q-Former may still lead to the             importance of each token and selectively retaining the most
loss of visual information, LLaVA-Mini [181] introduces an         valuable ones. Rather than relying on fixed-length queries
additional Modality-Pre Fusion module, which fuses visual          or simple pooling, these approaches identify the relative
representations with the instruction tokens before feeding         importance of tokens and selectively prune or merge less in-
them into the LLM, thereby mitigating such information             formative ones. Existing strategies include similarity-based
loss.                                                              methods, attention-based methods, saliency-based methods,
                                                                   and innovative metrics-based methods, which will be dis-
Cross-Attention-Based. Instead of relying solely on the            cussed in detail below. This perspective highlights how im-
Q-Former’s compressed token representations, some meth-            portance estimation shapes the trade-off between efficiency
ods [203] utilize the cross-attention mechanism to identify        and information preservation in MLLMs.
or extract task-relevant tokens.
                                                                   Various Similarity Metrics. There exist various approaches
    CATP [203] performs voting based on the cross-attention
                                                                   to measuring token similarity. DynTok [125] introduces a
probabilities between query tokens and image tokens, ac-
                                                                   dynamic token compression method based on local token
cumulates the scores across different layers and heads, and
                                                                   similarity. Its core idea is to exploit the varying information
prunes tokens according to their aggregated importance.
                                                                   density of image patches across video frames: DynTok adap-
    Several works [122]–[124], [176] move away from re-            tively groups visual tokens and merges them within each
lying on learnable queries for token compression. Token-           group, thereby preserving more tokens in high-information-
Packer [122] employs a coarse-to-fine visual information           density regions while achieving higher compression ratios
extraction strategy. It first downsamples the original visual      in less informative areas. Experiments show that computing
features to obtain low-resolution representations that act as      cosine similarity on CLIP-generated visual representations
point-based queries. These queries are then paired with their      yields better performance than directly measuring similarity
corresponding regions in the high-resolution features to           in the LLM embedding space. LLaVA-Scissor [126] proposes
form point–region pairs, which iteratively interact through        Semantic Connected Components (SCC), reframing token
Point-to-Region cross-attention, progressively injecting rich      compression as a graph connected components partitioning
visual information. Similarly, HiRes-LLaVA [123] abandons          task. By explicitly covering all semantic regions, this method
learnable queries and leverages downsampled visual fea-            alleviates the common bias of attention-based approaches,
tures as queries that interact with the original visual features   which often overemphasize only the most salient objects. In
via cross-attention, resulting in a compact, compressed se-        SCC, token similarity is likewise measured using the cosine
quence. mPLUG-DocOwl2 [124] uses global visual features            similarity of visual embeddings.
as queries and the cropped image features as keys and val-
ues, performing cross-attention to aggregate text semantics        Saliency-Based. SeqCompression [127] conducts a com-
while significantly reducing the number of visual tokens           parative study between saliency-based and importance-
for high-resolution images. QueCC [176] further injects the        agnostic token compression strategies, demonstrating that
textual features of the user query into the visual represen-       the saliency-based “Cluster and Aggregate” method offers
tations, enabling subsequent queries to carry task-specific        clear performance gains. Specifically, after the vision en-
semantics. Combined with the cross-attention between the           coder and projector, visual tokens are clustered with K-
downsampled tokens and their respective visual token re-           means++ according to embedding similarity, and tokens
gions, it achieves extreme visual token compression while          within each cluster are subsequently merged into a single
maintaining strong relevance to the textual task.                  representative token by averaging their embedding vectors.
    Other studies argue that relying solely on visual features     Innovative Metrics-Based. Some studies [128] depart
is insufficient to accurately identify the most informative to-    from the common attention-based similarity measures and
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                                                          11

instead propose novel definitions and designs for to-                                                                               Forward
ken importance, similarity, or diversity. For example, Di-                 Vision
                                                                                       Projector                                                                        vision token
                                                                          Encoder                                      LLM                              LLM
vPrune [128] formulates token pruning as a Max–Min Di-




                                                                                                                                                           · ··
                                                                                                                                    Token




                                                                                                      · ··
                                                                                                                       Layer       Reduction            Layer
versity Problem (MMDP), aiming to construct a token subset                                                              N                               N+1             text token

with the maximum minimum distance based on the original                              Tokenizer

token set.                                                          (a) Importance-based      (b) Learnable module-based       (c) Token Merging-based      (d) Fusion-based

                                                                   Importance
3.3   Token Compression in LLM                                       Metric
                                                                                                   Learnable Ranking
                                                                                                                                                                    LLM Layer N
                                                                                                                                                                K
                                                                                                   Module                                                           Cross Attention
Currently, the mainstream architectures for MLLMs typi-                             Drop
                                                                                                                                                                V
cally follow a classic design wherein visual information,                                                                                                                  Q
                                                                                                                                 Merge similar tokens
after being processed by a vision encoder and a projector,             Token Ranking

generates a large number of vision tokens. Given that the          Fig. 5: Illustration of token compression strategies applied
LLM component generally contains significantly more pa-            at the LLM module.
rameters than the vision encoder and projector, the resulting
sequence incurs substantial computational overhead when            to concatenate all vision and text tokens into a single long
forwarded through the LLM.                                         sequence.
    To address this issue, a growing body of research has
focused on reducing token redundancy within the LLM                Importance-based This category of methods typically uti-
component. These methods can be broadly categorized                lizes importance metrics to score vision token, followed by
based on the generation stage at which token reduction is          a ranking process to retain only the most important tokens.
applied, as illustrated in Figure 5. The first category (§3.3.1)   Among these metrics, the most commonly adopted is the
performs token compression during the prefilling stage, i.e.,      attention from textual tokens to visual tokens, which helps
it reduces the number of vision tokens at the first forward        preserve vision tokens that are relevant to the query. This
pass of the sequence through the LLM. This approach was            allows for aggressive pruning of redundant visual infor-
primarily motivated by early use cases such as short-form          mation without significantly affecting model performance.
visual question answering (VQA), where the cost of the             FastV [205] was among the first to observe that vision tokens
prefilling stage dominates that of decoding. However, with         receive substantially lower attention scores compared to text
the rapid advancement of chain-of-thought (CoT) and the            tokens within the LLM, revealing the extreme sparsity in the
increasing demand for long-form generation, attention has          information carried by vision tokens. Based on this observa-
shifted to methods that apply token reduction during the           tion, FastV prunes half of the vision tokens at the second
decoding stage (§3.3.2). These techniques typically reduce         layer of the LLM using the attention from the last textual
the memory and computational cost by selectively pruning           token. PyramidDrop [206] further extended this line of
or merging parts of the key-value cache (KV cache), which          analysis by identifying that the redundancy of vision tokens
proves especially beneficial for long-sequence generation          tends to increase with LLM depth. Leveraging this insight,
tasks.                                                             it introduced a multi-stage progressive pruning strategy.
                                                                   Following these pioneering works, a number of subsequent
3.3.1 Compression in Prefilling Stage                              studies, including [71], [134], [136], [137], [207]–[210], have
The prefilling stage refers to the first forward pass of all       adopted text-to-image attention ranking as a straightfor-
tokens through the LLM. Once a vision token is removed             ward and effective approach, applying either single-stage
in the shallow layers of the LLM, deeper layers are no             or multi-stage pruning schemes. Beyond simple attention-
longer able to access information from the corresponding           based ranking, some recent efforts have focused on refining
image region. As a result, achieving significant acceleration      the evaluation of token importance [135], [188], [211], [212].
during this stage while maintaining model performance is           For instance, SparseVLM [132] and AdaptInfer [213] pro-
particularly challenging. Existing approaches often rely on        pose more fine-grained methods for selecting text tokens
observations about the inherent behavior of LLMs when              that are most relevant to visual content, using these to more
processing tokens. By analyzing patterns in vision token           accurately assess the significance of vision tokens. Other
redundancy, researchers have proposed four representative          methods such as TransPrune [214] and VFlowOpt [214]
methods.                                                           combine attention scores with additional indicators, like
    One common line of work involves importance-based              information entropy maps, to improve the robustness of
approaches, which rank and retain vision tokens accord-            token importance estimation.
ing to predefined metrics that estimate their significance.            Additionally, VTW [131] takes a more radical approach
Another line involves learnable module-based approaches,           by entirely removing vision tokens from certain layers of
where additional trainable components are used to deter-           the model. In contrast, CrossMisalign [215] leverages vision-
mine token importance and the appropriate compression              to-vision attention mechanisms to assess token importance,
ratio. Both of these strategies primarily apply direct prun-       bypassing reliance on textual signals altogether.
ing, retaining only a subset of tokens without integrating             Attention Bias Problem. As attention-based pruning meth-
information from the discarded ones. In contrast, token            ods have advanced, several inherent issues have emerged.
merging-based methods adopt a softer strategy by merging           One such issue is the attention bias observed during impor-
similar vision tokens to preserve information. Lastly, fusion-     tance estimation. Feather [133] first noted that vision tokens
based approaches inject visual information through cross-          located near output tokens in the input sequence tend
attention or self-attention mechanisms, avoiding the need          to receive disproportionately high attention scores in the
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                   12

shallow layers of the LLM. This phenomenon is attributed           scores of all visual tokens. Beyond predicting token impor-
to the long-term decay property of Rotary Position Embed-          tance, learnable modules have also been applied to estimate
ding (RoPE). To mitigate this, Feather proposes computing          the compression rate of the entire sequence. For example,
importance without applying RoPE to eliminate positional           DyRate [142] incorporates a lightweight classifier to predict
bias. AdaTP [216] addresses attention bias challenge by            the optimal pruning ratio for each input sequence. ATP-
introducing a dedicated text encoder to compute cosine sim-        LLaVA [177] employs a MLP with dual prediction heads
ilarity between textual and visual features, thereby offering      to learn instance-specific thresholds for token pruning. This
a more balanced measure of token importance. VScan [71],           design enables adaptive token reduction during the genera-
on the other hand, avoids the issue altogether by initiating       tion process.
pruning from the intermediate layers of the LLM, rather
                                                                   Token Merging-based In contrast to the previous two
than at the shallow layers where attention bias is more
                                                                   approaches that perform direct pruning by discarding less
pronounced.
                                                                   important tokens, token merging offers a softer compres-
    Flash Attention Compatibility Problem. A technical chal-       sion strategy. Token merging techniques compute similarity
lenge arises when integrating these attention-based pruning        measures and apply grouping or clustering algorithms to
strategies with Flash Attention [217], which does not di-          fuse multiple vision tokens into fewer representative ones,
rectly expose attention scores due to its design. Using stan-      thereby achieving compression. Such methods were initially
dard attention mechanisms across all layers would degrade          popularized in the context of accelerating Vision Transform-
inference efficiency. A common solution involves applying          ers (ViTs). For example, ToMe [34] introduces a bipartite
Flash Attention at all layers, but selectively recomputing         soft matching algorithm to perform efficient token merging
the queries, keys, values, and attention maps only at the          based on pairwise similarity.
specific layers where attention-based ranking is needed.               In the MLLM setting, LLaVolta [144] is one of the first
While this solution mitigates the overhead for single-layer        works to apply token merging, using a simple and direct
pruning, inference latency increases significantly if pruning      average pooling strategy to aggressively compress vision
is conducted at multiple layers. To address this more fun-         tokens. To mitigate the loss of performance caused by
damentally, some studies have proposed alternative metrics         heavy compression, LLaVolta employs progressively lower
that bypass the need for attention scores entirely [218]. TopV     compression ratios with multiple training stages. Subse-
[138], for instance, ranks tokens using a combination of           quent methods have proposed more sophisticated designs
feature similarity, relative spatial distance, and absolute cen-   in both similarity computation and clustering mechanisms.
tral distance. PACT [139] incorporates hidden state norms          FiCoCo [86], for instance, first selects a subset of impor-
in conjunction with a global query vector to assess token          tant tokens and then computes a correlation matrix be-
importance. GreedyPrune [219] employs cosine similarity            tween these preserved tokens and the remaining ones. The
between text and vision tokens as a ranking criterion. It is       merging process is then guided by minimizing information
also worth noting that CATP [121] takes into account the           loss based on this matrix. In CrossMisalign [215], token
differences across layers, proposing a composite ranking           merging is used primarily as a visual information recovery
method that combines semantic relevance with layer-wise            mechanism. This method introduces a specialized recovery
attention variations to produce more robust token impor-           scheme that merges semantically redundant tokens with
tance estimations.                                                 their most similar counterparts, based on a dot-product
Learnable Module-based Unlike importance-based meth-               similarity calculated from reused attention key embeddings.
ods that rely on predefined metrics to rank tokens, learn-         In the video domain, to address inter-frame redundancy,
able module-based approaches introduce trainable com-              FrameFusion [145] computes cosine similarity between each
ponents that learn to assess token importance or deter-            visual token and its spatially corresponding token from the
mine the appropriate compression ratio during training,            preceding frame. This approach aims to minimize repetitive
thereby enabling dynamic compression. This paradigm was            information across consecutive frames by merging tokens
widely adopted in early Vision Transformer (ViT) research.         that represent similar spatial regions over time. Compared
For instance, DynamicViT [33] and AdaViT [220] attach              with such a sophisticated design, HoliTom [146] takes a
lightweight decision networks to the ViT backbone as learn-        relatively simple approach by directly merging those tokens
able modules, and employ Gumbel-Softmax [221] during               with lower attention scores.
training to render the framework fully differentiable. This        Fusion-based The previously discussed methods achieve
design significantly improves the computational efficiency         compression by directly pruning or merging tokens, thereby
of ViTs.                                                           shortening the overall sequence length. In contrast, fusion-
    In the MLLM domain, several works have adopted sim-            based approaches implement compression indirectly by
ilar strategies [141]. In p-MoD [140], a weight predictor is       leveraging cross-attention or self-attention module to in-
proposed to assign importance scores to each token. Before         tegrate visual information into other tokens, effectively
each layer, tokens are sorted by their predicted weights,          avoiding excessively long input sequences. An early ex-
and only the top R% of tokens are retained for further             ample is Flamingo [147], which introduced cross-attention
processing. This allows the model to dynamically preserve          layers called GATED XATTN-DENSE, between layers of a
informative visual tokens and skip less relevant ones in           pretrained language model. In this setup, the original text
a flexible manner. Similarly, GlimpsePrune [143] utilizes a        tokens serve as queries, while visual features are treated
visual token importance predictor to estimate the signifi-         as keys and values, enabling deep interaction between vi-
cance of each token at a given layer, based on the attention       sion and language representations. Building on this idea,
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                     13

mPLUG-Owl3 [148] adopts a similar architecture by com-                  Promising progress has also been made in KV-cache
bining intra-text self-attention with cross-modal attention         compression in video domain, where the visual input is
between text and image features. This design leverages tex-         especially long and redundant. DyCoke [152] proposes a
tual tokens as queries to selectively extract relevant visual       dynamic compression mechanism based on the text-vision
information, eliminating the need to pass through a long            attention. In each decoding step, only the KV pairs with high
sequence of visual tokens. More recently, CrossLMM [149]            attention scores are retained. If the attention distribution
has advanced this direction further by introducing a design         shifts significantly in subsequent decoding steps, the KV
where compressed visual tokens and text tokens are used as          cache is updated accordingly. Given the substantial redun-
queries, while the original long-sequence visual representa-        dancy present in video frames, Video-XL-2 [58] introduces a
tions act as keys and values. It incorporates visual-to-visual      novel Bi-level KVs decoding. Based on the current query, the
and text-to-visual cross-attention, ensuring that the LLM can       model dynamically selects whether to retrieve from dense
access high-resolution visual content while mitigating the          or sparse KV representations, allowing it to discard a large
performance degradation.                                            number of query-irrelevant KV pairs. In video streaming
    In addition to cross-attention-based fusion, another line       scenarios, several specialized strategies have been proposed.
of work explores extracting visual information via learnable        LiveVLM [155], for instance, first discards KV pairs of
tokens through self-attention. For example, VoCo-LLaMA              unimportant visual tokens based on attention scores, then
[150] introduces a single Vision Compression token and              merges the original KVs of each frame into a single KV
modifies the attention mechanism such that textual tokens           tuple. InfiniPot-V [154] explores the estimation of token
only attend to the VoCo token, effectively forcing the model        importance by integrating two distinct evaluation criteria:
to abstract visual information into this compressed represen-       Temporal-axis Redundancy (TaR) and Value Norm (VaN).
tation. Following a similar philosophy, Victor [222] appends        These jointly guide the model to retain only the most critical
learned visual register tokens after the visual tokens, which       tokens during KV-cache compression. In addition, Stream-
absorb visual content through attention. All original visual        Mem [156] implements KV-cache compression based on
tokens are subsequently discarded in deeper layers, thereby         attention scores between visual tokens and generic queries.
achieving efficient compression.                                    This is done while operating within a fixed-size KV memory
                                                                    to allow for efficient question answering.
3.3.2 Compression in Decoding Stage
Compression in the Decoding Stage typically refers to KV-           3.4   Token Compression in Multi-Module
cache compression, which aims to reduce the memory and              Beyond applying token compression within individual com-
computational overhead of cached key and value tensors in           ponents such as the vision encoder (Sec. 3.1), the projector
transformers during autoregressive decoding. This is com-           (Sec. 3.2), or the LLM (Sec. 3.3), an increasing number
monly achieved through pruning, quantization, or merging            of recent approaches explore compression strategies across
strategies, with the goal of preserving generation quality          multiple modules to achieve higher compression efficiency
while improving efficiency. Due to the inherently long out-         and improved representational quality.
puts produced by LLMs, there has long been a pressing need              Since most multi-module token compression approaches
for KV-cache optimization. Consequently, a wide range of            are essentially built by combining the single-module tech-
KV-cache compression techniques have been developed in              niques introduced earlier, we do not revisit their low-level
the LLM domain, as seen in works such as StreamLLM [223],           technical details here. Instead, we focus on how these meth-
FastGen [224], and H2o [225].                                       ods coordinate compression across different components
    In the multimodal setting, this challenge has become            and organize it as a multi-stage process to maximize overall
increasingly significant with the rise of multimodal chain-         efficiency and representational quality. In the following,
of-thought (CoT) reasoning. Output lengths have expanded            we analyze two emerging design paradigms: multi-module
from a few sentences to hundreds or even thousands of               collaborative compression, which emphasizes the joint and
tokens, making both computational load and KV-cache                 coordinated reduction of tokens across vision and language
memory consumption critical bottlenecks in the generation           pathways, and multi-stage progressive compression, which
process. As a result, a growing number of studies have              structures token reduction as a progressive pipeline span-
focused on KV-cache compression tailored for multimodal             ning early visual processing to late-stage LLM inference.
models [136], [141], [226]. One of the earliest works, LOOK-
M [151], proposes using cumulative attention scores to              3.4.1 Collaborative Compression.
estimate token importance. It preserves the KV pairs of the         CrossGET [82] is one of the earliest works to adopt multi-
most recent window and additionally retains a proportion            module token compression. It inserts CrossGET modules
of visual KV pairs ranked by their importance. Another              between the self-attention and FFN layers of both the visual
example, MustDrop [92], addresses both the prefilling and           and language branches, reducing the token count across
decoding stages. It stores only the KV pairs of visual tokens       layers. This design addresses the limitation of previous
that are retained in the final layer during the prefilling stage.   methods that required extracting visual information first
Recognizing that not all attention heads equally contribute         and thus lacked text-guided supervision in early stages,
to visual understanding, SparseMM [153] first identifies            enabling the earlier visual processing layers to be informed
visual heads using an OCR-based task, then allocates more           by subsequent textual features. LLaMA-VID [98] leverages
KV-cache budget to these heads. For non-visual heads, it            cross-modal interaction between visual tokens and textual
adopts a more aggressive compression policy, striking a             queries to extract task-relevant visual information and gen-
balance between performance and efficiency.                         erate context tokens. It further applies pooling on the visual
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                                          14

features to obtain content tokens, enabling each video frame




                                                                 How to select desirable strategy (§4)
to be represented by only two tokens (a context token and                                                      Temporal-Enhanced Compression for Videos (§4.1)
a content token), which facilitates efficient understanding
of long videos. PAR [100] provides a finer-grained analysis                                                    Purely-Visual vs. Text-guided Compression (§4.2)
of visual token redundancy, categorizing it into external
redundancy and internal redundancy. To address external
redundancy, PAR removes task-irrelevant tokens through                                                             Token Merging vs. Token Dropping (§4.3)
query rewriting, semantic clustering of visual tokens, and
semantic retrieval. For internal redundancy, it introduces a
                                                                                                                Plug-in Methods vs. Re-training Methods (§4.4)
token router mechanism that further eliminates redundant
tokens by applying predefined similarity and redundancy
thresholds.                                                                                                      Efficient Training vs. Efficient Inference (§4.5)

3.4.2 Progressive Compression.
To further improve the inference efficiency of MLLMs, sev-     Fig. 6: Decision taxonomy for selecting an appropriate token
eral studies [86], [92], [152], [227] have proposed rigorous   compression strategy (see §4).
multi-stage token compression strategies that span multiple
steps and phases of the inference process. MustDrop [92]
adopts a multi-stage token compression strategy with care-     (§ 4.2); (iii) token merging versus token pruning, comparing
fully designed mechanisms across the vision encoding, pre-     fundamental compression paradigms (§ 4.3); (iv) plug-in
filling, and decoding stages. It combines techniques such      versus retraining methods, weighing deployment flexibility
as merging highly similar spatial tokens, dual-attention       against performance optimization (§ 4.4); and (v) efficient
filtering, and output-aware KV cache to achieve end-to-        training versus efficient inference, distinguishing between
end acceleration throughout the entire inference pipeline.     optimization objectives (§ 4.5).
DyCoke [152] also employs a two-stage token compression            For each factor, we analyze underlying technical advan-
strategy. In the first stage, it merges and removes tokens     tages and disadvantages, providing practical recommenda-
by computing the cosine similarity between corresponding       tions based on deployment constraints.
tokens in adjacent video frames. In the second stage, it
performs dynamic pruning within the KV cache, adaptively
evaluating and retaining tokens based on their attention       4.1                                       Temporal-Enhanced Compression for Video Input
scores. FiCoCo [86] formulates token compression as a three-   Compared with static images, video input introduces an
stage process: filter, correlate, and compress, addressing     additional temporal dimension that substantially increases
three key questions: which tokens to discard, where to         computational demands. As video duration grows or frame
preserve discarded information, and how to fuse remaining      sampling rates rise, the number of visual tokens fed into lan-
tokens while retaining critical information.                   guage models increases explosively, creating a fundamental
     In summary, token compression in multi-module archi-      tension between inference efficiency and modeling fidelity.
tectures represents a shift from isolated, single-stage re-    Although existing spatial compression strategies (refer to
duction to a more holistic and system-level optimization       Sec 3.1) can be directly applied to individual frames, they
of MLLMs. Rather than limiting compression to a single         often fail to exploit cross-frame redundancy. To address
component such as the vision encoder, projector, or LLM,       this gap, recent research has proposed temporal-enhanced
these approaches strategically integrate multiple stages of    token compression methods that explicitly consider temporal
reduction — from early spatial downsampling and semantic       structure for efficient long-sequence modeling. Three central
clustering to query-guided selection and late-stage pruning    challenges emerge:
— in order to maximize both efficiency and representational     1) Spatial-temporal interaction: How to jointly compress
quality. This trend highlights an important direction for          across spatial (h, w) and temporal (t) dimensions to
future research: optimizing token reduction as a coordi-           form compact yet expressive representations (§ 4.1.1).
nated, end-to-end process rather than a set of independent,     2) Temporal structure preservation: How to retain spa-
module-specific techniques.                                        tiotemporal structure after compression for fine-grained
                                                                   perception tasks such as motion direction estima-
4  H OW TO S ELECT THE D ESIRABLE TOKEN C OM -                     tion [239], [240] and temporally grounded QA [9], [241]
PRESSION S TRATEGY                                                 (§ 4.1.2).
                                                                3) Scalability to extreme lengths: How to design com-
The proliferation of token compression designs necessitates        pression and memory mechanisms that scale to hour-
guidelines to help practitioners select optimal strategies         long videos containing tens of thousands of frames
for specific deployment scenarios. As Figure 6 illustrates,        (§ 4.1.3).
this section provides a comprehensive comparison of crit-
ical selection factors: (i) temporal-enhanced compression
for video inputs, which focuses on unique challenges of        4.1.1                                      Spatial-Temporal Compression
processing long temporal sequences (§ 4.1); (ii) text-guided   Joint spatial-temporal compression strategies can be broadly
versus purely visual compression, examining the trade-offs     divided into fixed and dynamic approaches, with hybrid
between cross-modal guidance and visual-only approaches        strategies emerging at the intersection.
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                  15

            TABLE 2: Overview of temporal-enhanced compression strategies for video input (details refer to §4.1).
 Category     Method                Key Idea                                               Representative Works
              Pooling               Average neighbor tokens across temporal dimen-         PLLaVA [108], Video-ChatGPT [228]
                                    sion.
 Fixed        Convolution           2D/3D convolutions for joint spatio-temporal           VideoLLaMA2 [114], Qwen2-VL [25]
                                    downsampling.
              Query-based           Learnable queries that attend over all video tokens    Clapper [78], LinVT [88], CrossLMM [149]
                                    (e.g., Q-former, Token Learner, Resampler).
              Sequential Models     Process tokens in temporal order with explicit         BLIP-3-Video [229], STORM [230]
                                    timestamp embeddings and recurrent memory.
              Token Merging         Merge redundant tokens across frames                   TESTA [83], AuroraCap [231], DyCoke [152]
 Dynamic
              Token Dropping        Drop temporally low-saliency or redundant tokens       LongVU [56], TimeChat-Online [57]
              Global-Local Fusion   Global event-level clustering with local frame-level   LongVLM [232], Video-XL [55], HiCom [233],
 Hybrid                             aggregation.                                           PruneVid [179], Chat-UniVi [81], FiLA-Video [80],
                                                                                           TempMe [174], Quicksviewer [234]
              Slow-Fast Pathways    Two-Stream architecture: a high-resolution slow        SlowFast-LLaVA [235], LLaVA-Video [236], Clap-
                                    pathway for spatial detail and a low-resolution fast   per [78], Keye-VL-1.5 [237]
                                    pathway for motion dynamics.
              Memory-bank           Long-term memory complemented by a short-term          MovieChat [173],    VidCompress     [238],   Flash-
                                    memory.                                                VStream [197]



Fixed Temporal Compression. Fixed strategies reduce the                 dependencies, yielding token representations enriched with
number of tokens per frame or clip from N to a prede-                   historical context.
fined M (M ≪ N ). Early Video-LLMs commonly adopted                     Dynamic Temporal Compression. In contrast to fixed-ratio
uniform frame sampling or downsampling to bound token                   compression that uniformly compresses videos of varying
budgets. Pooling-based designs (e.g., PLLaVA [108], Video-              information density into identical token counts, dynamic
ChatGPT [228], TC-LLaVA [107]) average patches across                   compression methods adaptively adjust the number of re-
adjacent frames to suppress redundancy, but often at the                tained tokens based on video content, enabling differenti-
cost of motion detail. Convolution-based designs integrate              ated modeling between static and dynamic segments.
temporal information more explicitly: VideoLLaMA2 intro-                    Temporal Token Merging. TESTA [83], AuroraCap [231],
duces a 3D spatio-temporal convolution (STC Connector)                  and DyCoke [152] merge similar or redundant tokens across
combined with RegStage to preserve local dynamics under                 frames, typically identifying merge candidates via token
reduced cost, while Qwen2-VL [25] applies 2D convolutions               similarity. Building upon this foundation, learnable merging
to fuse adjacent-frame features. To enhance temporal em-                strategies have emerged. InTI [243] introduces a lightweight
bedding, Qwen2.5-VL [4] adopts 3D convolutional module                  weight prediction network that generates dynamic weights
to downsample both spatially (4x) and temporally (2x).                  for spatially co-located tokens in adjacent frames, enabling
    Query-based designs represent another reserach line. In-            more adaptive fusion. Similarly, Learnable VTM [244] as-
stead of pooling all tokens, they learn a compact set of query          signs learnable saliency scores to each token, supporting
tokens (e.g., Q-former, Resampler, Token Learner) that ag-              dynamic merge ratios that substantially reduce token counts
gregate salient information through attention. For instance,            while preserving critical information.
Clapper’s TimePerceiver applies cross-attention to capture                  Temporal Token Pruning. Unlike merging, pruning directly
inter-frame dynamics, while LinVT and CrossLMM [149]                    discards less important tokens rather than fusing them [56],
leverage user queries to guide compression, producing                   [57], [245]. LongVU [56] proposes a three-stage compression
lightweight yet semantically aligned representations.                   pipeline where the final temporal-dependency-based spatial
    Sequential models leverage linear complexity O(n) to effi-          token pruning uses the first frame as an anchor within
ciently encode long video token sequences by first enhanc-              each sliding window, computing cosine similarity between
ing temporal modeling before compressing token counts.                  spatially aligned tokens across frames and discarding highly
BLIP-3-Video [229] proposes a Grouped Sequential Model                  similar ones to achieve extreme spatial compression. Simi-
that processes video tokens in temporal order while group-              larly, TimeChat-Online [57] retains only temporally dynamic
ing them by spatial location. Each group maintains in-                  information by measuring redundancy between temporally
dependent temporal memory augmented with timestamp                      adjacent, spatially co-located tokens and discarding redun-
positional encodings at each update step, ultimately aggre-             dant tokens in subsequent frames. This work also demon-
gating into merely 16-32 video-level tokens via attention               strates that feature-level redundancy measures outperform
mechanisms. Through systematic comparison against vari-                 pixel-level approaches.
ous fixed compression methods, BLIP-3-Video demonstrates                Hybrid Strategies combine multiple principles to balance
that its Grouped Sequential Model outperforms traditional               global coverage with local detail. A prevalent design is
pooling and attentional pooling by preserving absolute                  global-local fusion, which clusters video segments into key
temporal order alongside semantic completeness. Similarly,              events and then performs intra-event aggregation, thereby
STORM [230] leverages the Mamba State Space Model [242]                 capturing both coarse event structure and fine-grained dy-
to integrate temporal information, employing bidirectional              namics. Representative works include PruneVid [179], Chat-
scanning to simultaneously capture spatial and temporal                 UniVi [81], and FiLA-Video [80]. LongVLM [232] combines
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                    16

local token merging within clips with global semantic rep-           embedding as follows:
resentations across all frames. TempMe [174] and Video-
                                                                                                        T −2
                                                                                                              
XL [55] employ hierarchical merging or visual summariza-                                       1
                                                                                   t = 0,         ,...,      ,1              (11)
tion tokens (VSTs) to reduce redundancy while preserving                                     T −1       T −1
temporal context. HiCom [233] groups sampled frames                      Temporal Encoding Modules. Beyond positional embed-
in the spatiotemporal domain and performs instruction-               dings, dedicated architectural components can explicitly
conditioned compression, whereas Quicksviewer [234] uses             model temporal dependencies. STORM [230] leverages
Gumbel-Softmax to determine information density and per-             Mamba-based [248] state-space layers (MambaMixer) that
forms block-wise resampling to reduce irrelevant redun-              inject temporal awareness through bidirectional scanning
dancy.                                                               of token sequences, simultaneously capturing both spatial
    Slow-fast dual streams. Inspired by action recognition,          and temporal dependencies. PVC [178] adopts a progressive
SlowFast-LLaVA [235], LLaVA-Video [236], and Clapper                 encoding strategy where tokens of each frame are sequen-
process video through two pathways: a slow pathway with              tially encoded and adaptively compressed to supplement
low frame rate but high spatial detail, and a fast pathway           information not extracted from previous frames, ensuring
with high frame rate but compact tokens. Keye-VL 1.5 [237]           cumulative temporal context preservation.
further refines this by dynamically routing salient frames               Special Timestamp Tokens. An alternative strategy intro-
to the slow branch while assigning static frames to the fast         duces explicit timestamp representations as separate to-
branch, significantly improving token efficiency.                    kens. Video-XL-2 [58] interleaves timestamp tokens within
    Memory-bank mechanisms. Flash-VStream [197] intro-               the visual token sequence to enhance temporal awareness
duces STAR memory, consisting of (i) a Context Synopsis              throughout the model. Qwen3-VL [249] advances this ap-
Memory that clusters low-resolution features into centroids          proach by adopting a textual token-based time encoding
to preserve global temporal trends, and (ii) a Detail Aug-           strategy [193], [250], [251], wherein each video temporal
mentation Memory that selectively retains high-resolution            patch is prefixed with a timestamp expressed as a formatted
tokens for keyframes. This design offers flexible token              text string (e.g., <3.0 seconds>), moving beyond tradi-
budgets while balancing coverage and detail. Similarly,              tional Video-RoPE to achieve precise, timestamp-grounded
MovieChat [173] combines sliding windows with long-term              event localization for stronger video temporal modeling.
and short-term memory, periodically merging tokens when
capacity is exceeded, while VidCompress [238] enhances               4.1.3   Extreme-Long Video Compression
this approach with memory-augmented cross-clip attention.
                                                                     In hour-long video scenarios, MLLMs must process thou-
                                                                     sands of frames, posing severe challenges to computational
4.1.2   Temporal Structure Preservation                              efficiency and memory management. Addressing these chal-
                                                                     lenges necessitates specialized designs across multiple di-
During video compression, atomic operations such as token            mensions, including input sampling, encoding compression,
merging and pruning can blur or discard the spatiotemporal           memory storage, and inference acceleration.
positional information of visual tokens, thereby disrupting              Early explorations into long video understanding pri-
the original temporal structure of videos. This degradation          marily focused on memory bank-based approaches to
impairs MLLMs’ ability to perceive precise timestamp in-             store long-term temporal semantics. MovieChat [173] pi-
formation, adversely affecting tasks that require absolute           oneered the integration of sliding windows with dual-
temporal localization, such as temporal grounding [9], [246].        memory mechanisms, where short-term memory captures
To mitigate this issue, several works have introduced ex-            fine-grained details within the current window while long-
plicit time-aware mechanisms, which can be categorized               term memory aggregates global semantics from historical
into three main approaches: augmenting video tokens with             segments, enabling processing over 10,000 frames on a
temporal positional embeddings, incorporating dedicated              24GB GPU. Similarly, FlashVStream [252] proposes a more
temporal encoding modules within the overall architecture,           elaborate flash memory architecture to achieve real-time
and inserting special timestamp tokens.                              responses to user queries.
    Temporal Positional Embeddings. The most direct approach             Beyond the memory-bank foundations, the Video-XL
is to enrich visual tokens with temporal positional infor-           series demonstrates a clear evolution from adaptive com-
mation. BLIP-3-Video’s Grouped Sequential Model [229]                pression to comprehensive optimization. Video-XL [55] in-
processes frames sequentially with timestamp positional              troduces dynamic interval partitioning that assigns varying
encodings and grouped memory mechanisms, maintaining                 numbers of Visual Summarization Tokens (VSTs) to com-
separate temporal memory across different token groups to            press visual semantics. It enables processing 2,048 frames
preserve both local temporal details and absolute tempo-             with near-lossless 16x compression and 95% accuracy at
ral order. TimeChat-Online [57] retains the original Video-          32x compression. Video-XL-Pro [199] advances this by intro-
RoPE [4], [247] positional encodings for important tokens            ducing reconstructive capability through the ReCoT frame-
selected based on visual dynamics, thereby preserving spa-           work, which synthesizes dynamic tokens to capture motion
tiotemporal information relative to the original video even          patterns, employs semantic-guided masking to focus on
after pruning operations. PVC [178] also employs relative            dense regions, and incorporates query-aware selection to
timestamps to indicate video frames and obtains temporal             prune low-relevance tokens. This enables handling over
embeddings via MLP. Specifically, it uses either absolute            8,000 frames with near 99% accuracy. While these efforts
positional embedding t = [0, 1, . . . , T ] or relative positional   optimize training-time compression, Video-XL-2 [58] shifts
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                    17

TABLE 3: Comparison between Purely-Visual and Text-               ing on whether they leverage textual information, such as
Guided token compression strategies with representative           user instructions or questions, compared in Table 3.
works (details refer to §4.2).                                    Purely-visual Compression. These methods rely solely on
         Purely-Visual                 Text-Guided                visual cues to eliminate redundant information. They sys-
                                                                  tematically reduce tokens representing duplicate objects,
Method Retain informative visual       Select text semantic
       tokens according to             aligned visual tokens
                                                                  uniform backgrounds, or semantically equivalent regions.
       inherent vision                 according to textual       For video sequences, such approaches compress temporally
       redundancy                      instruction or query       static content while preserving dynamic motions. Specifi-
                                                                  cally, recent studies identify distinctive visual tokens [57],
Features (i) Suitable for multi-turn   Suitable for single-turn   [96], [183] or aggregate repetitive semantic tokens into com-
         dialogues, streaming          dialogues, long
         video understanding,          VideoQA, high-ratio        pact representations [34], [83]. As VisionZip [93] pointed
         visual captioning, (ii)       compression scenario,      out, these more compact visual tokens lead to better visual
         Easy to deployment            visual grounding           representations.
                                                                      Since purely-visual approaches are text-agnostic and
Works    DeCo [105], VisionZip         FastV [129],
                                                                  perform one-time compression, they are efficient for low-
         [93], DART [183], HoloV       SparseVLM [132],
         [96], TimeChat-Online         Q-Former [257], QueCC      latency applications such as multi-turn dialogue, online
         [57]                          [176], PyramidDrop         responses and streaming video understanding [258]–[262].
                                       [130], LLaVA-Mini [181]    Moreover, their general applicability to visually rich scenes
                                                                  makes them effective for captioning [263], [264] tasks. For
                                                                  deployment, they directly reduce visual tokens before the
focus to inference efficiency through KV cache sparsifica-        LLM, avoiding extensive computation and memory con-
tion, enabling processing over 10,000 frames on a single          sumption in the LLM’s shallow layers and enabling seam-
GPU.                                                              less adaptation across different LLM architectures.
    Under the video question-answering scenario, only a           Text-Guided Compression. In contrast, text-guided strate-
subset of frames is typically relevant to a given question.       gies [148], [265] use cross-modal information to select text-
This has motivated query-aware compression strategies.            relevant tokens according to a given instruction or query.
LinVT [88] identifies candidate regions through spatiotem-        Typical methods estimate text-to-vision attention or similar-
poral saliency analysis, then filters and aggregates them         ity or introduce proxy tokens for better cross-modal inter-
according to the text query to ensure retained tokens capture     action [73], [82]. By only focusing on task-relevant visual
both visual saliency and semantic relevance.                      semantics, these methods can achieve high compression
    Beyond query-aware innovations, practical deployment          ratios while maintaining accuracy in tasks such as visual
demands system-level efficiency. Long-VMNet [253] em-             question answering, grounding [9], [266], [267], and long-
ploys a fixed-size memory bank (e.g., 5,880 tokens) en-           video reasoning [28], [29], [268]. However, since user queries
abling memory reuse across queries after a single video           always vary across turns, text-guided compression often
scan, requiring less than 1GB memory while supporting             requires re-encoding historical tokens, limiting efficiency
10-hour videos. ReTaKe [254] detects keyframes via inter-         and reusability in multi-turn dialogue settings.
frame distance peaks and marks them as pivots, while              Takeaway. Purely-visual and text-guided strategies are com-
compressing non-pivot frames by pruning low-attention             plementary. A practical design is to first derive compact vi-
tokens in their KV cache. Leveraging LLM prior knowledge,         sual representations via purely-visual compression and then
it enables plug-and-play adaptation to existing VideoLLMs         apply text-guided selection within the language module to
for processing 8x longer sequences. TimeViper [255] adopts        refine tokens relevant to the given textual query.
a hybrid Mamba-Transformer architecture, combining linear
complexity with precise attention to process over 10,000          4.3   Token Merging vs. Token Dropping
frames.
                                                                  Token merging and token dropping (also referred to as
Summary. Extreme-long video understanding exhibits                pruning) are two fundamental operations in the token
multi-dimensional synergy: 1) adaptive key frame sam-             compression paradigm. Their core distinction lies in the
pling [56], [193], [256] and adaptive partitioning reduce         compression manner: merging is a soft strategy that aggre-
input redundancy; 2) multi-module collaboration enables           gates less informative tokens into representative ones, while
progressive encoding compression; 3) query-aware strate-          dropping is a hard operation that directly discards them. A
gies dynamically adjust based on user intent; 4) KV-cache         natural question arises: should these two operations be treated
sparsification improves inference efficiency. This evolution      identically? This subsection discusses their conceptual differ-
from isolated optimizations to systematic, task-aware, end-       ences, selection mechanisms, and practical implications.
to-end design establishes the foundation for practical hour-
long video understanding.                                         Merging or Dropping. Token merging and dropping each
                                                                  possess distinct advantages and drawbacks. As Table 4
                                                                  summarizes, token merging maintains holistic semantics
4.2   Purely-Visual vs. Text-guided Compression                   by smoothing token representations but may blur spa-
As discussed in § 2.2, token compression in MLLMs aims to         tial or temporal locality. Token dropping, in contrast, pre-
reduce two major types of redundancy: intra-visual (vision-       serves sparse and salient semantics yet risks losing fine-
to-vision) and cross-modal (text-to-vision). Accordingly, ex-     grained contextual information. Quantitative analyses from
isting methods can be grouped into two branches depend-           LLMC+ [270] reveal that for spatial redundancy, drop-based
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                           18

TABLE 4: Comparison between token merging and token                 TABLE 5: Comparison between plug-in and re-training
dropping strategies with representative works (details refer        methods with representative works (details refer to §4.4).
to §4.3).                                                                    Plug-in                     Re-training
         Token Merging                Token Dropping
                                                                    Method A parameter-free strategy     A trainable strategy that
Method A soft strategy that           A hard strategy that                 that can be directly          requires additional
       aggregates visually            directly discards tokens             integrated into existing      training to obtain
       redundant tokens into          considered less                      models without                learnable compression
       compact and                    informative or                       additional training.          capability.
       representative                 task-irrelevant.
                                                                    Features (i) Training-free and       (i) Higher performance
       embeddings.
                                                                             parameter-free, (ii)        ceiling, (ii)Require
Pros     (i) Preserves holistic and   (i) Retains sparse and                 Lightweight and efficient   additional training, (iii)
         fine-grained semantics,      salient semantics (ii)                 for deployment, (iii)       Limited transferability
         (ii) Suitable for            Suitable for compressing               Performance degradation     across models
         compressing low-level        high-level visual features.            on fine-grained tasks.
         visual features, (iii)
                                                                    Works    FastV [129],                Honeybee [113],
         Effective for spatial
                                                                             SparseVLM [132],            DeCo [105],
         redundancy.
                                                                             PyramidDrop [130],          TokenPacker [122],
Cons     May blur spatial or          May overlook subtle                    MustDrop [92]               HiCo [180]
         temporal locality due to     contextual cues that are
         averaging across             removed during pruning.
         multiple tokens.
                                                                    4.4   Plug-in Methods vs. Re-training Methods
Works    ToMe [34], TESTA [83],       VisPruner [68], DivPrune      From the perspective of model adaptation, existing token
         HoliTom [146],               [128], MADTP [73],
         MustDrop [92]                DART [183], FlexSelect        compression methods can be broadly categorized into two
                                      [189], CDPruner [269],        groups: plug-in methods, which can be seamlessly inte-
                                      DTD [57]                      grated into pre-trained models without the need for extra
                                                                    training, and re-training approaches, which require addi-
                                                                    tional fine-tuning or end-to-end optimization. As illustrated
                                                                    in Tab. 5, although both aim to reduce token redundancy
strategies generally outperform merge-based ones in both            and accelerate inference, they differ markedly in design
the vision encoder and the LLM component.                           philosophy, deployment cost, and the level of performance
Attention-based or Similarity-based strategies for token            they can ultimately achieve.
selection. Both token merging and dropping rely on iden-            Plug-in Methods.           Plug-in approaches focus on
tifying “unimportant” tokens to aggregate or discard. Early         lightweight modules that require minimal or no
works primarily used attention scores as indicators of token        training and can be seamlessly integrated into frozen
importance. However, recent studies have exposed several            backbones.     Representative     strategies    include:   (1)
limitations of attention-based selection. DART [183] and            parameter-free spatial transformations, such as global
FEATHER [133] reported that attention scores introduce              or adaptive pooling employed in TC-LLaVA [107],
a positional bias, favoring tokens located at the lower-            PLLaVA [108], DeCo [105], and AVG-LLaVA [106]; (2)
right region of the image—typically appearing later in              pixel rearrangement operations, exemplified by pixel shuf-
the sequence—regardless of their semantic significance.             fle and space-to-depth transformations in NVLM [201] and
HoloV [96] further highlighted that MLLMs often over-               InternVL 1.5 [109]; (3) similarity-based token compression,
fit to “highlighted tokens” and overlook holistic context,          where DynTok [125] dynamically groups video tokens
leading to local overemphasis on salient regions. Moreover,         and performs intra-group merging, LLaVA-Scissor [126]
attention-based selection can be incompatible with Flash At-        leverages Semantic Connected Components to preserve
tention implementations, reducing efficiency and sometimes          semantic regions while reducing redundancy, and
even underperforming random reduction baselines. To ad-             DivPrune [128] selects informative tokens by maximizing
dress these issues, recent approaches increasingly adopt            diversity; and (4) inference-time KV cache compression,
similarity-based token selection [57], [145], [271], where          where DyCoke [152] prunes the KV cache guided by
redundancy is measured via feature-level similarity rather          attention scores, and MustDrop [92] adopts an output-
than attention magnitude, enabling more stable and context-         aware KV Cache policy to reduce memory consumption
aware compression.                                                  and accelerate decoding without backbone modification.
Takeaway. Merging and dropping are complementary                        These methods are easy to deploy, highly compatible
rather than competing strategies [79], [84], [175], [272], [273].   across models, and cost-efficient, making them ideal for
Merging provides smooth aggregation suitable for dense              rapid inference acceleration or flexible deployment. How-
or temporally redundant visual inputs, whereas dropping             ever, because they are often task-agnostic and rely on heuris-
is preferable when sparse, high-level semantics suffice. Fu-        tics such as similarity thresholds or clustering, their seman-
ture frameworks may benefit from adaptive hybrid designs            tic retention and performance upper bound can degrade
that dynamically switch between soft aggregation and hard           under aggressive compression or complex task demands.
pruning according to modality characteristics and redun-            Re-training Methods. Re-training approaches introduce
dancy types.                                                        learnable modules or require end-to-end optimization, aim-
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                            19

TABLE 6: Comparison between Efficient Training and Effi-            TABLE 7: Representative MLLMs and their efficiency-
cient Inference strategies (details refer to §4.5).                 oriented training compression strategies.
         Efficient Training          Efficient Inference                    Representative MLLMs           Compression Strategy
Method Aim to mitigate training      Aim to lower inference         2022    Flamingo [147]                 GATED XATTN–DENSE
       costs by reducing the         costs by performing
       number of image tokens        token reduction during         2023    BLIP-2 [115], mPLUG-Owl        Q-former and its variants
       during the forward            the prefill or decoding                [275], Qwen-VL [118], Video-
       process.                      stage.                                 LLaMA [276], MiniGPT-4
                                                                            [116]
Features (i) The methodological      (i) The methodological
                                                                            Video-ChatGPT [228]            Temporal and Spatial Pooling
         design is relatively        design is more diverse,
         simple, with a limited      with a greater body of
                                                                    2024    PLLaVA [108], LongVLM          Temporal and Spatial Pooling
         number of studies in this   research in this field. (ii)           [232], VideoLLaMA 2 [99]
         area. (ii) The validation   The validation cost is
         cost is substantial.        minimal.                               LLaVA-OneVision [10]           Bilinear Interpolation
                                                                            LLaVA-Video [236]              Average Spatial Pooling
Works    Flamingo [147],             FastV [129],
         Q-Former [257],             SparseVLM [132],               2025    InternVL series [52], [109],   Pixel Shuffle
         LLaVA-OneVision [10],       PyramidDrop [130],                     Qwen2VL series [4], [112]
         Qwen2.5-VL [274],           VisionZip [93],
         InternVL3.5 [52]            SparseMM [153]                         Seed1.5-VL [277]               Average Pooling



ing for task-adaptive and semantically aware token com-             token compression methods suffer from notable perfor-
pression. Representative methods include query-based de-            mance drops on tasks requiring high-resolution visual un-
signs, such as Q-Former and its variants, including the             derstanding and complex text reasoning. In contrast, re-
BLIP-2 [115], the simplified single-layer cross-attention in        training methods excel in task-specific scenarios and fine-
Qwen-VL [118], the C-/D-Abstractor modules in Honey-                grained multimodal understanding, offering higher per-
bee [113] for better locality modeling, and MQT [119],              formance ceilings and greater stability under aggressive
which adapts the number of query tokens. Another line               compression, though at the cost of substantial additional
of work employs downsampled-as-query cross-attention, as            training overhead and poor transferability across models.
seen in TokenPacker [122] and HiRes-LLaVA [123], which                  In practice, hybrid strategies have gained increasing
use downsampled features as queries to interact with high-          attention as a promising compromise between efficiency and
resolution regions and achieve coarse-to-fine information in-       adaptability. A common design is to apply lightweight plug-
jection. There are also text- and concept-guided compression        in techniques such as pooling or pixel unshuffle for early
methods. TG-LLaVA [120] performs text-driven masking,               spatial reduction, then incorporate re-trained modules such
QueCC [176] incorporates user query semantics through               as cross-attention or query-guided compression for semantic
local cross-attention aggregation, and VCM [186] models             refinement, and finally adopt key-value cache pruning to
vision concepts to dynamically determine concept granu-             improve decoding efficiency. This progressive integration,
larity and spatial alignment. Finally, several multi-module         exemplified by the multi-stage design of MustDrop [92],
and multi-stage token compression frameworks have been              reflects a trend toward combining the deployment flexibility
proposed. CrossGET [82] breaks the sequential visual-first          of plug-in methods with the task adaptivity and perfor-
processing paradigm, LLaMA-VID [98] constructs context              mance advantages of re-training approaches.
and content tokens for each video frame, PAR [100] differ-
entiates between external and internal redundancy through           4.5    Efficient Training vs. Efficient Inference
query rewriting and token routing, and MustDrop [92]                Efficient training and efficient inference respectively address
accelerates inference through a three-stage “vision-prefill-        the problem of token reduction during the training and
decoding” merging strategy with dual-attention filtering.           inference phases. In this section, we focus on discussing
    These methods typically achieve stronger semantic               the distinctions between these two approaches, compared
preservation and task relevance because they are able to            in Table 6.
leverage cross-modal attention, textual guidance, and pro-              Efficient training typically aims to mitigate costs during
gressive refinement. They also tend to reach higher com-            pretraining and SFT, which demands hundreds of billions
pression ratios without causing severe performance degra-           to trillions of tokens. In practice, most state-of-the-art mul-
dation. However, their use comes with additional training           timodal models rely on relatively simple mechanisms for
costs, greater data requirements, and increased engineering         token reduction as presented in Table 7. LLaVA-OneVision
complexity, and their effectiveness may vary depending on           [10] utilizes bilinear interpolation to reduce tokens per
the target task or application domain.                              frame. The InternVL [278] and Qwen2 series [274] apply
Comparative Insights. Plug-in methods are well-suited for           pixel shuffle strategies, reducing the number of vision to-
rapid deployment and inference acceleration when training           kens while expanding their feature dimensions. Similarly,
resources are limited or when task requirements are rela-           Seed1.5-VL [277] employs a basic average pooling method.
tively moderate. However, their performance upper bound             It is worth noting that a variety of new methods have
is relatively limited. Recent studies such as FCoT-VL [182]         recently been proposed to accelerate training. For exam-
further empirically demonstrate that current training-free          ple, LLaVolta [144] introduces staged training, where more
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                     20

TABLE 8: Summary of benchmarks widely-used in visual token pruning studies. MQA denotes multiple-choice question
answering, Open denotes open-ended question answering, Y/N denotes Yes/No question answering.

 Benchmark                  Answer Type              Metric             Num Examples               Focus              Data Link
Image Domain
 GQA-testdev-balanced          Open                Accuracy                 12,578       General Image Perception       Link
 VQA-v2-testdev                Open                Accuracy                107,394       General Image Perception       Link
 VizWiz-val                    Open                Accuracy                  4,319       General Image Perception       Link
 POPE                          Y/N                  F1-Score                 3,000       General Image Perception       Link
 TextVQA-val                   Open                Accuracy                  5,000                 OCR                  Link
 ScienceQA-Image-test        MQA,Y/N               Accuracy                  2,017             Knowledge                Link
 MathVista-testmini          MQA,Open              Accuracy                  1,000        Knowledge,Reasoning           Link
 MathVerse-testmini          MQA,Open              Accuracy                  3,940        Knowledge,Reasoning           Link
 MMMU                        MQA,Open              Accuracy                 11,550        Knowledge,Reasoning           Link
 MME                           Y/N              Perception Score             2,374              Integrated              Link
 MMBench-en-dev                MQA                 Accuracy                  4,329              Integrated              Link
 MM-Vet                        Open                GPT-Score                  218               Integrated              Link
 SeedBench-Image               MQA                 Accuracy                 14,280              Integrated              Link
 LLaVA-BenchW                  Open                GPT-Score                   60               Integrated              Link
Video Domain
 ActivityNet-QA-test           Open           Accuracy,GPT-Score            8,000              Integrated               Link
 MVBench                       MQA                Accuracy                  4,000        Temporal Understanding         Link
 EgoSchema                     MQA                Accuracy                  5,063             Long Video                Link
 LongVideoBench-val            MQA                Accuracy                  1,337         Long Video,Integrated         Link
 MLVU-dev                    MQA,Open         Accuracy,GPT-Score            2,593         Long Video,Integrated         Link
 Next-QA-MC-test               MQA                Accuracy                  8,564              Integrated               Link
 Video-ChatGPT                 Open               GPT-Score                 3,493              Integrated               Link
 Video-MME                     MQA                Accuracy                  2,700              Integrated               Link


aggressive token reduction is applied in the early stages           become an urgent requirement.
and the compression ratio is gradually decreased over time.
PyramidDrop [130] removes tokens layer by layer inside the
                                                                    5     B ENCHMARKS AND M ETRICS
LLM. From the perspective of task similarity, both LLM
prefilling and training involve a single forward pass of            In this section, we first provide a detailed overview of the
a sequence through the LLM. Therefore, in principle, all            benchmarks (§5.1) and evaluation metrics (§5.2) commonly
strategies that can be applied during LLM prefilling could          used in MLLM token compression studies.
also be used for efficient training.
    However, why have these diverse methods not been                5.1   Benchmarks
widely adopted by mainstream LVLMs? We identify three               Table 8 summarizes the image and video understanding
main reasons. First, compatibility issues: many prefilling ac-      benchmarks commonly used in token pruning studies. De-
celeration methods are not compatible with Flash Attention,         pending on the primary capability being evaluated, these
which directly affects training efficiency. Second, validation      benchmarks can be grouped into several categories.
cost: adopting a new strategy requires validation during                For image understanding benchmarks, the categories
training, which is far more expensive than inference, making        include:
researchers more conservative. As long as current costs               • General Image Perception: Evaluates basic visual
remain acceptable, new methods are unlikely to be adopted                recognition skills in natural images, such as identifying
unless they prove to be a breakthrough. Third, inductive bias:           objects, scenes, attributes, and spatial relationships.
existing compression techniques often design customized               • Optical Character Recognition (OCR): Measures the
strategies based on observations from certain tasks or bench-            ability to recognize and interpret textual content em-
marks, thereby introducing strong inductive bias. Such                   bedded in unstructured visual formats. This skill is cru-
methods may lead to performance degradation in scenarios                 cial for enabling effective interaction between MLLMs
where visual information is denser or task distributions                 and humans.
differ significantly. Since current MLLMs are intended for            • Knowledge: Assesses the integration of visual percep-
general-purpose use, any degradation in certain capabilities             tion with domain-specific or general world knowledge
is unacceptable.                                                         across diverse disciplines.
    In the field of efficient inference, nearly all of the afore-     • Reasoning: Goes beyond perception, requiring logical
mentioned methods are designed for this scenario, and                    inference and problem-solving based on visual content
the area is evolving rapidly. The popularity of this topic               combined with specific prior knowledge.
is largely due to its low exploration cost. Moreover, for             • Integrated Image Understanding: Provides a holistic
the practical deployment of large multimodal models, the                 evaluation by combining visual perception and rea-
volume of API requests is extremely high, and minimizing                 soning tasks into a single benchmark, thereby testing
latency is crucial. As a result, controlling inference cost has          comprehensive multimodal understanding.
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                  21

    For video understanding benchmarks, the categories in-       efforts have focused on translating these methods into real-
clude:                                                           world applications, thereby maximizing their societal im-
   • Temporal Understanding: Measures the ability to cap-        pact. In this section, we provide a concise overview of the
     ture and interpret temporal dynamics, such as action        key aspects that are critical for the applications.
     sequences, motion patterns, and event localizations.
   • Long Video Understanding: Evaluates the capacity
     to process and reason over long-form videos, ranging        6.1   Image Understanding
     from several to tens of minutes.                            In image understanding, current algorithms primarily focus
   • Integrated Video Understanding: Offers a holistic as-       on accelerating the processing of high-resolution inputs,
     sessment of perception and reasoning skills in video        which is essential for downstream tasks.
     contexts by combining multiple evaluation dimensions.
                                                                 Medical Image Processing. A key application lies in med-
                                                                 ical imaging, where MLLMs must rapidly and accurately
5.2     Metrics                                                  interpret clinical data, underscoring the need to balance
The evaluation of MLLM token compression methods                 efficiency and accuracy. Extensive research [279], [280] has
primarily considers two perspectives: downstream task            been devoted to evaluating the capabilities of these models.
performance (effectiveness) and computational efficiency         However, despite the rapid advancement, current mod-
(efficiency), either theoretical or practical.                   els remain limited in effectively handling high-resolution
                                                                 medical imaging examination results. The incorporation of
                                                                 efficient token compression algorithms presents a promising
5.2.1    Effectiveness
                                                                 avenue to further improve both efficiency and effectiveness
Effectiveness evaluation typically follows the standard of       in such settings.
original benchmarks. Most benchmarks adopt Accuracy as
the primary metric, which measures whether the model’s           Multi-page Document Understanding. Another valuable
prediction matches the ground-truth answer. For open-            application is document understanding, where models must
ended tasks without a single correct answer (e.g., image         process long documents and generate concise summaries
captioning), GPT-Score is often employed to provide a nu-        or meaningful solutions from the input. Prior studies [281]
merical rating of the MLLM’s response.                           have primarily focused on improving accuracy and expand-
                                                                 ing the range of document lengths that models can handle.
                                                                 Inspired by advances in high-resolution image processing,
5.2.2    Efficiency
                                                                 where algorithms accelerate computation without sacrific-
Efficiency can be evaluated from several complementary           ing accuracy, similar techniques [124], [148] can be applied
aspects:                                                         to document understanding. Such integration would allow
  • Token Retention Count/Ratio: Measures the absolute           models to manage longer inputs within limited context
     number or relative percentage of visual tokens pre-         lengths while also improving overall efficiency.
     served after compression. Token compression methods
     are commonly compared under the same retention              Satellite and Remote Sensing Imagery. In industrial ap-
     count/ratio in downstream tasks. However, identical         plications, MLLMs have been deployed to interpret satellite
     retention levels do not guarantee equal inference la-       and remote sensing imagery [282]. These images typically
     tency, as factors such as the compression position can      contain rich structural information at high resolutions, yet
     significantly influence runtime.                            practical deployments face computational resource con-
  • Prefilling/Decoding FLOPs: Captures the theoretical
                                                                 straints. Efficiently processing such imagery remains a sig-
     computational cost of the query prefilling and decoding     nificant challenge. Recent studies [283], [284] have explored
     stages, measured in floating-point operations.              token compression strategies to address this bottleneck,
  • Prefilling/Decoding Latency: Reports the actual wall-
                                                                 achieving notable progress by enabling models to handle
     clock time required for the model to process input          higher-resolution inputs more efficiently—an advancement
     (prefilling) and generate output tokens (decoding). Un-     of considerable importance for industrial deployment.
     like FLOPs, which is hardware-agnostic, this metric is
     dependent on the specific infrastructure and implemen-      6.2   Video Understanding
     tation.
  • Memory Usage: This metric quantifies the peak mem-           In the realm of video understanding tasks, previous research
     ory footprint during inference, which is especially crit-   has primarily concentrated on addressing the inherent chal-
     ical for deploying MLLMs on resource-constrained de-        lenges associated with comprehending lengthy videos. Ef-
     vices. Token compression can reduce the memory re-          forts have been made to reduce data redundancy and en-
     quired for attention key–value caches and intermediate      hance efficiency in processing extended video content.
     representations, but the reduction is highly dependent      Embodied AI. A practical application of such algorithms
     on how compression is implemented.                          lies in the development of robot learning and embodied AI.
                                                                 In these settings, embodied agents or robots must respond
                                                                 in real time to the visual input they receive during contin-
6     A PPLICATION S CENARIOS                                    uous video perception. The token compression strategy [74]
Following the development of advanced algorithms that            addresses this challenge by efficiently capturing both spatial
significantly enhance the efficiency of MLLMs, subsequent        and temporal information, thereby enabling fine-grained
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                    22

video understanding while maintaining computational effi-         validation. Consequently, they often exhibit poor transfer-
ciency. This capability is essential for the real-world deploy-   ability across datasets, architectures, and modalities, as well
ment of robots and embodied agents, making them more              as insufficient robustness under distribution shift.
suitable for practical applications.                                  A key weakness lies in the absence of a principled theory
Streaming Video Understanding. Another significant ap-            of token importance. Current practices—such as ranking
plication domain is streaming video understanding, where          tokens by attention weights, pairwise similarity, or mutual
models must process continuous video streams and deliver          information—lack causal or generalization-based justifica-
real-time responses with minimal latency. Prior studies [57],     tion. These metrics indicate correlation rather than necessity,
[285]–[287] have adopted token compression techniques             offering little explanation of whether the retained tokens
to address the high temporal redundancy in dense video            are truly sufficient for the downstream objective or merely
streams (e.g., 1-10 FPS), store compact historical representa-    coincidental with good performance.
tions through memory mechanisms, and efficiently retrieve             By connecting token selection to sufficiency, causality,
question-relevant KV caches during inference. These strate-       and robustness, future work can move beyond ad-hoc
gies enable models to maintain responsiveness and accuracy        heuristics toward a principled understanding of why com-
while managing computational resources effectively, a criti-      pression works, enabling generalizable and theoretically
cal requirement for real-time applications.                       sound compression strategies for MLLMs.

Instructional Video Summary. Other real-world appli-              7.2   Lack of Task- and Content-Aware Adaptivity
cations, such as meeting summarization and lecture key-           Most existing token compression strategies operate in a
point extraction, also require models to achieve efficient        task-agnostic and content-agnostic manner, applying a fixed
video understanding while preserving fine-grained details.        compression ratio or heuristic rule regardless of the task
Several studies [5], [6] have investigated these challenging      type or the visual complexity of the input. However, the
scenarios and proposed a variety of solutions. A central idea     granularity of information required to fulfill a given ques-
underlying these approaches is the selective retention of in-     tion varies substantially. As M 3 [91] observed, for most
formative tokens while discarding redundant ones, thereby         benchmarks, especially those mainly crafted from natural
improving overall efficiency and facilitating the practical       scenes (such as COCO [291]), can be handled well with only
adoption of such methods in real-world tasks.                     9 tokens per image. In contrast, dense visual perception
                                                                  tasks such as document understanding or OCR require a
6.3   Other Applications                                          greater amount of tokens (144 ∼ 576 tokens) per image to
                                                                  handle the task well. A uniform compression policy thus
Beyond accelerating the processing of high-resolution im-         risks either retaining redundant tokens for simple tasks or
ages and long videos through redundant token reduction,           discarding crucial details for complex ones, leading to ineffi-
token pruning demonstrates considerable potential across          ciency and degraded understanding. Similarly, multimodal
diverse applications. A key advantage of this approach is its     inputs such as images or video clips exhibit vastly different
ability to guide model attention toward the most relevant         levels of informational richness. Compressing them under
image or video regions [288]. By filtering out background         a single fixed strategy ignores variations in object density,
noise and irrelevant objects, models can allocate compu-          scene complexity, and visual salience. Yet few existing meth-
tational capacity to critical visual information essential for    ods explicitly model this heterogeneity or incorporate adap-
accurately interpreting and responding to prompts. Prior          tive mechanisms conditioned on either the task semantics or
studies [289], [290] have shown that this improved focus          the visual content itself.
can mitigate visual hallucinations, where models generate              Future research should explore task- and content-aware
text inconsistent with visual input. Through selective token      compression, where the model dynamically determines the
pruning, these strategies improve the grounding of model          degree and manner of token reduction. Some recent stud-
outputs in the actual visual context.                             ies [100], [101], [186] have begun to move in this direction,
                                                                  introducing adaptive mechanisms that modulate compres-
                                                                  sion according to textual queries or visual content complex-
7     O PEN C HALLENGES AND F UTURE W ORK                         ity. However, how to further couple such adaptive compres-
Despite the rapid progress in token compression for               sion with improved training strategies to achieve stronger
MLLMs, several open challenges remain that warrant fur-           generalization across diverse tasks remains an open ques-
ther investigation. We discuss remaining open challenges          tion. For instance, VisionThink [190] proposes a reinforce-
and future research directions in this section.                   ment learning-based approach that enables the model to
                                                                  autonomously decide whether the higher-resolution visual
                                                                  input is necessary for a given task, offering a promising step
7.1   Lack of Theoretical Understanding                           toward fully adaptive token compression. Such adaptive
Although token compression has achieved notable em-               strategies would align the compression process more closely
pirical success, most existing approaches remain largely          with the cognitive demands of multimodal understanding,
experience-driven and lack rigorous theoretical ground-           improving both efficiency and fidelity across diverse tasks.
ing. Apart from a few works, such as DeCo [105] and
DART [183], which analyze how compression influences              7.3   Performance Degradation in Practical Tasks
representation learning within MLLMs, the majority of             Although many token compression methods demonstrate
methods rely on heuristic intuition and limited empirical         competitive results on general Visual QA tasks [292], often
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                 23

maintaining comparable accuracy even when reducing vi-            R EFERENCES
sual tokens to 1/3 or 1/4 of the original, this performance       [1]    H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,”
stability does not generalize well to real-world applica-                in Advances in Neural Information Processing Systems 36: Annual
tions. Tasks that require fine-grained perception, such as               Conference on Neural Information Processing Systems 2023, NeurIPS
OCR [293], [294], document understanding [295], and dense                2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh,
                                                                         T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine,
reasoning over structured visual layouts, tend to experience             Eds., 2023. 1
a substantial drop in accuracy after compression. These           [2]    H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llava-
scenarios demand precise localization, text recognition, and             next: Improved reasoning, ocr, and world knowledge,” 2024. 1
                                                                  [3]    K. Ataallah, X. Shen, E. Abdelrahman, E. Sleiman, D. Zhu, J. Ding,
structural alignment, where the loss of subtle spatial or                and M. Elhoseiny, “Minigpt4-video: Advancing multimodal llms
semantic cues introduced by aggressive compression be-                   for video understanding with interleaved visual-textual tokens,”
comes detrimental. This performance gap highlights a key                 ArXiv preprint, vol. abs/2404.03413, 2024. 1
limitation: current compression schemes prioritize average        [4]    S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang,
                                                                         S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan,
efficiency rather than task-specific fidelity, which constrains          P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng,
their applicability in practical multimodal systems requiring            H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical
high-resolution understanding or domain-level precision.                 report,” arXiv preprint arXiv:2502.13923, 2025. 1, 15, 16, 19
                                                                  [5]    K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang,
7.4 Limitations of Existing Evaluation                                   and Y. Qiao, “Videochat: Chat-centric video understanding,”
                                                                         ArXiv preprint, vol. abs/2305.06355, 2023. 1, 22
From an evaluation perspective, the efficiency and effective-     [6]    B. Zhang, K. Li, Z. Cheng, Z. Hu, Y. Yuan, G. Chen, S. Leng,
ness of existing token compression methods are primarily                 Y. Jiang, H. Zhang, X. Li et al., “Videollama 3: Frontier multimodal
                                                                         foundation models for image and video understanding,” arXiv
assessed through downstream multimodal tasks. We iden-                   preprint arXiv:2501.13106, 2025. 1, 22
tify three key limitations in current MLLM token compres-         [7]    P. Tong, E. Brown, P. Wu, S. Woo, A. J. V. IYER, S. C. Akula,
sion evaluation practices:                                               S. Yang, J. Yang, M. Middepogu, Z. Wang et al., “Cambrian-1:
    Lack of systematic task categorization. As shown in                  A fully open, vision-centric exploration of multimodal llms,”
                                                                         Advances in Neural Information Processing Systems, vol. 37, pp.
Table 8, benchmarks are grouped into broad categories,                   87 310–87 356, 2024. 1, 3, 7
offering limited insight into how token compression af-           [8]    D. Li, Y. Liu, H. Wu, Y. Wang, Z. Shen, B. Qu, X. Niu, F. Zhou,
fects specific visual understanding capabilities (e.g., spatial          C. Huang, Y. Li et al., “Aria: An open multimodal native mixture-
                                                                         of-experts model,” arXiv preprint arXiv:2410.05993, 2024. 1
relation reasoning or object motion tracking) and content
                                                                  [9]    S. Ren, L. Yao, S. Li, X. Sun, and L. Hou, “Timechat: A time-
domains (e.g., table or chart interpretation).                           sensitive multimodal large language model for long video under-
    Inefficient evaluation processes. Current evaluations                standing,” in Proceedings of the IEEE/CVF Conference on Computer
typically employ at least ten benchmarks encompassing tens               Vision and Pattern Recognition, 2024, pp. 14 313–14 323. 1, 14, 16,
                                                                         17
of thousands of examples. Many benchmarks exhibit sub-            [10]   B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang,
stantial overlap in evaluation focus, leading to redundant               P. Zhang, Y. Li, Z. Liu et al., “Llava-onevision: Easy visual task
assessments and inefficient resource utilization.                        transfer,” arXiv preprint arXiv:2408.03326, 2024. 1, 4, 19
    Absence of consistent evaluation standards. The se-           [11]   Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li,
                                                                         “Video instruction tuning with synthetic data,” 2024. [Online].
lection of benchmarks and metrics varies widely across                   Available: https://arxiv.org/abs/2410.02713 1
studies, with each work emphasizing different strengths.          [12]   S. Tong, E. L. Brown II, P. Wu, S. Woo, A. J. IYER, S. C. Akula,
This inconsistency hinders fair cross-method comparison.                 S. Yang, J. Yang, M. Middepogu, Z. Wang et al., “Cambrian-1: A
    Although recent efforts have introduced more challeng-               fully open, vision-centric exploration of multimodal llms,” in The
                                                                         Thirty-eighth Annual Conference on Neural Information Processing
ing evaluation settings tailored for token compression ap-               Systems. 1
proaches [296], a systematic and standardized evaluation          [13]   Z. Sun, S. Shen, S. Cao, H. Liu, C. Li, Y. Shen, C. Gan, L.-Y.
framework remains necessary to enable fair comparisons                   Gui, Y.-X. Wang, Y. Yang, K. Keutzer, and T. Darrell, “Aligning
                                                                         large multimodal models with factually augmented rlhf,” ArXiv
and advance progress in this field.                                      preprint, vol. abs/2309.14525, 2023. 1
                                                                  [14]   L.-C.-T. Xiaomi, “Mimo-vl technical report,” 2025. [Online].
8   C ONCLUSION                                                          Available: https://arxiv.org/abs/2506.03569 1
                                                                  [15]   T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu,
MLLMs represent a significant advancement in cross-modal                 H. Zhao, J. Yang, J. Zeng, J. Wang, J. Zhang, J. Zhou, J. Wang,
understanding, yet computational efficiency remains a crit-              J. Chen, K. Zhu, K. Zhao, K. Yan, L. Huang, M. Feng, N. Zhang,
ical bottleneck. Token compression emerges as a promis-                  P. Li, P. Wu, R. Chu, R. Feng, S. Zhang, S. Sun, T. Fang, T. Wang,
ing solution by reducing redundancy across MLLM com-                     T. Gui, T. Weng, T. Shen, W. Lin, W. Wang, W. Wang, W. Zhou,
                                                                         W. Wang, W. Shen, W. Yu, X. Shi, X. Huang, X. Xu, Y. Kou, Y. Lv,
ponents, enhancing both training and inference efficiency                Y. Li, Y. Liu, Y. Wang, Y. Zhang, Y. Huang, Y. Li, Y. Wu, Y. Liu,
while alleviating long-context reasoning complexity. The                 Y. Pan, Y. Zheng, Y. Hong, Y. Shi, Y. Feng, Z. Jiang, Z. Han, Z.-
field has evolved from single-module to multi-module com-                F. Wu, and Z. Liu, “Wan: Open and advanced large-scale video
                                                                         generative models,” arXiv preprint arXiv:2503.20314, 2025. 1
pression, from fixed-rate to adaptive dynamic approaches,         [16]   L. Li, Y. Liu, L. Yao, P. Zhang, C. An, L. Wang, X. Sun,
and from static images to complex video sequences. How-                  L. Kong, and Q. Liu, “Temporal reasoning transfer from text
ever, key challenges persist: the absence of unified evalua-             to video,” in ICLR 2025. OpenReview.net, 2025. [Online].
tion frameworks for token compression, limited integration               Available: https://openreview.net/forum?id=sHAvMp5J4R 1
                                                                  [17]   K. Ouyang, Y. Liu, L. Yao, Y. Cai, H. Zhou, J. Zhou, F. Meng, and
with mainstream training or inference acceleration libraries,            X. Sun, “Conan: Progressive learning to reason like a detective
and insufficient synergy with other MLLM efficiency tech-                over multi-scale visual evidence,” arXiv preprint arXiv:2510.20470,
niques. This survey provides a systematic foundation for                 2025. 1
advancing efficient, scalable, and practically deployable         [18]   H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi,
                                                                         Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al.,
multimodal large language models through strategic token                 “Llama 2: Open foundation and fine-tuned chat models,” arXiv
compression methodologies.                                               preprint arXiv:2307.09288, 2023. 1, 3
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                                    24

[19]   A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Let-              [36]   R. Choudhury, G. Zhu, S. Liu, K. Niinuma, K. Kitani, and L. Jeni,
       man, A. Mathur, A. Schelten, A. Yang, A. Fan, A. Goyal,                             “Don’t look twice: Faster video transformers with run-length
       A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev,                        tokenization,” Advances in Neural Information Processing Systems,
       A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson,                          vol. 37, pp. 28 127–28 149, 2024. 1
       A. Spataru, B. Rozière, B. Biron, B. Tang, B. Chern, C. Caucheteux,         [37]   R. Koner, G. Jain, P. Jain, V. Tresp, and S. Paul, “Lookupvit:
       C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret,                      Compressing visual information to a limited number of tokens,”
       C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song,                 in European Conference on Computer Vision. Springer, 2024, pp.
       D. Pintz, D. Livshits, D. Esiobu, D. Choudhary, D. Mahajan,                         322–337. 1, 3, 7
       D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. Al-                   [38]   K. Zhou, “Lvp: Language-guide visual projector for efficient
       Badawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic,                           multimodal llm.” 1
       F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Nail, G. Mi-               [39]   Y. He, F. Chen, J. Liu, W. Shao, H. Zhou, K. Zhang, and B. Zhuang,
       alon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu,                          “Zipvl: Efficient large vision-language models with dynamic
       H. Touvron, I. Zarov, I. A. Ibarra, I. M. Kloumann, I. Misra, I. Ev-                token sparsification and kv cache compression,” 2024. 1
       timov, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar,      [40]   Q. Wu, W. Lin, Y. Zhou, W. Ye, Z. Zen, X. Sun, and R. Ji,
       J. Shah, J. van der Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi,              “Accelerating multimodal large language models via dynamic
       J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca,          visual-token exit and the empirical findings,” arXiv preprint
       J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Upasani, K. Plawiak,                 arXiv:2411.19628, 2024. 1
       K. Li, K. Heafield, K. Stone, and et al., “The llama 3 herd of               [41]   H. Wang, Y. Nie, Y. Ye, D. GuanYu, Y. Wang, S. Li, H. Yu, J. Lu,
       models,” ArXiv preprint, vol. abs/2407.21783, 2024. 1                               and C. Huang, “Dynamic-vlm: Simple dynamic visual token
[20]   OpenAI, “Introducing chatgpt,” 2022. 1                                              compression for videollm,” arXiv preprint arXiv:2412.09530, 2024.
[21]   A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li,                   1, 5
       D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang,                 [42]   Y. Jiang, Q. Wu, W. Lin, W. Yu, and Y. Zhou, “What kind of visual
       J. Yang, J. Tu, J. Zhang, J. Ma, J. Xu, J. Zhou, J. Bai, J. He, J. Lin,             tokens do we need? training-free visual token pruning for multi-
       K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang,                   modal large language models from the perspective of graph,” in
       P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan,                  Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39,
       T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang,                   no. 4, 2025, pp. 4075–4083. 1
       X. Wei, X. Ren, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu,            [43]   ——, “What kind of visual tokens do we need? training-free vi-
       Z. Cui, Z. Zhang, and Z. Fan, “Qwen2 technical report,” ArXiv                       sual token pruning for multi-modal large language models from
       preprint, vol. abs/2407.10671, 2024. 1                                              the perspective of graph,” in Proceedings of the AAAI Conference
[22]   D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu,                        on Artificial Intelligence, vol. 39, no. 4, 2025, pp. 4075–4083. 1
       S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning          [44]   R. Zhang, R. Shao, G. Chen, M. Zhang, K. Zhou, W. Guan, and
       capability in llms via reinforcement learning,” arXiv preprint                      L. Nie, “Falcon: Resolving visual redundancy and fragmentation
       arXiv:2501.12948, 2025. 1                                                           in high-resolution multimodal large language models via visual
[23]   A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,                   registers,” arXiv preprint arXiv:2501.16297, 2025. 1, 5
       T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly               [45]   H. Wang, Z. Yu, G. Spadaro, C. Ju, V. Quétu, S. Xiao, and
       et al., “An image is worth 16x16 words: Transformers for image                      E. Tartaglione, “Folder: Accelerating multi-modal large lan-
       recognition at scale,” in International Conference on Learning Repre-               guage models with enhanced performance,” arXiv preprint
       sentations, 2020. 1                                                                 arXiv:2501.02430, 2025. 1
[24]   H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,”            [46]   Z. Wen, Y. Gao, W. Li, C. He, and L. Zhang, “Token pruning
       arXiv preprint arXiv:2304.08485, 2023. 1                                            in multimodal large language models: Are we solving the right
[25]   P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu,                  problem?” arXiv preprint arXiv:2502.11501, 2025. 1
       J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu,              [47]   H. Zhang, M. Lyu, C. He, Y. Ao, and Y. Lin, “Towards adaptive vi-
       C. Zhou, J. Zhou, and J. Lin, “Qwen2-vl: Enhancing vision-                          sual token pruning for large multimodal models,” arXiv preprint
       language model’s perception of the world at any resolution,”                        arXiv:2509.00320, 2025. 1
       2024. 1, 15                                                                  [48]   J. Ma, Q. Zhang, M. Lu, Z. Wang, Q. Zhou, J. Song, and
[26]   OpenAI, “Hello gpt-4o,” 2024. [Online]. Available: https:                           S. Zhang, “Mmg-vid: Maximizing marginal gains at segment-
       //openai.com/index/hello-gpt-4o/ 1                                                  level and token-level for efficient video llms,” arXiv preprint
[27]   P. Wu and S. Xie, “V*: Guided visual search as a core mechanism                     arXiv:2508.21044, 2025. 1
       in multimodal llms,” arXiv preprint arXiv:2312.14135, 2023. 1                [49]   K. Zeng, G. Zhong, J. Cheng, J. Yuan, and Z. Li, “Avam: Uni-
[28]   C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou,                   versal training-free adaptive visual anchoring embedded into
       Y. Shen, M. Zhang et al., “Video-mme: The first-ever compre-                        multimodal large language model for multi-image question an-
       hensive evaluation benchmark of multi-modal llms in video                           swering,” arXiv preprint arXiv:2508.17860, 2025. 1
       analysis,” ArXiv preprint, vol. abs/2405.21075, 2024. 1, 17                  [50]   Z. Tang, Z. Ma, S. Wang, Z. Li, L. Zhang, H. Zhao, Y. Li,
[29]   J. Zhou, Y. Shu, B. Zhao, B. Wu, S. Xiao, X. Yang, Y. Xiong,                        and Q. Wang, “Covipal: Layer-wise contextualized visual to-
       B. Zhang, T. Huang, and Z. Liu, “Mlvu: A comprehensive bench-                       ken pruning for large vision-language models,” arXiv preprint
       mark for multi-task long video understanding,” ArXiv preprint,                      arXiv:2508.17243, 2025. 1
       vol. abs/2406.04264, 2024. 1, 17                                             [51]   K. Zhao, W. Yuan, A. L. Hung, and D. Zeng, “Pore: Position-
[30]   Y. Liang, C. Ge, Z. Tong, Y. Song, J. Wang, and P. Xie, “Not all                    reweighted visual token pruning for vision language models,”
       patches are what you need: Expediting vision transformers via                       arXiv preprint arXiv:2508.17807, 2025. 1
       token reorganizations,” arXiv preprint arXiv:2202.07800, 2022. 1             [52]   W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye,
[31]   H. Yin, A. Vahdat, J. Alvarez, A. Mallya, J. Kautz, and                             J. Shao et al., “Internvl3. 5: Advancing open-source multimodal
       P. Molchanov, “Adavit: Adaptive tokens for efficient vision trans-                  models in versatility, reasoning, and efficiency,” arXiv preprint
       former,” arXiv preprint arXiv:2112.07658, 2021. 1                                   arXiv:2508.18265, 2025. 1, 19
[32]   M. Fayyaz, S. A. Koohpayegani, F. R. Jafari, S. Sengupta, H. R. V.           [53]   J. Liu, J. Lin, Y. Wei, K. Shao, K. Tao, J. Huang, X. Yang,
       Joze, E. Sommerlade, H. Pirsiavash, and J. Gall, “Adaptive token                    Z. Chen, H. Wang, and X. Jin, “Revisiting mllm token technol-
       sampling for efficient vision transformers,” in European conference                 ogy through the lens of classical visual coding,” arXiv preprint
       on computer vision. Springer, 2022, pp. 396–414. 1                                  arXiv:2508.13460, 2025. 1
[33]   Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C.-J. Hsieh, “Dynam-            [54]   Z. Zhang, S. Liu, W. Yu, X. Wang et al., “Top-down compression:
       icvit: Efficient vision transformers with dynamic token sparsifi-                   Revisit efficient vision token projection for visual instruction
       cation,” Advances in neural information processing systems, vol. 34,                tuning,” arXiv preprint arXiv:2505.11945, 2025. 1
       pp. 13 937–13 949, 2021. 1, 12                                               [55]   Y. Shu, Z. Liu, P. Zhang, M. Qin, J. Zhou, Z. Liang, T. Huang, and
[34]   D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and                         B. Zhao, “Video-xl: Extra-long vision language model for hour-
       J. Hoffman, “Token merging: Your vit but faster,” arXiv preprint                    scale video understanding,” in Proceedings of the Computer Vision
       arXiv:2210.09461, 2022. 1, 5, 6, 12, 17, 18                                         and Pattern Recognition Conference, 2025, pp. 26 160–26 169. 1, 5,
[35]   S. Peng, D. Fu, B. Wei, Y. Cao, L. Gao, and Z. Tang, “Vote&mix:                     15, 16
       Plug-and-play token reduction for efficient vision transformer,”             [56]   X. Shen, Y. Xiong, C. Zhao, L. Wu, J. Chen, C. Zhu, Z. Liu,
       arXiv preprint arXiv:2408.17062, 2024. 1                                            F. Xiao, B. Varadarajan, F. Bordes et al., “Longvu: Spatiotemporal
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                               25

       adaptive compression for long video-language understanding,”              [77]   P. K. A. Vasu, F. Faghri, C.-L. Li, C. Koc, N. True, A. Antony,
       arXiv preprint arXiv:2410.17434, 2024. 1, 3, 5, 8, 15, 17                        G. Santhanam, J. Gabriel, P. Grasch, O. Tuzel et al., “Fastvlm: Effi-
[57]   L. Yao, Y. Li, Y. Wei, L. Li, S. Ren, Y. Liu, K. Ouyang, L. Wang,                cient vision encoding for vision language models,” in Proceedings
       S. Li, S. Li et al., “Timechat-online: 80% visual tokens are naturally           of the Computer Vision and Pattern Recognition Conference, 2025, pp.
       redundant in streaming videos,” arXiv preprint arXiv:2504.17343,                 19 769–19 780. 3, 7
       2025. 1, 3, 5, 15, 16, 17, 18, 22                                         [78]   L. Kong, H. Zhang, J. Zhang, J. Huang, K. Li, Q. Wang, and
[58]   M. Qin, X. Liu, Z. Liang, Y. Shu, H. Yuan, J. Zhou, S. Xiao,                     F. Zhang, “Clapper: Compact learning and video representation
       B. Zhao, and Z. Liu, “Video-xl-2: Towards very long-video un-                    in vlms,” arXiv preprint arXiv:2505.15529, 2025. 3, 15
       derstanding through task-aware kv sparsification,” arXiv preprint         [79]   R. Zhang, Y. Lyu, R. Shao, G. Chen, W. Guan, and L. Nie, “Token-
       arXiv:2506.19225, 2025. 1, 3, 5, 13, 16                                          level correlation-guided compression for efficient multimodal
[59]   Z. Liu, Y. Dong, Z. Liu, W. Hu, J. Lu, and Y. Rao, “Oryx mllm: On-               document understanding,” arXiv preprint arXiv:2407.14439, 2024.
       demand spatial-temporal understanding at arbitrary resolution,”                  3, 7, 18
       arXiv preprint arXiv:2409.12961, 2024. 1                                  [80]   Y. Guo, W. Dong, J. Song, S. Zhu, X. Zhang, H. Yang, Y. Wang,
[60]   Z. Liu, Y. Dong, J. Wang, Z. Liu, W. Hu, J. Lu, and Y. Rao, “Ola:                Y. Du, X. Chen, and B. Zheng, “Fila-video: Spatio-temporal
       Pushing the frontiers of omni-modal language model,” arXiv                       compression for fine-grained long video understanding,” arXiv
       preprint arXiv:2502.04328, 2025. 1                                               preprint arXiv:2504.20384, 2025. 3, 7, 15
[61]   Y. Ji, J. Zhang, H. Xia, J. Chen, L. Shou, G. Chen, and H. Li,            [81]   P. Jin, R. Takanobu, W. Zhang, X. Cao, and L. Yuan, “Chat-univi:
       “Specvlm: Enhancing speculative decoding of video llms via                       Unified visual representation empowers large language models
       verifier-guided token pruning,” arXiv preprint arXiv:2508.16201,                 with image and video understanding,” in Proceedings of the
       2025. 1, 5                                                                       IEEE/CVF Conference on Computer Vision and Pattern Recognition,
[62]   S. Dong, J. Hu, M. Zhang, M. Yin, Y. Fu, and Q. Qian, “Mm-                       2024, pp. 13 700–13 710. 3, 7, 15
       tok: Multimodal coverage maximization for efficient inference of          [82]   D. Shi, C. Tao, A. Rao, Z. Yang, C. Yuan, and J. Wang, “Cross-
       vlms,” arXiv preprint arXiv:2508.18264, 2025. 1                                  get: Cross-guided ensemble of tokens for accelerating vision-
[63]   J. Chen, X. Liu, Z. Wen, Y. Wang, S. Huang, and H. Chen,                         language transformers,” arXiv preprint arXiv:2305.17455, 2023. 3,
       “Variation-aware vision token dropping for faster large vision-                  7, 13, 17, 19
       language models,” arXiv preprint arXiv:2509.01552, 2025. 1                [83]   S. Ren, S. Chen, S. Li, X. Sun, and L. Hou, “Testa: Temporal-spatial
                                                                                        token aggregation for long-form video-language understanding,”
[64]   X. Wang, J. Zhang, T. Wang, H. Zhang, and F. Zheng, “Seeing
                                                                                        arXiv preprint arXiv:2310.19060, 2023. 3, 7, 15, 17, 18
       more, saying more: Lightweight language experts are dynamic
                                                                                 [84]   Y. Shang, M. Cai, B. Xu, Y. J. Lee, and Y. Yan, “Llava-prumerge:
       video token compressors,” arXiv preprint arXiv:2509.00969, 2025.
                                                                                        Adaptive token reduction for efficient large multimodal models,”
       1, 5
                                                                                        arXiv preprint arXiv:2403.15388, 2024. 3, 5, 7, 8, 18
[65]   K. H. I. Arif, J. Yoon, D. S. Nikolopoulos, H. Vandierendonck,
                                                                                 [85]   M. Huang, R. Huang, H. Shi, Y. Chen, C. Zheng, X. Sun, X. Jiang,
       D. John, and B. Ji, “Hired: Attention-guided token dropping for
                                                                                        Z. Li, and H. Cheng, “Efficient multi-modal large language mod-
       efficient inference of high-resolution vision-language models,” in
                                                                                        els via visual token grouping,” arXiv preprint arXiv:2411.17773,
       Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39,
                                                                                        2024. 3, 7
       no. 2, 2025, pp. 1773–1781. 3, 5
                                                                                 [86]   Y. Han, X. Liu, P. Ding, D. Wang, H. Chen, Q. Yan, and
[66]   D. Song, W. Wang, S. Chen, X. Wang, M. Guan, and B. Wang,
                                                                                        S. Huang, “Rethinking token reduction in mllms: Towards a
       “Less is more: A simple yet effective token reduction method
                                                                                        unified paradigm for training-free acceleration,” arXiv e-prints,
       for efficient multi-modal llms,” 2024. [Online]. Available:
                                                                                        pp. arXiv–2411, 2024. 3, 7, 12, 14
       https://arxiv.org/abs/2409.10994 3, 6
                                                                                 [87]   J. Hyun, S. Hwang, S. H. Han, T. Kim, I. Lee, D. Wee, J.-Y.
[67]   A. Jeddi, N. Baghbanzadeh, E. Dolatabadi, and B. Taati,                          Lee, S. J. Kim, and M. Shim, “Multi-granular spatio-temporal
       “Similarity-aware token pruning: Your vlm but faster,” arXiv                     token merging for training-free acceleration of video llms,” arXiv
       preprint arXiv:2503.11549, 2025. 3, 6                                            preprint arXiv:2507.07990, 2025. 3, 5, 7
[68]   Q. Zhang, A. Cheng, M. Lu, R. Zhang, Z. Zhuo, J. Cao, S. Guo,             [88]   L. Gao, Y. Zhong, Y. Zeng, H. Tan, D. Li, and Z. Zhao, “Linvt:
       Q. She, and S. Zhang, “Beyond text-visual attention: Exploiting                  Empower your image-level large language model to understand
       visual cues for effective token pruning in vlms,” arXiv preprint                 videos,” arXiv preprint arXiv:2412.05185, 2024. 3, 7, 15, 17
       arXiv:2412.01818, 2024. 3, 5, 6, 18                                       [89]   Q. Zhu, X. Wang, Z. Lu, J. Lao, C. Jin, J. Chen, Y. Peng, Q. Zhu,
[69]   J. Liu, F. Du, G. Zhu, N. Lian, J. Li, and B. Chen, “Hiprune:                    L. Zhong, J. Liu et al., “Admire: Adaptive method to enhance
       Training-free visual token pruning via hierarchical attention in                 multiple image resolutions in text-rich multi-image understand-
       vision-language models,” arXiv preprint arXiv:2508.00553, 2025.                  ing,” in Proceedings of the 31st ACM SIGKDD Conference on Knowl-
       3, 6                                                                             edge Discovery and Data Mining V. 2, 2025, pp. 5237–5248. 3, 7
[70]   S. Yang, R. Xu, C. Cui, T. Wang, D. Lin, and J. Pang, “Vflowopt:          [90]   X. Li, Y. Wang, J. Yu, X. Zeng, Y. Zhu, H. Huang, J. Gao,
       A token pruning framework for lmms with visual information                       K. Li, Y. He, C. Wang et al., “Videochat-flash: Hierarchical
       flow-guided optimization,” arXiv preprint arXiv:2508.05211, 2025.                compression for long-context video modeling,” arXiv preprint
       3, 5, 6                                                                          arXiv:2501.00574, 2024. 3
[71]   C. Zhang, K. Ma, T. Fang, W. Yu, H. Zhang, Z. Zhang, Y. Xie,              [91]   M. Cai, J. Yang, J. Gao, and Y. J. Lee, “Matryoshka multimodal
       K. Sycara, H. Mi, and D. Yu, “Vscan: Rethinking visual token re-                 models,” arXiv preprint arXiv:2405.17430, 2024. 3, 5, 7, 22
       duction for efficient large vision-language models,” arXiv preprint       [92]   T. Liu, L. Shi, R. Hong, Y. Hu, Q. Yin, and L. Zhang, “Multi-
       arXiv:2505.22654, 2025. 3, 7, 8, 11, 12                                          stage vision token dropping: Towards efficient multimodal large
[72]   Z. Wang, J. Chen, W. Zhou, H. Zhu, J. Liang, L. Shan, M. Liu,                    language model,” arXiv preprint arXiv:2411.10803, 2024. 3, 8, 13,
       D. Xu, Q. Yang, and B. Qin, “Smarttrim: Adaptive tokens and                      14, 18, 19
       attention pruning for efficient vision-language models,” arXiv            [93]   S. Yang, Y. Chen, Z. Tian, C. Wang, J. Li, B. Yu, and J. Jia,
       preprint arXiv:2305.15033, 2023. 3, 6                                            “Visionzip: Longer is better but not necessary in vision language
[73]   J. Cao, P. Ye, S. Li, C. Yu, Y. Tang, J. Lu, and T. Chen, “Madtp: Mul-           models,” in Proceedings of the Computer Vision and Pattern Recogni-
       timodal alignment-guided dynamic token pruning for accelerat-                    tion Conference, 2025, pp. 19 792–19 802. 3, 4, 5, 7, 17, 19
       ing vision-language transformer,” in Proceedings of the IEEE/CVF          [94]   L. Jiang, W. Huang, T. Liu, Y. Zeng, J. Li, L. Cheng, and X. Xu,
       conference on computer vision and pattern recognition, 2024, pp.                 “Fopru: Focal pruning for efficient large vision-language mod-
       15 710–15 719. 3, 6, 17, 18                                                      els,” arXiv preprint arXiv:2411.14164, 2024. 3, 8
[74]   J. Li, K. Li, C. Gao, Y. Li, and X. Chen, “Egoprune: Efficient token      [95]   B. Xu, Y. Shang, Y. Ge, Q. Lou, and Y. Yan, “freepruner: A
       pruning for egomotion video reasoning in embodied agent,”                        training-free approach for large multimodal model acceleration,”
       arXiv preprint arXiv:2507.15428, 2025. 3, 6, 21                                  arXiv preprint arXiv:2411.15446, 2024. 3, 8
[75]   J. Liu, L. Niu, W. Chen, J. Zhou, and F. Meng, “Laco: Efficient           [96]   X. Zou, D. Lu, Y. Wang, Y. Yan, Y. Lyu, X. Zheng, L. Zhang,
       layer-wise compression of visual tokens for multimodal large                     and X. Hu, “Don’t just chase ”highlighted tokens” in mllms:
       language models,” arXiv preprint arXiv:2507.02279, 2025. 3, 7                    Revisiting visual holistic context retention,” 2025. [Online].
[76]   H. Tang and C. Shen, “Learning compact vision tokens for effi-                   Available: https://arxiv.org/abs/2510.02912 3, 5, 8, 17, 18
       cient large multimodal models,” arXiv preprint arXiv:2506.07138,          [97]   H. Wang, J. Kai, H. Bai, L. Hou, B. Jiang, Z. He, and
       2025. 3, 7                                                                       Z. Lin, “Fourier-vlm: Compressing vision tokens in the fre-
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                              26

      quency domain for large vision-language models,” arXiv preprint                 language models with instruction tuning,” Advances in neural
      arXiv:2508.06038, 2025. 3, 7                                                    information processing systems, vol. 36, pp. 49 250–49 267, 2023. 3, 9
[98] Y. Li, C. Wang, and J. Jia, “Llama-vid: An image is worth 2 tokens         [118] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou,
      in large language models,” in European Conference on Computer                   and J. Zhou, “Qwen-vl: A versatile vision-language model for
      Vision. Springer, 2024, pp. 323–340. 3, 8, 13, 19                               understanding, localization, text reading, and beyond,” 2023.
[99] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu,                     [Online]. Available: https://arxiv.org/abs/2308.12966 3, 9, 19
      W. Zhang, Z. Luo, D. Zhao et al., “Videollama 2: Advancing                [119] W. Hu, Z.-Y. Dou, L. Li, A. Kamath, N. Peng, and K.-W. Chang,
      spatial-temporal modeling and audio understanding in video-                     “Matryoshka query transformer for large vision-language mod-
      llms,” arXiv preprint arXiv:2406.07476, 2024. 3, 8, 19                          els,” Advances in Neural Information Processing Systems, vol. 37, pp.
[100] Y. Liu, F. Wu, R. Li, Z. Tang, and K. Li, “Par: Prompt-aware token              50 168–50 188, 2024. 3, 9, 10, 19
      reduction method for efficient large multimodal models,” arXiv            [120] D. Yan, P. Li, Y. Li, H. Chen, Q. Chen, W. Luo, W. Dong,
      preprint arXiv:2410.07278, 2024. 3, 8, 14, 19, 22                               Q. Yan, H. Zhang, and C. Shen, “Tg-llava: Text guided llava
[101] S. Li, J. Xu, X.-H. Li, C. Deng, and L.-L. Huang, “Qg-vtc:                      via learnable latent embeddings,” 2024. [Online]. Available:
      Question-guided visual token compression in mllms for efficient                 https://arxiv.org/abs/2409.09564 3, 9, 10, 19
      vqa,” arXiv preprint arXiv:2504.00654, 2025. 3, 8, 22                     [121] Y. Li, J. Yang, Z. Shen, L. Han, H. Xu, and R. Tang, “Catp:
[102] Y. Chen, J. Xu, X.-Y. Zhang, W.-Z. Liu, Y.-Y. Liu, and C.-L. Liu,               Contextually adaptive token pruning for efficient and enhanced
      “Recoverable compression: A multimodal vision token recovery                    multimodal in-context learning,” arXiv preprint arXiv:2508.07871,
      mechanism guided by text information,” in Proceedings of the                    2025. 3, 5, 12
      AAAI Conference on Artificial Intelligence, vol. 39, no. 2, 2025, pp.     [122] W. Li, Y. Yuan, J. Liu, D. Tang, S. Wang, J. Qin, J. Zhu, and
      2293–2301. 3, 8                                                                 L. Zhang, “Tokenpacker: Efficient visual projector for multimodal
[103] D. Wang, J. Cui, M. Li, W. Lin, B. Chen, and H. Zhang, “Instruc-                llm,” International Journal of Computer Vision, pp. 1–19, 2025. 3, 5,
      tion tuning-free visual token complement for multimodal llms,”                  10, 18, 19
      in European Conference on Computer Vision. Springer, 2024, pp.            [123] R. Huang, X. Ding, C. Wang, J. Han, Y. Liu, H. Zhao, H. Xu,
      446–462. 3, 8                                                                   L. Hou, W. Zhang, and X. Liang, “Hires-llava: Restoring frag-
[104] X. Chu, L. Qiao, X. Zhang, S. Xu, F. Wei, Y. Yang, X. Sun, Y. Hu,               mentation input in high-resolution large vision-language mod-
      X. Lin, B. Zhang et al., “Mobilevlm v2: Faster and stronger base-               els,” in Proceedings of the Computer Vision and Pattern Recognition
      line for vision language model,” arXiv preprint arXiv:2402.03766,               Conference, 2025, pp. 29 814–29 824. 3, 5, 10, 19
      2024. 3, 5, 8, 9                                                          [124] A. Hu, H. Xu, L. Zhang, J. Ye, M. Yan, J. Zhang, Q. Jin, F. Huang,
[105] L. Yao, L. Li, S. Ren, L. Wang, Y. Liu, X. Sun, and L. Hou, “Deco:              and J. Zhou, “mplug-docowl2: High-resolution compressing for
      Decoupling token compression from semantic abstraction in mul-                  ocr-free multi-page document understanding,” arXiv preprint
      timodal large language models,” arXiv preprint arXiv:2405.20985,                arXiv:2409.03420, 2024. 3, 10, 21
      2024. 3, 5, 8, 9, 17, 18, 22                                              [125] H. Zhang, J. Zhang, X. Ji, Q. Wang, and F. Zhang, “Dyntok:
[106] Z. Lan, L. Niu, F. Meng, W. Li, J. Zhou, and J. Su, “Avg-llava: A               Dynamic compression of visual tokens for efficient and effective
      large multimodal model with adaptive visual granularity,” arXiv                 video understanding,” arXiv preprint arXiv:2506.03990, 2025. 3, 5,
      preprint arXiv:2410.02745, 2024. 3, 9, 18                                       10, 18
[107] M. Gao, J. Liu, M. Li, J. Xie, Q. Liu, B. Zhao, X. Chen, and              [126] B. Sun, J. Zhao, X. Wei, and Q. Hou, “Llava-scissor: Token com-
      H. Xiong, “Tc-llava: Rethinking the transfer from image to video                pression with semantic connected components for video llms,”
      understanding with temporal considerations,” 2024. [Online].                    arXiv preprint arXiv:2506.21862, 2025. 3, 5, 10, 18
      Available: https://arxiv.org/abs/2409.03206 3, 9, 15, 18                  [127] Y. Omri, P. Shroff, and T. Tambe, “Token sequence com-
[108] L. Xu, Y. Zhao, D. Zhou, Z. Lin, S. K. Ng, and J. Feng, “Pllava:                pression for efficient multimodal computing,” arXiv preprint
      Parameter-free llava extension from images to videos for video                  arXiv:2504.17892, 2025. 3, 10
      dense captioning,” arXiv preprint arXiv:2404.16994, 2024. 3, 8, 9,        [128] S. R. Alvar, G. Singh, M. Akbari, and Y. Zhang, “Divprune:
      15, 18, 19                                                                      Diversity-based visual token pruning for large multimodal mod-
[109] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu,               els,” in Proceedings of the Computer Vision and Pattern Recognition
      J. Luo, Z. Ma et al., “How far are we to gpt-4v? closing the gap to             Conference, 2025, pp. 9392–9401. 3, 5, 10, 11, 18
      commercial multimodal models with open-source suites,” Science            [129] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang,
      China Information Sciences, vol. 67, no. 12, p. 220101, 2024. 3, 9, 18,         “An image is worth 1/2 tokens after layer 2: Plug-and-play infer-
      19                                                                              ence acceleration for large vision-language models,” in European
[110] Z. Liu, L. Zhu, B. Shi, Z. Zhang, Y. Lou, S. Yang, H. Xi, S. Cao,               Conference on Computer Vision. Springer, 2024, pp. 19–35. 3, 5, 17,
      Y. Gu, D. Li et al., “Nvila: Efficient frontier visual language                 18, 19
      models,” arXiv preprint arXiv:2412.04468, 2024. 3                         [130] L. Xing, Q. Huang, X. Dong, J. Lu, P. Zhang, Y. Zang, Y. Cao,
[111] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye,                C. He, J. Wang, F. Wu et al., “Pyramiddrop: Accelerating your
      H. Tian, Z. Liu et al., “Expanding performance boundaries of                    large vision-language models via pyramid visual redundancy
      open-source multimodal models with model, data, and test-time                   reduction,” arXiv preprint arXiv:2410.17247, 2024. 3, 5, 17, 18, 19,
      scaling,” arXiv preprint arXiv:2412.05271, 2024. 3                              20
[112] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu,        [131] Z. Lin, M. Lin, L. Lin, and R. Ji, “Boosting multimodal large
      J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu,                 language models with visual tokens withdrawal for rapid infer-
      C. Zhou, J. Zhou, and J. Lin, “Qwen2-vl: Enhancing vision-                      ence,” in Proceedings of the AAAI Conference on Artificial Intelli-
      language model’s perception of the world at any resolution,”                    gence, vol. 39, no. 5, 2025, pp. 5334–5342. 3, 11
      arXiv preprint arXiv:2409.12191, 2024. 3, 4, 19                           [132] Y. Zhang, C.-K. Fan, J. Ma, W. Zheng, T. Huang, K. Cheng,
[113] J. Cha, W. Kang, J. Mun, and B. Roh, “Honeybee: Locality-                       D. Gudovskiy, T. Okuno, Y. Nakata, K. Keutzer et al., “Sparsevlm:
      enhanced projector for multimodal llm,” in Proceedings of the                   Visual token sparsification for efficient vision-language model
      IEEE/CVF Conference on Computer Vision and Pattern Recognition,                 inference,” arXiv preprint arXiv:2410.04417, 2024. 3, 5, 11, 17, 18,
      2024, pp. 13 817–13 827. 3, 8, 9, 18, 19                                        19
[114] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu,              [133] M. Endo, X. Wang, and S. Yeung-Levy, “Feather the throttle:
      W. Zhang, Z. Luo, D. Zhao et al., “Videollama 2: Advancing                      Revisiting visual token pruning for vision-language model ac-
      spatial-temporal modeling and audio understanding in video-                     celeration,” arXiv preprint arXiv:2412.13180, 2024. 3, 5, 8, 11, 18
      llms,” arXiv preprint arXiv:2406.07476, 2024. 3, 15                       [134] X. Ye, Y. Gan, Y. Ge, X.-P. Zhang, and Y. Tang, “Atp-llava:
[115] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping                   Adaptive token pruning for large vision language models,” in
      language-image pre-training with frozen image encoders and                      Proceedings of the Computer Vision and Pattern Recognition Confer-
      large language models,” in International conference on machine                  ence, 2025, pp. 24 972–24 982. 3, 11
      learning. PMLR, 2023, pp. 19 730–19 742. 3, 8, 9, 19                      [135] Y. Zhong, Z. Liu, Y. Li, and L. Wang, “Aim: Adaptive inference of
[116] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4:                  multi-modal llms via token merging and pruning,” arXiv preprint
      Enhancing vision-language understanding with advanced large                     arXiv:2412.03248, 2024. 3, 11
      language models,” arXiv preprint arXiv:2304.10592, 2023. 3, 9, 19         [136] J. Zhuang, L. Lu, M. Dai, R. Hu, J. Chen, Q. Liu, and H. Hu,
[117] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N.                  “St3: Accelerating multimodal large language model by spatial-
      Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-                temporal visual token trimming,” in Proceedings of the AAAI
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                                27

      Conference on Artificial Intelligence, vol. 39, no. 10, 2025, pp. 11 049–   [156] Y. Yang, Z. Zhao, S. N. Shukla, A. Singh, S. K. Mishra, L. Zhang,
      11 057. 3, 11, 13                                                                 and M. Ren, “Streammem: Query-agnostic kv cache memory for
[137] S. Zhao, Z. Wang, F. Juefei-Xu, X. Xia, M. Liu, X. Wang, M. Liang,                streaming video understanding,” arXiv preprint arXiv:2508.15717,
      N. Zhang, D. N. Metaxas, and L. Yu, “Accelerating multimodal                      2025. 3, 13
      large language models by searching optimal vision token reduc-              [157] A. Wang, F. Sun, H. Chen, Z. Lin, J. Han, and G. Ding, “[cls]
      tion,” in Proceedings of the Computer Vision and Pattern Recognition              token tells everything needed for training-free efficient mllms,”
      Conference, 2025, pp. 29 869–29 879. 3, 11                                        arXiv preprint arXiv:2412.05819, 2024. 3, 7
[138] C. Yang, Y. Sui, J. Xiao, L. Huang, Y. Gong, C. Li, J. Yan, Y. Bai,         [158] Y. Liu, Y. Wang, B. Shi, X. Zhang, W. Dai, C. Li, H. Xiong,
      P. Sadayappan, X. Hu et al., “Topv: Compatible token pruning                      and Q. Tian, “Meteor: Multi-encoder collaborative token prun-
      with inference time optimization for fast and low-memory mul-                     ing for efficient vision language models,” arXiv preprint
      timodal vision language model,” in Proceedings of the Computer                    arXiv:2507.20842, 2025. 3, 5, 6, 7
      Vision and Pattern Recognition Conference, 2025, pp. 19 803–19 813.         [159] Y. Jin, J. Li, Y. Liu, T. Gu, K. Wu, Z. Jiang, M. He, B. Zhao, X. Tan,
      3, 5, 12                                                                          Z. Gan et al., “Efficient multimodal large language models: A
[139] M. Dhouib, D. Buscaldi, S. Vanier, and A. Shabou, “Pact: Pruning                  survey,” arXiv preprint arXiv:2405.10739, 2024. 2
      and clustering-based token reduction for faster visual language             [160] G. Shinde, A. Ravi, E. Dey, S. Sakib, M. Rampure, and N. Roy,
      models,” in Proceedings of the Computer Vision and Pattern Recogni-               “A survey on efficient vision-language models,” Wiley Interdis-
      tion Conference, 2025, pp. 14 582–14 592. 3, 12                                   ciplinary Reviews: Data Mining and Knowledge Discovery, vol. 15,
[140] J. Zhang, D. Meng, Z. Zhang, Z. Huang, T. Wu, and L. Wang,                        no. 3, p. e70036, 2025. 2
      “p-mod: Building mixture-of-depths mllms via progressive ratio              [161] P. Nguyen and N.-M. Cheung, “Token compression meets com-
      decay,” arXiv preprint arXiv:2412.04449, 2024. 3, 12                              pact vision transformers: A survey and comparative evaluation
[141] W. Huang, Z. Zhai, Y. Shen, S. Cao, F. Zhao, X. Xu, Z. Ye,                        for edge ai,” arXiv preprint arXiv:2507.09702, 2025. 2, 6
      Y. Hu, and S. Lin, “Dynamic-llava: Efficient multimodal large               [162] K. Shao, K. Tao, K. Zhang, S. Feng, M. Cai, Y. Shang, H. You,
      language models via dynamic vision-language context sparsifi-                     C. Qin, Y. Sui, and H. Wang, “When tokens talk too much: A
      cation,” arXiv preprint arXiv:2412.00876, 2024. 3, 12, 13                         survey of multimodal long-context token compression across im-
[142] X. Liang, C. Guan, J. Lu, H. Chen, H. Wang, and H. Hu, “Dynamic                   ages, videos, and audios,” arXiv preprint arXiv:2507.20198, 2025.
      token reduction during generation for vision language models,”                    2
      arXiv preprint arXiv:2501.14204, 2025. 3, 12                                [163] OpenAI., “Gpt-4 technical report,” 2023. 3
[143] Q.-S. Zeng, Y. Li, Q. Wang, P.-T. Jiang, Z. Wu, M.-M. Cheng,                [164] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhari-
      and Q. Hou, “A glimpse to compress: Dynamic visual to-                            wal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Lan-
      ken pruning for large vision-language models,” arXiv preprint                     guage models are few-shot learners,” NeurIPS, 2020. 3
      arXiv:2508.01548, 2025. 3, 12                                               [165] R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin,
[144] J. Chen, L. Ye, J. He, Z.-Y. Wang, D. Khashabi, and A. Yuille,                    P. Liang, and T. B. Hashimoto, “Stanford alpaca: An instruction-
      “Efficient large multi-modal models via visual context compres-                   following llama model,” 2023. 3
      sion,” Advances in Neural Information Processing Systems, vol. 37,          [166] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chap-
      pp. 73 986–74 007, 2024. 3, 12, 19                                                lot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier
[145] T. Fu, T. Liu, Q. Han, G. Dai, S. Yan, H. Yang, X. Ning, and                      et al., “Mistral 7b,” arXiv preprint arXiv:2310.06825, 2023. 3
      Y. Wang, “Framefusion: Combining similarity and importance for              [167] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss
      video token reduction on large visual language models,” arXiv                     for language image pre-training,” in Proceedings of the IEEE/CVF
      preprint arXiv:2501.01986, 2024. 3, 12, 18                                        International Conference on Computer Vision, 2023, pp. 11 975–
[146] K. Shao, K. Tao, C. Qin, H. You, Y. Sui, and H. Wang, “Holitom:                   11 986. 3
      Holistic token merging for fast video large language models,”               [168] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agar-
      arXiv preprint arXiv:2505.21334, 2025. 3, 5, 12, 18                               wal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning
[147] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson,                  transferable visual models from natural language supervision,”
      K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a                 in International conference on machine learning. PmLR, 2021, pp.
      visual language model for few-shot learning,” Advances in neural                  8748–8763. 3
      information processing systems, vol. 35, pp. 23 716–23 736, 2022. 3,        [169] R. Bavishi, E. Elsen, C. Hawthorne, M. Nye, A. Odena, A. Somani,
      12, 19                                                                            and S. Taşırlar, “Introducing our multimodal models,” 2023. 3
[148] J. Ye, H. Xu, H. Liu, A. Hu, M. Yan, Q. Qian, J. Zhang, F. Huang,           [170] C. Team, “Chameleon: Mixed-modal early-fusion foundation
      and J. Zhou, “mplug-owl3: Towards long image-sequence under-                      models,” arXiv preprint arXiv:2405.09818, 2024. 3
      standing in multi-modal large language models,” arXiv preprint              [171] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
      arXiv:2408.04840, 2024. 3, 5, 13, 17, 21                                          Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
[149] S. Yan, J. Han, J. Tsai, H. Xue, R. Fang, L. Hong, Z. Guo,                        Advances in neural information processing systems, vol. 30, 2017. 4
      and R. Zhang, “Crosslmm: Decoupling long video sequences                    [172] J. Li, D. Li, S. Savarese, and S. C. H. Hoi, “BLIP-2: bootstrapping
      from lmms via dual cross-attention mechanisms,” arXiv preprint                    language-image pre-training with frozen image encoders and
      arXiv:2505.17020, 2025. 3, 13, 15                                                 large language models,” in International Conference on Machine
[150] X. Ye, Y. Gan, X. Huang, Y. Ge, and Y. Tang, “Voco-llama: Towards                 Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, ser.
      vision compression with large language models,” in Proceedings                    Proceedings of Machine Learning Research, A. Krause, E. Brun-
      of the Computer Vision and Pattern Recognition Conference, 2025, pp.              skill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol.
      29 836–29 846. 3, 5, 13                                                           202, 2023, pp. 19 730–19 742. 5
[151] J. Chen, L. Ye, J. He, Z.-Y. Wang, D. Khashabi, and A. Yuille,              [173] E. Song, W. Chai, G. Wang, Y. Zhang, H. Zhou, F. Wu, H. Chi,
      “Efficient large multi-modal models via visual context compres-                   X. Guo, T. Ye, Y. Zhang et al., “Moviechat: From dense token to
      sion,” Advances in Neural Information Processing Systems, vol. 37,                sparse memory for long video understanding,” in Proceedings of
      pp. 73 986–74 007, 2024. 3, 13                                                    the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
[152] K. Tao, C. Qin, H. You, Y. Sui, and H. Wang, “Dycoke: Dynamic                     tion, 2024, pp. 18 221–18 232. 5, 15, 16
      compression of tokens for fast video large language models,” in             [174] L. Shen, T. Hao, T. He, S. Zhao, Y. Zhang, P. Liu, Y. Bao, and
      Proceedings of the Computer Vision and Pattern Recognition Confer-                G. Ding, “Tempme: Video temporal token merging for efficient
      ence, 2025, pp. 18 992–19 001. 3, 13, 14, 15, 18                                  text-video retrieval,” arXiv preprint arXiv:2409.01156, 2024. 5, 15,
[153] J. Wang, Z. Liu, Y. Rao, and J. Lu, “Sparsemm: Head sparsity                      16
      emerges from visual concept responses in mllms,” arXiv preprint             [175] Z. Wang, D. Gong, S. Wang, Z. Huang, and Y. Luo, “Is less
      arXiv:2506.05344, 2025. 3, 5, 13, 19                                              more? exploring token condensation as training-free test-time
[154] M. Kim, K. Shim, J. Choi, and S. Chang, “Infinipot-v: Memory-                     adaptation,” arXiv preprint arXiv:2410.14729, 2024. 5, 18
      constrained kv cache compression for streaming video under-                 [176] K. Y. Li, S. Goyal, J. D. Semedo, and J. Z. Kolter, “Inference
      standing,” arXiv preprint arXiv:2506.15745, 2025. 3, 13                           optimal vlms need fewer visual tokens and more parameters,”
[155] Z. Ning, G. Liu, Q. Jin, W. Ding, M. Guo, and J. Zhao, “Livevlm:                  arXiv preprint arXiv:2411.03312, 2024. 5, 10, 17, 19
      Efficient online video understanding via streaming-oriented kv              [177] X. Ye, Y. Gan, Y. Ge, X.-P. Zhang, and Y. Tang, “Atp-llava:
      cache and retrieval,” arXiv preprint arXiv:2505.15269, 2025. 3, 13                Adaptive token pruning for large vision language models,” in
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                              28

      Proceedings of the Computer Vision and Pattern Recognition Confer-        [199] X. Liu, Y. Shu, Z. Liu, A. Li, Y. Tian, and B. Zhao, “Video-xl-
      ence, 2025, pp. 24 972–24 982. 5, 12                                            pro: Reconstructive token compression for extremely long video
[178] C. Yang, X. Dong, X. Zhu, W. Su, J. Wang, H. Tian, Z. Chen,                     understanding,” arXiv preprint arXiv:2503.18478, 2025. 8, 16
      W. Wang, L. Lu, and J. Dai, “Pvc: Progressive visual token                [200] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer,
      compression for unified image and video processing in large                     “High-resolution image synthesis with latent diffusion models,”
      vision-language models,” in Proceedings of the Computer Vision and              in Proceedings of the IEEE/CVF conference on computer vision and
      Pattern Recognition Conference, 2025, pp. 24 939–24 949. 5, 16                  pattern recognition, 2022, pp. 10 684–10 695. 8
[179] X. Huang, H. Zhou, and K. Han, “Prunevid: Visual token                    [201] W. Dai, N. Lee, B. Wang, Z. Yang, Z. Liu, J. Barker,
      pruning for efficient video large language models,” 2024.                       T. Rintamaki, M. Shoeybi, B. Catanzaro, and W. Ping, “Nvlm:
      [Online]. Available: https://arxiv.org/abs/2412.16117 5, 15                     Open frontier-class multimodal llms,” 2024. [Online]. Available:
[180] X. Li, Y. Wang, J. Yu, X. Zeng, Y. Zhu, H. Huang, J. Gao,                       https://arxiv.org/abs/2409.11402 9, 18
      K. Li, Y. He, C. Wang et al., “Videochat-flash: Hierarchical              [202] Z. Xia, X. Pan, S. Song, L. E. Li, and G. Huang, “Vision trans-
      compression for long-context video modeling,” arXiv preprint                    former with deformable attention,” in Proceedings of the IEEE/CVF
      arXiv:2501.00574, 2024. 5, 7, 18                                                conference on computer vision and pattern recognition, 2022, pp.
[181] S. Zhang, Q. Fang, Z. Yang, and Y. Feng, “Llava-mini: Efficient                 4794–4803. 10
      image and video large multimodal models with one vision to-               [203] R. Liao, C. Zhao, J. Li, W. Feng, Y. Lyu, B. Chen, and H. Yang,
      ken,” arXiv preprint arXiv:2501.03895, 2025. 5, 9, 10, 17                       “Catp: Cross-attention token pruning for accuracy preserved
[182] J. Li, J. Fan, F. Tang, G. Huang, S. Zhu, S. Liu, N. Xie, W. Liu, and           multimodal model inference,” in 2025 IEEE Conference on Artificial
      Y. Liao, “Fcot-vl: Advancing text-oriented large vision-language                Intelligence (CAI). IEEE, 2025, pp. 1100–1104. 10
      models with efficient visual token compression,” arXiv preprint           [204] J. Han, L. Du, Y. Wu, X. Zhou, H. Du, and W. Zheng, “Adafv:
      arXiv:2502.18512, 2025. 5, 19                                                   Rethinking of visual-language alignment for vlm acceleration,”
[183] Z. Wen, Y. Gao, S. Wang, J. Zhang, Q. Zhang, W. Li, C. He, and                  arXiv preprint arXiv:2501.09532, 2025. 10
      L. Zhang, “Stop looking for important tokens in multimodal                [205] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang,
      language models: Duplication matters more,” arXiv preprint                      “An image is worth 1/2 tokens after layer 2: Plug-and-play infer-
      arXiv:2502.11494, 2025. 5, 17, 18, 22                                           ence acceleration for large vision-language models,” in European
[184] L. Shen, G. Gong, T. He, Y. Zhang, P. Liu, S. Zhao, and G. Ding,                Conference on Computer Vision. Springer, 2024, pp. 19–35. 11
      “Fastvid: Dynamic density pruning for fast video large language           [206] L. Xing, Q. Huang, X. Dong, J. Lu, P. Zhang, Y. Zang, Y. Cao,
      models,” arXiv preprint arXiv:2503.11187, 2025. 5                               C. He, J. Wang et al., “Pyramiddrop: Accelerating your large
[185] W. Zeng, Z. Huang, K. Ji, and Y. Yan, “Skip-vision: Efficient                   vision-language models via pyramid visual redundancy reduc-
      and scalable acceleration of vision-language models via adaptive                tion,” arXiv preprint arXiv:2410.17247, 2024. 11
      token skipping,” arXiv preprint arXiv:2503.21817, 2025. 5                 [207] Y. Zhu, C. Xie, S. Liang, B. Zheng, and S. Guo, “Focusllava: A
                                                                                      coarse-to-fine approach for efficient and effective visual token
[186] R. Luo, R. Shan, L. Chen, Z. Liu, L. Wang, M. Yang, and X. Xia,
                                                                                      compression,” arXiv preprint arXiv:2411.14228, 2024. 11
      “Vcm: Vision concept modeling based on implicit contrastive
      learning with vision-language instruction fine-tuning,” arXiv             [208] X. Huang, H. Zhou, and K. Han, “Prunevid: Visual token prun-
      preprint arXiv:2504.19627, 2025. 5, 10, 19, 22                                  ing for efficient video large language models,” arXiv preprint
                                                                                      arXiv:2412.16117, 2024. 11
[187] D. Li, Z. Yang, and S. Lu, “Todre: Visual token pruning via
      diversity and task awareness for efficient large vision-language          [209] B. Cheng, Y. Ma, L. Wu, S. Liu, A. Ma, X. Wu, D. Leng, and Y. Yin,
      models,” arXiv preprint arXiv:2505.18757, 2025. 5                               “Hico: Hierarchical controllable diffusion model for layout-to-
                                                                                      image generation,” arXiv preprint arXiv:2410.14324, 2024. 11
[188] K. Li, X. Chen, C. Gao, Y. Li, and X. Chen, “Balanced token
                                                                                [210] T. Fu, T. Liu, Q. Han, G. Dai, S. Yan, H. Yang, X. Ning, and
      pruning: Accelerating vision language models beyond local opti-
                                                                                      Y. Wang, “Framefusion: Combining similarity and importance for
      mization,” arXiv preprint arXiv:2505.22038, 2025. 5, 11
                                                                                      video token reduction on large visual language models,” arXiv
[189] Y. Zhang, Y. Lu, T. Wang, F. Rao, Y. Yang, and L. Zhu, “Flexselect:             preprint arXiv:2501.01986, 2024. 11
      Flexible token selection for efficient long video understanding,”
                                                                                [211] J. Lee, K. Xuan, C. Ekbote, S. Polisetty, Y. R. Fung, and P. P. Liang,
      arXiv preprint arXiv:2506.00993, 2025. 5, 18
                                                                                      “Tamp: Token-adaptive layerwise pruning in multimodal large
[190] S. Yang, J. Li, X. Lai, B. Yu, H. Zhao, and J. Jia, “Visionthink: Smart         language models,” arXiv preprint arXiv:2504.09897, 2025. 11
      and efficient vision language model via reinforcement learning,”          [212] W. Ye, Q. Wu, W. Lin, and Y. Zhou, “Fit and prune:
      arXiv preprint arXiv:2507.13348, 2025. 5, 22                                    Fast and training-free visual token pruning for multi-
[191] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey            modal large language models,” 2024. [Online]. Available:
      on multimodal large language models,” National Science Review,                  https://arxiv.org/abs/2409.10197 11
      vol. 11, no. 12, p. nwae403, 2024. 4                                      [213] W. Zhang, Z. Zhu, N. Li, K. Liu, and Y. Liu, “Adaptinfer:
[192] J. Wu, W. Gan, Z. Chen, S. Wan, and P. S. Yu, “Multimodal large                 Adaptive token pruning for vision-language model inference
      language models: A survey,” in 2023 IEEE International Conference               with dynamical text guidance,” arXiv preprint arXiv:2508.06084,
      on Big Data (BigData). IEEE, 2023, pp. 2247–2256. 4                             2025. 11
[193] L. Yao, H. Wu, K. Ouyang, Y. Zhang, C. Xiong, B. Chen, X. Sun,            [214] A. Li, Y. Duan, J. Zhang, C. Ma, Y. Xie, G. Carneiro, M. Yaqub,
      and J. Li, “Generative frame sampler for long video understand-                 and H. Wang, “Transprune: Token transition pruning for efficient
      ing,” arXiv preprint arXiv:2503.09146, 2025. 4, 16, 17                          large vision-language model,” arXiv preprint arXiv:2507.20630,
[194] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec,                       2025. 11
      V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al.,       [215] R. Xu, Y. Wang, Y. Luo, and B. Du, “Rethinking visual token re-
      “Dinov2: Learning robust visual features without supervision,”                  duction in lvlms under cross-modal misalignment,” arXiv preprint
      arXiv preprint arXiv:2304.07193, 2023. 7                                        arXiv:2506.22283, 2025. 11, 12
[195] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agar-            [216] F. Sun, L. Shen, H. Chen, S. Zhao, J. Han, and G. Ding, “Adatp:
      wal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning               Attention-debiased token pruning for video large language mod-
      transferable visual models from natural language supervision,”                  els,” arXiv preprint arXiv:2505.20100, 2025. 12
      in International conference on machine learning. PMLR, 2021, pp.          [217] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré, “Flashattention:
      8748–8763. 7                                                                    Fast and memory-efficient exact attention with io-awareness,”
[196] H. Wu, M. Tang, X. Zheng, and H. Jiang, “When language over-                    Advances in neural information processing systems, vol. 35, pp.
      rules: Revealing text dominance in multimodal large language                    16 344–16 359, 2022. 12
      models,” arXiv preprint arXiv:2508.10552, 2025. 7                         [218] Q. Wang, H. Ye, M.-Y. Chung, Y. Liu, Y. Lin, M. Kuo, M. Ma,
[197] H. Zhang, Y. Wang, Y. Tang, Y. Liu, J. Feng, and X. Jin,                        J. Zhang, and Y. Chen, “Corematching: A co-adaptive sparse
      “Flash-vstream: Efficient real-time understanding for long video                inference framework with token and neuron pruning for compre-
      streams,” arXiv preprint arXiv:2506.23825, 2025. 8, 15, 16                      hensive acceleration of vision-language models,” arXiv preprint
[198] S. Jie, Y. Tang, J. Guo, Z.-H. Deng, K. Han, and Y. Wang, “To-                  arXiv:2505.19235, 2025. 12
      ken compensator: Altering inference cost of vision transformer            [219] R. Pei, W. Sun, Z. Fu, and J. Wang, “Greedyprune: Retenting
      without re-tuning,” in European conference on computer vision.                  critical visual token set for large vision language models,” arXiv
      Springer, 2024, pp. 76–94. 8                                                    preprint arXiv:2506.13166, 2025. 12
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                            29

[220] L. Meng, H. Li, B.-C. Chen, S. Lan, Z. Wu, Y.-G. Jiang, and S.-N.       [242] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with
      Lim, “Adavit: Adaptive vision transformers for efficient image                selective state spaces,” in First conference on language modeling,
      recognition,” in Proceedings of the IEEE/CVF conference on computer           2024. 15
      vision and pattern recognition, 2022, pp. 12 309–12 318. 12             [243] G. Zhang, J. Liu, S. Cao, X. Zhao, K. Zhao, K. Ma, and L. Wang,
[221] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with            “Dynamic and compressive adaptation of transformers from
      gumbel-softmax,” arXiv preprint arXiv:1611.01144, 2016. 12                    images to videos,” arXiv preprint arXiv:2408.06840, 2024. 15
[222] Y. Wen, Q. Cao, Q. Fu, S. Mehta, and M. Najibi, “Efficient vision-      [244] S.-H. Lee, J. Wang, Z. Zhang, D. Fan, and X. Li, “Video to-
      language models by summarizing visual tokens into compact                     ken merging for long-form video understanding,” arXiv preprint
      registers,” arXiv preprint arXiv:2410.14072, 2024. 13                         arXiv:2410.23782, 2024. 15
[223] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, “Efficient             [245] J. Cho, J. Lee, M. Hayat, K. Hwang, F. Porikli, and S. Choi, “Floc:
      streaming language models with attention sinks,” arXiv preprint               Facility location-based efficient visual token compression for long
      arXiv:2309.17453, 2023. 13                                                    video understanding,” arXiv preprint arXiv:2511.00141, 2025. 15
[224] S. Ge, Y. Zhang, L. Liu, M. Zhang, J. Han, and J. Gao, “Model tells     [246] J. Gao, C. Sun, Z. Yang, and R. Nevatia, “Tall: Temporal activity
      you what to discard: Adaptive kv cache compression for llms,”                 localization via language query,” in Proceedings of the IEEE inter-
      arXiv preprint arXiv:2310.01801, 2023. 13                                     national conference on computer vision, 2017, pp. 5267–5275. 16
[225] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,        [247] X. Wei, X. Liu, Y. Zang, X. Dong, P. Zhang, Y. Cao, J. Tong,
      Y. Tian, C. Ré, C. Barrett et al., “H2o: Heavy-hitter oracle for             H. Duan, Q. Guo, J. Wang et al., “Videorope: What makes
      efficient generative inference of large language models,” Advances            for good video rotary position embedding?” arXiv preprint
      in Neural Information Processing Systems, vol. 36, pp. 34 661–34 710,         arXiv:2502.05173, 2025. 16
      2023. 13                                                                [248] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with
[226] Z. Liu, B. Liu, J. Wang, Y. Dong, G. Chen, Y. Rao, R. Krishna, and            selective state spaces,” in First Conference on Language Modeling,
      J. Lu, “Efficient inference of vision instruction-following models            2024. 16
      with elastic cache,” arXiv preprint arXiv:2407.18121, 2024. 13          [249] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng,
[227] D. Zheng, M. Huang, B. Jiang, H. Hu, and X. Chen, “Towards                    W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang,
      lossless ultimate vision token compression for vlms,” arXiv                   F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin,
      preprint arXiv:2512.09010, 2025. 14                                           J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo,
[228] M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, “Video-chatgpt:                 C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang,
      Towards detailed video understanding via large vision and lan-                J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu,
      guage models,” arXiv preprint arXiv:2306.05424, 2023. 15, 19                  H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang,
[229] M. S. Ryoo, H. Zhou, S. Kendre, C. Qin, L. Xue, M. Shu, J. Park,              H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou,
      K. Ranasinghe, S. Savarese, R. Xu et al., “xgen-mm-vid (blip-3-               J. Zhou, Y. Zhu, and K. Zhu, “Qwen3-vl technical report,” arXiv
      video): You only need 32 tokens to represent a video even in                  preprint arXiv:2511.21631, 2025. 16
      vlms,” arXiv preprint arXiv:2410.16267, 2024. 15, 16                    [250] S. Chen, X. Lan, Y. Yuan, Z. Jie, and L. Ma, “Timemarker: A versa-
[230] J. Jiang, X. Li, Z. Liu, M. Li, G. Chen, Z. Li, D.-A. Huang, G. Liu,          tile video-llm for long and short video understanding with supe-
      Z. Yu, K. Keutzer et al., “Token-efficient long video understanding           rior temporal localization ability,” arXiv preprint arXiv:2411.18211,
      for multimodal llms,” arXiv preprint arXiv:2503.04130, 2025. 15,              2024. 16
      16                                                                      [251] Y. Wu, X. Hu, Y. Sun, Y. Zhou, W. Zhu, F. Rao, B. Schiele, and
[231] W. Chai, E. Song, Y. Du, C. Meng, V. Madhavan, O. Bar-Tal, J.-                X. Yang, “Number it: Temporal grounding videos like flipping
      N. Hwang, S. Xie, and C. D. Manning, “Auroracap: Efficient,                   manga,” arXiv preprint arXiv:2411.10332, 2024. 16
      performant video detailed captioning and a new benchmark,”              [252] H. Zhang, Y. Wang, Y. Tang, Y. Liu, J. Feng, J. Dai, and X. Jin,
      arXiv preprint arXiv:2410.03051, 2024. 15                                     “Flash-vstream: Memory-based real-time understanding for long
[232] Y. Weng, M. Han, H. He, X. Chang, and B. Zhuang, “Longvlm:                    video streams,” arXiv preprint arXiv:2406.08085, 2024. 16
      Efficient long video understanding via large language models,”          [253] S. Gurukar and A. Kadav, “Long-vmnet: Accelerating long-
      in European Conference on Computer Vision. Springer, 2024, pp.                form video understanding via fixed memory,” arXiv preprint
      453–470. 15, 19                                                               arXiv:2503.13707, 2025. 17
[233] Z. Liu, C.-W. Xie, P. Li, L. Zhao, L. Tang, Y. Zheng, C. Liu, and       [254] X. Wang, Q. Si, J. Wu, S. Zhu, L. Cao, and L. Nie, “Retake:
      H. Xie, “Hybrid-level instruction injection for video token com-              Reducing temporal and knowledge redundancy for long video
      pression in multi-modal large language models,” in Proceedings                understanding,” arXiv preprint arXiv:2412.20504, 2024. 17
      of the Computer Vision and Pattern Recognition Conference, 2025, pp.    [255] B. Xu, Z. Xiao, J. Li, J. Ju, Z. Luo, J. Luan, and Q. Jin, “Timeviper:
      8568–8578. 15, 16                                                             A hybrid mamba-transformer model for efficient long video
[234] J. Qi, Y. Yao, Y. Bai, B. Xu, J. Li, Z. Liu, and T.-S. Chua, “An lmm          understanding,” arXiv preprint arXiv:2511.16595, 2025. 17
      for efficient video understanding via reinforced compression of         [256] S. Wang, T. Niu, R. Yang, D. Liu, X. He, Z. Wen, C. He, X. Hu, and
      video cubes,” arXiv preprint arXiv:2504.15270, 2025. 15, 16                   L. Zhang, “Videocompressa: Data-efficient video understanding
[235] M. Xu, M. Gao, Z. Gan, H.-Y. Chen, Z. Lai, H. Gang, K. Kang, and              via joint temporal compression and spatial reconstruction,” arXiv
      A. Dehghan, “Slowfast-llava: A strong training-free baseline for              preprint arXiv:2511.18831, 2025. 17
      video large language models,” arXiv preprint arXiv:2407.15841,          [257] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping
      2024. 15, 16                                                                  language-image pre-training with frozen image encoders and
[236] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li, “Llava-              large language models,” in International conference on machine
      video: Video instruction tuning with synthetic data,” Transactions            learning. PMLR, 2023, pp. 19 730–19 742. 17, 19
      on Machine Learning Research, 2025. 15, 16, 19                          [258] Z. Ning, J. Zhao, Q. Jin, W. Ding, and M. Guo, “Inf-mllm: Efficient
[237] B. Yang, B. Wen, B. Ding, C. Liu, C. Chu, C. Song, C. Rao, C. Yi,             streaming inference of multimodal large language models on a
      D. Li, D. Zang et al., “Kwai keye-vl 1.5 technical report,” arXiv             single gpu,” arXiv preprint arXiv:2409.09086, 2024. 17
      preprint arXiv:2509.01563, 2025. 15, 16                                 [259] J. Lin, Z. Fang, C. Chen, Z. Wan, F. Luo, P. Li, Y. Liu, and
[238] X. Lan, Y. Yuan, Z. Jie, and L. Ma, “Vidcompress: Memory-                     M. Sun, “Streamingbench: Assessing the gap for mllms to achieve
      enhanced temporal compression for video understanding in large                streaming video understanding,” arXiv preprint arXiv:2411.03628,
      language models,” arXiv preprint arXiv:2410.11417, 2024. 15, 16               2024. 17
[239] Y. Liu, S. Li, Y. Liu, Y. Wang, S. Ren, L. Li, S. Chen, X. Sun,         [260] H. Xiong, Z. Yang, J. Yu, Y. Zhuge, L. Zhang, J. Zhu, and
      and L. Hou, “Tempcompass: Do video llms really understand                     H. Lu, “Streaming video understanding and multi-round in-
      videos?” arXiv preprint arXiv:2403.00476, 2024. 14                            teraction with memory-enhanced knowledge,” arXiv preprint
[240] Z. Shangguan, C. Li, Y. Ding, Y. Zheng, Y. Zhao, T. Fitzgerald,               arXiv:2501.13468, 2025. 17
      and A. Cohan, “Tomato: Assessing visual temporal reasoning              [261] Z. Huang, X. Li, J. Li, J. Wang, X. Zeng, C. Liang, T. Wu,
      capabilities in multimodal foundation models,” arXiv preprint                 X. Chen, L. Li, and L. Wang, “Online video understanding: A
      arXiv:2410.23266, 2024. 14                                                    comprehensive benchmark and memory-augmented method,”
[241] Y. Liu, Z. Ma, Z. Qi, Y. Wu, Y. Shan, and C. W. Chen, “Et bench:              arXiv preprint arXiv:2501.00584, 2024. 17
      Towards open-ended event-level video-language understand-               [262] P. Zhang, X. Dong, Y. Cao, Y. Zang, R. Qian, X. Wei, L. Chen,
      ing,” Advances in Neural Information Processing Systems, vol. 37,             Y. Li, J. Niu, S. Ding et al., “Internlm-xcomposer2. 5-omnilive:
      pp. 32 076–32 110, 2024. 14                                                   A comprehensive multimodal system for long-term streaming
JOURNAL OF LATEX CLASS FILES, NOVEMBER 2025                                                                                                             30

      video and audio interactions,” arXiv preprint arXiv:2412.09596,                  mote sensing imagery: Coarse-to-fine text-guided token prun-
      2024. 17                                                                         ing,” arXiv preprint arXiv:2503.07588, 2025. 21
[263] L. Yuan, J. Wang, H. Sun, Y. Zhang, and Y. Lin, “Tarsier2:                 [284] Y. Niu, Z. Song, Q. Luo, G. Chen, M. Ma, and F. Li, “Atmformer:
      Advancing large vision-language models from detailed video de-                   An adaptive token merging vision transformer for remote sens-
      scription to comprehensive video understanding,” arXiv preprint                  ing image scene classification,” Remote Sensing, vol. 17, no. 4, p.
      arXiv:2501.07888, 2025. 17                                                       660, 2025. 21
[264] L. Yao, Y. Zhang, Z. Wang, X. Hou, T. Ge, Y. Jiang, X. Sun, and            [285] J. Chen, Z. Lv, S. Wu, K. Q. Lin, C. Song, D. Gao, J.-W. Liu,
      Q. Jin, “Edit as you wish: Video caption editing with multi-                     Z. Gao, D. Mao, and M. Z. Shou, “Videollm-online: Online video
      grained user control,” in Proceedings of the 32nd ACM International              large language model for streaming video,” in Proceedings of the
      Conference on Multimedia, 2024, pp. 1924–1933. 17                                IEEE/CVF Conference on Computer Vision and Pattern Recognition,
[265] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson,                 2024, pp. 18 407–18 418. 22
      K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo:            [286] S. Di, Z. Yu, G. Zhang, H. Li, T. Zhong, H. Cheng, B. Li,
      a visual language model for few-shot learning,” in Advances in                   W. He, F. Shu, and H. Jiang, “Streaming video question-
      Neural Information Processing Systems, 2022. 17                                  answering with in-context video kv-cache retrieval,” arXiv
[266] Z. Liu, P. Han, H. Yu, H. Li, and J. You, “Time-r1: To-                          preprint arXiv:2503.00540, 2025. 22
      wards comprehensive temporal reasoning in llms,” arXiv preprint            [287] Y. Wang, X. Liu, X. Gui, X. Lin, B. Yang, C. Liao, T. Chen,
      arXiv:2505.13508, 2025. 17                                                       and L. Zhang, “Accelerating streaming video large language
[267] X. Zeng, K. Li, C. Wang, X. Li, T. Jiang, Z. Yan, S. Li, Y. Shi, Z. Yue,         models via hierarchical token compression,” arXiv preprint
      Y. Wang et al., “Timesuite: Improving mllms for long video under-                arXiv:2512.00891, 2025. 22
      standing via grounded tuning,” arXiv preprint arXiv:2410.19702,            [288] L. Lei, J. Gu, X. Ma, C. Tang, J. Chen, and T. Xu, “Generic
      2024. 17                                                                         token compression in multimodal large language models from an
[268] H. Wu, D. Li, B. Chen, and J. Li, “Longvideobench: A bench-                      explainability perspective,” arXiv preprint arXiv:2506.01097, 2025.
      mark for long-context interleaved video-language understand-                     22
      ing,” ArXiv preprint, vol. abs/2407.15754, 2024. 17                        [289] Z. Kong, Y. Li, F. Zeng, L. Xin, S. Messica, X. Lin, P. Zhao,
[269] Q. Zhang, M. Liu, L. Li, M. Lu, Y. Zhang, J. Pan, Q. She,                        M. Kellis, H. Tang, and M. Zitnik, “Token reduction should go
      and S. Zhang, “Beyond attention or similarity: Maximizing con-                   beyond efficiency in generative models–from vision, language to
      ditional diversity for token pruning in mllms,” arXiv preprint                   multimodality,” arXiv preprint arXiv:2505.18227, 2025. 22
      arXiv:2506.10967, 2025. 18                                                 [290] X. Zhang, L. Zhu, H. He, S. Zeng, O. Fu, J. Hu, Z. Yao, and
[270] C. Lv, B. Zhang, Y. Yong, R. Gong, Y. Huang, S. Gu, J. Wu, Y. Shi,               Y. Lu, “Adatok: Adaptive token compression with object-aware
      J. Guo, and W. Wang, “Llmc+: Benchmarking vision-language                        representations for efficient multimodal llms,” arXiv preprint
      model compression with a plug-and-play toolkit,” arXiv preprint                  arXiv:2511.14169, 2025. 22
      arXiv:2508.09981, 2025. 17                                                 [291] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,
[271] X. Liu, Y. Wang, J. Ma, and L. Zhang, “Video compression                         P. Dollár, and C. L. Zitnick, “Microsoft coco: Common objects in
      commander: Plug-and-play inference acceleration for video large                  context,” in Computer Vision–ECCV 2014: 13th European Confer-
      language models,” arXiv preprint arXiv:2505.14454, 2025. 18                      ence, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V
                                                                                       13. Springer, 2014, pp. 740–755. 22
[272] Q. Cao, B. Paranjape, and H. Hajishirzi, “Pumer: Pruning and
                                                                                 [292] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang,
      merging tokens for efficient vision language models,” arXiv
                                                                                       X. Zheng, K. Li, X. Sun, Y. Wu, and R. Ji, “Mme: A comprehensive
      preprint arXiv:2305.17530, 2023. 18
                                                                                       evaluation benchmark for multimodal large language models,”
[273] X. Wu, F. Zeng, X. Wang, and X. Chen, “Ppt: Token pruning
                                                                                       2024. [Online]. Available: https://arxiv.org/abs/2306.13394 22
      and pooling for efficient vision transformers,” arXiv preprint
                                                                                 [293] Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X.-C. Yin, C.-L.
      arXiv:2310.01812, 2023. 18
                                                                                       Liu, L. Jin, and X. Bai, “Ocrbench: on the hidden mystery of ocr
[274] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang,
                                                                                       in large multimodal models,” Science China Information Sciences,
      S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv
                                                                                       vol. 67, no. 12, p. 220102, 2024. 23
      preprint arXiv:2502.13923, 2025. 19
                                                                                 [294] Y. Shi, H. Wang, W. Xie, H. Zhang, L. Zhao, Y.-F. Zhang, X. Li,
[275] Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu,                     C. Fu, Z. Wen, W. Liu et al., “Mme-videoocr: Evaluating ocr-based
      P. Shi, Y. Shi, C. Jiang, C. Li, Y. Xu, H. Chen, J. Tian, Q. Qi,                 capabilities of multimodal llms in video scenarios,” arXiv preprint
      J. Zhang, and F. Huang, “mplug-owl: Modularization empowers                      arXiv:2505.21333, 2025. 23
      large language models with multimodality,” 2023. 19                        [295] M. Mathew, D. Karatzas, and C. Jawahar, “Docvqa: A dataset for
[276] H. Zhang, X. Li, and L. Bing, “Video-llama: An instruction-tuned                 vqa on document images,” in Proceedings of the IEEE/CVF winter
      audio-visual language model for video understanding,” arXiv                      conference on applications of computer vision, 2021, pp. 2200–2209.
      preprint arXiv:2306.02858, 2023. 19                                              23
[277] D. Guo, F. Wu, F. Zhu, F. Leng, G. Shi, H. Chen, H. Fan, J. Wang,          [296] C. Liao, W. Wang, Z. Wen, X. Zheng, Y. Wang, H. He, Y. Lyu,
      J. Jiang, J. Wang et al., “Seed1. 5-vl technical report,” arXiv preprint         L. Jiang, X. Zou, Y. Fu et al., “Are we using the right benchmark:
      arXiv:2505.07062, 2025. 19                                                       An evaluation framework for visual token compression meth-
[278] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong,                      ods,” arXiv preprint arXiv:2510.07143, 2025. 23
      Q. Zhang, X. Zhu, L. Lu, B. Li, P. Luo, T. Lu, Y. Qiao, and
      J. Dai, “Internvl: Scaling up vision foundation models and
      aligning for generic visual-linguistic tasks,” ArXiv preprint, vol.
      abs/2312.14238, 2023. 19
[279] H. Xiao, F. Zhou, X. Liu, T. Liu, Z. Li, X. Liu, and X. Huang,
      “A comprehensive survey of large language models and multi-
      modal large language models in medicine,” Information Fusion, p.
      102888, 2024. 21
[280] Y. Hu, C. Xu, B. Lin, W. Yang, and Y. Y. Tang, “Medical multi-
      modal large language models: A systematic review,” Intelligent
      Oncology, 2025. 21
[281] Y. Ding, S. Luo, Y. Dai, Y. Jiang, Z. Li, G. Martin, and Y. Peng,
      “A survey on mllm-based visually rich document understand-
      ing: Methods, challenges, and emerging trends,” arXiv preprint
      arXiv:2507.09861, 2025. 21
[282] F. Wang, H. Wang, Z. Guo, D. Wang, Y. Wang, M. Chen, Q. Ma,
      L. Lan, W. Yang, J. Zhang et al., “Xlrs-bench: Could your mul-
      timodal llms understand extremely large ultra-high-resolution
      remote sensing imagery?” in Proceedings of the Computer Vision
      and Pattern Recognition Conference, 2025, pp. 14 325–14 336. 21
[283] J. Luo, Y. Zhang, X. Yang, K. Wu, Q. Zhu, L. Liang, J. Chen,
      and Y. Li, “When large vision-language model meets large re-
