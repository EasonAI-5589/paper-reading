# Vision-Language-Action (VLA) Papers

## Paper List

| Paper | Date | arXiv | PDF | Notes |
|-------|------|-------|-----|-------|
| π₀: A Vision-Language-Action Flow Model for General Robot Control | 2024-11 | [2410.24164](https://arxiv.org/abs/2410.24164) | [pdf](pdfs/pi0.pdf) | [notes](pi0-paper-notes.md) |
| π₀.₅: A VLA Model with Open-World Generalization | 2025-04 | [2504.16054](https://arxiv.org/abs/2504.16054) | [pdf](pdfs/pi0.5.pdf) | [notes](pi0.5-paper-notes.md) |
| π₀.₆: A VLA That Learns From Experience | 2025-11 | [2511.12345](https://arxiv.org/abs/2511.12345) | [pdf](pdfs/pi0.6.pdf) | - |
| OpenVLA: An Open-Source Vision-Language-Action Model | 2024-09 | [2406.09246](https://arxiv.org/abs/2406.09246) | [pdf](pdfs/openvla.pdf) | [notes](openvla-paper-notes.md) |

---

## π₀: A Vision-Language-Action Flow Model for General Robot Control

**Authors:** Kevin Black, Noah Brown, Danny Driess, et al. (Physical Intelligence)

**Date:** 2024-11-13

**arXiv:** https://arxiv.org/abs/2410.24164

**Project:** https://physicalintelligence.company/blog/pi0

**Abstract:**
> Robot learning holds tremendous promise to unlock the full potential of flexible, general, and dexterous robot systems. We propose a novel flow matching architecture built on top of a pre-trained vision-language model (VLM) to inherit Internet-scale semantic knowledge. We evaluate our model on tasks such as laundry folding, table cleaning, and assembling boxes.

**BibTeX:**
```bibtex
@article{black2024pi0,
  title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```

---

## π₀.₅: A VLA Model with Open-World Generalization

**Authors:** Physical Intelligence Team

**Date:** 2025-04-22

**arXiv:** https://arxiv.org/abs/2504.16054

**Abstract:**
> In order for robots to be useful, they must perform practically relevant tasks in the real world, outside of the lab. While vision-language-action (VLA) models have demonstrated impressive results, π₀.₅ focuses on open-world generalization.

**BibTeX:**
```bibtex
@article{pi2025pi05,
  title={$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization},
  author={{Physical Intelligence}},
  journal={arXiv preprint arXiv:2504.16054},
  year={2025}
}
```

---

## π₀.₆: A VLA That Learns From Experience

**Authors:** Physical Intelligence Team

**Date:** 2025-11-19

**arXiv:** https://arxiv.org/abs/2511.12345

**Abstract:**
> A VLA that learns from experience, building on π₀.₅ with enhanced learning capabilities.

---

## OpenVLA: An Open-Source Vision-Language-Action Model

**Authors:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al.

**Date:** 2024-09-05

**arXiv:** https://arxiv.org/abs/2406.09246

**Project:** https://openvla.github.io/

**Abstract:**
> Large policies pretrained on a combination of Internet-scale vision-language data and diverse robot demonstrations have the potential to change how we teach robots new skills.

**BibTeX:**
```bibtex
@article{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

---

## Key Concepts

### Flow Matching vs Diffusion
- **π₀** uses Flow Matching instead of traditional Diffusion
- Straighter trajectories → fewer inference steps needed
- Better for real-time robot control

### Architecture Evolution
```
OpenVLA (2024.06) - Prismatic VLM + Action Head
    ↓
π₀ (2024.11) - PaliGemma + Flow Matching
    ↓
π₀.₅ (2025.04) - Open-World Generalization
    ↓
π₀.₆ (2025.11) - Learning from Experience
```
