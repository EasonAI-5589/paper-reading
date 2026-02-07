# Chapter-Llama

**会议/期刊**: CVPR 2025
**作者**: [待填写]
**链接**: [arXiv / OpenAccess]

---

## 一句话总结

[待填写]

---

## 目录

- [Abstract](sections/00-abstract.md)
- [Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs](sections/01-chapter-llama-efficient-chaptering-in-hour-long-vi.md)
- [Abstract](sections/02-abstract.md)
- [1. Introduction](sections/03-introduction.md)
- [2. Related Work](sections/04-related-work.md)
- [3. Chapter-Llama: LLM-based Video Chaptering](sections/05-chapter-llama-llm-based-video-chaptering.md)
- [4. Experiments](sections/06-experiments.md)
- [4.1. Data and evaluation](sections/07-data-and-evaluation.md)
- [4.2. Comparison with the state of the art](sections/08-comparison-with-the-state-of-the-art.md)
- [4.3. Ablation studies](sections/09-ablation-studies.md)
- [4.4. Iterative prediction on longer videos](sections/10-iterative-prediction-on-longer-videos.md)
- [5. Conclusions](sections/11-conclusions.md)
- [References](sections/12-references.md)
- [APPENDIX](sections/13-appendix.md)
- [A. Implementation Details 13](sections/14-a-implementation-details-13.md)
- [B. Data Analysis and Statistics 14](sections/15-b-data-analysis-and-statistics-14.md)
- [C. Additional Quantitative Results](sections/16-additional-quantitative-results.md)
- [14](sections/17-.md)
- [D. Additional Qualitative Analyses 18](sections/18-d-additional-qualitative-analyses-18.md)
- [A. Implementation Details](sections/19-a-implementation-details.md)
- [A.1. Finetuning the LLM](sections/20-a1-finetuning-the-llm.md)
- [A.2. Prompt details](sections/21-a2-prompt-details.md)
- [A.3. Training data format](sections/22-a3-training-data-format.md)
- [A.4. Iterative prediction details](sections/23-a4-iterative-prediction-details.md)
- [B. Data Analysis and Statistics](sections/24-b-data-analysis-and-statistics.md)
- [B.1. Video duration distribution](sections/25-b1-video-duration-distribution.md)
- [B.2. Video category distribution](sections/26-b2-video-category-distribution.md)
- [B.3. Videos within 15k window token limit](sections/27-b3-videos-within-15k-window-token-limit.md)
- [C. Additional Quantitative Results](sections/28-additional-quantitative-results.md)
- [C.1. Predicting timestamps without chapter titles](sections/29-1-predicting-timestamps-without-chapter-titles.md)
- [C.2. ASR timestamp representation](sections/30-2-asr-timestamp-representation.md)
- [C.3. Modality prefixes](sections/31-3-modality-prefixes.md)
- [C.4. Alternative frame selection strategies](sections/32-4-alternative-frame-selection-strategies.md)
- [C.5. Training data size on the frame selection model](sections/33-5-training-data-size-on-the-frame-selection-model.md)
- [C.6. Separate training data for frame selector and Chapter-Llama](sections/34-6-separate-training-data-for-frame-selector-and-ch.md)
- [C.7. LLM variants](sections/35-7-llm-variants.md)
- [C.8. LoRA rank](sections/36-8-lora-rank.md)
- [C.9. Training on videos of various durations](sections/37-9-training-on-videos-of-various-durations.md)
- [C.10. Oracle experiments with partial ground truth input](sections/38-10-oracle-experiments-with-partial-ground-truth-in.md)
- [C.11. Performance on videos that have no speech](sections/39-11-performance-on-videos-that-have-no-speech.md)
- [C.12. Full set of metrics](sections/40-12-full-set-of-metrics.md)
- [C.13. Repetition analysis](sections/41-13-repetition-analysis.md)
- [C.14. Accuracy of number of chapter predictions](sections/42-14-accuracy-of-number-of-chapter-predictions.md)
- [D. Additional Qualitative Analyses](sections/43-d-additional-qualitative-analyses.md)
- [D.1. Evaluation metrics](sections/44-d1-evaluation-metrics.md)
- [D.2. Visualizing captions](sections/45-d2-visualizing-captions.md)
- [D.3. Chapter-Llama prediction examples](sections/46-d3-chapter-llama-prediction-examples.md)
- [Ground truth](sections/47-ground-truth.md)
- [Frame selector(S: 49, C: 187)](sections/48-frame-selectors-49-c-187.md)
- [Chapter-Llama(S: 54, C: 225)](sections/49-chapter-llamas-54-c-225.md)
- [Captions](sections/50-captions.md)
- [Ground truth](sections/51-ground-truth.md)
- [Chapter-Llama(S:38, C:296)](sections/52-chapter-llamas38-c296.md)

---

## 核心贡献

1. 
2. 
3. 

---

## 阅读进度

- [ ] Abstract
- [ ] Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs
- [ ] Abstract
- [ ] 1. Introduction
- [ ] 2. Related Work
- [ ] 3. Chapter-Llama: LLM-based Video Chaptering
- [ ] 4. Experiments
- [ ] 4.1. Data and evaluation
- [ ] 4.2. Comparison with the state of the art
- [ ] 4.3. Ablation studies
- [ ] 4.4. Iterative prediction on longer videos
- [ ] 5. Conclusions
- [ ] References
- [ ] APPENDIX
- [ ] A. Implementation Details 13
- [ ] B. Data Analysis and Statistics 14
- [ ] C. Additional Quantitative Results
- [ ] 14
- [ ] D. Additional Qualitative Analyses 18
- [ ] A. Implementation Details
- [ ] A.1. Finetuning the LLM
- [ ] A.2. Prompt details
- [ ] A.3. Training data format
- [ ] A.4. Iterative prediction details
- [ ] B. Data Analysis and Statistics
- [ ] B.1. Video duration distribution
- [ ] B.2. Video category distribution
- [ ] B.3. Videos within 15k window token limit
- [ ] C. Additional Quantitative Results
- [ ] C.1. Predicting timestamps without chapter titles
- [ ] C.2. ASR timestamp representation
- [ ] C.3. Modality prefixes
- [ ] C.4. Alternative frame selection strategies
- [ ] C.5. Training data size on the frame selection model
- [ ] C.6. Separate training data for frame selector and Chapter-Llama
- [ ] C.7. LLM variants
- [ ] C.8. LoRA rank
- [ ] C.9. Training on videos of various durations
- [ ] C.10. Oracle experiments with partial ground truth input
- [ ] C.11. Performance on videos that have no speech
- [ ] C.12. Full set of metrics
- [ ] C.13. Repetition analysis
- [ ] C.14. Accuracy of number of chapter predictions
- [ ] D. Additional Qualitative Analyses
- [ ] D.1. Evaluation metrics
- [ ] D.2. Visualizing captions
- [ ] D.3. Chapter-Llama prediction examples
- [ ] Ground truth
- [ ] Frame selector(S: 49, C: 187)
- [ ] Chapter-Llama(S: 54, C: 225)
- [ ] Captions
- [ ] Ground truth
- [ ] Chapter-Llama(S:38, C:296)

---

## 笔记

### 亮点


### 局限性


### 与我的工作的关系


---

*Generated by Paper Reader Skill*
