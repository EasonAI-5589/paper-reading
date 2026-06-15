# Repository Format

Use these templates as a baseline and adapt them to the paper. Prefer the conventions already used by nearby entries when they differ.

## Full README

```markdown
# Paper Title

| 元信息 | |
|---|---|
| **标题** | Full title |
| **作者** | Authors |
| **会议** | Confirmed venue or arXiv preprint |
| **arXiv** | [ID](official URL) |
| **代码** | [Repository](official URL) |
| **日期** | YYYY-MM-DD |

## 一句话总结

State the problem, central mechanism, and strongest supported result.

## 核心贡献

1. Contribution and why it differs from prior work.
2. Method or resource contribution.
3. Experimental evidence with exact numbers.

## 关键结果

Use a table when several benchmarks or baselines matter.

## Section 导航

| Section | 文件 | 内容 |
|---|---|---|
| Abstract | [00-abstract.md](sections/00-abstract.md) | ... |

## BibTeX

Use metadata verified against an official source.
```

## Collection README

```markdown
# Paper or Model Name

**论文**: Full title<br>
**发表**: Confirmed venue or arXiv preprint<br>
**arXiv**: [ID](official URL)<br>
**代码/数据/权重**: [Official source](URL)

One concise paragraph describing the work.

**与主题的关系**: Explain why it is collected and whether it is a citation, baseline, benchmark, or later extension.

> 当前仅完成基础收录。
```

## Section Notes

```markdown
[← 返回 README](../README.md)

# 1 Introduction

Original paper paragraph, preserved verbatim.

> **批注标题**：Explain the research meaning, hidden assumption, comparison, or consequence in Chinese.

![Figure N: concise Chinese description](../images/file.jpg)
> **Figure N 解读**：Explain axes, components, comparison, and the conclusion supported by the figure.
```

Keep equations and tables adjacent to their explanation. Do not detach an annotation so far from its source that the relationship becomes ambiguous.

## Annotation Quality

A useful annotation should do at least one of the following:

- explain why the paragraph matters;
- connect it to a prior or later method;
- unpack an equation or implementation choice;
- identify what an ablation actually proves;
- interpret a figure or table using exact values;
- expose assumptions, limitations, or reproducibility risks;
- suggest a concrete research extension.

Avoid generic praise, sentence-by-sentence translation, unsupported SOTA claims, and conclusions not established by the cited experiment.

## Research Map Tables

For benchmarks:

```markdown
| Benchmark | Venue | Task | Data/metric | Relationship to seed paper |
```

For models:

```markdown
| Model | Base/scale | Open weights | Metric/result | Role in seed paper |
```

When a paper compares many systems, create a dedicated comparison page rather than overloading the paper README.

## Index Rules

- Make the seed paper and expansion logic visible near the top of the topic README.
- Link every row to a local entry when one exists.
- Keep counts synchronized with actual directories.
- Separate confirmed conference publications from arXiv-only papers.
- Keep Web3, generic agents, or other adjacent domains out of an AI Finance map unless they directly support the stated research question.
- Preserve concise root-level summaries; detailed discussion belongs in topic indexes and comparison pages.
