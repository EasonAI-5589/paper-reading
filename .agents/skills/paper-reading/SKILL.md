---
name: paper-reading
description: Read, collect, research, compare, and maintain academic papers in the paper-reading repository. Use for requests to add a paper, perform section-by-section 批读, preserve paper text with inline Chinese annotations, extract figures and experiments, trace references or compared models, build benchmark/model research maps, update topic and root indexes, verify metadata, or continue any literature-review work in this project.
---

# Paper Reading

Follow the repository's existing structure and the user's requested depth. Treat the paper and official project pages as primary sources; never infer publication venue, model availability, experimental values, or citation relationships without evidence.

Read [references/repository-format.md](references/repository-format.md) before creating or substantially restructuring a paper entry.

## Choose the work mode

Determine the mode from the user's wording:

- **Research**: answer questions from papers, trace references, compare models, or identify benchmarks. Browse primary sources and update files only when requested or clearly implied.
- **Collection**: when the user says "收录", "放上去", or "不用批读", create only a concise paper/model/benchmark card and update indexes. Do not manufacture section notes.
- **Full reading**: when the user says "批读", "继续读", or asks about sections, preserve original paper content in `sections/` and insert Chinese annotations close to the relevant paragraphs, figures, equations, and tables.
- **Maintenance**: reorganize topic maps, fix metadata and links, remove off-topic entries, or update README files while preserving unrelated user work.

If the requested depth is unambiguous, act without asking for confirmation.

## Gather authoritative sources

1. Inspect the existing topic directory, README conventions, git status, and nearby high-quality entries before editing.
2. Prefer the official paper page/PDF, conference proceedings, OpenReview, official code repository, and official model or dataset card.
3. Verify time-sensitive facts such as latest paper version, venue status, code availability, licenses, and model weights online.
4. Distinguish direct paper statements from interpretation. Label inferred relationships as analysis rather than fact.
5. Record exact experiment values from the paper's tables or appendix. Do not copy values from secondary posts when the paper is available.

## Create the paper entry

Use `[Venue Year] Short-Name/` when the venue is confirmed. Use `[Arxiv YYMM.NNNNN] Short-Name/` or the topic's established capitalization when it is only a preprint. Do not rename existing directories solely to normalize capitalization.

For collection mode:

1. Create `README.md` with title, paper/venue/arXiv links, code/data/weights when available, a concise description, and its relationship to the topic.
2. State that only basic collection is complete.
3. Update the nearest topic index and any relevant research map.

For full reading mode:

1. Keep the original English text complete and in paper order.
2. Split it into `sections/00-abstract.md`, numbered section files, and an appendix file when useful.
3. Add Chinese annotations as Markdown blockquotes immediately after the material they explain.
4. Explain motivation, assumptions, method mechanics, equations, experimental controls, key numbers, comparisons, limitations, and implications. Avoid annotations that merely translate or repeat the paragraph.
5. Embed relevant local figures with working relative paths and explain what each figure establishes.
6. Build a navigable README containing metadata, one-sentence summary, core contributions, section navigation, key results, and verified BibTeX.

Do not claim full reading is complete unless all substantive sections and experiments have been covered.

## Expand research maps

When tracing a paper outward:

1. Separate direct citations, experimental baselines, datasets/benchmarks, evaluation methods, and later related work.
2. For model comparisons, capture exact model variant, scale, base model, open-source status, official link, metric, and score when reported.
3. Group entries by useful research categories instead of producing one flat list.
4. Make the seed paper explicit in the topic README and explain how the collection expanded beyond it.
5. Avoid retaining adjacent topics that do not support the current research question.

## Update repository indexes

Update the paper README, topic README, specialized benchmark/model index, and root README when their visible counts or descriptions change. Keep the root README concise and link to detailed maps instead of duplicating them.

Preserve existing user changes. Never remove or rewrite unrelated entries unless the user explicitly asks for cleanup.

## Verify before finishing

Run:

```bash
python3 scripts/check_repository.py
git diff --check
git status --short
```

For a completed paper with BibTeX, run when network access is available:

```bash
scripts/verify_bibtex.sh "path/to/paper"
```

Report new errors separately from pre-existing warnings. Check that README counts match actual directories and that every new local link resolves.

Commit or push only when the user requests it. When asked to push, include all completed changes from the current literature task, verify the remote branch, and confirm the pushed commit hash.
