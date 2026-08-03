# Datasheet for FinExam-10K, released public partition

Structured after Gebru et al., *Datasheets for Datasets*. Prepared for anonymous review.

## Motivation

**Why was the dataset created?** Existing financial reasoning benchmarks either cover a single
certification level, aggregate heterogeneous certifications under coarse labels, or provide no
item-level difficulty. None preserves CFA Levels I to III and FRM Parts I and II under one
evaluation protocol with item-level expert reannotation. FinExam-10K fills that gap so that model
comparisons can be read against externally defined professional curricula rather than ad hoc task
labels.

**Who created it and who funded it?** Withheld for anonymous review.

## Composition

**What do instances represent?** Each instance is one multiple-choice professional finance
examination item: a stem, answer options, a gold letter, a reference rationale, the examination
program and stage, and derived difficulty and context-completeness labels.

**How many instances?** The full benchmark holds 10,198 items. This release carries the
5,110-item public mock and practice partition. The remaining 5,088 held-out items are reserved for
the leaderboard protocol and are **not released**.

| Stage | Items |
|---|---:|
| CFA Level I | 3,109 |
| CFA Level II | 1,030 |
| CFA Level III | 179 |
| FRM Part I | 516 |
| FRM Part II | 276 |

| Difficulty band | Items |
|---|---:|
| easy | 3,164 |
| medium | 1,087 |
| hard | 859 |

**Is any information missing?** Yes, and it is labelled rather than hidden. 1,704 of 5,110
items (33.3 percent) refer to background the release does not carry, almost always a
table or a shared case vignette that existed as an image in the source and was therefore not
extracted. **No field is missing from any record.** All nineteen fields are present on all 5,110
items, every stem is non-empty, and every gold letter indexes a real option. See
`data/context_completeness_public.json` for the audit and worked examples of each defect class.

**Does it contain confidential or offensive content?** No. Items are professional examination
questions on finance and risk management.

**Does it identify individuals?** No. Items reference fictional analysts and firms as examination
scenarios do.

## Collection

**How was the data acquired?** From CFA and FRM aligned preparatory materials available for
academic research. Official CFA Institute and GARP examination content is excluded.

**What preprocessing was done?** Rationale filtering, normalisation, global deduplication, then
two-stage expert reannotation by a four-member finance-qualified team. A second full review by the
lead curator integrated corrections and applied deterministic checks for schema validity,
identifier uniqueness, answer-option mapping and content duplication. The paper's dataset section
documents the full chain including the counts removed at each step.

**Were identifiers re-minted for release?** Yes. Release identifiers are a keyed hash of the
internal identifier. The internal identifiers were database object identifiers whose bytes decoded
to an ingestion timestamp and whose value space was shared between the two partitions. Both
properties leaked information unrelated to the data, so the release uses opaque identifiers and the
key is not distributed.

## Uses

**What tasks is it suitable for?** Multiple-choice professional financial reasoning, difficulty-
stratified evaluation, retrieval-augmentation ablation, and context-completeness sensitivity
analysis.

**What should it not be used for?** Certification preparation advice, and any claim about a
model's readiness for professional practice. Accuracy on examination items is not a competence
assessment.

**Known limitations.** 105 items
carry a short Chinese case-analysis prefix in the stem, and
364 items carry a
Chinese reference rationale, both inherited from the source material. Rationales are not model
input. The benchmark is otherwise English.

## Distribution

**How is it distributed?** As JSON in this bundle. The held-out partition is not distributed.

**Licence.** See `LICENSE.md`. Items are provided for non-commercial research use. CFA Institute
and GARP are not affiliated with and do not endorse this work.

## Maintenance

**Who maintains it?** Withheld for anonymous review. The paper describes a maintained leaderboard
and quarterly refreshes.

**Will the held-out partition ever be released?** Its release would end the protocol that makes it
useful. Any future release would re-mint identifiers under the same key so that the two partitions
remain consistent.
