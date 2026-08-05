# FinExam-10K

Anonymous review artifact for **FinExam-10K**, a professional financial examination reasoning
benchmark spanning CFA Levels I--III and FRM Parts I--II.

The paper separates two complementary evaluation scopes:

- **Full-Coverage Track:** all 10,198 expert-reannotated items; this is the maintained leaderboard
  universe.
- **Context-Complete Reasoning Track:** 7,625 items whose answer-necessary evidence is locally
  attached; this is the primary basis for claims about reasoning from the supplied record.

This artifact releases the 5,110 public mock/practice items. The 5,088 held-out items remain
sequestered. The public release is designed to make the dataset structure, public predictions,
public difficulty labels, public RQ1/RQ2 diagnostics, and public-only gate training path auditable
without exposing held-out questions.

## Contents

```text
README.md
LICENSE.md
requirements.txt
pyproject.toml
leaderboard.html
run_tests.sh

code/
  public_reproduce.py              one-command public reproduction
  verify_aggregate_claims.py       arithmetic checks for sequestered aggregates
  models17.py                      released 17-model prediction registry
  conditions.py                    released intervention registry
  data/
    build_public_exports.py        canonical JSONL/CSV exports for the 5,110 public records
  tables/
    make_public_tables.py          public summary table generation
  figures/
    make_public_figures.py         public smoke-test diagnostic figures
  analysis/
    rq1_public_diagnostics.py      public members of the clean-Hard and universal-failure sets
    rq2_public.py                  public rescue/harm and judge-stratified analyses
  diagnostics/
    difficulty_public.py           public difficulty reconstruction and stage association
  router/
    features.py                    single authoritative 27-feature implementation
    train_gate.py                  public five-fold OOF selection and frozen-model verification
    gate_infer.py                  deterministic frozen-gate inference
  selector/
    audit_frozen.py                PoT/CoT artifact and graph audit
    pot_candidate_protocol.py      one-hop candidate protocol
    pot_frozen_selector.py         frozen four-feature PoT scoring rule
    cot_selector.py                frozen 56-feature CoT selector implementation

data/
  finexam10k_public_5110.json      original released 19-field public records
  public/                          canonical JSONL/CSV/XLSX exports for public inspection
  response_matrix_public_5110.json
  intervention_matrix_public_5110.json
  difficulty_labels_public_5110.json
  context_completeness_public.json
  diagnostic_context_complete_hard.json
  diagnostic_zero_solve.json
  leaderboard_data.json
  router/
  selector/
  aggregates/
```

## Installation

```bash
python -m pip install -r requirements.txt
```

The audited environment uses Python 3.11. No GPU or network access is required because the release
contains frozen option predictions rather than model weights or raw generations.

## Quick start

Run the full offline audit:

```bash
./run_tests.sh
```

The six public reproduction entry points below are also tested in CI:

```bash
python code/data/build_public_exports.py --out /tmp/finexam10k_public_exports
python code/tables/make_public_tables.py --out /tmp/finexam10k_public_tables
python code/figures/make_public_figures.py --out /tmp/finexam10k_public_figures
python code/public_reproduce.py
python code/router/train_gate.py
python code/selector/audit_frozen.py
```

Additional focused commands:

```bash
python code/diagnostics/difficulty_public.py
python code/analysis/rq1_public_diagnostics.py
python code/analysis/rq2_public.py
python code/router/gate_infer.py
python code/verify_aggregate_claims.py
```

## Released public data

The most reviewer-friendly files are in `data/public/`:

| File | Purpose |
|---|---|
| `finexam10k_public_5110_canonical.json` | canonical public records with metadata and stable field order |
| `finexam10k_public_5110_canonical.jsonl` | one public record per line |
| `finexam10k_public_5110_table.csv` | flat inspection table |
| `finexam10k_public_5110.xlsx` | spreadsheet with public items, summaries, dictionary, model scores, and intervention scores |
| `data_dictionary.csv` | field definitions |
| `stage_summary.csv` | public counts by CFA/FRM stage |
| `difficulty_context_summary.csv` | public counts by difficulty and context completeness |
| `model_public_scores.csv` | public 17-model scores recomputed from the response matrix |
| `intervention_public_scores.csv` | public intervention scores and rescue/harm counts |

The public partition contains 5,110 items:

| Stage | Items |
|---|---:|
| CFA Level I | 3,109 |
| CFA Level II | 1,030 |
| CFA Level III | 179 |
| FRM Part I | 516 |
| FRM Part II | 276 |

Context completeness is a local-answerability property, not a schema property. All released public
records have a nonempty stem, at least two options, a gold answer, and a rationale. A subset of
records refers to a detached table, vignette, figure, exhibit, or option text and is marked
accordingly in `data/context_completeness_public.json`.

## Gate reproduction

`code/router/features.py` is the only implementation of the 27 Direct-stage features. Both
`train_gate.py` and `gate_infer.py` import it. Numeric-option detection therefore uses the same
regular expression during training and inference.

The release identifiers were re-minted after the original experiment. Re-sorting those opaque ids
would change the shuffled cross-validation folds, so `data/router/public_cv_folds.json` maps every
opaque public id to its original fold. Running `train_gate.py` reproduces:

- five-fold OOF correct: 3,518 / 5,110 (68.8454%);
- selected regularization: `C=0.5`;
- selected threshold: `0.68`;
- the frozen coefficient vector and intercept to numerical tolerance.

`gate_infer.py` reproduces the public frozen decision vector and records its hash in
`public_decision_manifest.json`. The held-out partition is not distributed; the held-out result is
recorded by aggregate counts and decision-vector hash in `heldout_decision_manifest.json`.

The reported `1 + trigger rate` value is an implied branch-invocation count under lazy execution.
It is not a measurement of latency, tokens, money, or energy.

## Selector boundary

The PoT and CoT FunctionGraph-RAG chains use different frozen selectors. Their exact retrievers,
graph relations, feature counts, output budgets, label counts, and split counts are recorded in
`data/selector/provenance_manifest.json` and checked by `code/selector/audit_frozen.py`.

The upstream FinanceReasoning function corpus, embedding index, and relevance-label files are
third-party research assets and are not redistributed here. Therefore, this bundle supports exact
inspection of the frozen selector rules, graph file, and public per-item selector state, but does
not claim end-to-end regeneration of retrieval candidates or selector training from this bundle
alone.

## Reproducibility tiers

1. **Exact from released per-item data:** public data exports, public 17-model scores, public
   difficulty diagnostics, public RQ1 slices, public RQ2 transitions, public gate cross-validation,
   frozen public gate decisions, and selector artifact integrity.
2. **Aggregate arithmetic only:** Full-Coverage/Context-Complete RQ2 results, held-out gate results,
   and bounded agentic probes. Their per-item held-out outputs are sequestered.
3. **External-source dependent:** end-to-end FunctionGraph-RAG candidate generation and selector
   training, which require the cited FinanceReasoning assets.


## Anonymity

This bundle contains no author names, institution identifiers, Git history, credentials, or local
absolute paths. During anonymous review, do not expose a public non-anonymous mirror containing
byte-identical files or unique frozen hashes.

## Licence and intended use

Code is released under MIT. The released data is for non-commercial academic research under the
notice in `LICENSE.md`. Official CFA Institute and GARP examination content is excluded, and the
benchmark is not affiliated with either organization. Benchmark scores are not evidence of fitness
for professional financial practice.
