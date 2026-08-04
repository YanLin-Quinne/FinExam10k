# FinExam-10K

Benchmark and evaluation code for **FinExam-10K**, a professional financial reasoning benchmark
spanning all five stages of the CFA and FRM programs under one evaluation protocol.

Anonymous artifact for review. Author, institution and repository identifiers are withheld.

## Why this benchmark exists

Professional finance examinations state, through published curricula, what a practitioner is
expected to know at each stage. That makes them a rare thing in evaluation: a difficulty axis
defined by someone other than the benchmark author.

Existing English financial benchmarks either cover one certification level, or aggregate several
certifications under coarse labels, or provide no item-level difficulty. None preserves CFA Levels
I to III and FRM Parts I and II together, which is what a reader needs in order to ask whether a
model's failure is a knowledge gap at one stage or a reasoning gap across all of them.

The benchmark also exists to test a specific claim. Retrieval augmentation is widely assumed to
repair financial reasoning errors. Under matched conditions it does not: on this data it is
significantly *worse* than not retrieving at all, and the aggregate hides two significant effects
with opposite signs. Reproducing that is the point of the code here.

## What is in this release

5,110 of the 10,198 benchmark items. The remaining 5,088 are held out for the leaderboard
protocol and are **not distributed**.

| Stage | Released items |
|---|---:|
| CFA Level I | 3,109 |
| CFA Level II | 1,030 |
| CFA Level III | 179 |
| FRM Part I | 516 |
| FRM Part II | 276 |
| **Total** | **5,110** |

| Difficulty band | Released items | Definition |
|---|---:|---|
| easy | 3,164 | at least two thirds of the frozen panel correct |
| medium | 1,087 | between one third and two thirds |
| hard | 859 | fewer than one third |

| | Items | Share |
|---|---:|---:|
| Answerable from the item's own text | 3,406 | 66.7% |
| Refers to background the release does not carry | 1,704 | 33.3% |

**Every record is structurally complete.** All 19 fields are present on all 5,110 items, no stem is
empty, and every gold letter indexes a real option. What a subset of items lacks is not a *field*
but a *referent*: a table or shared case vignette that was an image in the source and was
therefore not extracted. Those items carry an explicit flag naming what is absent.

| `answerability_defect` | Items | Meaning |
|---|---:|---|
| `absent_table` | 847 | the stem refers to a numeric table not extracted from the source |
| `absent_vignette` | 738 | the item belongs to a shared case vignette not carried with it |
| `absent_labelled_object` | 105 | the stem names a figure, chart or panel that is not present |
| `option_mismatch` | 10 | the options do not correspond to the quantity the stem asks for |
| `dangling_reference` | 4 | the stem points at a referent that resolves to nothing |

`data/context_completeness_public.json` holds the full audit and one worked example per defect
class.

## Repository layout

```
.
  leaderboard.html                         open in any browser, no server needed
  data/
    finexam10k_public_5110.json            5,110 released items, 19 fields each
    response_matrix_public_5110.json       17 leaderboard systems, predicted letter per item
    intervention_matrix_public_5110.json   9 retrieval conditions, predicted letter per item
    difficulty_labels_public_5110.json     frozen consensus band per item
    context_completeness_public.json       structural audit and defect examples
    diagnostic_context_complete_hard.json  138 released members of the 372-item set
    diagnostic_zero_solve.json             118 released members of the 188-item set
    selector/                              frozen selector coefficients, the function graph,
                                           and per-item selector state for the public partition
    leaderboard_data.json                  the four leaderboard views, precomputed
    SCHEMA.md                              every field, described
  code/
    paths.py            the only file that writes a path down
    selector/           how retrieval candidates are built and chosen, both chains
    models17.py         the 17-system registry, backed by the response matrix
    conditions.py             the 9 retrieval conditions, backed by the intervention matrix
    analysis/           results reported in the paper body
    rag_variants/       retrieval configurations that were tried and not reported
    diagnostics/        error analysis and red-team probes
    figures/            figure generation
  tex/main/             tables and sections as they appear in the paper
  tex/appendix/         includes appendix_selector.tex, both selectors side by side
  DATASHEET.md          provenance, composition, collection, uses, maintenance
  DATA_STATEMENT.md     language variety, annotator characteristics, text characteristics
    REPRODUCIBILITY.md    what reproduces, what does not, and why
  MANIFEST.sha256
```

## Item format

```json
{
  "id": "f8bcec19cbad5d55",
  "exam": "CFA",
  "level": "Level I",
  "category": "2025 Mock Exam 1 Session 2",
  "content": "At initiation, the price of a forward contract i  mo t likely:...",
  "options": [{"id": "A", "content": "..."}, {"id": "B", "content": "..."}, ...],
  "answer": "C",
  "explanation": "...",
  "publication_split": "public_mock_practice",
  "difficulty": "hard",
  "consensus_pass_rate": 0.3,
  "answerable": true,
  "answerability_defect": null,
  "missing_referent": null
}
```

* `difficulty` is a frozen band from the 17-system consensus, not a curriculum level. Thresholds
  are 2/3 and 1/3 of a two-group access-mode balanced pass rate, so the ten API-served and seven
  locally-served systems weigh equally. The appendix gives the construction and its sensitivity.
* `consensus_pass_rate` is the quantity those thresholds are applied to.
* `answerable` and `answerability_defect` are adjudicated labels. They never alter the question,
  the gold answer, the partition or the frozen band.
* CFA items carry three options and FRM items four. Chance is therefore per item, and the paper
  reports chance-normalised accuracy alongside raw accuracy.

## Response and intervention matrices

Raw model generations are not distributed. Their file names encode provider account structure and
they are large. What ships instead is the predicted option letter from each system on each
released item, which is what every reported number is computed from.

```json
{
  "systems": [{"name": "...", "group": "Proprietary"}, ...],
  "predictions": {"<item id>": ["A", "C", "", "B", ...]}
}
```

An empty string means no option letter could be parsed, which is scored incorrect. Correctness is
this letter against the item's `answer`, so accuracies, bands, error-concentration statistics and
rescue-harm decompositions are all recomputable without trusting our arithmetic.

`intervention_matrix_public_5110.json` has the same shape over nine retrieval conditions:
`pot_direct`, `pot_function`, `pot_graph`, `pot_verifier`, `cot_direct`, `cot_function_bm25`,
`cot_function_judge`, `cot_graph`, `cot_graph_nojudge`. Its `aux` block carries
`judge_selected_count`, the number of functions the Function-RAG judge chose to inject, which is
the stratifying variable behind the paper's central result.

## Diagnostic subsets

Two files carve out the item sets the error analysis turns on. **Both are defined over all 10,198
items and release only their public members, so counts here do not match the paper.** Each file
states its full size, its released size, and the number withheld.

| File | Full set | Released | Withheld | What it is |
|---|---:|---:|---:|---|
| `diagnostic_context_complete_hard.json` | 372 | 138 | 234 | Hard-band items answerable from their own text. A failure here cannot be blamed on absent background. |
| `diagnostic_zero_solve.json` | 188 | 118 | 70 | Items no system in the frozen panel answered correctly. Accuracy is zero by construction, so the informative quantity is where the systems went instead. |

Each record carries a `diagnostic` block with the per-system predicted letters, the modal wrong
choice and its share, and a flag for the items where every system converged on one wrong option.

## Leaderboard

`leaderboard.html` is a single self-contained page with no external resources. Open it directly in
a browser. It gives four views of the same seventeen systems, and the point is how the ranking
changes between them.

| View | Items | Chance | Systems below chance |
|---|---:|---:|---:|
| Full corpus | 10,198 | 31.24% | 0 of 17 |
| Context-complete | 7,625 | 30.97% | 0 of 17 |
| Hard and context-complete | 372 | 30.40% | **14 of 17** |
| Zero-solve | 188 | 31.52% | 17 of 17, by construction |

The zero-solve view is deliberately not a ranking. Accuracy there is zero for every system by
definition, so it reports where the systems went instead. On 41 of those 188 items all seventeen
converged on the same wrong option, which difficulty alone would not produce.

Numbers are recomputable from `data/response_matrix_public_5110.json` for the released partition.
The page itself reports over all 10,198 items, so its totals cover both partitions and will not
match a recomputation from the released half alone.

## Verifying the bundle

```
./run_tests.sh
```

Offline, no GPU, a few seconds. It compiles every Python file, runs a fifteen-case self-test over
the data and the frozen models, and performs deterministic gate inference on the public partition.
The same three steps run in CI on every push, so the badge rather than our word is the claim.

What the self-test asserts, because each of these is a claim made somewhere in this README or the
paper: every file compiles, every manifest checksum matches, all 5,110 records carry all 19 fields
with an in-range gold letter, the response and intervention matrices align with the items file,
the intervention matrix exposes the key the loader actually reads, the shipped function graph
hashes to the value the frozen selector was built against, both selectors declare their training
source and label count, the frozen gate is complete and unscaled with 27 coefficients matching 27
named features in order, and no file anywhere carries a local path, a credential, or a held-out
item.

## The frozen gate

`data/router/gate_frozen.json` holds the coefficients, the intercept, the threshold, and the
feature names in the order the coefficients expect. There is no scaler, and the file says so
explicitly: every feature is a binary indicator or is already divided by a fixed constant inside
the feature function, so a reimplementation that standardises will not reproduce the decisions.

`code/router/gate_infer.py` runs it. On the public partition it runs end to end here. On the
held-out partition, which is not released, it cannot. What ships instead is
`data/router/heldout_decision_manifest.json`: the fire rate, the reported outcome, and a sha256 of
the per-item decision vector in ascending id order. Anyone who obtains the held-out partition under
the leaderboard protocol can rerun the script and check that hash. A match proves the frozen model
reproduced the reported evaluation, without us releasing the items or the per-item mapping.

## What is shipped for reference but does not run here

`code/upstream_reference/` holds three files that need modules outside this release, such as the
upstream evaluation harness or a GPU embedding stack. They ship so the procedure can be read, they
carry a banner saying so, and nothing in the reproduction path imports them.

## Installation

```
python -m pip install -r requirements.txt
```

Python 3.11, versions pinned. No GPU is needed for anything here, because the bundle ships
predictions rather than weights and no inference is rerun.

## Quick start

Start here, which compiles everything, runs the 22-assertion self-test, and then executes each
command below:

```
bash run_tests.sh
```

The same commands individually:

```
cd code/analysis
python why_hard372.py          # error concentration against an exact per-item null
python where_graph_wins.py     # 220-test slice sweep, Benjamini-Hochberg corrected
python rq2_error_analysis.py   # rescue and harm decomposition per retrieval condition

cd ../router
python gate_infer.py           # the frozen gate, deterministic, prints a decision-vector sha256

cd ../figures
python make_taxonomy_figure.py # curriculum taxonomy
```

CI runs exactly these on every push, from a clean checkout, so a command that stops working fails
visibly rather than waiting to be found by a reader.

Every script prints a scope line naming the partition and the item count it actually ran on.
Bootstrap and cross-validation use seed 202607, fixed in the scripts, and repeated runs of
`gate_infer.py` produce a byte-identical decision vector.

**These recompute over the public partition, not the full corpus.** The paper reports over all
10,198 items; this release carries the 5,110 public ones, so the same definitions select
proportionally smaller sets. The 372 context-complete Hard items become 138 here, and the 188
zero-solve items become 118. The scope line makes this visible on every run. Subsetting is sound
because the difficulty score is computed per item, so restricting the file selects a subset of each
band rather than recomputing the bands.

Some scripts in `code/` deliberately do not run here: the held-out router evaluation, the
curriculum-subject figures, and the program-of-thought selector's upstream retrieval stack. Each
stops with a message explaining why and pointing at what ships instead. `REPRODUCIBILITY.md`
lists them.

## What will not reproduce, and why

**Every held-out number, which includes the entire routing evaluation.** The router is fitted on
the public partition and scored once on the 5,088 held-out items, and the held-out partition is
not released. That separation is what makes the leaderboard protocol worth anything, so the gap is
deliberate rather than an oversight. The routing code ships in full so the protocol can be
inspected: `code/analysis/router_gate.py` is the configuration reported in the paper, and
`code/analysis/router.py` with `code/rag_variants/router_per_condition.py` are the variants evaluated
alongside it. Selection among the four used out-of-fold accuracy inside the public partition,
never held-out accuracy.

**Statistics quoted over all 10,198 items**, since those cover both partitions.

## Licence

Code is MIT. Data is released for non-commercial research use and derives from CFA and FRM aligned
preparatory materials available for academic research. Official CFA Institute and GARP examination
content is excluded. CFA Institute and GARP are not affiliated with and do not endorse this work.
See `LICENSE.md`.

## Citation

```bibtex
@misc{finexam10k2026,
  title  = {FinExam-10K: Benchmarking Professional Financial Reasoning and
            Diagnosing the Limits of Retrieval Augmentation},
  author = {Anonymous Authors},
  year   = {2026}
}
```
