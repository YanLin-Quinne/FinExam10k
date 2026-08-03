# Field schema, released public partition

`finexam10k_public_5110.json` holds 5,110 records with 19 fields each. Every field is
present on every record. No stem is empty and every gold letter indexes a real option.

| Field | Meaning | Example |
|---|---|---|
| `answer` | gold option letter | `"A"` |
| `answerability_defect` | defect category when answerable is false, null otherwise | `"absent_vignette"` |
| `answerable` | adjudicated flag: can the item be answered from its own text | `false` |
| `category` | curriculum subject where the source metadata carries one, otherwise a paper identifier such as a mock exam name | `"2025 Mock Exam 4 Session 2"` |
| `consensus_pass_rate` | share of the 17 frozen systems answering correctly. The thresholds 2/3 and 1/3 are applied to this | `0.9286` |
| `content` | question stem | `"Tugba Aydin Case Scenario Tugba Aydin is a foreign exchange analyst for a g` |
| `difficulty` | frozen band from the 17-system consensus: easy, medium or hard | `"easy"` |
| `difficulty_answerable_pool` | band recomputed on the answerable pool, reported for sensitivity only | `"easy"` |
| `difficulty_band_3group` | band under the two-group access-mode balanced rule used throughout the paper. Identical to difficulty | `"easy"` |
| `difficulty_band_flat_17` | band under an unweighted 17-system rule, reported for sensitivity only | `"easy"` |
| `difficulty_version` | identifier of the band definition, see the appendix | `"difficulty_v1"` |
| `exam` | CFA or FRM. Items are aligned to those programmes, not drawn from official examinations | `"CFA"` |
| `explanation` | reference rationale as provided by the source. Not model input in any reported experiment | `"A. Correct because a decline in the ratio of exports to imports is consider` |
| `id` | opaque release identifier. A keyed hash of an internal identifier, 16 hex characters. It carries no timestamp and no ordering, and the key is not distributed | `"a88f86f671515a9b"` |
| `level` | examination stage: Level I, Level II, Level III, Part I or Part II | `"Level II"` |
| `missing_exhibit` | whether the stem references an exhibit that was not recoverable | `false` |
| `missing_referent` | free text naming what the stem points at that is not present. Written during adjudication, so its cardinality is high by design | `"Exhibit 1 and specific data trends are absent"` |
| `options` | answer options. Three for CFA aligned items, four for FRM aligned items | `[{"id": "A", "content": "ratio of exports to imports."}, {"id": "B", "conten` |
| `publication_split` | always public_mock_practice in this file | `"public_mock_practice"` |

## A note on identifiers

Release identifiers are a keyed hash of the internal identifier used during curation. The
internal identifiers were database object identifiers whose bytes decoded to a creation
timestamp and whose value space was shared between the released and held-out partitions. Both
properties leaked information unrelated to the data, so the release uses opaque identifiers.
They are stable: the same item keeps the same identifier across releases, and the held-out
partition would be re-minted under the same key if it were ever released.

## Response matrix

`response_matrix_public_5110.json` gives, for each item, the predicted option letter from each of
the seventeen frozen leaderboard systems, in the order listed under `systems`. An empty string
means no option letter could be parsed, which is scored as incorrect. Correctness is that letter
against the item's `answer`, so every accuracy, band and error-concentration statistic in the
paper is recomputable from this file plus the items file.

## Intervention matrix

`intervention_matrix_public_5110.json` has the same shape over 9 retrieval
conditions across the two reasoning chains:

```
  pot_direct
  pot_function
  pot_graph
  pot_verifier
  cot_direct
  cot_function_bm25
  cot_function_judge
  cot_graph
  cot_graph_nojudge
```

Its `aux` block carries `judge_selected_count`, the number of functions the Function-RAG judge
chose to inject, which is the stratifying variable behind the paper's central RQ2 result, plus
the parser and executor verdict for the direct condition.

## Selector state

`selector/per_item_sidecar_public_5110.json` gives, for each of the 5,110 released
items, the candidate pool size, the number of functions actually injected, and the parser and
executor verdict, separately for Function-RAG and for FunctionGraph-RAG. This is what makes the
stratified analysis checkable without rerunning anything.

`selector/pot_selector_frozen.json` and `selector/cot_selector_frozen.json` hold the frozen
coefficients and full provenance for the two selectors. `selector/pot_function_graph.json` is the
static function graph the Program-of-Thought condition expands over.

## Scope

These files cover the released public partition only. Statistics the paper reports over all
10,198 items will not reproduce from this bundle alone, by design. The 5,110 items here are the
mock and practice half. The held-out half is reserved for the leaderboard protocol and is not
distributed.

## Known data characteristics

* 105 items carry a short Chinese case-analysis prefix in the stem, all of them CFA Level
  II, inherited from the source material. The benchmark is otherwise English.
* 364 items carry a Chinese reference rationale. Rationales are not model input.
* 1,704 items (33.3 percent) refer to background the release does not carry.
  They are flagged individually rather than silently retained.
* Option letters are positional. `answer` is within range on every record.

## Diagnostic subsets

Two files carve out the item sets the error analysis turns on. Both are defined over all 10,198
benchmark items and release only their public members, so **counts here do not match the paper**.
Each file states its full size, its released size and the number withheld.

| File | Full set | Released | Withheld |
|---|---:|---:|---:|
| `diagnostic_context_complete_hard.json` | 372 | 138 | 234 |
| `diagnostic_zero_solve.json` | 188 | 118 | 70 |

`diagnostic_context_complete_hard.json` holds Hard-band items an adjudicator judged answerable
from their own text, so a failure there cannot be blamed on absent background. This is the
cleanest difficulty evidence in the benchmark.

`diagnostic_zero_solve.json` holds items no system in the frozen panel answered correctly.
Accuracy on it is zero for those seventeen by construction, so the informative quantities are
elsewhere: which option the systems chose instead, and how much of the set is a context defect
rather than genuine difficulty.

Every record in both carries the full field schema plus a `diagnostic` block:

| Key | Meaning |
|---|---|
| `n_systems_correct` | how many of the seventeen answered correctly |
| `n_systems_unparsed` | how many produced no parsable option letter |
| `modal_choice` | the option letter the largest number of systems chose |
| `modal_choice_share` | that count over seventeen |
| `all_systems_chose_same_wrong_option` | true when every system converged on one wrong option |
| `per_system` | the predicted letter from each of the seventeen |
| `in_both_diagnostic_sets` | whether the item is in the 47-item intersection |

The two sets overlap in 47 items across the full benchmark, of which 15 are released here.
