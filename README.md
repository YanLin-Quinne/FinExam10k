# FinExam-10K

FinExam-10K is a dataset for professional financial examination reasoning across CFA Levels I through III and FRM Parts I and II. The benchmark contains 10,198 expert reannotated multiple choice items. This anonymous review artifact releases 5,110 mock and practice items and keeps 5,088 evaluation items held out.

The benchmark supports a 10,198 item Full Coverage Track and a 7,625 item Context Complete Reasoning Track. The latter contains records with all answer necessary evidence available in the local item representation. Difficulty is an empirical property derived once from a frozen panel of 17 models.

## Data Access

The public release contains exactly 5,110 records in both primary formats.

| Format | Direct link | Contents |
|---|---|---|
| JSON | [Download the 5,110 item public JSON](data/public/finexam10k_public_5110_canonical.json) | Canonical 19 field records under `records_data` |
| XLSX | [Download the 5,110 item public XLSX](data/public/finexam10k_public_5110.xlsx) | The same 5,110 records plus dictionary and summary sheets |

The pipeline source JSON with the original nested option structure is also available at [data/finexam10k_public_5110.json](data/finexam10k_public_5110.json). Other table and streaming formats are documented in [data/public/README.md](data/public/README.md).

## Verified Key Facts

| Property | Verified value |
|---|---:|
| Full benchmark | 10,198 |
| Public mock and practice partition | 5,110 |
| Held out evaluation partition | 5,088 |
| Context Complete subset | 7,625 |
| Frozen model panel | 17 models |
| Frozen Hard band | 1,437 |
| Context Complete Hard subset | 372 |
| Universal failure core | 188 |

> Difficulty is frozen once on all 10,198 items from the 17 model panel. Subsetting by access partition or context completeness never recomputes or changes an item's Easy, Medium, or Hard label.

## Artifact Map

Counts in the paper refer to the full benchmark. Item level files in this repository contain public records only. Held out question content is not released.

| Scope | Definition | Full count | Public item records released here | Artifact |
|---|---|---:|---:|---|
| Full benchmark | Every curated item | 10,198 | 5,110 | [public JSON](data/public/finexam10k_public_5110_canonical.json), [public XLSX](data/public/finexam10k_public_5110.xlsx) |
| Context Complete subset | Items with answer necessary evidence in the local record | 7,625 | 3,406 | Flags in the public exports, with audit totals in [context_completeness_public.json](data/context_completeness_public.json) |
| Frozen Hard band | Items with the frozen group balanced consensus score at or below one third | 1,437 | 859 | Public labels in [difficulty_labels_public_5110.json](data/difficulty_labels_public_5110.json) |
| Context Complete Hard subset | Intersection of Context Complete and the frozen Hard band | 372 | 138 | [diagnostic_context_complete_hard.json](data/diagnostic_context_complete_hard.json) |
| Universal failure core | Items missed by every model in the frozen 17 model panel | 188 | 118 | [diagnostic_zero_solve.json](data/diagnostic_zero_solve.json) |

The 372 item Context Complete Hard subset and the 188 item universal failure core are distinct diagnostic subsets within the 1,437 item frozen Hard band. The universal failure core is not a second difficulty band. The two diagnostic JSON files state their full count, released public count, and withheld count in metadata. Aggregate views over the full scopes are available in [leaderboard_data.json](data/leaderboard_data.json) without releasing held out questions.

Additional context artifacts include [difficulty_context_summary.csv](data/public/difficulty_context_summary.csv), [stage_summary.csv](data/public/stage_summary.csv), and the item level completeness fields in every canonical public record.

## Schema and Data Quality

The canonical JSON and the `Public Items` XLSX sheet use the same 19 columns. They cover identifiers and source metadata, item content and answer fields, the frozen difficulty label and consensus score, and context completeness annotations. Exact field definitions are in [data_dictionary.csv](data/public/data_dictionary.csv).

Every benchmark record is structurally complete with a stem, options, a gold answer, and a rationale. Context completeness measures a different property. It asks whether the standalone record contains all answer necessary evidence. Detached vignettes, tables, figures, images, or exhibits can make a structurally valid record context incomplete. Missing evidence is flagged rather than invented, and claims about reasoning from supplied evidence use the Context Complete subset.

## Public and Held Out Access

Only the 5,110 mock and practice items are public. The 5,088 held out items remain sequestered for controlled evaluation. This repository includes selected aggregate results and cryptographic hashes for held out evaluations, but no held out question records or per item predictions. Public reproduction does not reconstruct private records or recompute held out decisions.

## Evaluation and Reproduction

Use Python 3.11. No GPU or network access is required because the artifact uses frozen predictions.

```bash
python -m pip install -r requirements.txt
python code/public_reproduce.py
```

Run the complete repository audit with:

```bash
./run_tests.sh
```

This checks compilation, requirement driven tests, regenerated public exports, public diagnostics, aggregate arithmetic, router and selector audits, manifest integrity, XLSX hygiene, and deterministic anonymous packaging. The frozen result boundary and optional export commands are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Static Leaderboard

Open [leaderboard.html](leaderboard.html) directly in a browser. It is a self contained static snapshot with four views for the Full Coverage Track, Context Complete Reasoning Track, Context Complete Hard subset, and universal failure core. It does not contact a server and is not a live public leaderboard.

A maintained web leaderboard is planned after paper acceptance.

## Checksums and Release Integrity

[MANIFEST.sha256](MANIFEST.sha256) records a SHA 256 digest for every included repository file except the manifest itself. CI verifies both complete file coverage and every digest. The anonymous release builder also refuses to package files when a manifest entry is missing, stale, or mismatched.

Verify the checked out artifact with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

Regenerate the manifest only after an intentional artifact change with `python code/release/update_manifest.py`.

## Responsible Use

See [LICENSE.md](LICENSE.md) for the code and data terms. The public data is intended for noncommercial academic research. Official CFA Institute and GARP examination content is excluded, and the benchmark is not affiliated with either organization. Benchmark scores are not evidence of fitness for professional financial practice, and the artifact is not financial advice.

For anonymous packaging, release boundaries, and reproducibility scope, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation

An archival citation will be added after paper acceptance. During anonymous review, cite the accompanying submission as *FinExam-10K: When Retrieval Helps Financial Reasoning?*
