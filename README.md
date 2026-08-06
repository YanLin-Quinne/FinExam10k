# FinExam-10K Anonymous Reviewer Artifact

This package supports anonymous review and offline reproduction of FinExam-10K, an academic
benchmark for professional financial reasoning. It contains the complete 5,110 item public
partition, frozen predictions and selectors, aggregate results, reproduction code, tests, Figure 1,
and a static leaderboard. No network access or model inference is required for the included checks.

[Open the offline leaderboard](docs/index.html)

[![FinExam-10K leaderboard preview](docs/assets/leaderboard-preview.png)](docs/index.html)

## Figure 1

![Figure 1. FinExam-10K corpus, empirical difficulty construction, and observed failure patterns.](docs/assets/figure-1-overview.png)

Figure 1 summarizes the corpus composition, expert review and public and held out split, frozen
17 model difficulty construction, and benchmark views used to study model failures.

## Public dataset

The released partition contains exactly 5,110 mock and practice records. The JSON and XLSX exports
contain the same public item IDs.

- [Canonical JSON](data/public/finexam10k_public_5110_canonical.json)
- [Canonical JSONL](data/public/finexam10k_public_5110_canonical.jsonl)
- [Table CSV](data/public/finexam10k_public_5110_table.csv)
- [Workbook XLSX](data/public/finexam10k_public_5110.xlsx)
- [Data dictionary](data/public/data_dictionary.csv)

The fixed Context-Complete Hard view contains 372 items in the full benchmark. The fixed
universal-failure core contains 188 items in the full benchmark. Their aggregate 17 system results
are included in the [leaderboard data](docs/data/leaderboard.json). To prevent disclosure of held
out item content, the canonical public diagnostic files contain only the released members: 138 in
the [Context-Complete Hard public extract](data/diagnostic_context_complete_hard.json) and 118 in
the [universal-failure public extract](data/diagnostic_zero_solve.json). The files also record the
full diagnostic counts of 372 and 188.

## Methods and experiments

- **Direct** answers from the supplied item without function retrieval.
- **Function-RAG** retrieves individual FinanceReasoning functions as structured support.
- **FunctionGraph-RAG** expands retrieval through relations between functions before selection.
- **FunctionGraph-RAG + Verifier** executes the selected PoT branch and checks its execution before
  finalizing the answer.
- **Selective Branch Router** is a frozen public trained pre execution gate applied after Direct and
  before retrieval branch execution. It uses the item and completed Direct output without observing
  the gold answer, retrieval outcome, or competing branch result.

Selected aggregate results:

| Evaluation | Direct | Intervention | Change |
|---|---:|---:|---:|
| DeepSeek-R1 CoT, full coverage, FunctionGraph-RAG | 79.75 | 79.84 | +0.09 points |
| GPT-4o PoT, full coverage, FunctionGraph-RAG | 69.37 | 68.06 | -1.30 points |
| GPT-4o PoT, full coverage, FunctionGraph-RAG + Verifier | 69.37 | 68.92 | -0.45 points |
| Selective Branch Router, held out | 70.83 | 71.23 | +0.39 points |
| Selective Branch Router, held out context complete | 78.34 | 78.88 | +0.55 points |

The held out router invoked FunctionGraph-RAG on 404 of 5,088 items, with 55 rescues and 35 harms.
Only aggregate held out results and manifests are included. Held out questions, answers, and per
item predictions are not included.

## Reproduction

Python 3.11 or 3.12 is required. Install the pinned dependencies, run the public reproduction, and
run the normal test entry point from the package root.

```bash
python -m pip install -r requirements.txt
python code/public_reproduce.py
./run_tests.sh
```

The leaderboard uses only included relative assets. To view it without network access, serve this
directory locally and open `http://127.0.0.1:8000/docs/` in a browser.

```bash
python -m http.server 8000
```

See the [reproducibility notes](docs/REPRODUCIBILITY.md) for component commands and scope.
