# Reproducibility

All commands run from the repository root with Python 3.11. Public reproduction uses released
records, frozen predictions, frozen Gate coefficients, selector state, and aggregate result files.
It does not rerun model inference or reconstruct the held-out partition.

## Complete public reproduction

```bash
python code/public_reproduce.py
```

This runs public difficulty diagnostics, RQ1 and RQ2 comparisons, the five-fold Gate training check,
public Gate inference, and selector checks. The Gate check preserves feature order, 27-dimensional
inputs, the frozen threshold, 373 public triggers, and the published public routed outcome.

## Public exports

```bash
python code/data/build_public_exports.py --out /tmp/finexam10k_public_exports
```

The command rebuilds equivalent JSON, JSONL, CSV, and XLSX representations for 5,110 public records.
The workbook is constructed from the released JSON and contains the same public fields and summary
tables.

## Full repository tests

```bash
./run_tests.sh
```

The suite checks row counts, required fields, unique IDs, format equivalence, public/held-out
separation, model and result matrix alignment, frozen feature dimensions, aggregate arithmetic,
published Gate and selector results, and the static leaderboard contract.

## Compute boundary

The reported implied branch-call count is `1 + trigger rate` under lazy execution. It is not measured
latency, token use, monetary cost, energy use, or a rerun of the model panel.
