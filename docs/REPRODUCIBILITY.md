# Reproducibility contract

All commands below run from the repository root on Python 3.11. The public artifact contains the
required per-item inputs, frozen Gate coefficients, selector state, and prediction matrices. No
GPU or network access is required.

## One-command public reproduction

```bash
python code/public_reproduce.py
```

CI executes this exact command through `run_tests.sh`. It runs public diagnostics, public RQ1 and
RQ2 checks, the frozen public Gate training audit, the real Gate inference CLI, and the frozen
selector audit. The Gate CLI must verify 373 public triggers and the decision-vector hash in
`data/router/public_decision_manifest.json`.

The paper's held-out Gate result is separate: 404 triggers over 5,088 sequestered items, with 55
rescues and 35 harms. Public reproduction verifies that the frozen held-out manifest is present
and internally bounded, but cannot and must not regenerate sequestered per-item decisions.

## Public exports

```bash
python code/data/build_public_exports.py --out /tmp/finexam10k_public_exports
```

The output directory contains canonical JSON, JSONL, CSV, summary CSVs, and a freshly built XLSX.
The XLSX is constructed from the released JSON rather than copied from a workbook. Its six sheets,
five summary tables, and intervention chart contain the same public records and summaries. The
builder clears creator and last-modifier properties, removes absolute-path extensions, fixes ZIP
metadata, and rejects remaining local paths or spreadsheet formula errors.

No exporter uses the gold answer to invent missing question, option, vignette, table, figure, or
exhibit content. Records with detached context retain their published completeness flags.

## Anonymous review bundle

```bash
python code/release/update_manifest.py
python code/release/build_anonymous_bundle.py --out /tmp/FinExam10k-anonymous.zip
```

Regenerate the manifest only after rebuilding the checked-in public exports. The anonymous ZIP is
deterministic for a fixed checkout, has an empty comment, contains no Git directory or GitHub
workflow metadata, and is scanned for identity strings, local paths, public repository URLs,
badges, commit hashes, unsafe archive names, OOXML leaks, and operating-system ZIP metadata.

## Compute boundary

The paper reports `1 + trigger rate` as an implied branch-invocation count under lazy execution.
It is not measured latency, token use, monetary cost, energy, or a rerun of the H100 model panel.
The public reproduction consumes frozen predictions and does not silently rerun full model
inference, change the held-out result, or re-band empirical difficulty.
