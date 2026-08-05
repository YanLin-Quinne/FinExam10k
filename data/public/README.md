# Public FinExam-10K data exports

This directory contains reviewer-facing exports of the 5,110 public mock/practice records.

Files:

- `finexam10k_public_5110_canonical.json`: metadata plus canonical public records.
- `finexam10k_public_5110_canonical.jsonl`: one canonical public record per line.
- `finexam10k_public_5110_table.csv`: flat table view for inspection.
- `finexam10k_public_5110.xlsx`: spreadsheet view of the same public table and summary sheets.
- `data_dictionary.csv`: field descriptions.
- `stage_summary.csv`, `difficulty_context_summary.csv`: count summaries.
- `model_public_scores.csv`: public 17-model scores recomputed from the released response matrix.
- `intervention_public_scores.csv`: public intervention-arm scores and rescue/harm counts.
- `public_export_manifest.json`: row counts and SHA-256 checksums for the export files.

Scope:

- Public records: 5,110
- Context complete: 3,406
- Context incomplete: 1,704

Context completeness is a local-answerability flag, not a structural schema flag. Every public record has a nonempty stem, at least two options, a gold answer, and a rationale.
