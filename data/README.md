# FinExam-10K public partition

5,110 mock and practice items, one record per question, in four equivalent formats:

| File | Format |
|---|---|
| `finexam10k_public_5110.jsonl` | one JSON record per line (canonical) |
| `finexam10k_public_5110.json` | `{"name", "version", "records", "fields", "records_data"}` with the same records |
| `finexam10k_public_5110.csv` | flat table, one row per record |
| `finexam10k_public_5110.xlsx` | the table, with the data dictionary and count summaries as extra sheets |

`data_dictionary.csv` defines every field. `summary/stage_summary.csv` and `summary/difficulty_context_summary.csv`
give the counts by stage, difficulty band, and context completeness. `summary/panel_public_scores.csv` gives the
accuracy of the frozen 17-model panel on the public items, for comparison with new runs.

Counts: 3,406 context-complete and 1,704 context-incomplete records. By band, 3,164 are Easy, 1,087 Medium, and 859 Hard.
The 5,088 held-out items of the benchmark are not part of this release.

Text is released as curated, with one normalization: five question stems (`406c8c0d559dba63`, `4ab6df58d8335594`,
`549883a13477f91b`, `a2b2e20a3a49e5dc`, `b43d3a34b8a425d2`) carried a stray control character where a hyphen belongs
(for example "four-year"), and it was replaced by the hyphen.
