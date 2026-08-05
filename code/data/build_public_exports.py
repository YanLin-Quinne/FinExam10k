"""Build public data exports from released FinExam-10K public JSON files.

This script writes canonical JSON/JSONL, flat CSV tables, summary CSVs, and a deterministic XLSX.
It is used by CI to verify the complete public data path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.xlsx_artifact import write_public_workbook  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PUBLIC_FIELDS = [
    "id","exam","level","program_stage","source_category","publication_split",
    "difficulty","consensus_pass_rate","context_complete","answerability_defect",
    "missing_referent","question","option_A","option_B","option_C","option_D",
    "answer","answer_text","rationale"
]
FIELD_DEFINITIONS = [
    ("id", "Opaque 16-character public release identifier."),
    ("exam", "Examination family: CFA or FRM."),
    ("level", "Stage within the examination family, e.g. Level II or Part I."),
    ("program_stage", "Concatenated exam and stage label."),
    ("source_category", "Public mock/practice source category or paper/session descriptor."),
    ("publication_split", "Release partition. Public records use public_mock_practice."),
    ("difficulty", "Frozen empirical difficulty band from the 17-model panel."),
    ("consensus_pass_rate", "Group-balanced frozen consensus pass rate used for difficulty."),
    ("context_complete", "Whether the standalone record has all answer-necessary local evidence."),
    ("answerability_defect", "Defect class when context_complete is false, else blank."),
    ("missing_referent", "Free-text description of detached table, exhibit, vignette, or other referent."),
    ("question", "Question stem as released."),
    ("option_A", "Option A text."),
    ("option_B", "Option B text."),
    ("option_C", "Option C text."),
    ("option_D", "Option D text; blank for three-option CFA items."),
    ("answer", "Gold answer letter."),
    ("answer_text", "Text of the gold option."),
    ("rationale", "Reference rationale supplied with the released record."),
]

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def option_map(item: dict) -> dict[str, str]:
    out = {"A": "", "B": "", "C": "", "D": ""}
    for option in item.get("options", []):
        key = str(option.get("id", "")).strip().upper()
        if key in out:
            out[key] = str(option.get("content", "")).strip()
    return out

def flatten_item(item: dict) -> dict:
    opts = option_map(item)
    answer = str(item.get("answer", "")).strip().upper()
    return {
        "id": item["id"],
        "exam": item["exam"],
        "level": item["level"],
        "program_stage": f"{item['exam']} {item['level']}",
        "source_category": item["category"],
        "publication_split": item["publication_split"],
        "difficulty": item["difficulty"],
        "consensus_pass_rate": item["consensus_pass_rate"],
        "context_complete": bool(item["answerable"]),
        "answerability_defect": item.get("answerability_defect") or "",
        "missing_referent": item.get("missing_referent") or "",
        "question": str(item["content"]).strip(),
        "option_A": opts["A"],
        "option_B": opts["B"],
        "option_C": opts["C"],
        "option_D": opts["D"],
        "answer": answer,
        "answer_text": opts.get(answer, ""),
        "rationale": str(item["explanation"]).strip(),
    }

def validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("public export has zero rows")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate public ids")
    for row in rows:
        for field in ("id", "question", "answer", "rationale"):
            if row.get(field) in ("", None):
                raise SystemExit(f"empty required field {field} for {row.get('id')}")
        if row["answer"] not in {"A", "B", "C", "D"}:
            raise SystemExit(f"invalid answer letter for {row['id']}")
        if row["answer_text"] == "":
            raise SystemExit(f"answer option text missing for {row['id']}")
        if sum(bool(row[f"option_{letter}"]) for letter in "ABCD") < 2:
            raise SystemExit(f"too few nonempty options for {row['id']}")

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def stage_summary(rows: list[dict]) -> list[dict]:
    order = ["CFA Level I", "CFA Level II", "CFA Level III", "FRM Part I", "FRM Part II"]
    out = []
    for stage in order:
        subset = [row for row in rows if row["program_stage"] == stage]
        if not subset:
            raise SystemExit(f"zero items for stage {stage}")
        out.append({
            "program_stage": stage,
            "items": len(subset),
            "context_complete": sum(row["context_complete"] for row in subset),
            "context_incomplete": sum(not row["context_complete"] for row in subset),
            "easy": sum(row["difficulty"] == "easy" for row in subset),
            "medium": sum(row["difficulty"] == "medium" for row in subset),
            "hard": sum(row["difficulty"] == "hard" for row in subset),
        })
    return out

def difficulty_summary(rows: list[dict]) -> list[dict]:
    out = []
    for band in ["easy", "medium", "hard"]:
        subset = [row for row in rows if row["difficulty"] == band]
        if not subset:
            raise SystemExit(f"zero items for difficulty {band}")
        out.append({
            "difficulty": band,
            "items": len(subset),
            "context_complete": sum(row["context_complete"] for row in subset),
            "context_incomplete": sum(not row["context_complete"] for row in subset),
        })
    return out

def model_scores(rows: list[dict]) -> list[dict]:
    answers = {row["id"]: row["answer"] for row in rows}
    matrix = load_json(DATA / "response_matrix_public_5110.json")
    if matrix["n_items"] != len(rows):
        raise SystemExit("response-matrix count mismatch")
    out = []
    for idx, system in enumerate(matrix["systems"]):
        correct = parsed = 0
        for item_id, predictions in matrix["predictions"].items():
            pred = str(predictions[idx]).strip().upper()
            parsed += bool(pred)
            correct += pred == answers[item_id]
        out.append({
            "model": system["name"],
            "group": system["group"],
            "items": len(rows),
            "accuracy_percent": round(100 * correct / len(rows), 4),
            "parse_rate_percent": round(100 * parsed / len(rows), 4),
        })
    return out

def intervention_scores(rows: list[dict]) -> list[dict]:
    answers = {row["id"]: row["answer"] for row in rows}
    matrix = load_json(DATA / "intervention_matrix_public_5110.json")
    if matrix["n_items"] != len(rows):
        raise SystemExit("intervention-matrix count mismatch")
    conditions = matrix["conditions"]
    predictions = matrix["predictions"]
    direct_idx = {"pot": conditions.index("pot_direct"), "cot": conditions.index("cot_direct")}
    out = []
    for idx, condition in enumerate(conditions):
        chain = "pot" if condition.startswith("pot") else "cot"
        d_idx = direct_idx[chain]
        correct = parsed = rescue = harm = 0
        for item_id, preds in predictions.items():
            branch_pred = str(preds[idx]).strip().upper()
            direct_pred = str(preds[d_idx]).strip().upper()
            parsed += bool(branch_pred)
            branch_correct = branch_pred == answers[item_id]
            direct_correct = direct_pred == answers[item_id]
            correct += branch_correct
            rescue += (not direct_correct) and branch_correct
            harm += direct_correct and (not branch_correct)
        out.append({
            "condition": condition,
            "chain": chain,
            "items": len(rows),
            "accuracy_percent": round(100 * correct / len(rows), 4),
            "parse_rate_percent": round(100 * parsed / len(rows), 4),
            "rescue_vs_direct": "" if idx == d_idx else rescue,
            "harm_vs_direct": "" if idx == d_idx else harm,
            "delta_vs_direct_pp": "" if idx == d_idx else round(100 * (rescue - harm) / len(rows), 4),
        })
    return out

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DATA / "public")
    args = parser.parse_args()
    rows = [flatten_item(item) for item in load_json(DATA / "finexam10k_public_5110.json")]
    rows.sort(key=lambda row: row["id"])
    validate_rows(rows)
    out = args.out
    if out.exists() and not out.is_dir():
        raise SystemExit("public export output path is not a directory")
    out.mkdir(parents=True, exist_ok=True)

    dictionary_rows = [{"field": key, "definition": definition} for key, definition in FIELD_DEFINITIONS]
    stage_rows = stage_summary(rows)
    difficulty_rows = difficulty_summary(rows)
    model_rows = model_scores(rows)
    intervention_rows = intervention_scores(rows)

    write_json(out / "finexam10k_public_5110_canonical.json", {
        "name": "finexam10k_public_5110_canonical",
        "version": "public_v1",
        "records": len(rows),
        "fields": PUBLIC_FIELDS,
        "records_data": rows,
    })
    write_jsonl(out / "finexam10k_public_5110_canonical.jsonl", rows)
    write_csv(out / "finexam10k_public_5110_table.csv", rows, PUBLIC_FIELDS)
    write_csv(out / "data_dictionary.csv", dictionary_rows, ["field", "definition"])
    write_csv(out / "stage_summary.csv", stage_rows, ["program_stage", "items", "context_complete", "context_incomplete", "easy", "medium", "hard"])
    write_csv(out / "difficulty_context_summary.csv", difficulty_rows, ["difficulty", "items", "context_complete", "context_incomplete"])
    write_csv(out / "model_public_scores.csv", model_rows, ["model", "group", "items", "accuracy_percent", "parse_rate_percent"])
    write_csv(out / "intervention_public_scores.csv", intervention_rows, ["condition", "chain", "items", "accuracy_percent", "parse_rate_percent", "rescue_vs_direct", "harm_vs_direct", "delta_vs_direct_pp"])
    write_public_workbook(
        out / "finexam10k_public_5110.xlsx",
        rows,
        dictionary_rows,
        stage_rows,
        difficulty_rows,
        model_rows,
        intervention_rows,
    )

    readme = f"""# Public FinExam-10K data exports

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

- Public records: {len(rows):,}
- Context complete: {sum(row['context_complete'] for row in rows):,}
- Context incomplete: {sum(not row['context_complete'] for row in rows):,}

Context completeness is a local-answerability flag, not a structural schema flag. Every public record has a nonempty stem, at least two options, a gold answer, and a rationale.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "record_count": len(rows),
        "context_complete": sum(row["context_complete"] for row in rows),
        "context_incomplete": sum(not row["context_complete"] for row in rows),
        "files": {},
    }
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "public_export_manifest.json":
            manifest["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(out / "public_export_manifest.json", manifest)
    print(f"wrote {len(rows)} public records to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
