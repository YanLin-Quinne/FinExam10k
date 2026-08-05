"""Generate public summary tables used to audit the public-data release path."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write zero-row table {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/public_tables"))
    args = parser.parse_args()
    items = load(DATA / "finexam10k_public_5110.json")
    if not items:
        raise SystemExit("zero public items")
    out = args.out
    stage_rows = []
    for stage in ["Level I", "Level II", "Level III", "Part I", "Part II"]:
        subset = [item for item in items if item["level"] == stage]
        if not subset:
            raise SystemExit(f"zero items for {stage}")
        stage_rows.append({
            "stage": stage,
            "items": len(subset),
            "context_complete": sum(bool(item["answerable"]) for item in subset),
            "context_incomplete": sum(not bool(item["answerable"]) for item in subset),
            "easy": sum(item["difficulty"] == "easy" for item in subset),
            "medium": sum(item["difficulty"] == "medium" for item in subset),
            "hard": sum(item["difficulty"] == "hard" for item in subset),
        })
    write_rows(out / "public_stage_table.csv", stage_rows,
               ["stage", "items", "context_complete", "context_incomplete", "easy", "medium", "hard"])

    context_rows = []
    for status, subset in [("context_complete", [item for item in items if item["answerable"]]),
                           ("context_incomplete", [item for item in items if not item["answerable"]])]:
        if not subset:
            raise SystemExit(f"zero {status} items")
        context_rows.append({
            "scope": status,
            "items": len(subset),
            "easy": sum(item["difficulty"] == "easy" for item in subset),
            "medium": sum(item["difficulty"] == "medium" for item in subset),
            "hard": sum(item["difficulty"] == "hard" for item in subset),
        })
    write_rows(out / "public_context_table.csv", context_rows,
               ["scope", "items", "easy", "medium", "hard"])
    print(f"wrote public tables to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
