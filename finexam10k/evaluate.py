"""Score a predictions file against the public partition.

Each line of the predictions file is a JSON object with an ``id`` and either a ``prediction`` letter or the raw
model ``output`` text. Example:

    python -m finexam10k.evaluate predictions.jsonl --track context-complete
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_PATH, load_items
from .scoring import parse_answer, score


def read_predictions(path: str | Path) -> dict[str, str | None]:
    predictions: dict[str, str | None] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "id" not in row:
            raise ValueError(f"line {number}: missing id")
        if row.get("finish_reason") == "length" or row.get("error"):
            letter = None  # truncated or failed calls are incorrect, as in the paper
        elif "prediction" in row:
            letter = str(row["prediction"] or "").strip().upper() or None
        else:
            letter = parse_answer(row.get("output"))
        predictions[str(row["id"])] = letter
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--track", choices=("all", "context-complete"), default="all",
                        help="all 5,110 public items, or the 3,406 context-complete public items")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH, help="path to finexam10k_public_5110.jsonl")
    args = parser.parse_args(argv)
    items = load_items(args.data, context_complete=True if args.track == "context-complete" else None)
    result = score(items, read_predictions(args.predictions))
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"items {result['n']}  answered {result['answered']}  accuracy {result['accuracy']:.2f}%")
    for title, key in (("stage", "by_stage"), ("difficulty", "by_difficulty"), ("context", "by_context")):
        print(f"  by {title}: " + ", ".join(f"{k} {v:.2f}" for k, v in result[key].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
