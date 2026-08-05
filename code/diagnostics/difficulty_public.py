"""Recompute difficulty diagnostics on the released 5,110-item public partition."""
from __future__ import annotations

import collections
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from models17 import GROUP_OF, NAMES, correct, load_questions  # noqa: E402

API_SERVED = {model for model in NAMES if GROUP_OF[model] == "Proprietary"}


def band(score: float, high: float = 2 / 3, low: float = 1 / 3) -> str:
    return "easy" if score >= high else "hard" if score <= low else "medium"


def score_flat(item_id: str, models: list[str]) -> float:
    return sum(correct(model, item_id) for model in models) / len(models)


def score_two_group(item_id: str, models: list[str]) -> float:
    api = [model for model in models if model in API_SERVED]
    local = [model for model in models if model not in API_SERVED]
    return 0.5 * (
        sum(correct(model, item_id) for model in api) / len(api)
        + sum(correct(model, item_id) for model in local) / len(local)
    )


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    x = rankdata(left)
    y = rankdata(right)
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def cramers_v(pairs: list[tuple[str, str]]) -> float:
    rows = sorted({row for row, _ in pairs})
    columns = sorted({column for _, column in pairs})
    table = collections.Counter(pairs)
    row_sum = collections.Counter(row for row, _ in pairs)
    column_sum = collections.Counter(column for _, column in pairs)
    n = len(pairs)
    chi_square = 0.0
    for row in rows:
        for column in columns:
            expected = row_sum[row] * column_sum[column] / n
            if expected:
                chi_square += (table[(row, column)] - expected) ** 2 / expected
    return math.sqrt(chi_square / (n * (min(len(rows), len(columns)) - 1)))


def main() -> int:
    questions = load_questions()
    ids = sorted(questions)
    scores_flat = [score_flat(item_id, NAMES) for item_id in ids]
    scores_balanced = [score_two_group(item_id, NAMES) for item_id in ids]
    bands = {item_id: band(score_two_group(item_id, NAMES)) for item_id in ids}
    counts = collections.Counter(bands.values())

    print("scope: released public partition only")
    print("items:", len(ids), "models:", len(NAMES))
    print("balanced band counts:", dict(counts))
    print("flat-vs-balanced Spearman:", f"{spearman(scores_flat, scores_balanced):.4f}")
    print("stage-band Cramer's V:", f"{cramers_v([(questions[i]['level'], bands[i]) for i in ids]):.4f}")
    print("\nstage composition")
    for level in ("Level I", "Level II", "Level III", "Part I", "Part II"):
        subset = [item_id for item_id in ids if questions[item_id]["level"] == level]
        counter = collections.Counter(bands[item_id] for item_id in subset)
        print(
            f"{level:<10} n={len(subset):4d} "
            f"easy={100*counter['easy']/len(subset):5.1f}% "
            f"medium={100*counter['medium']/len(subset):5.1f}% "
            f"hard={100*counter['hard']/len(subset):5.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
