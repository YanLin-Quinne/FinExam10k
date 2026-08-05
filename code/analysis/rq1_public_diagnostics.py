"""Public-partition diagnostics for the clean Hard and universal-failure slices.

The released files contain only public members of sets defined on all 10,198 items. Results printed
here are therefore public-slice diagnostics and are not expected to equal the full-paper counts.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def multinomial_probability(counts: tuple[int, ...]) -> float:
    n = sum(counts)
    d = len(counts)
    coefficient = math.factorial(n)
    for count in counts:
        coefficient //= math.factorial(count)
    return coefficient * (1 / d) ** n


def compositions(n: int, d: int):
    if d == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in compositions(n - first, d - 1):
            yield (first,) + rest


def expected_modal_share(n_wrong: int, n_distractors: int) -> float:
    if n_wrong <= 0:
        return float("nan")
    expectation = 0.0
    for counts in compositions(n_wrong, n_distractors):
        expectation += max(counts) / n_wrong * multinomial_probability(counts)
    return expectation


def concentration(record: dict) -> tuple[float, float] | None:
    gold = str(record["answer"]).strip().upper()
    predictions = [str(value or "").strip().upper()
                   for value in record["diagnostic"]["per_system"].values()]
    wrong = [value for value in predictions if value and value != gold]
    if not wrong:
        return None
    counts = {letter: wrong.count(letter) for letter in set(wrong)}
    observed = max(counts.values()) / len(wrong)
    null = expected_modal_share(len(wrong), len(record["options"]) - 1)
    return observed, null


def exact_sign_p(n_positive: int, n_total: int) -> float:
    if n_total == 0:
        return 1.0
    return sum(math.comb(n_total, k) for k in range(n_positive, n_total + 1)) / (2 ** n_total)


def main() -> int:
    hard_payload = json.loads((DATA / "diagnostic_context_complete_hard.json").read_text(encoding="utf-8"))
    zero_payload = json.loads((DATA / "diagnostic_zero_solve.json").read_text(encoding="utf-8"))
    hard = hard_payload["items"]
    zero = zero_payload["items"]

    pairs = [pair for record in hard if (pair := concentration(record)) is not None]
    observed = [pair[0] for pair in pairs]
    nulls = [pair[1] for pair in pairs]
    positive = sum(left > right for left, right in pairs)
    single_distractor = 0
    for record in hard:
        gold = str(record["answer"]).strip().upper()
        wrong = {
            str(value or "").strip().upper()
            for value in record["diagnostic"]["per_system"].values()
            if str(value or "").strip().upper()
            and str(value or "").strip().upper() != gold
        }
        single_distractor += int(len(wrong) == 1)

    print(hard_payload["warning"])
    print(f"public Context-Complete Hard: {len(hard)} items")
    print(f"concentration defined: {len(pairs)}")
    print(f"median observed modal wrong share: {statistics.median(observed):.4f}")
    print(f"median exact item-specific null: {statistics.median(nulls):.4f}")
    print(f"observed > null: {positive}/{len(pairs)}")
    print(f"one-sided exact sign p: {exact_sign_p(positive, len(pairs)):.6g}")
    print(f"all erroneous votes on one distractor: {single_distractor}/{len(hard)}")

    context_complete = sum(bool(record.get("answerable")) for record in zero)
    same_wrong = sum(record["diagnostic"]["all_systems_chose_same_wrong_option"] for record in zero)
    print("\n" + zero_payload["warning"])
    print(f"public universal-failure members: {len(zero)}")
    print(f"context complete: {context_complete}")
    print(f"all 17 choose the same wrong option: {same_wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
