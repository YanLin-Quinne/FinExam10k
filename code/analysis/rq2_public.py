"""Recompute matched intervention outcomes on the released public partition."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def exact_mcnemar(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    k = min(rescue, harm)
    return min(1.0, 2.0 * float(binom.cdf(k, n, 0.5)))


def outcome(ids, gold, predictions, direct_index, branch_index):
    direct_correct = sum(predictions[item_id][direct_index] == gold[item_id] for item_id in ids)
    branch_correct = sum(predictions[item_id][branch_index] == gold[item_id] for item_id in ids)
    rescue = sum(
        predictions[item_id][direct_index] != gold[item_id]
        and predictions[item_id][branch_index] == gold[item_id]
        for item_id in ids
    )
    harm = sum(
        predictions[item_id][direct_index] == gold[item_id]
        and predictions[item_id][branch_index] != gold[item_id]
        for item_id in ids
    )
    n = len(ids)
    return {
        "direct": 100 * direct_correct / n,
        "branch": 100 * branch_correct / n,
        "rescue": rescue,
        "harm": harm,
        "delta": 100 * (rescue - harm) / n,
        "p": exact_mcnemar(rescue, harm),
    }


def main() -> int:
    items = json.loads((DATA / "finexam10k_public_5110.json").read_text(encoding="utf-8"))
    gold = {row["id"]: str(row["answer"]).strip().upper() for row in items}
    ids = sorted(gold)
    matrix = json.loads((DATA / "intervention_matrix_public_5110.json").read_text(encoding="utf-8"))
    index = {name: position for position, name in enumerate(matrix["conditions"])}
    predictions = matrix["predictions"]

    comparisons = [
        ("PoT Function-RAG", "pot_direct", "pot_function"),
        ("PoT FunctionGraph-RAG", "pot_direct", "pot_graph"),
        ("PoT Branch Verifier", "pot_direct", "pot_verifier"),
        ("CoT Function-RAG (judge)", "cot_direct", "cot_function_judge"),
        ("CoT FunctionGraph-RAG", "cot_direct", "cot_graph"),
    ]
    print("scope: released public partition only")
    for label, direct, branch in comparisons:
        result = outcome(ids, gold, predictions, index[direct], index[branch])
        print(
            f"{label:<29} direct={result['direct']:6.2f} branch={result['branch']:6.2f} "
            f"rescue={result['rescue']:4d} harm={result['harm']:4d} "
            f"delta={result['delta']:+.3f} p={result['p']:.6g}"
        )

    print("\nPoT Function-RAG versus FunctionGraph-RAG by judge-retained count")
    for count in range(4):
        subset = [item_id for item_id in ids
                  if int(matrix["aux"][item_id]["judge_selected_count"]) == count]
        result = outcome(subset, gold, predictions, index["pot_function"], index["pot_graph"])
        print(
            f"#Fn={count} n={len(subset):4d} Fn={result['direct']:6.2f} FG={result['branch']:6.2f} "
            f"rescue={result['rescue']:3d} harm={result['harm']:3d} "
            f"delta={result['delta']:+.3f} p={result['p']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
