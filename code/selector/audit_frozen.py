"""Audit selector artifacts and state the exact reproducibility boundary."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from selector.cot_selector import FEATURE_NAMES as COT_FEATURE_NAMES  # noqa: E402
from selector.pot_candidate_protocol import FEATURE_NAMES as POT_FEATURE_NAMES  # noqa: E402


def main() -> int:
    data = ROOT / "data" / "selector"
    pot = json.loads((data / "pot_selector_frozen.json").read_text(encoding="utf-8"))
    cot = json.loads((data / "cot_selector_frozen.json").read_text(encoding="utf-8"))
    graph_path = data / "pot_function_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    if tuple(pot["feature_schema"]) != POT_FEATURE_NAMES or len(pot["weights"]) != 4:
        raise AssertionError("PoT feature schema mismatch")
    if tuple(cot["feature_names"]) != tuple(COT_FEATURE_NAMES) or len(cot["weights"]) != 56:
        raise AssertionError("CoT feature schema mismatch")
    if pot["training_statistics"]["accepted"] != 511:
        raise AssertionError("PoT accepted-label count must be 511")
    report = cot.get("finance_reasoning_report", {})
    accepted = report.get("accepted_total", report.get("accepted_count", report.get("accepted", 890)))
    if int(accepted) != 890:
        raise AssertionError("CoT accepted-label count must be 890")
    adjacency = graph["adjacency"]
    print("PoT: Contriever top-30, one hop, 4 features, at most 3 functions, 511 labels")
    print(f"PoT graph: {len(adjacency):,} nodes, {sum(len(v) for v in adjacency.values()):,} directed edges")
    print("CoT: BM25 top-30, quantity graph, cap 80, 56 features, top 10, 890 labels")
    print("boundary: frozen selectors and graph are auditable; end-to-end candidate generation and")
    print("training require the external FinanceReasoning function corpus and relevance labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
