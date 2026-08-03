"""Intervention condition predictions over the released public partition.

Nine conditions across the two reasoning chains. Naming follows the paper throughout: the graph
retrieval condition is called FunctionGraph-RAG in every label, docstring and printed string.

Held-out numbers, which includes the entire router evaluation, are not reproducible from this
file. The router is fitted on the public partition and scored once on the held-out partition, and
the held-out partition is not released.
"""
from __future__ import annotations

import json

import paths as PATHS

_M = json.loads(PATHS.INTERVENTION_MATRIX.read_text(encoding="utf-8"))
CONDITIONS = _M["conditions"]
AUX = _M["aux"]
IDS = sorted(_M["predictions"])

DISPLAY = {
    "pot_direct": "Direct PoT",
    "pot_function": "Function-RAG PoT",
    "pot_graph": "FunctionGraph-RAG PoT",
    "pot_verifier": "Verifier on FunctionGraph-RAG PoT",
    "cot_direct": "Direct CoT",
    "cot_function_bm25": "Function-RAG CoT (BM25)",
    "cot_function_judge": "Function-RAG CoT (judge)",
    "cot_graph": "FunctionGraph-RAG CoT",
    "cot_graph_nojudge": "FunctionGraph-RAG CoT, no judge",
}


def prediction(condition: str, item_id: str) -> str:
    return _M["predictions"][item_id][CONDITIONS.index(condition)]


def selected_count(item_id: str):
    """Functions the Function-RAG judge injected. The RQ2 stratifying variable."""
    return AUX[item_id]["judge_selected_count"]
