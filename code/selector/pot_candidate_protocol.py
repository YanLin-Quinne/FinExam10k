"""One-hop PoT candidate expansion and four-feature scoring protocol."""
from __future__ import annotations

from typing import Any, Iterable

FEATURE_NAMES = ("reciprocal_rank", "reciprocal_depth", "edge_weight", "degree_scale")


def stable_deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or candidate.get("source_id") or candidate.get("id") or "")


def feature_vector(candidate: dict[str, Any]) -> list[float]:
    rank = max(0, int(candidate.get("retrieval_rank", 0)))
    depth = max(0, int(candidate.get("graph_depth", 0)))
    edge_weight = float(candidate.get("edge_weight", 0.0))
    degree = max(0, int(candidate.get("graph_degree", 0)))
    return [1.0 / (rank + 1), 1.0 / (depth + 1), edge_weight, degree / (degree + 1)]


def expand_graph_candidates(
    top_candidates: list[dict[str, Any]],
    adjacency: dict[str, list[str]],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect the top-30 roots followed by their unique one-hop neighbours."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(top_candidates):
        item = dict(raw)
        item_id = candidate_id(item)
        if not item_id or item_id in seen:
            continue
        item.setdefault("candidate_id", item_id)
        item.setdefault("retrieval_rank", rank)
        item.setdefault("graph_depth", 0)
        item.setdefault("edge_weight", 1.0)
        item.setdefault("graph_degree", len(adjacency.get(item_id, [])))
        result.append(item)
        seen.add(item_id)
    roots = list(result)
    for root in roots:
        root_id = candidate_id(root)
        for neighbour_id in adjacency.get(root_id, []):
            if neighbour_id in seen or neighbour_id not in catalog:
                continue
            item = dict(catalog[neighbour_id])
            item["candidate_id"] = neighbour_id
            item.setdefault("retrieval_rank", int(root["retrieval_rank"]))
            item["graph_depth"] = 1
            item.setdefault("edge_weight", 1.0)
            item.setdefault("graph_degree", len(adjacency.get(neighbour_id, [])))
            result.append(item)
            seen.add(neighbour_id)
    return result
