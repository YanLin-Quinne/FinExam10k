from __future__ import annotations

from typing import Any, Iterable

try:
    from .core import canonical_hash
except ImportError as exc:                                  # pragma: no cover
    from . import upstream_required
    raise upstream_required("`core`") from exc


FEATURE_NAMES = ("reciprocal_rank", "reciprocal_depth", "edge_weight", "degree_scale")
FEATURE_SCHEMA_HASH = canonical_hash({"version": 1, "features": FEATURE_NAMES})


def stable_deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or candidate.get("source_id") or candidate.get("id") or "")


def feature_vector(candidate: dict[str, Any]) -> list[float]:
    rank = max(0, int(candidate.get("retrieval_rank", 0)))
    depth = max(0, int(candidate.get("graph_depth", 0)))
    weight = float(candidate.get("edge_weight", 0.0))
    degree = max(0, int(candidate.get("graph_degree", 0)))
    return [1.0 / (rank + 1), 1.0 / (depth + 1), weight, degree / (degree + 1)]


def expand_graph_candidates(top_candidates: list[dict[str, Any]], adjacency: dict[str, list[str]],
                            catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
        for neighbor_id in adjacency.get(root_id, []):
            if neighbor_id in seen or neighbor_id not in catalog:
                continue
            item = dict(catalog[neighbor_id])
            item["candidate_id"] = neighbor_id
            item.setdefault("retrieval_rank", int(root["retrieval_rank"]))
            item["graph_depth"] = 1
            item.setdefault("edge_weight", 1.0)
            item.setdefault("graph_degree", len(adjacency.get(neighbor_id, [])))
            result.append(item)
            seen.add(neighbor_id)
    return result
