"""Frozen four-feature PoT selector.

The function corpus and Contriever index are external FinanceReasoning assets and are not bundled.
This module scores an already constructed candidate list and is sufficient to inspect the frozen
ranking rule without claiming end-to-end retrieval reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from selector.pot_candidate_protocol import FEATURE_NAMES, candidate_id, feature_vector


@dataclass(frozen=True)
class FrozenPoTSelector:
    weights: tuple[float, ...]
    bias: float
    threshold: float

    @classmethod
    def load(cls, path: Path) -> "FrozenPoTSelector":
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if tuple(artifact["feature_schema"]) != FEATURE_NAMES:
            raise ValueError("unexpected PoT feature schema")
        weights = tuple(float(value) for value in artifact["weights"])
        if len(weights) != len(FEATURE_NAMES):
            raise ValueError("unexpected PoT coefficient count")
        return cls(weights, float(artifact.get("bias", 0.0)), float(artifact.get("threshold", 0.0)))

    def linear_score(self, candidate: dict[str, Any]) -> float:
        return self.bias + sum(weight * value for weight, value in zip(self.weights, feature_vector(candidate)))

    def probability(self, candidate: dict[str, Any]) -> float:
        score = max(-30.0, min(30.0, self.linear_score(candidate)))
        return 1.0 / (1.0 + math.exp(-score))

    def select(self, candidates: list[dict[str, Any]], max_selected: int = 3) -> list[str]:
        if not 0 <= max_selected <= 3:
            raise ValueError("PoT output budget must be between zero and three")
        ranked: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for position, candidate in enumerate(candidates):
            item_id = candidate_id(candidate)
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            score = self.linear_score(candidate)
            if score >= self.threshold:
                ranked.append((-score, position, item_id))
        ranked.sort()
        return [item_id for _, _, item_id in ranked[:max_selected]]
