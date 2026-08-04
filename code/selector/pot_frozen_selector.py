from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from .core import canonical_hash
    from .graph_features import FEATURE_NAMES, FEATURE_SCHEMA_HASH, candidate_id, feature_vector
    from .label_graph_training import (
        CANDIDATE_PROTOCOL, CANDIDATE_PROTOCOL_FINGERPRINT, LABEL_SEMANTICS,
        SOURCE_COMMIT, SOURCE_DATASET, SOURCE_KIND, SOURCE_SPLITS, TRAINING_OBJECTIVE,
        TRAINING_STATISTICS, reject_answer_or_outcome_supervision,
        validate_candidate_protocol, validate_training_statistics,
    )
except ImportError as exc:                                  # pragma: no cover
    from . import upstream_required
    raise upstream_required("`core`, `graph_features`, `label_graph_training`") from exc


@dataclass(frozen=True)
class FrozenGraphSelector:
    weights: list[float]
    bias: float = 0.0
    threshold: float = 0.5
    training_source_kind: str = "unspecified"

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "FrozenGraphSelector":
        weights = artifact.get("weights")
        if weights is None:
            weights = artifact.get("positive_weight", artifact.get("promotion_weight"))
        if not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES):
            raise ValueError("selector_weights_invalid")
        return cls([float(value) for value in weights], float(artifact.get("bias", 0.0)),
                   float(artifact.get("threshold", 0.5)),
                   str(artifact.get("training_source_kind") or "legacy_label"))

    def score(self, candidate: dict[str, Any]) -> float:
        value = self.bias + sum(weight * feature for weight, feature in zip(self.weights, feature_vector(candidate)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))

    def select(self, candidates: list[dict[str, Any]], *, max_selected: int = 3) -> list[str]:
        if max_selected < 0 or max_selected > 3:
            raise ValueError("max_selected_must_be_at_most_3")
        ranked: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for index, candidate in enumerate(candidates):
            item_id = candidate_id(candidate)
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            score = self.score(candidate)
            if score >= self.threshold:
                ranked.append((-score, index, item_id))
        ranked.sort()
        return [item_id for _, _, item_id in ranked[:max_selected]]


def require_full_selector_artifact(artifact: dict[str, Any], catalog_hash: str,
                                   artifact_path: Path | None = None,
                                   expected_file_sha256: str | None = None,
                                   expected_label_source_fingerprint: str | None = None,
                                   expected_finexam_id_fingerprint: str | None = None,
                                   expected_candidate_protocol_fingerprint: str | None = None,
                                   expected_graph_sha256: str | None = None) -> None:
    reject_answer_or_outcome_supervision(artifact)
    required = {
        "schema_version", "source_dataset", "source_commit", "training_source_kind",
        "source_splits", "label_semantics", "training_objective", "training_statistics",
        "candidate_protocol", "candidate_protocol_fingerprint",
        "label_source_sha256", "label_source_fingerprint", "finexam_id_manifest_sha256",
        "finexam_id_fingerprint", "function_catalog_hash", "candidate_id_set_hash",
        "graph_artifact_sha256", "feature_schema",
        "feature_schema_hash", "source_split_counts", "split_counts", "seed", "weights",
        "bias", "threshold", "training_metrics", "finexam_overlap_audit", "artifact_sha256",
    }
    if set(artifact) != required or artifact.get("schema_version") != 3:
        raise ValueError("selector_artifact_schema_invalid")
    if artifact.get("source_dataset") != SOURCE_DATASET:
        raise ValueError("selector_source_dataset_invalid")
    if artifact.get("training_source_kind") != SOURCE_KIND:
        raise ValueError("selector_must_be_label_supervised")
    if artifact.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("selector_source_commit_invalid")
    if (artifact.get("source_splits") != list(SOURCE_SPLITS) or
            artifact.get("label_semantics") != LABEL_SEMANTICS or
            artifact.get("training_objective") != TRAINING_OBJECTIVE or
            sum((artifact.get("source_split_counts") or {}).values()) !=
            TRAINING_STATISTICS["accepted"] or
            set((artifact.get("source_split_counts") or {})) != set(SOURCE_SPLITS)):
        raise ValueError("selector_label_contract_invalid")
    validate_training_statistics(artifact.get("training_statistics"))
    validate_candidate_protocol(artifact.get("candidate_protocol"),
                                artifact.get("candidate_protocol_fingerprint"))
    if artifact.get("finexam_overlap_audit") != {
            "count": 0, "method": "code_computed_id_intersection"}:
        raise ValueError("FinExam_overlap_must_be_zero")
    if artifact.get("function_catalog_hash") != catalog_hash:
        raise ValueError("selector_catalog_hash_mismatch")
    if artifact.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
        raise ValueError("selector_feature_schema_mismatch")
    if artifact.get("feature_schema") != list(FEATURE_NAMES):
        raise ValueError("selector_feature_schema_mismatch")
    fingerprints = ("label_source_sha256", "label_source_fingerprint",
                    "finexam_id_manifest_sha256", "finexam_id_fingerprint",
                    "candidate_id_set_hash", "graph_artifact_sha256")
    if any(not isinstance(artifact.get(name), str) or len(artifact[name]) != 64
           or any(character not in "0123456789abcdef" for character in artifact[name])
           for name in fingerprints):
        raise ValueError("selector_provenance_fingerprint_invalid")
    if not expected_label_source_fingerprint:
        raise ValueError("selector_expected_label_source_fingerprint_required")
    if artifact.get("label_source_fingerprint") != expected_label_source_fingerprint:
        raise ValueError("selector_label_source_fingerprint_mismatch")
    if not expected_finexam_id_fingerprint:
        raise ValueError("selector_expected_finexam_id_fingerprint_required")
    if artifact.get("finexam_id_fingerprint") != expected_finexam_id_fingerprint:
        raise ValueError("selector_finexam_id_fingerprint_mismatch")
    if not expected_candidate_protocol_fingerprint:
        raise ValueError("selector_expected_candidate_protocol_fingerprint_required")
    if (expected_candidate_protocol_fingerprint != CANDIDATE_PROTOCOL_FINGERPRINT or
            artifact.get("candidate_protocol_fingerprint") !=
            expected_candidate_protocol_fingerprint):
        raise ValueError("selector_candidate_protocol_fingerprint_mismatch")
    if not expected_graph_sha256:
        raise ValueError("selector_expected_graph_sha256_required")
    if artifact.get("graph_artifact_sha256") != expected_graph_sha256:
        raise ValueError("selector_graph_artifact_sha256_mismatch")
    expected_artifact_hash = artifact.get("artifact_sha256")
    content = dict(artifact)
    content.pop("artifact_sha256", None)
    if expected_artifact_hash != canonical_hash(content):
        raise ValueError("selector_artifact_hash_mismatch")
    if artifact_path is not None:
        raw = artifact_path.read_bytes()
        if json.loads(raw) != artifact:
            raise ValueError("selector_artifact_file_mismatch")
        if not expected_file_sha256:
            raise ValueError("selector_expected_file_sha256_required")
        if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
            raise ValueError("selector_file_sha256_mismatch")
