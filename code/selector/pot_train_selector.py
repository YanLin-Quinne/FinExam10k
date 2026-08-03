from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .core import canonical_hash
from .graph_features import FEATURE_NAMES, FEATURE_SCHEMA_HASH, candidate_id, feature_vector
from .graph_training import _accuracy, _atomic_write, _load_id_manifest, _probability, _split
from .retrieval import (BUPT_REPO_COMMIT, CONTRIEVER_MODEL_ID,
                        CONTRIEVER_REVISION)


SOURCE_DATASET = "BUPT FinanceReasoning"
SOURCE_COMMIT = BUPT_REPO_COMMIT
SOURCE_KIND = "direct_function_relevance_labels"
SOURCE_SPLITS = ("Easy", "Medium")
LABEL_SEMANTICS = "direct_function_id_relevance"
TRAINING_OBJECTIVE = "linear_pairwise_function_relevance"
TOTAL_LABELED_EXAMPLES = 752
ACCEPTED_EXAMPLES = 511
REJECTED_EXAMPLES = 241
HISTORICAL_BM25_ACCEPTED_EXAMPLES = 736
RETRIEVAL_STAGE_EXCLUSIONS = {"label_not_in_candidate_pool": REJECTED_EXAMPLES}
TRAINING_STATISTICS = {
    "total_labeled": TOTAL_LABELED_EXAMPLES,
    "accepted": ACCEPTED_EXAMPLES,
    "rejected": REJECTED_EXAMPLES,
    "retrieval_stage_exclusions": RETRIEVAL_STAGE_EXCLUSIONS,
}
CANDIDATE_PROTOCOL = {
    "lineage": "frozen_contriever_question_only_top30_graph_expansion",
    "query_input": "question_only",
    "model_provenance": {
        "model_id": CONTRIEVER_MODEL_ID,
        "revision": CONTRIEVER_REVISION,
    },
    "pooling": "CLS last_hidden_state[:,0,:]",
    "normalization": False,
    "similarity": "inner_product",
    "index": "faiss.IndexFlatIP",
    "top_k": 30,
    "graph_expansion": {"hops": 1, "order": "after_top_k"},
}
CANDIDATE_PROTOCOL_FINGERPRINT = canonical_hash(CANDIDATE_PROTOCOL)
ROW_KEYS = frozenset({"record_type", "example_id", "split", "candidates",
                      "relevant_function_ids"})
CANDIDATE_KEYS = frozenset({"candidate_id", "retrieval_rank", "graph_depth",
                            "edge_weight", "graph_degree"})
FORBIDDEN_OUTCOME_KEYS = frozenset({
    "answer", "answers", "referenceanswer", "finalanswer", "correct", "correctness",
    "iscorrect", "gold", "goldanswer", "groundtruth", "trajectory", "trajectories",
    "trajectoryoutcome", "outcome", "outcomes", "answerdelta", "hybrid", "reward", "rewards",
})


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def reject_answer_or_outcome_supervision(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if (normalized in FORBIDDEN_OUTCOME_KEYS or "trajectory" in normalized or
                    "answerdelta" in normalized or "correctness" in normalized):
                raise ValueError("answer_or_trajectory_outcome_supervision_forbidden")
            reject_answer_or_outcome_supervision(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_answer_or_outcome_supervision(nested)


def _sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def validate_training_statistics(statistics: object) -> None:
    if not isinstance(statistics, dict) or set(statistics) != set(TRAINING_STATISTICS):
        raise ValueError("contriever_training_statistics_invalid")
    values = [statistics.get(name) for name in ("total_labeled", "accepted", "rejected")]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("contriever_training_statistics_invalid")
    total, accepted, rejected = values
    if accepted == HISTORICAL_BM25_ACCEPTED_EXAMPLES:
        raise ValueError("contriever_accepted_736_is_historical_bm25_lineage")
    exclusions = statistics.get("retrieval_stage_exclusions")
    if (not isinstance(exclusions, dict) or set(exclusions) != set(RETRIEVAL_STAGE_EXCLUSIONS) or
            any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in exclusions.values())):
        raise ValueError("contriever_retrieval_stage_exclusions_invalid")
    if total != accepted + rejected or rejected != sum(exclusions.values()):
        raise ValueError("contriever_training_partition_mismatch")
    if total != TOTAL_LABELED_EXAMPLES:
        raise ValueError("contriever_total_labeled_count_must_be_752")
    if accepted != ACCEPTED_EXAMPLES:
        raise ValueError("contriever_accepted_count_must_be_511")
    if rejected != REJECTED_EXAMPLES or exclusions != RETRIEVAL_STAGE_EXCLUSIONS:
        raise ValueError("contriever_rejected_count_must_be_241")


def validate_candidate_protocol(protocol: object, fingerprint: object) -> None:
    if protocol != CANDIDATE_PROTOCOL:
        raise ValueError("contriever_candidate_protocol_mismatch")
    if fingerprint != CANDIDATE_PROTOCOL_FINGERPRINT:
        raise ValueError("contriever_candidate_protocol_fingerprint_mismatch")


def _load_labels(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("label_jsonl_malformed") from exc
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError("label_jsonl_object_required")
    reject_answer_or_outcome_supervision(values)
    metadata, rows = values[0], values[1:]
    expected_identity = {
        "record_type": "metadata",
        "source_dataset": SOURCE_DATASET,
        "source_commit": SOURCE_COMMIT,
        "training_source_kind": SOURCE_KIND,
        "source_splits": list(SOURCE_SPLITS),
        "label_semantics": LABEL_SEMANTICS,
        "training_objective": TRAINING_OBJECTIVE,
    }
    required = set(expected_identity) | {
        "training_statistics", "candidate_protocol", "candidate_protocol_fingerprint",
        "function_catalog_hash", "graph_artifact_sha256",
    }
    if set(metadata) != required:
        raise ValueError("label_metadata_schema_invalid")
    if any(metadata.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("label_metadata_schema_invalid")
    validate_training_statistics(metadata.get("training_statistics"))
    validate_candidate_protocol(metadata.get("candidate_protocol"),
                                metadata.get("candidate_protocol_fingerprint"))
    if not _sha256(metadata.get("function_catalog_hash")):
        raise ValueError("label_function_catalog_hash_invalid")
    if not _sha256(metadata.get("graph_artifact_sha256")):
        raise ValueError("label_graph_artifact_sha256_invalid")
    if len(rows) != ACCEPTED_EXAMPLES:
        raise ValueError("label_accepted_count_must_be_511")
    return metadata, rows, canonical_hash(values)


def _pairwise_examples(rows: list[dict[str, Any]]) -> tuple[
        list[tuple[str, list[float], int]], str, list[str], dict[str, int]]:
    examples: list[tuple[str, list[float], int]] = []
    catalog_ids: set[str] = set()
    example_ids: list[str] = []
    source_split_counts = {name: 0 for name in SOURCE_SPLITS}
    for row in rows:
        if set(row) != ROW_KEYS or row.get("record_type") != "function_relevance_label":
            raise ValueError("function_relevance_label_schema_invalid")
        example_id = str(row.get("example_id") or "").strip()
        split = str(row.get("split") or "")
        candidates = row.get("candidates")
        relevant = row.get("relevant_function_ids")
        if (not example_id or example_id in example_ids or split not in SOURCE_SPLITS or
                not isinstance(candidates, list) or not isinstance(relevant, list)):
            raise ValueError("function_relevance_label_schema_invalid")
        candidate_vectors: dict[str, list[float]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_KEYS:
                raise ValueError("function_relevance_candidate_schema_invalid")
            item_id = candidate_id(candidate)
            if not item_id or item_id in candidate_vectors:
                raise ValueError("function_relevance_candidate_id_invalid")
            candidate_vectors[item_id] = feature_vector(candidate)
        relevant_ids = {str(item).strip() for item in relevant}
        irrelevant_ids = set(candidate_vectors) - relevant_ids
        if (not relevant_ids or any(not item for item in relevant_ids) or
                not relevant_ids <= set(candidate_vectors) or not irrelevant_ids):
            raise ValueError("direct_function_relevance_labels_invalid")
        for positive_id in sorted(relevant_ids):
            for negative_id in sorted(irrelevant_ids):
                difference = [left - right for left, right in
                              zip(candidate_vectors[positive_id], candidate_vectors[negative_id])]
                examples.append((example_id, difference, 1))
                examples.append((example_id, [-value for value in difference], 0))
        example_ids.append(example_id)
        source_split_counts[split] += 1
        catalog_ids.update(candidate_vectors)
    return examples, canonical_hash(sorted(catalog_ids)), example_ids, source_split_counts


def train_label_supervised_selector(source: Path, output: Path, *,
                                    finexam_evaluation_ids: Path,
                                    seed: int = 202607) -> dict[str, Any]:
    metadata, rows, label_fingerprint = _load_labels(source)
    examples, candidate_id_set_hash, example_ids, source_split_counts = _pairwise_examples(rows)
    finexam_ids, finexam_manifest_sha, finexam_fingerprint = _load_id_manifest(
        finexam_evaluation_ids, kind="finexam_evaluation")
    if set(example_ids) & set(finexam_ids):
        raise ValueError("FinExam_overlap_must_be_zero")
    groups = {name: [item for item in examples if _split(item[0], seed) == name]
              for name in ("train", "validation", "test")}
    weights = [0.0] * len(FEATURE_NAMES)
    bias = 0.0
    train = groups["train"] or examples
    for _ in range(200):
        gradient = [0.0] * len(weights)
        bias_gradient = 0.0
        for _, features, label in train:
            error = _probability(weights, bias, features) - label
            bias_gradient += error
            for index, feature in enumerate(features):
                gradient[index] += error * feature
        scale = 0.1 / len(train)
        weights = [weight - scale * gradient[index] for index, weight in enumerate(weights)]
        bias -= scale * bias_gradient
    payload = {
        "schema_version": 3,
        "source_dataset": SOURCE_DATASET,
        "source_commit": SOURCE_COMMIT,
        "training_source_kind": SOURCE_KIND,
        "source_splits": list(SOURCE_SPLITS),
        "label_semantics": LABEL_SEMANTICS,
        "training_objective": TRAINING_OBJECTIVE,
        "training_statistics": dict(TRAINING_STATISTICS),
        "candidate_protocol": dict(CANDIDATE_PROTOCOL),
        "candidate_protocol_fingerprint": CANDIDATE_PROTOCOL_FINGERPRINT,
        "label_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "label_source_fingerprint": label_fingerprint,
        "finexam_id_manifest_sha256": finexam_manifest_sha,
        "finexam_id_fingerprint": finexam_fingerprint,
        "function_catalog_hash": metadata["function_catalog_hash"],
        "candidate_id_set_hash": candidate_id_set_hash,
        "graph_artifact_sha256": metadata["graph_artifact_sha256"],
        "feature_schema": list(FEATURE_NAMES),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "source_split_counts": source_split_counts,
        "split_counts": {name: len({item[0] for item in values}) for name, values in groups.items()},
        "seed": seed,
        "weights": weights,
        "bias": bias,
        "threshold": 0.0,
        "training_metrics": {name: {"pair_accuracy": _accuracy(values, weights, bias),
                                     "pairs": len(values)} for name, values in groups.items()},
        "finexam_overlap_audit": {"count": 0, "method": "code_computed_id_intersection"},
    }
    payload["artifact_sha256"] = canonical_hash(payload)
    _atomic_write(output, payload)
    return payload
