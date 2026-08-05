"""Shared 27-feature construction for the Direct-conditioned gate.

This module is imported by both training and inference. Keeping one implementation avoids the
feature drift that previously changed routing decisions for numeric answer options.
"""
from __future__ import annotations

import math
import re
from typing import Any

LETTERS = {"A", "B", "C", "D"}
ERROR_CODES = (
    "call_not_allowed",
    "nested_function_not_allowed",
    "function_missing_return",
    "assignment_target_not_allowed",
)
NUMERIC_OPTION_RE = re.compile(
    r"^[^A-Za-z]*[-+]?[\d,]+(\.\d+)?\s*(%|bp|bps|million|billion|x)?[^A-Za-z]*$"
)
COMPUTE_CUE_RE = re.compile(
    r"\b(closest to|calculate|compute|value of|equals|estimate the)\b", re.I
)
JUDGMENT_CUE_RE = re.compile(
    r"\b(most likely|least likely|most appropriate|best describes|which of the following)\b",
    re.I,
)
LEVELS = ("Level I", "Level II", "Level III", "Part I", "Part II")

FEATURE_NAMES = (
    "direct_unparsed",
    "direct_exec_ok",
    "direct_parse_ok",
    "err_call_not_allowed",
    "err_nested_function_not_allowed",
    "err_function_missing_return",
    "err_assignment_target_not_allowed",
    "out_ktok",
    "in_ktok",
    "log_latency",
    "http_attempts",
    "direct_pred_A",
    "direct_pred_B",
    "direct_pred_C",
    "direct_pred_D",
    "is_cfa",
    "lv_LevelI",
    "lv_LevelII",
    "lv_LevelIII",
    "lv_PartI",
    "lv_PartII",
    "n_options",
    "log_stem_len",
    "numeric_options",
    "cue_compute",
    "cue_judgment",
    "digit_density",
)


def build_features(item: dict[str, Any], sidecar_item: dict[str, Any]) -> list[float]:
    """Build the exact unscaled feature vector expected by ``gate_frozen.json``.

    ``sidecar_item`` is one entry from ``per_item_sidecar_public_5110.json`` and must contain
    a ``direct`` block. No gold label, retrieval state, FunctionGraph-RAG output, or cross-branch
    feature is read.
    """
    direct = sidecar_item["direct"]
    prediction = str(direct.get("prediction") or "").strip().upper()
    stem = str(item["content"])
    options = [
        str(option.get("content") if isinstance(option, dict) else option)
        for option in item["options"]
    ]

    values: list[float] = []
    values.extend(
        [
            float(prediction not in LETTERS),
            float(direct.get("executor_status") == "ok"),
            float(direct.get("parser_status") == "ok"),
        ]
    )
    values.extend(float(direct.get("error_code") == code) for code in ERROR_CODES)
    values.extend(
        [
            float(direct.get("output_tokens") or 0) / 1000.0,
            float(direct.get("input_tokens") or 0) / 1000.0,
            math.log1p(float(direct.get("latency_s") or 0.0)),
            float(direct.get("http_attempts") or 1),
        ]
    )
    values.extend(float(prediction == letter) for letter in "ABCD")
    values.append(float(item["exam"] == "CFA"))
    values.extend(float(item["level"] == level) for level in LEVELS)
    values.extend(
        [
            float(len(options)),
            math.log1p(len(stem)) / 10.0,
            float(all(NUMERIC_OPTION_RE.match(option.strip()) for option in options)),
            float(bool(COMPUTE_CUE_RE.search(stem))),
            float(bool(JUDGMENT_CUE_RE.search(stem))),
            float(sum(character.isdigit() for character in stem)) / 100.0,
        ]
    )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(f"built {len(values)} features, expected {len(FEATURE_NAMES)}")
    return values
