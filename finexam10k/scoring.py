"""Answer extraction and fixed-denominator scoring."""
from __future__ import annotations

from collections import defaultdict
import json
import re

from .data import BANDS, STAGES
from .prompt import valid_letters

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*|\s*```$")
_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"\s*([A-Da-d])\s*"')
# the keyword is case-insensitive, the option letter must be a capital A to D (so "the answer is a ..." is not A)
_ANSWER_TEXT = re.compile(r"\b(?i:(?:final\s+)?answer\s*(?:is\s*:?|:))\s*\(?([A-D])\)?(?![A-Za-z])")


def parse_answer(text: str | None) -> str | None:
    """Return the answer letter in a model output, or None.

    The JSON ``answer`` field is read first, then the last ``"answer": "X"`` pair embedded in other text, then the
    last phrase such as ``Answer: X``. Anything else is unparseable and is scored as incorrect.
    """
    if not text:
        return None
    value = _FENCE.sub("", text.strip()).strip()
    try:
        obj = json.loads(value)
        if isinstance(obj, dict):
            answer = str(obj.get("answer") or "").strip().upper()
            return answer if len(answer) == 1 and answer in "ABCD" else None
    except json.JSONDecodeError:
        pass
    matches = _ANSWER_FIELD.findall(value) or _ANSWER_TEXT.findall(value)
    return matches[-1].upper() if matches else None  # the last answer wins over drafts and self-corrections


def _percent(correct: int, total: int) -> float | None:
    return round(100 * correct / total, 2) if total else None


def score(items: list[dict], predictions: dict[str, str | None]) -> dict:
    """Accuracy (%) over all given items, overall and by stage, difficulty band, and context completeness.

    Every item stays in the denominator: a missing prediction or a letter outside the item's options is incorrect.
    """
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    normalized = {key: (str(value).strip().upper() or None) if value is not None else None
                  for key, value in predictions.items()}
    for item in items:
        prediction = normalized.get(item["id"])
        hit = int(prediction in valid_letters(item) and prediction == item["answer"])
        keys = ("overall", f"stage:{item['program_stage']}", f"difficulty:{item['difficulty']}",
                "context_complete" if item["context_complete"] else "context_incomplete")
        for key in keys:
            groups[key][0] += hit
            groups[key][1] += 1
    result = {"n": groups["overall"][1], "accuracy": _percent(*groups["overall"])}
    result["by_stage"] = {s: _percent(*groups[f"stage:{s}"]) for s in STAGES if groups[f"stage:{s}"][1]}
    result["by_difficulty"] = {b: _percent(*groups[f"difficulty:{b}"]) for b in BANDS if groups[f"difficulty:{b}"][1]}
    result["by_context"] = {k: _percent(*groups[k]) for k in ("context_complete", "context_incomplete") if groups[k][1]}
    result["answered"] = sum(1 for item in items if normalized.get(item["id"]) is not None)
    return result
