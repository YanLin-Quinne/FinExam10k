"""Bundle-local stand-in for the 17-system leaderboard registry.

The original registry pointed at seventeen raw run files whose names encode provider account
structure. The bundle ships their predictions over the released public partition instead, and this
module exposes the same surface the analysis scripts already import.

Scope, which matters when reading any number this produces: the released public partition only.
Statistics the paper quotes over all 10,198 items cover both partitions and will not reproduce
here. That gap is the leaderboard protocol working as intended, not a packaging error.
"""
from __future__ import annotations

import json

import paths as PATHS

_M = json.loads(PATHS.RESPONSE_MATRIX.read_text(encoding="utf-8"))
_ITEMS = json.loads(PATHS.PUBLIC_ITEMS.read_text(encoding="utf-8"))
GOLD = {r["id"]: str(r["answer"]).strip().upper() for r in _ITEMS}

NAMES = [s["name"] for s in _M["systems"]]
GROUP_OF = {s["name"]: s["group"] for s in _M["systems"]}
MODELS = [(n, None) for n in NAMES]
PROPRIETARY, OPEN_GENERAL, OPEN_FINANCE = (
    "Proprietary", "Open-weight reasoning", "Finance-specialized")

# Examination stages in curriculum order, which is the order every table and figure prints them in.
# Derived from the released items rather than hard coded, so a stage cannot silently go missing.
_ORDER = ["Level I", "Level II", "Level III", "Part I", "Part II"]
LEVELS = [lv for lv in _ORDER if any(r["level"] == lv for r in _ITEMS)]


def load_all():
    """(questions by id, responses[system][item] -> predicted letter, system names).

    The first element is the item mapping, not a bare id list. Callers that only want the ids use
    `sorted(questions)`, which yields the same sorted keys either way, and callers that need the
    stem or the stage index into it directly.
    """
    R = {m: {i: _M["predictions"][i][k] for i in _M["predictions"]}
         for k, m in enumerate(NAMES)}
    return load_questions(), R, NAMES


def pred(letter):
    """Responses are already the predicted letter, so this is the identity."""
    return letter or ""


def ok(letter, item_id=None):
    """True when `letter` is the gold answer for `item_id`.

    The item id is not optional. An earlier version of this shim defaulted it to None and returned
    False in that case, which turned every one-argument call into a silent zero: the script ran,
    exited 0, and reported that no system answered anything correctly. Raising here means a call
    site that forgets the id fails at once instead of producing a plausible wrong table.
    """
    if item_id is None:
        raise TypeError(
            "ok() needs the item id: correctness is a property of the (prediction, item) pair, "
            "not of the letter alone. Call ok(records[system][item], item), or use "
            "correct(system, item).")
    return bool(letter) and letter == GOLD[item_id]


def correct(system: str, item_id: str) -> bool:
    return _M["predictions"][item_id][NAMES.index(system)] == GOLD[item_id]


def load_questions():
    return {r["id"]: r for r in _ITEMS}
