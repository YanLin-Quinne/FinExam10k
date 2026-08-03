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


def load_all():
    """(item ids, responses[system][item] -> predicted letter, system names)."""
    R = {m: {i: _M["predictions"][i][k] for i in _M["predictions"]}
         for k, m in enumerate(NAMES)}
    return sorted(_M["predictions"]), R, NAMES


def pred(letter):
    """Responses are already the predicted letter, so this is the identity."""
    return letter or ""


def ok(letter, item_id=None):
    """Correctness needs the item, so scripts should prefer correct() below."""
    return bool(letter) and item_id is not None and letter == GOLD[item_id]


def correct(system: str, item_id: str) -> bool:
    return _M["predictions"][item_id][NAMES.index(system)] == GOLD[item_id]


def load_questions():
    return {r["id"]: r for r in _ITEMS}
