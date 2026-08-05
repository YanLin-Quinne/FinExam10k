"""Released 17-model prediction registry over the public partition."""
from __future__ import annotations

import json

import paths as PATHS

_MATRIX = json.loads(PATHS.RESPONSE_MATRIX.read_text(encoding="utf-8"))
_ITEMS = json.loads(PATHS.PUBLIC_ITEMS.read_text(encoding="utf-8"))
GOLD = {row["id"]: str(row["answer"]).strip().upper() for row in _ITEMS}
QUESTIONS = {row["id"]: row for row in _ITEMS}

NAMES = [system["name"] for system in _MATRIX["systems"]]
INDEX = {name: position for position, name in enumerate(NAMES)}
GROUP_OF = {system["name"]: system["group"] for system in _MATRIX["systems"]}
MODELS = [(name, None) for name in NAMES]
LEVELS = ("Level I", "Level II", "Level III", "Part I", "Part II")
PROPRIETARY = "Proprietary"
OPEN_GENERAL = "Open-weight reasoning"
OPEN_FINANCE = "Finance-specialized"


def load_all():
    responses = {
        name: {item_id: _MATRIX["predictions"][item_id][position]
               for item_id in _MATRIX["predictions"]}
        for position, name in enumerate(NAMES)
    }
    return sorted(_MATRIX["predictions"]), responses, list(NAMES)


def pred(letter: str | None) -> str:
    return str(letter or "").strip().upper()


def ok(letter: str | None, item_id: str | None = None) -> bool:
    return item_id is not None and pred(letter) == GOLD[item_id]


def correct(model: str, item_id: str) -> bool:
    return pred(_MATRIX["predictions"][item_id][INDEX[model]]) == GOLD[item_id]


def prediction(model: str, item_id: str) -> str:
    return pred(_MATRIX["predictions"][item_id][INDEX[model]])


def load_questions():
    return dict(QUESTIONS)
