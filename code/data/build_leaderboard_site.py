"""Build the offline leaderboard data from the included aggregate results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "leaderboard_data.json"
DESTINATION = ROOT / "docs" / "data" / "leaderboard.json"

CATEGORY_NAMES = {
    "Proprietary": "Proprietary API served",
    "Open-weight reasoning": "Open weight reasoning",
    "Finance-specialized": "Finance specialized",
}
VIEW_META = {
    "full": ("Full Coverage", "All 10,198 curated items across CFA Levels I–III and FRM Parts I–II."),
    "cc": ("Context Complete", "The 7,625 items with all answer-relevant evidence present in the local record."),
    "h372": ("Context Complete Hard", "The 372 context-complete items in the frozen Hard band."),
    "z188": ("Universal Failure Core", "The 188 items missed by every system in the frozen panel."),
}


def build_payload(source: dict) -> dict:
    views = {}
    for key, (title, description) in VIEW_META.items():
        block = source[key]
        rows = [{
            "model": row["model"],
            "category": CATEGORY_NAMES[row["group"]],
            "accuracy": row["acc"],
            "adjusted_accuracy": row["astar"],
            "agreement": row["agree"],
            "modal_choice": row["own"],
            "modal_share": row["own_share"],
            "stages": {
                "CFA Level I": row["Level I"],
                "CFA Level II": row["Level II"],
                "CFA Level III": row["Level III"],
                "FRM Part I": row["Part I"],
                "FRM Part II": row["Part II"],
            },
        } for row in block["rows"]]
        if len(rows) != 17:
            raise ValueError(f"{key} must contain all 17 systems")
        if {row["category"] for row in rows} != set(CATEGORY_NAMES.values()):
            raise ValueError(f"{key} does not contain the three paper categories")
        views[key] = {
            "title": title,
            "description": description,
            "items": block["n"],
            "chance_accuracy": block["chance"],
            "cfa_items": block["cfa"],
            "frm_items": block["frm"],
            "below_chance_systems": block["below"],
            "unanimous_wrong_items": block["unanimous"],
            "rows": rows,
        }
    return {"schema_version": 1, "categories": list(CATEGORY_NAMES.values()), "views": views}


def main() -> int:
    payload = build_payload(json.loads(SOURCE.read_text(encoding="utf-8")))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DESTINATION.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
